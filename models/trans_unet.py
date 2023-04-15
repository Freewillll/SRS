import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys, os
import copy
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from modules import ConvDropoutNonlinNorm, ConvDropoutNormNonlin, Upsample

def pair3d(t):
    return t if isinstance(t, tuple) else (t, t, t)

def posemb_sincos_1d(seq, temperature = 10000, dtype = torch.float32):
    _, n, dim, device, dtype = *seq.shape, seq.device, seq.dtype

    n = torch.arange(n, device = device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)


def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, d, w, h, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(d, device = device),
        torch.arange(w, device = device),
        torch.arange(h, device = device),
        indexing='ij'
    )

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)


class UpBlock(nn.Module):
    def __init__(self, up_in_channels, in_channels=None, out_channels=None, up_stride=(2,2,2), has_nonlin=True):
        super(UpBlock, self).__init__()
        self.has_nonlin = has_nonlin
        self.up = nn.ConvTranspose3d(up_in_channels, out_channels, up_stride, stride=up_stride)

        if in_channels is None:
            conv_in_channels = out_channels
            self.skip_input = False
        else:
            self.skip_input = True
            conv_in_channels = out_channels + in_channels

        if has_nonlin:
            self.conv = nn.Sequential(
                    ConvDropoutNormNonlin(conv_in_channels, out_channels), 
                    ConvDropoutNormNonlin(out_channels, out_channels))

    def forward(self, x, x_skip=None):
        x = self.up(x)
        if x_skip is not None and self.skip_input:
            x = torch.cat((x, x_skip), dim=1)
        if self.has_nonlin:
            x = self.conv(x)

        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_kernel=(3,3,3), down_stride=(2,2,2)):
        super(DownBlock, self).__init__()
        padding = tuple((k-1)//2 for k in down_kernel)
        down_kwargs = {
            'kernel_size': down_kernel,
            'stride': down_stride,
            'padding': padding,
            'dilation': 1,
            'bias': True,
        }
        self.down = ConvDropoutNormNonlin(in_channels, out_channels, conv_kwargs=down_kwargs)
        conv_kwargs = copy.deepcopy(down_kwargs)
        conv_kwargs['stride'] = 1
        self.conv = ConvDropoutNormNonlin(out_channels, out_channels, conv_kwargs=conv_kwargs)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Encoderlayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Encoder(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64):
        super().__init__()
        image_depth, image_width, image_height = pair3d(image_size)
        patch_depth, patch_width, patch_height = pair3d(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0 and image_depth % patch_depth == 0,\
            'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_depth // patch_depth) * (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_depth * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (d p1) (w p2) (h p3) -> b d w h (p1 p2 p3 c)', p1=patch_depth, p2=patch_width, p3=patch_height),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.layers = Encoderlayer(dim, depth, heads, dim_head, mlp_dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        pe = posemb_sincos_3d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe
        x = self.layers(x)
        x = self.norm(x)
        # x = x.mean(dim=1)
        return x


class BottleNeck(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.proj = ConvDropoutNormNonlin(dim, self.head_dim, norm_op=nn.BatchNorm3d)

    def forward(self, x):
        _, n, dim = x.shape
        patch_size = int(np.round(np.power(n, 1/3)))
        x = rearrange(x, 'b (d h w) dim -> b dim d h w', d=patch_size, h=patch_size)
        x = self.proj(x)
        return x


class TransUnet(nn.Module):
    def __init__(self, in_channels, base_num_filters, scale_factor, head_dim, class_num, down_kernel_list, stride_list, patch_size,
                 dim, depth, heads, dim_head, mlp_dim, img_shape):
        super(TransUnet, self).__init__()
        assert len(down_kernel_list) == len(stride_list)
        self.downs = []

        # the first layer to process the input image
        self.pre_layer = nn.Sequential(
            ConvDropoutNormNonlin(in_channels, base_num_filters, norm_op=nn.BatchNorm3d),
            ConvDropoutNormNonlin(base_num_filters, base_num_filters, norm_op=nn.BatchNorm3d)
        )
        patch_size = tuple(patch_size)
        in_channels = base_num_filters
        out_channels = 2*base_num_filters
        down_filters = []
        self.down_d = 1
        self.down_h = 1
        self.down_w = 1
        for i in range(len(down_kernel_list)):
            down_kernel = down_kernel_list[i]
            stride = stride_list[i]
            self.down_d *= stride[0]
            self.down_w *= stride[1]
            self.down_h *= stride[2]
            down_filters.append((in_channels, out_channels))
            down = DownBlock(in_channels, out_channels, down_kernel=down_kernel, down_stride=stride)
            self.downs.append(down)
            in_channels = out_channels
            out_channels = out_channels * 2

        out_channels = int(out_channels / 2)
        *_, d, w, h = img_shape
        d = int(d / self.down_d)
        h = int(h / self.down_h)
        w = int(w / self.down_w)
        self.encoder = Encoder(
            image_size=(d, w, h),
            patch_size=patch_size,
            channels=out_channels,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim
        )
        self.bottleneck = BottleNeck(dim, head_dim)

        self.ups = []
        down_filters.append((out_channels, head_dim))
        in_channels, up_in_channels = down_filters[-1]
        out_channels = down_filters[-1][0]
        final_channels = -1
        down_kernel_list.append([3,3,3])
        stride_list.append([2,2,2])
        for i in range(len(down_kernel_list)-1, -1, -1):
            stride = stride_list[i]
            if i == 0:
                final_channels = min(up_in_channels // 2, 8 * class_num)
                self.ups.append(UpBlock(up_in_channels, None, final_channels, up_stride=stride))
            else:
                self.ups.append(UpBlock(up_in_channels, in_channels, out_channels, up_stride=stride))

            in_channels, up_in_channels = down_filters[i-1]
            out_channels = down_filters[i-1][0]

        self.downs = nn.ModuleList(self.downs)
        self.ups = nn.ModuleList(self.ups)
        self.class_conv = nn.Conv3d(final_channels, class_num, 1)
        self.rescale = Upsample(scale_factor=scale_factor, mode='trilinear')


    def forward(self, x):
        assert x.ndim == 5
        skip_feats = []
        x = self.pre_layer(x)
        ndown = len(self.downs)

        for i in range(ndown):
            x = self.downs[i](x)
            skip_feats.append(x)

        x = self.encoder(x)
        x = self.bottleneck(x)

        for i in range(ndown):
            x = self.ups[i](x, skip_feats[ndown-i-1])

        x = self.ups[-1](x)
        x = self.rescale(x)
        x = self.class_conv(x)
        return x


if __name__ == '__main__':
    import json
    from torchinfo import summary

    conf_file = 'configs/transunet_config.json'
    with open(conf_file) as fp:
        configs = json.load(fp)
    print('Initialize model...')
    #model = UNet(in_channels, base_num_filters, class_num, down_kernel_list, stride_list, num_side_loss, output_bias=output_bias, direct_supervision=direct_supervision)

    input = torch.randn(1, configs['in_channels'], 64,128,128)
    model = TransUnet(**configs)
    # print(model)

    output = model(input)
    print('output size: ', output.shape)

    summary(model, input_size=(1,1,64,128,128))