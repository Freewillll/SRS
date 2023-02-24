import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys
import os 

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from datasets.generic_dataset import *



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
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, memory=None, attn_mask=None):
        x = self.norm(x)

        if memory is None:
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        else:
            q = self.to_q(x)
            k = self.to_k(memory)
            v = self.to_v(memory)
            q, k, v = map(lambda t: rearrange(q, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        #  q, k, v dim:  b, h, n, d
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #  dots dim:  b, h, n, n
        if attn_mask is not None:
            assert attn_mask.size() == dots.size()
            dots.masked_fill_(attn_mask, float("-inf"))

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
        return x 


class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)


    def forward(self, x, im_size):
        D, W, H = im_size
        p_h = H // self.patch_size
        p_w = W // self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (d w h) c -> b c d w h", h=p_h, w=p_w)
        return x


class SRS(nn.Module):
    def __init__(self, in_channels, patch_size, n_cls, dim, depth, heads, dim_head, mlp_dim, img_shape):
        super(SRS, self).__init__()

        *_, d, w, h = img_shape
        self.im_size = (d, w, h)

        self.encoder = Encoder(
            image_size=(d, w, h),
            patch_size=patch_size,
            channels=in_channels,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim
        )

        self.decoder = DecoderLinear(
            n_cls= n_cls,
            d_encoder=dim,
            patch_size=patch_size
        )


    def forward(self, img):
        assert img.ndim == 5
        x = self.encoder(img)
        masks = self.decoder(x, self.im_size)
        masks = F.interpolate(masks, size=self.im_size, mode="bilinear")
        return masks

if __name__ == '__main__':
    from torchinfo import summary
    import json
    conf_file = 'configs/default_config.json'
    with open(conf_file) as fp:
        configs = json.load(fp)
    print('Initialize model...')

    img = torch.randn(2, 1, 32, 64, 64)
    seq = torch.randn(2, 10, 8, 4)
    model = NTT(**configs)
    print(model)
    outputs = model(img, seq)

    for output in outputs:
        print('output size: ', output.size())

    summary(model, input_size=[(2, 1, 32, 64, 64), (2, 10, 8, 4)])