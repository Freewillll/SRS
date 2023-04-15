

import numpy as np
import pickle
import torch.utils.data as tudata
import SimpleITK as sitk
import torch
import sys, os
from skimage.transform import resize

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swc_handler import parse_swc
from augmentation.generic_augmentation import InstanceAugmentation
from datasets.swc_processing import swc_to_image, trim_out_of_box

# To avoid the recursionlimit error, maybe encountered in trim_swc
sys.setrecursionlimit(30000)

class GenericDataset(tudata.Dataset):

    def __init__(self, split_file, phase='train', imgshape=(256,512,512), res_rescale=(4,1,1)):
        self.data_list = self.load_data_list(split_file, phase)
        self.imgshape = imgshape
        print(f'Image shape of {phase}: {imgshape}')
        self.phase = phase
        # augmentations
        self.augment = InstanceAugmentation(p=0.2, imgshape=imgshape, phase=phase)
        self.res_rescale=res_rescale
    
    @staticmethod
    def load_data_list(split_file, phase):
        # define helper function for single soma extraction

        with open(split_file, 'rb') as fp:
            data_dict = pickle.load(fp)
        #return data_dict[phase]

        if phase == 'train' or phase == 'val':
            return data_dict[phase]

        elif phase == 'test':
            dd = data_dict['test']
            # filtering, for debug only!
            #new_dd = []
            #for di in dd:
            #    prefix = os.path.splitext(os.path.split(di[0])[-1])[0]
            #    if not os.path.exists(f'/media/data/lyf/SEU-ALLEN/neuronet_1741/debug_{prefix}_test_img.tiff'):
            #        new_dd.append(di)
            #dd = new_dd
            #print(len(dd))

            ##list_file = './data/img_singleSoma.list'
            #list_file = './data/additional_crops/single_soma.list'
            #dd = extract_single_soma(dd, list_file)
            return dd 
        else:
            raise ValueError

    def __getitem__(self, index):
        img, gt, imgfile, swcfile = self.pull_item(index)
        return img, gt, imgfile, swcfile

    def __len__(self):
        return len(self.data_list)

    def pull_item(self, index):
        imgfile, swcfile, somafile = self.data_list[index]
        # parse, image should in [c,z,y,x] format

        img = np.load(imgfile)['data']

        if img.ndim == 3:
            img = img[None]
        
        if self.phase == 'test':
            hr_shape = tuple([img[0].shape[i] * self.res_rescale[i] for i in range(3)])
            lab = np.random.random(hr_shape) > 0.5
            return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(lab.astype(np.uint8)), imgfile, imgfile
        else:
            if somafile is not None:
                soma = np.load(somafile)['data']      #  0 or 1
                soma_pad = np.zeros(img.shape, dtype=soma.dtype)
                center = [dim // 2 for dim in img[0].shape]
                soma_shape = soma[0].shape

                soma_pad[:, center[0] - soma_shape[0] // 2 : center[0] + soma_shape[0] // 2,
                center[1] - soma_shape[1] // 2 : center[1] + soma_shape[1] // 2,
                center[2] - soma_shape[2] // 2 : center[2] + soma_shape[2] // 2] = soma

                if soma_pad.ndim == 3:
                    soma_pad = soma_pad[None]
            else:
                soma_pad = None


            if swcfile is not None:
                tree = parse_swc(swcfile)
            else:
                tree = None

            # random augmentation
            img, tree, soma = self.augment(img, tree, soma_pad)
            new_shape = tuple([img[0].shape[i] * self.res_rescale[i] for i in range(3)])

            if tree is not None:
                # convert swc to image
                # firstly trim_swc via deleting out-of-box points
                tree = trim_out_of_box(tree, img[0].shape, True)
                # rescale for super resolution
                new_tree = []
                for line in tree:
                    idx, type_, x, y, z, r, p = line
                    x = x * self.res_rescale[2]
                    y = y * self.res_rescale[1]
                    z = z * self.res_rescale[0]
                    new_tree.append((idx, type_, x, y, z, r, p))
                new_soma = np.zeros((soma.shape[0], *new_shape), dtype=soma.dtype)
                for c in range(soma.shape[0]):
                    new_soma[c] = resize(soma[c], new_shape, order=0, mode='edge', anti_aliasing=False)

                lab = swc_to_image(new_tree, new_soma, r_exp=(3,3,3),imgshape=new_shape)
                return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(lab.astype(np.uint8)), imgfile, swcfile    


if __name__ == '__main__':
    split_file = '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task003_srs_256/data_splits.pkl'
    idx = 0
    imgshape = (32,64,64)
    dataset = GenericDataset(split_file, 'train', imgshape=imgshape)
    img, lab, _, _ = dataset.pull_item(idx)
    print(torch.max(img))
    print(torch.max(lab))
    print(lab.shape)
    print(img.shape)
    import matplotlib.pyplot as plt
    from utils.image_util import *
    plt.imshow(unnormalize_normal(img.numpy())[0, 0])
    from file_io import *
    img = unnormalize_normal(img.numpy()).astype(np.uint8)
    lab = unnormalize_normal(lab.numpy()).astype(np.uint8)

    save_image('img.v3draw', img)
    save_image('lab.v3draw', lab)
     
    # plt.savefig('img.png')
    # # plt.imshow(unnormalize_normal(lab[None].numpy())[0, 0])
    # plt.savefig('lab.png')

