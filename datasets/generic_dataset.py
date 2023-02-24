#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : dataset.py
#   Author       : Yufeng Liu
#   Date         : 2021-04-01
#   Description  : 
#
#================================================================

import numpy as np
import pickle
import torch.utils.data as tudata
import SimpleITK as sitk
import torch
import sys
import os

from swc_handler import parse_swc
from augmentation.generic_augmentation import InstanceAugmentation
from datasets.swc_processing import trim_swc, swc_to_image, trim_out_of_box

# To avoid the recursionlimit error, maybe encountered in trim_swc
sys.setrecursionlimit(30000)

class GenericDataset(tudata.Dataset):

    def __init__(self, split_file, phase='train', imgshape=(256,512,512)):
        self.data_list = self.load_data_list(split_file, phase)
        self.imgshape = imgshape
        print(f'Image shape of {phase}: {imgshape}')
        self.phase = phase
        # augmentations
        self.augment = InstanceAugmentation(p=0.2, imgshape=imgshape, phase=phase)
    
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

        if somafile is not None and self.phase != 'test':
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


        if swcfile is not None and self.phase != 'test':
            tree = parse_swc(swcfile)
        else:
            tree = None

        # random augmentation
        img, tree, soma, _ = self.augment(img, tree, soma_pad)

        if tree is not None and self.phase != 'test':
            # convert swc to image
            # firstly trim_swc via deleting out-of-box points
            tree = trim_out_of_box(tree, img[0].shape, True)
            lab = swc_to_image(tree, soma, imgshape=img[0].shape)
            return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(lab.astype(np.uint8)), imgfile, swcfile
        else:
            lab = np.random.random(img[0].shape) > 0.5
            return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(lab.astype(np.uint8)), imgfile, imgfile
        
        


if __name__ == '__main__':
    split_file = '/PBshare/SEU-ALLEN/Users/Gaoyu/neuronSegSR/Task501_neuron/data_splits.pkl'
    idx = 5
    imgshape = (64,128,128)
    dataset = GenericDataset(split_file, 'train', imgshape=imgshape, is_2d=True)
    img, lab, _, _ = dataset.pull_item(idx)
    print(torch.max(img))
    print(torch.max(lab))
    print(lab.shape)
    print(img.shape)
    import matplotlib.pyplot as plt
    from neuronet.utils.image_util import *
    plt.imshow(unnormalize_normal(img.numpy())[0, 0])
    from pylib.file_io import *
    img = unnormalize_normal(img.numpy()).astype(np.uint8)
    # lab = unnormalize_normal(lab.numpy()).astype(np.uint8)

    # save_image('img.v3draw', img)
    # save_image('lab.v3draw', lab)
     
    plt.savefig('img.png')
    plt.imshow(unnormalize_normal(lab[None].numpy())[0, 0])
    plt.savefig('lab.png')

