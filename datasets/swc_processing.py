

import os, glob, sys
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from copy import deepcopy
import SimpleITK as sitk
from multiprocessing.pool import Pool
import pickle
from skimage.draw import line_nd
import skimage.morphology as morphology

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swc_handler import parse_swc, write_swc
from path_util import get_file_prefix


def is_in_box(x, y, z, imgshape):
    """
    imgshape must be in (z,y,x) order
    """
    if x < 0 or y < 0 or z < 0 or \
        x > imgshape[2] - 1 or \
        y > imgshape[1] - 1 or \
        z > imgshape[0] - 1:
        return False
    return True


def trim_out_of_box(tree_orig, imgshape, keep_candidate_points=True):
    """ 
    Trim the out-of-box leaves
    """
    # execute trimming
    child_dict = {}
    for leaf in tree_orig:
        if leaf[-1] in child_dict:
            child_dict[leaf[-1]].append(leaf[0])
        else:
            child_dict[leaf[-1]] = [leaf[0]]
    
    pos_dict = {}
    for i, leaf in enumerate(tree_orig):
        pos_dict[leaf[0]] = leaf

    tree = []
    for i, leaf in enumerate(tree_orig):
        idx, type_, x, y, z, r, p = leaf
        ib = is_in_box(x,y,z,imgshape)
        if ib: 
            tree.append(leaf)
        elif keep_candidate_points:
            if p in pos_dict and is_in_box(*pos_dict[p][2:5], imgshape):
                tree.append(leaf)
            elif idx in child_dict:
                for ch_leaf in child_dict[idx]:
                    if is_in_box(*pos_dict[ch_leaf][2:5], imgshape):
                        tree.append(leaf)
                        break
    return tree


def swc_to_image(tree, soma, r_exp=(1, 3, 3), imgshape=(256,512,512), flipy=False):
    # Note imgshape in (z,y,x) order
    # initialize empty image
    img = np.zeros(shape=imgshape, dtype=np.uint8)
    # get the position tree and parent tree
    pos_dict = {}
    soma_node = None
    for i, leaf in enumerate(tree):
        idx, type_, x, y, z, r, p = leaf
        if p == -1:
            soma_node = leaf
        
        leaf = (idx, type_, x, y, z, r, p, is_in_box(x,y,z,imgshape))
        pos_dict[idx] = leaf
        tree[i] = leaf
        
    xl, yl, zl = [], [], []
    for _, leaf in pos_dict.items():
        idx, type_, x, y, z, r, p, ib = leaf
        if idx == 1: continue   # soma
       
        if p not in pos_dict: 
            continue
        parent_leaf = pos_dict[p]
        if (not ib) and (not parent_leaf[ib]):
            print('All points are out of box! do trim_swc before!')
            raise ValueError
        
        # draw line connect each pair
        cur_pos = leaf[2:5]
        par_pos = parent_leaf[2:5]
        lin = line_nd(cur_pos[::-1], par_pos[::-1], endpoint=True)

        xl.extend(list(lin[2]))
        yl.extend(list(lin[1]))
        zl.extend(list(lin[0]))

    xn, yn, zn = [], [], []
    for (xi,yi,zi) in zip(xl,yl,zl):
        if is_in_box(xi,yi,zi,imgshape):
            xn.append(xi)
            yn.append(yi)
            zn.append(zi)
    img[zn, yn, xn] = 1

    # do morphology expansion
    selem = np.ones(r_exp, dtype=np.uint8)
    for z in range(r_exp[0]):
        if z == 1:
            selem[z, 0, r_exp[2] - 1] = 0
            selem[z, r_exp[1] - 1, 0] = 0
            selem[z, 0, 0] = 0
            selem[z, r_exp[1] - 1, r_exp[2] - 1] = 0
        else:
            selem[z, ...] = 0
            selem[z, 1, 1] = 1

    img = morphology.dilation(img, selem)
    img += soma[0]

    if flipy:
        img = img[:, ::-1]

    return img.astype(bool).astype(np.uint8)
    

if __name__ == '__main__':
    from file_io import *
    imgfile = f'/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/dataset/img/256/raw/18465_26006.02_11073.60_5371.90.v3draw'
    img = load_image(imgfile)
    img = np.repeat(img, 3, axis=0)
    print(img.shape)

    swc_file = f'/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/dataset/swc/256/final/18465_26006.02_11073.60_5371.90.swc'
    tree = parse_swc(swc_file)
    new_tree = []
    for i, leaf in enumerate(tree):
        idx, type_, x, y, z, r, p = leaf
        new_tree.append((idx, type_, x, y, z*4, r, p))
    somafile = f'/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/dataset/img/256/somaSeg/18465_26006.02_11073.60_5371.90.v3draw'
    soma = load_image(somafile)[0]
    imgshape = (128,256,256)
    soma_pad = np.zeros(imgshape, dtype=soma.dtype)
    center = [dim // 2 for dim in imgshape]
    soma_shape = soma.shape
    soma_pad[center[0] - soma_shape[0] // 2 : center[0] + soma_shape[0] // 2,
    center[1] - soma_shape[1] // 2 : center[1] + soma_shape[1] // 2,
    center[2] - soma_shape[2] // 2 : center[2] + soma_shape[2] // 2] = soma
    soma_pad = soma_pad[None]
    new_soma_pad = resize(soma_pad, (512,256,256), order=0, mode='edge', anti_aliasing=False)
    lab_img = swc_to_image(new_tree, new_soma_pad, imgshape=(512,256,256))
    print(lab_img.shape)
    lab_img = lab_img*255
    lab_img = lab_img[None]
    lab_img = 255 - lab_img
    save_image('/home/freewill/lab_18465_26006.v3draw', lab_img)
    # sitk.WriteImage(sitk.GetImageFromArray(lab_img), f'{prefix}_label.tiff')

