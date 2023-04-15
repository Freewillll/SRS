#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : evaluation.py
#   Author       : Yufeng Liu
#   Date         : 2021-05-11
#   Description  : 
#
#================================================================

import os, sys, glob
import numpy as np
import subprocess
from skimage.draw import line_nd
from scipy.spatial import distance_matrix

from swc_handler import parse_swc, write_swc, is_in_box

def tree_to_voxels(tree, crop_box):
    # initialize position dict
    pos_dict = {}
    new_tree = []
    for i, leaf in enumerate(tree):
        idx, type_, x, y, z, r, p = leaf
        leaf_new = (*leaf, is_in_box(x,y,z,crop_box))
        pos_dict[leaf[0]] = leaf_new
        new_tree.append(leaf_new)
    tree = new_tree

    xl, yl, zl = [], [], []
    for _, leaf in pos_dict.items():
        idx, type_, x, y, z, r, p, ib = leaf
        if p == -1: continue # soma
        
        if p not in pos_dict:
            continue

        parent_leaf = pos_dict[p]
        if (not ib) and (not parent_leaf[ib]):
            print('All points are out of box! do trim_swc before!')
            raise ValueError

        # draw line connecting each pair
        cur_pos = leaf[2:5]
        par_pos = parent_leaf[2:5]
        lin = line_nd(cur_pos[::-1], par_pos[::-1], endpoint=True)
        xl.extend(list(lin[2]))
        yl.extend(list(lin[1]))
        zl.extend(list(lin[0]))

    voxels = []
    for (xi,yi,zi) in zip(xl,yl,zl):
        if is_in_box(xi,yi,zi,crop_box):
            voxels.append((xi,yi,zi))
    # remove duplicate points
    voxels = np.array(list(set(voxels)), dtype=np.float32)
    return voxels
     

def get_specific_neurite(tree, type_id):
    if (not isinstance(type_id, list)) and (not isinstance(type_id, tuple)):
        type_id = (type_id,)
    
    new_tree = []
    for leaf in tree:
        if leaf[1] in type_id:
            new_tree.append(leaf)
    return new_tree

class DistanceEvaluation(object):
    def __init__(self, crop_box, neurite_type='all'):
        self.crop_box = crop_box
        self.neurite_type = neurite_type

    def calc_ESA(self, voxels1, voxels2, dist_type):
        if len(voxels1) > 200000 or len(voxels2) > 200000:
            if dist_type in ('ESA', 'DSA'):
                return 0, 198., 99.0
            elif dist_type == 'PDS':
                return 0.0, 1.0, 0.5
        elif len(voxels1) == 0:
            return None
            #if dist_type in ('ESA', 'DSA'):
            #    return 0., 198., 99.
            #elif dist_type == 'PDS':
            #    return 0., 1., .5
        elif len(voxels2) == 0:
            return None
            #if dist_type in ('ESA', 'DSA'):
            #    return 198., 0., 99.
            #elif dist_type == 'PDS':
            #    return 1., 0., .5

        pdist = distance_matrix(voxels1, voxels2)
        dists1 = pdist.min(axis=1)  # 
        dists2 = pdist.min(axis=0)
        if dist_type == 'DSA':
            dists1 = dists1[dists1 > 2.0]
            dists2 = dists2[dists2 > 2.0]
            if dists1.shape[0] == 0:
                dists1 = np.array([0.])
            if dists2.shape[0] == 0:
                dists2 = np.array([0.])
        elif dist_type == 'PDS':
            dists1 = (dists1 > 2.0).astype(np.float32)
            dists2 = (dists2 > 2.0).astype(np.float32)
        print(f'Distance shape: {dists1.shape}, {dists2.shape}')
        esa1 = dists1.mean()
        esa2 = dists2.mean()
        esa = (esa1 + esa2) / 2.
        return esa1, esa2, esa
    
    def calc_precision(self, voxels1, voxels2):
        if len(voxels1) == 0:
            return None
        elif len(voxels2) == 0:
            return None
        
        pdist = distance_matrix(voxels1, voxels2)
        dist1 = pdist.min(axis=0)
        dist2 = pdist.min(axis=1)
        tp = (dist1 <= 2.0).sum()
        fp = (dist1 > 2.0).sum()
        fn = (dist2 > 2.0).sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall  / (precision + recall)
        return precision, recall, f1


    def calc_DIADEM(self, swc_file1, swc_file2, jar_path='/home/freewill/Diadem/DiademMetric.jar'):
        exec_str = f'java -jar {jar_path} -G {swc_file1} -T {swc_file2} -x 6 -R 3 -z 2 --xyPathThresh 0.08 --zPathThresh 0.20 --excess-nodes false'
        #print(exec_str)
        output = subprocess.check_output(exec_str, shell=True)
        #print(output)
        score1 = float(output.split()[-1])

        exec_str = f'java -jar {jar_path} -G {swc_file2} -T {swc_file1} -x 6 -R 3 -z 2 --xyPathThresh 0.08 --zPathThresh 0.20 --excess-nodes false -r 17'
        output = subprocess.check_output(exec_str, shell=True)
        print(output)
        score2 = float(output.split()[-1])

        score = (score1 + score2) / 2.
        return score1, score2, score
        

    def calc_distance(self, swc_file1, swc_file2, dist_type='ESA'):
        if dist_type in ('ESA', 'DSA', 'PDS'):
            tree1 = parse_swc(swc_file1)
            tree2 = parse_swc(swc_file2)
            print(f'Length of nodes in tree1 and tree2: {len(tree1)}, {len(tree2)}')
            if self.neurite_type == 'all':
                pass
            elif self.neurite_type == 'dendrite':
                type_id = (3,4)
                tree1 = get_specific_neurite(tree1, type_id)
            elif self.neurite_type == 'axon':
                type_id = 2
                tree1 = get_specific_neurite(tree1, type_id)
            else:
                raise NotImplementedError

            # to successive voxels
            voxels1 = tree_to_voxels(tree1, self.crop_box)
            voxels2 = tree_to_voxels(tree2, self.crop_box)
            dist = self.calc_ESA(voxels1, voxels2, dist_type=dist_type)
        elif dist_type == 'DIADEM':
            dist = self.calc_DIADEM(swc_file1, swc_file2)
        elif dist_type == 'F1':
            tree1 = parse_swc(swc_file1)
            tree2 = parse_swc(swc_file2)
            print(f'Length of nodes in tree1 and tree2: {len(tree1)}, {len(tree2)}')

            # to successive voxels
            voxels1 = tree_to_voxels(tree1, self.crop_box)
            voxels2 = tree_to_voxels(tree2, self.crop_box)
            dist = self.calc_precision(voxels1, voxels2)
        else:
            raise NotImplementedError

        return dist

    def calc_distance_triple(self, swc_gt, swc_cmp1, swc_cmp2, dist_type='ESA'):
        if dist_type in ('ESA', 'DSA', 'PDS'):
            tree_gt = parse_swc(swc_gt)
            tree_cmp1 = parse_swc(swc_cmp1)
            tree_cmp2 = parse_swc(swc_cmp2)
            print(f'Length of nodes for gt, cmp1 and cmp2: {len(tree_gt)}, {len(tree_cmp1)}, {len(tree_cmp2)}')
            if self.neurite_type == 'all':
                pass
            elif self.neurite_type == 'dendrite':
                type_id = (3,4)
                tree_gt = get_specific_neurite(tree_gt, type_id)
            elif self.neurite_type == 'axon':
                type_id = 2
                tree_gt = get_specific_neurite(tree_gt, type_id)
            else:
                raise NotImplementedError

            # to successive voxels
            voxels_gt = tree_to_voxels(tree_gt, self.crop_box).astype(np.float32)
            voxels_cmp1 = tree_to_voxels(tree_cmp1, self.crop_box).astype(np.float32)
            voxels_cmp2 = tree_to_voxels(tree_cmp2, self.crop_box).astype(np.float32)
            dist1 = self.calc_ESA(voxels_gt, voxels_cmp1, dist_type=dist_type)
            dist2 = self.calc_ESA(voxels_gt, voxels_cmp2, dist_type=dist_type)
        elif dist_type == 'DIADEM':
            dist1 = self.calc_DIADEM(swc_gt, swc_cmp1)
            dist2 = self.calc_DIADEM(swc_gt, swc_cmp2)
        else:
            raise NotImplementedError
        
        return dist1, dist2

if __name__ == '__main__':

    crop_box = (128,256,256)
    gt_dir = f'/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/dataset/benchmark/gold_standard'
    result_dir = f'/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/dataset/benchmark/ablation/res_128'
    # pred_dir = f'/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/dataset/benchmark/proposed'

    for dist_type in ['ESA']:
        print(f'dist_type: {dist_type}')
        dists_app2 = []
        dists_pred = []
        n_success_app2 = 0
        n_success_pred = 0
        deval = DistanceEvaluation(crop_box)
        
        for swc in glob.glob(os.path.join(result_dir, '*rescale.swc')):
            swc_name = os.path.split(swc)[-1][:-4]
            img_name = swc_name[:-len('_test_pred_local_trace_rescale')]
            print(f'----- {img_name}-------')
            # for swc_gt in glob.glob(os.path.join(gt_dir, f'{img_name}*')):
            
            swc_gt = os.path.join(gt_dir, f'{img_name}.swc')

            try:
                dist_app2 = deval.calc_distance(swc_gt, swc, dist_type=dist_type)
                # dist_app2 = deval.calc_DIADEM()
            except:
                continue

            print(f'{dist_type} distance between test and gt: {dist_app2}')
            print(f'\n')
        
            if dist_app2 is not None:
                n_success_app2 += 1
                dists_app2.append(dist_app2)

        dists_app2 = np.array(dists_app2)
        print(f'Succeed number of proposed are: {n_success_app2}\n')
        print(f'Statistics for proposed: ')
        print(f'mean: ')
        print(f'    {dists_app2.mean(axis=0)}')
        print(f'std: ')
        print(f'    {dists_app2.std(axis=0)}\n')


#  diadem  app2    [0.26983933 0.35440533 0.31212233] mean   
#                  [0.22186778 0.27893651 0.23116059] std

# pds app2
# mean:
#    [0.23034471 0.48184582 0.35609527]
# std:
#    [0.22177585 0.29547983 0.21288479]


#esa proposed
# mean [2.39953981 4.24924248 3.32439114]
#std: [4.78371582 5.43279647 4.25797183]

# diadem proposed
# mean:
#    [0.59373067 0.59351774 0.59362421]
# std:
#    [0.1669646  0.17083899 0.14647336]

# dsa proposed
# mean:
#    [10.80608358 12.76183424 11.78395891]
# std:
#    [7.98137738 9.23317871 6.20515978]

# pds proposed 
# mean:
#    [0.12327574 0.20762758 0.16545166]
# std:
#    [0.08789356 0.14932639 0.08967303]


#  pds neutube
# mean:
#    [0.21184603 0.3190764  0.26546121]
# std:
#    [0.24471449 0.16526642 0.14224005]

# esa neutube
# mean:
#    [5.43094541 7.58975096 6.51034819]
# std:
#    [10.40283699  6.12871363  5.65059203]

# dsa neutube
# mean:
#    [10.4388319  18.72998662 14.58440926]
# std:
#    [11.57546049  9.48572179  6.22075249]


# diadem fmst
# mean:
#    [0.35106207 0.35122069 0.35114138]
# std:
#    [0.19160207 0.23578881 0.19108864]

# esa fmst:
# mean:
#    [6.2304406  6.99020784 6.61032422]
# std:
#    [8.76132152 6.43322147 5.08706311]

# dsa fmst
# mean:
#    [10.83033073 15.29675743 13.06354408]
# std:
#    [ 9.22714144 10.15571773  5.83341762]

# pds fmst
# mean: 
#    [0.39114911 0.36402655 0.37758783]
# std:
#    [0.20032817 0.17045727 0.15359904]

# f1 fmst
# mean:
#    [0.63597345 0.59724869 0.59054628]
# std:
#    [0.17045727 0.20287206 0.18865691]



# pds rivulet
# mean:
#    [0.4260075 0.3539239 0.3899657]
# std:
#    [0.25858005 0.15238819 0.15904175]

# esa rivulet
# mean:
#    [12.30764802  5.00555111  8.65659956]
# std:
#    [15.0932224   4.93303281  7.80276346]