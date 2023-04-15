

import os, glob, sys
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
from copy import deepcopy
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from multiprocessing.pool import Pool
import pickle
from fnmatch import fnmatch, fnmatchcase

from swc_handler import parse_swc, write_swc
from path_util import get_file_prefix
from file_io import *

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.image_util import normalize_normal

def load_data(data_dir, img_shape=[128,256,256], soma_shape=[64, 128, 128], spacing=(1,1,1), is_train=True):

    # load the spacing file
    img_dir = os.path.join(data_dir, 'img', str(img_shape[1]), 'raw')
    somaseg_dir = os.path.join(data_dir, 'img', str(img_shape[1]), 'somaSeg')
    # get all annotated data
    data_list = []
    for img_file in glob.glob(os.path.join(img_dir, '*.v3draw')):
        swc_dir = os.path.join(data_dir, 'swc', str(img_shape[1]), 'final')
        prefix = get_file_prefix(img_file)
        if is_train:
            swc_file = os.path.join(swc_dir, f'{prefix}.swc')
            soma_file = os.path.join(somaseg_dir, f'{prefix}.v3draw')
        else:
            swc_file = None
            soma_file = None
        data_list.append((img_file, swc_file, soma_file, img_shape, soma_shape, spacing))

    return data_list

def calc_spacing_anisotropy(spacing):
    """
    spacing in (z,y,x) format
    """
    assert spacing[1] == spacing[2] and spacing[1] <= spacing[0], "Spacing in X- and Y-dimension must be the same, and the must smaller than Z-axis"
    spacing_multi = 1.0 * spacing[0] / spacing[2]
    return spacing_multi


class GenericPreprocessor(object):

    def remove_nans(self, data):
        # inplace modification of nans
        print(np.isnan(data))
        data[np.isnan(data)] = 0
        return data

    def _preprocess_sample(self, imgfile, swcfile, somafile, imgfile_out, swcfile_out, somafile_out):
        #if os.path.exists(imgfile_out) and os.path.exists(swcfile_out):
        #    return 
        
        print(f'--> Processing for image: {imgfile}')
        # load the image and annotated tree
        image = load_image(imgfile)
        if image.ndim == 3:
            image = image[None]

        tree = None

        if somafile is not None:
            soma = load_image(somafile)
            if soma.ndim == 3:
                soma = soma[None]
            np.savez_compressed(somafile_out, data=soma.astype(bool))

        if swcfile is not None:
            tree = parse_swc(swcfile)
        # remove nans
        # image = self.remove_nans(image)
        # normalize the image
        image = image.astype(np.float32)
        image = normalize_normal(image, mask=None)
        # write the image and tree as well
        np.savez_compressed(imgfile_out, data=image.astype(np.float32))
        #np.save(imgfile_out, image.astype(np.float32))
        if tree is not None:
            write_swc(tree, swcfile_out)
            #with open(swcfile_out, 'wb') as fp:
            #    pickle.dump(tree, fp)

    def run(self, data_dir, output_dir, img_shape=[128, 256, 256], soma_shape=[64, 128, 128], spacing=(1, 1, 1), is_train=True, num_threads=8):
        print('Processing for dataset, should be run at least once for each dataset!')
        # get all files
        data_list = load_data(data_dir, img_shape, soma_shape, spacing, is_train=is_train)
        print(f'Total number of samples found: {len(data_list)}')

        maybe_mkdir_p(output_dir)
        # execute preprocessing
        args_list = []
        for imgfile, swcfile, somafile, _, _, _ in data_list:
            prefix = get_file_prefix(imgfile)
            #print(imgfile, swcfile)
            imgfile_out = os.path.join(output_dir, f'{prefix}.npz')
            #swcfile_out = os.path.join(output_dir, f'{prefix}.pkl')
            #imgfile_out = os.path.join(output_dir, f'{prefix}.npy')
            swcfile_out = os.path.join(output_dir, f'{prefix}.swc')
            somafile_out = os.path.join(output_dir, f'{prefix}_soma.npz')
            args = imgfile, swcfile, somafile, imgfile_out, swcfile_out, somafile_out
            args_list.append(args)

        # execute in parallel
        pt = Pool(num_threads)
        pt.starmap(self._preprocess_sample, args_list)
        pt.close()
        pt.join()

    def dataset_split(self, task_dir, val_ratio=0.1, test_ratio=0.1, seed=1024, img_ext='.npz', lab_ext='.swc', is_train=True):
        samples = []
        for imgfile in glob.glob(os.path.join(task_dir, f'*{img_ext}')):
            if fnmatch(imgfile, f'*soma{img_ext}'):
                continue
            if is_train:
                labfile = f'{imgfile[:-len(img_ext)]}{lab_ext}'
                somafile = f'{imgfile[:-len(lab_ext)]}_soma.npz'
            else:
                labfile = None
                somafile = None
            samples.append((imgfile, labfile, somafile))
        # data splitting
        num_tot = len(samples)
        num_val = int(round(num_tot * val_ratio))
        num_test = int(round(num_tot * test_ratio))
        num_train = num_tot - num_val - num_test
        print(f'Number of samples of total/train/val/test are {num_tot}/{num_train}/{num_val}/{num_test}')
        
        np.random.seed(seed)
        np.random.shuffle(samples)
        splits = {}
        splits['train'] = samples[:num_train]
        splits['val'] = samples[num_train:num_train+num_val]
        splits['test'] = samples[-num_test:]
        # write to file
        with open(os.path.join(output_dir, 'data_splits.pkl'), 'wb') as fp:
            pickle.dump(splits, fp)
        

        
if __name__ == '__main__':
    data_dir = '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/dataset/benchmark/packed_neurites'
    output_dir = '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task007_srs_packed'
    is_train = False
    num_threads = 16
    gp = GenericPreprocessor()
    gp.run(data_dir, output_dir, img_shape=[32, 64, 64], is_train=is_train, num_threads=num_threads)
    gp.dataset_split(output_dir, val_ratio=0.0, test_ratio=1.0, seed=1024, img_ext='.npz', lab_ext='.swc')
    

