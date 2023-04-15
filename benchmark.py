import subprocess
import numpy as np
import os
import glob
from fnmatch import fnmatch, fnmatchcase
from multiprocessing import Process
from multiprocessing.pool import Pool
import shutil

from path_util import *
from swc_handler import *

def runcmd(command, timeout=600):
    ret = subprocess.run(command, shell=True, encoding="utf-8", timeout=timeout)
    if ret.returncode == 0:
        print("success:", ret)
    else:
        print("error:", ret)


class Tracer():
    def __init__(self, vaa3d_path):
        self.vaa3d = vaa3d_path
    
    def app2(self, img_file, swc_name):
        outfile = f'{swc_name}_app2.swc'
        if not os.path.exists(outfile):
            command=f'xvfb-run -a {self.vaa3d} -x libvn2 -f app2 -i {img_file} -o {outfile}'
            runcmd(command)

    def rivulet(self, img_file, swc_name):
        outfile = f'{swc_name}_rivulet.swc'
        if not os.path.exists(outfile):
            command=f'xvfb-run -a {self.vaa3d} -x libRivulet -f tracing_func -i {img_file} -o {outfile} -p 1 40'
            runcmd(command)

    def fmst(self, img_file, swc_name):
        outfile = f'{swc_name}_fmst.swc'
        if not os.path.exists(outfile):
            command=f'xvfb-run -a {self.vaa3d} -x libfastmarching_spanningtree -f tracing_func -i {img_file} -o {outfile}'
            runcmd(command)

    def neutube(self, img_file, swc_name):
        outfile = f'{swc_name}_neutube.swc'
        if not os.path.exists(outfile):
            command=f'xvfb-run -a {self.vaa3d} -x libneuTube -f neutube_trace -i {img_file} -o {outfile}'
            runcmd(command)
    
    def smart_trace(self, img_file, swc_name):
        outfile = f'{swc_name}_smartrace.swc'
        command=f'xvfb-run -a {self.vaa3d} -x libsmartTrace -f smartTrace -i {img_file} -o {outfile}'
        runcmd(command)
    
    def snake(self, img_file, swc_name):
        outfile = f'{swc_name}_snake.swc'
        if not os.path.exists(outfile):
            command=f'xvfb-run -a {self.vaa3d} -x libsnake_tracing -f snake_trace -i {img_file} -o {outfile}'
            runcmd(command)

    def gpstree(self, img_file, swc_name):
        outfile = f'{swc_name}_gpstree.swc'
        if not os.path.exists(outfile):
            command=f'xvfb-run -a {self.vaa3d} -x libNeuroGPSTree -f tracing_func -i {img_file} -o {outfile}'
            runcmd(command)

    def tremap(self, img_file, swc_name):
        outfile = f'{swc_name}_tremap.swc'
        if not os.path.exists(outfile):
            command=f'xvfb-run -a {self.vaa3d} -x libneurontracing_mip -f trace_mip -i {img_file} -o {outfile}'
            runcmd(command)

    def local_trace(self, img_file, swc_name):
        outfile = f'{swc_name}_local_trace.swc'
        # if not os.path.exists(outfile):
        command= f'xvfb-run -a {self.vaa3d} -x liblocal_tracing -f local_tracing -i {img_file} -o {outfile} -p NULL 0.98 1 1 5 15 8 2.4 1'
        runcmd(command)

    def trace(self, img_file, swc_name):
        # self.app2(img_file, swc_name)
        # self.rivulet(img_file, swc_name)
        # self.fmst(img_file, swc_name)
        # self.neutube(img_file, swc_name)
        # self.smart_trace(img_file, swc_name)
        # self.gpstree(img_file, swc_name)
        # self.snake(img_file, swc_name)
        # self.tremap(img_file, swc_name)
        self.local_trace(img_file, swc_name)

    def batch(self, img_dir, out_dir, numThreads=16):
        # print(img_dir)
        args_list = []
        for img_file in glob.glob(os.path.join(img_dir, '*.v3draw')):
            # print(img_file)
            img_name = get_file_prefix(img_file)
            swc_name = os.path.join(out_dir, img_name)
            args = img_file, swc_name
            args_list.append(args)
        pl = Pool(numThreads)
        pl.starmap(self.trace, args_list)
        pl.close()
        pl.join()

def rename(img_dir):
    for imgfile in glob.glob(os.path.join(img_dir, '*.v3draw')):
        imgname = get_file_prefix(imgfile)
        new_name = f'{imgname}_pred.v3draw'
        dir_path = os.path.split(imgfile)[0]
        new_path = os.path.join(dir_path, new_name)
        shutil.move(imgfile, new_path)

def rescale(swc_dir):
    for swcfile in glob.glob(os.path.join(swc_dir, '*_pred_local_trace.swc')):
        new_tree = []
        tree = parse_swc(swcfile)
        swcname = get_file_prefix(swcfile)
        dir_path = os.path.split(swcfile)[0]
        out_path = os.path.join(dir_path, f'{swcname}_rescale.swc')
        for line in tree:
            idx, type_, x, y, z, r, p = line
            new_tree.append((idx, type_, x, y, z/4, r, p))
        write_swc(new_tree, out_path)




if __name__ == '__main__':
    vaa3d_path = '/PBshare/SEU-ALLEN/Users/Gaoyu/bin/start_vaa3d.sh'
    img_dir = '/PBshare/SEU-ALLEN/Users/Gaoyu/SRS/exps/exp0013_ablation_res_128_patch_4/evaluation'
    out_dir = '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/dataset/benchmark/ablation/res_128'
    swc_dir = '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/dataset/benchmark/ablation/res_128'
    # tracer = Tracer(vaa3d_path)
    # tracer.batch(img_dir, out_dir)
    rescale(swc_dir)



