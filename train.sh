#!/bin/bash

exp_folder="exps/exp0015_ablation_no_seg"
mkdir -p $exp_folder


#export NUM_NODES=1
#export NUM_GPUS_PER_NODE=2
#export NODE_RANK=0
#export WORLD_SIZE=$((NUM_NODES * $NUM_GPUS_PER_NODE))

# launch our script w/ `torch.distributed.launch`
CUDA_VISIBLE_DEVICES=1 nohup \
python -u train.py \
    --deterministic \
    --max_epochs 300 \
    --save_folder ${exp_folder}/debug \
    --amp \
    --step_per_epoch 200 \
    --decay_type 'ploy' \
    --test_frequency 3 \
    --image_shape '32,64,64' \
    --res_rescale '1,1,1' \
    --gce_q 0.2 \
    --batch_size 16 \
    --net_config './models/configs/transunet_config.json'\
    --data_file '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task003_srs_256/data_splits.pkl' \
    > ${exp_folder}/train.log &
