exp_folder="exps/exp0014_ablation_no_tras"
mkdir -p $exp_folder
save_folder=${exp_folder}/evaluation

#export NUM_NODES=1
#export NUM_GPUS_PER_NODE=2
#export NODE_RANK=0
#export WORLD_SIZE=$((NUM_NODES * $NUM_GPUS_PER_NODE))

# launch our script w/ `torch.distributed.launch`
CUDA_VISIBLE_DEVICES=0 nohup \
python -u train.py \
    --deterministic \
    --max_epochs 300 \
    --save_folder ${save_folder} \
    --amp \
    --step_per_epoch 200 \
    --decay_type 'ploy' \
    --test_frequency 3 \
    --phase 'test'\
    --evaluation \
    --checkpoint 'exps/exp003/final_model.pt'\
    --image_shape '64,128,128' \
    --res_rescale '4,1,1' \
    --batch_size 1 \
    --net_config './models/configs/default_config.json'\
    --data_file '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task003_srs_256/data_splits.pkl' \
    > ${exp_folder}/evaluation.log &