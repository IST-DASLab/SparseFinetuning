#!/bin/bash

export HF_DATASETS_CACHE=/localhome/ekurtic/hf_cache
# export CUDA_HOME=/nfs/scistore14/alistgrp/ekurtic/miniconda3/envs/llmfoundry/lib/python3.10/site-packages/triton/third_party/cuda

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False

export SPARSITY=50
export WANDB_PROJECT=gsm8k_sp${SPARSITY}
export CUDA_VISIBLE_DEVICES=4,5,6,7

# export MDL=mosaicml/mpt-7b
export MDL=/nfs/scistore14/alistgrp/ekurtic/github/eldarkurtic/neuralmagicml/research/sparsegpt/mpt_7b_base/gsm8k/oneshot_sparsegpt_sp${SPARSITY}/hf_ckpt
# export MDL=meta-llama/Llama-2-7b-chat-hf

# try 5e-5
for NUM_EPOCHS in 1 2 3 4;
do
    for LR in 5e-5;
    do
        export MAX_DURATION=${NUM_EPOCHS}ep
        export BS=64
        export RUN_NAME=oneshot_sparsegpt_sp${SPARSITY}_${MAX_DURATION}_lr${LR}_bs${BS}
        # save_folder
        composer train_sparse_KD.py yamls/finetune/mpt/FT_gsm8k_noGradClip.yaml model_name_or_path=${MDL} max_duration=${MAX_DURATION} run_name=${RUN_NAME} optimizer.lr=${LR} global_train_batch_size=${BS}
    done
done
