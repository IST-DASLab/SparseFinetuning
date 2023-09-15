#!/bin/bash

export HF_DATASETS_CACHE=/localhome/ekurtic/hf_cache
# export CUDA_HOME=/nfs/scistore14/alistgrp/ekurtic/miniconda3/envs/llmfoundry/lib/python3.10/site-packages/triton/third_party/cuda

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False

export SPARSITY=40
export WANDB_PROJECT=gsm8k_sp${SPARSITY}
export CUDA_VISIBLE_DEVICES=4,5,6,7

# export MDL=mosaicml/mpt-7b
export MDL=/nfs/scistore14/alistgrp/ekurtic/github/eldarkurtic/neuralmagicml/research/sparsegpt/mpt_7b_base/gsm8k/oneshot_sparsegpt_sp${SPARSITY}/hf_ckpt
export TEACHER=/nfs/scistore14/alistgrp/ekurtic/github/eldarkurtic/llm-foundry/scripts/train/output_dir/mpt-7b_dense_2ep_lr1e-5_bs64/hf_ckpt

# try 5e-5
export ELDAR_HDSTATES_HACK=1
#KDHACK_kdtemp2.0_kdhard0.5_oneshot_sparsegpt_sp40_2ep_lr3e-5_bs64_noGradClip

# try 2, 3 epochs
for NUM_EPOCHS in 2;
do
    for LR in 1e-5 3e-5 5e-5;
    do
        export KD_TEMPERATURE=2.0
        export HARDNESS_CE=0.5
        export HARDNESS_KD_OUT=0.5
        export HARDNESS_KD_LAYERWISE=0.5

        export MAX_DURATION=${NUM_EPOCHS}ep
        export BS=32
        export WARMUP=20ba
        export RUN_NAME=KDall_KDce${HARDNESS_CE}_KDout${HARDNESS_KD_OUT}_KDlayeriwse${HARDNESS_KD_LAYERWISE}_oneshot_sparsegpt_sp${SPARSITY}_${MAX_DURATION}_lr${LR}_bs${BS}_noGradClip_warmup${WARMUP}
        # save_folder
        composer train_sparse_KD.py \
            yamls/finetune/mpt/FT_gsm8k_noGradClip_KDall_smallBS.yaml \
            model_name_or_path=${MDL} \
            max_duration=${MAX_DURATION} \
            run_name=${RUN_NAME} \
            optimizer.lr=${LR} \
            global_train_batch_size=${BS} \
            knowledge_distillation.teacher_name_or_path=${TEACHER} \
            knowledge_distillation.temperature=${KD_TEMPERATURE} \
            knowledge_distillation.hardness_ce=${HARDNESS_CE} \
            knowledge_distillation.hardness_kd_out=${HARDNESS_KD_OUT} \
            knowledge_distillation.hardness_kd_layerwise=${HARDNESS_KD_LAYERWISE} \
            scheduler.t_warmup=${WARMUP}
    done
done
