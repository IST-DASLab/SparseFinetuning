#!/bin/bash

# in case it can't find your cuda installation, uncomment the following line with your path:
# export CUDA_HOME=/nfs/scistore14/alistgrp/ekurtic/miniconda3/envs/llmfoundry/lib/python3.10/site-packages/triton/third_party/cuda

# in case you store datasets somewhere else, uncomment the following line with your path:
# export HF_DATASETS_CACHE=/ssdpool/eldar/hf_cache

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False

export NUM_EPOCHS=2
export BS=32
export WARMUP_STEPS=20
export OPTIM=decoupled_adamw
export LR=1e-4
export PACKING_RATIO=13
export SPARSITY=0.7
export SPARSE_CKPT_PATH=/nfs/scistore14/alistgrp/ekurtic/to_be_removed/oneshot_sparsegpt_sp${SPARSITY}_nsamples128.pt

export RUN_NAME=sparsegpt_sp${SPARSITY}_${OPTIM}_LR${LR}_warm${WARMUP_STEPS}_noGradNormClipping_bf16_${NUM_EPOCHS}ep_bs${BS}_wPacking${PACKING_RATIO}
export WANDB_PROJECT=mpt-7b-instruct_sparsefinetune_${SPARSITY}

composer train_sparse.py yamls/finetune/sparse_finetune_wPacking.yaml load_path=${SPARSE_CKPT_PATH} run_name=${RUN_NAME}