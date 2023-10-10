#!/bin/bash

# ==== Hyperparams to reproduce results from the paper ====

# 1) Unstructured sparsity results in Figure 4 and Table 3:
# sparsity=40% -> NUM_EPOCHS=2 and LR=3e-5
# sparsity=50% -> NUM_EPOCHS=2 and LR=3e-5
# sparsity=60% -> NUM_EPOCHS=2 and LR=1e-4
# sparsity=70% -> NUM_EPOCHS=4 and LR=8e-5
# sparsity=80% -> NUM_EPOCHS=4 and LR=1e-4

# 2) N:M sparsity results in Table 4:
# sparsity=2:4   -> NUM_EPOCHS=4 and LR=1e-4
# sparsity=16:32 -> NUM_EPOCHS=4 and LR=1e-4
# sparsity=16:64 -> NUM_EPOCHS=4 and LR=1e-4

export CUDA_VISIBLE_DEVICES=0,1,2,3

export SPARSITY=40
export NUM_EPOCHS=2

export WANDB_DISABLED=False
export WANDB_ENTITY=<your_wandb_entity>
export WANDB_PROJECT=<your_wandb_project>

export SPARSE_MDL=<path_to_your_sparse_model>
export TEACHER_MDL=<path_to_your_teacher_model>

export HARDNESS_CE=1.0
export HARDNESS_SQUAREHEAD=1.0
export HARDNESS_KLDIV=0.0
export KLDIV_TEMPERATURE=1.0

export LR=3e-5
export WARMUP=20ba
export MAX_DURATION=${NUM_EPOCHS}ep
export BS=32
export PER_DEVICE_BS=8
export RUN_NAME=<your_name_for_the_run>

composer train_sparse.py \
    yamls/finetune/mpt/sparse_finetune_with_distillation.yaml \
    model_name_or_path=${SPARSE_MDL} \
    max_duration=${MAX_DURATION} \
    run_name=${RUN_NAME} \
    optimizer.lr=${LR} \
    global_train_batch_size=${BS} \
    device_train_microbatch_size=${PER_DEVICE_BS} \
    device_eval_batch_size=${PER_DEVICE_BS} \
    knowledge_distillation.teacher_name_or_path=${TEACHER_MDL} \
    knowledge_distillation.hardness_ce=${HARDNESS_CE} \
    knowledge_distillation.hardness_kldiv=${HARDNESS_KLDIV} \
    knowledge_distillation.temperature=${KLDIV_TEMPERATURE} \
    knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD} \
    eval_first=True \
    scheduler.t_warmup=${WARMUP}
