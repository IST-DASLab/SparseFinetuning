#!/bin/bash

#export HF_DATASETS_CACHE=/localhome/ekurtic/hf_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False

export WANDB_PROJECT=test_sparse

export MDL=/nfs/scistore14/alistgrp/ekurtic/seafile-client/seafile/eldar_random_5/oneshot_sparsegpt_sp50/hf_ckpt 

export RUN_NAME=test_sparse

composer train_sparse_KD.py \
	yamls/finetune/mpt/test_sparse.yaml \
	model_name_or_path=${MDL} \
	max_duration=1ep \
	run_name=${RUN_NAME} \
	optimizer.lr=5e-5 \
	global_train_batch_size=16 \
	device_train_microbatch_size=4 \
	device_eval_batch_size=4 \
	eval_first=True \
	scheduler.t_warmup=20ba
