# Library for Sparse-Finetuning of LLMs with support for Knowledge Distillation

This is a modified version of [MosaicML/llmfoundry](https://github.com/mosaicml/llm-foundry) library with support for sparse-finetuning of LLMs and with support for knowledge distillation (KL-divergence and layerwise SquareHead). It has been used to produce results for MPT-7B and GSM8K dataset in the paper [Sparse Finetuning for Inference Acceleration of Large Language Models](https://arxiv.org/abs/2310.06927).

# Repository structure
The main python script is [`scripts/train/train_sparse.py`](https://github.com/IST-DASLab/SparseFinetuning/blob/main/scripts/train/train_sparse.py). It is a modified version of llmfoundry's [`scripts/train/train.py`](https://github.com/IST-DASLab/SparseFinetuning/blob/main/scripts/train/train.py) with support for keeping the fixed mask of the sparse model and two variants of knowledge distillation (KL-divergence and layerwise SquareHead, as described in the paper).

# How to create environment for this project?
```bash
1. conda create --name sparse_finetuning python=3.10 -y
2. conda activate sparse_finetuning
3. git clone git@github.com:IST-DASLab/SparseFinetuning.git
4. cd SparseFinetuning
5. pip install -e .
```

# How to reproduce results from the paper?
```bash
1. conda activate sparse_finetuning
2. cd SparseFinetuning/llmfoundry/scripts/train
3. bash scripts/mpt/run_sparse_finetune.sh  <-- look here for precise hyperparams
```
- Results in the paper were obtained with the following versions of libraries:
```
pytorch = 2.0.1 py3.10_cuda11.8_cudnn8.7.0_0
transformers = 4.31.0
```

# How to evaluate models via [EleutherAI/lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)?
```bash
1. conda create --name lm_eval_harness python=3.10 -y
2. conda activate lm_eval_harness
3. git clone git@github.com:EleutherAI/lm-evaluation-harness.git (we ran evals with master/2c18e367c6ded428863cd1fd4cf9558ca49d68dc commit)
4. cd lm-evaluation-harness
5. pip install -e .
6. run the following bash script with your sparse model (30mins on a single A6000, occupies ~15GB of GPU memory)
```

```bash
export CUDA_VISIBLE_DEVICES=0

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=<path_to_your_sparse_model>,use_accelerate=True,dtype=bfloat16 \
    --tasks gsm8k \
    --num_fewshot 0 \
    --batch_size 1 \
    --write_out \
    --output_base_path results/<my_logs> \
    --no_cache \
    --device cuda
```

# Citation info
```
@article{kurtic2023sparse,
  title={Sparse Finetuning for Inference Acceleration of Large Language Models},
  author={Kurtic, Eldar and Kuznedelev, Denis and Frantar, Elias and Goin, Michael and Alistarh, Dan},
  journal={arXiv preprint arXiv:2310.06927},
  year={2023}
}
```
