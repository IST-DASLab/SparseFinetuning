# Library for Sparse-Finetuning of LLMs with support for Knowledge Distillation

This is a modified version of [MosaicML/llmfoundry](https://github.com/mosaicml/llm-foundry) library with support for sparse-finetuning of LLMs and with support for knowledge distillation (KL-divergence and layerwise SquareHead).

# How to create environment for this project?
```bash
1. conda create --name sparse_finetuning python=3.10 -y
2. conda activate sparse_finetuning
3. git clone git@github.com:eldarkurtic/llm-foundry.git
4. cd SparseFinetuning
5. pip install -e .
```

# How to reproduce the results?
```bash
1. conda activate sparse_finetuning
2. cd SparseFinetuning/llmfoundry
3. bash scripts/train/scripts/mpt/run_sparse_finetune.sh
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
