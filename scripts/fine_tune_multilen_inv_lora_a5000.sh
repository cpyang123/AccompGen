#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# LoRA fine-tune on the inversion-format multi-length dataset (built 2026-09-10):
#   %motif:v1:step_skip_leap: <pattern>
#   %motif:count: <rectus occurrences>
#   %motif:abc: <rectus excerpt>
#   %motif:inversion_count: <inversion occurrences>
#   %motif:abc:inversion_instance: <inversion excerpt>   (only when count > 0)
# Same recipe otherwise (lengths 4-10, 21 crops/piece, bias4, 2 syn + 5 real
# epochs) but with USE_LORA=1: base weights frozen, rank-16 adapters on the GPT2
# attention/MLP projections of both encoder and decoder, LR 2e-4. Checkpoints hold
# the merged weights under 'model' (inference loaders unchanged) + 'lora' adapter.

set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate notagen
python -c "import torch, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'peft', peft.__version__)"

export EXP_TAG="multilen4to10_inv_v1_bias4_syn2real5"
export USE_LORA=1
export LORA_R=16
export LORA_ALPHA=32
export LEARNING_RATE=2e-4
export WANDB_CACHE_DIR=/usr/xtmp/cy232/accompgen/wandb_cache

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 21.6 GiB peak on a 24 GiB card: avoid fragmentation OOMs
cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
