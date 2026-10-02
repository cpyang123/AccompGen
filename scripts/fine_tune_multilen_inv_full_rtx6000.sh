#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# FULL-parameter fine-tune on the inversion-format multi-length dataset (built
# 2026-09-10; same data the LoRA run 12569798 used):
#   %motif:v1:step_skip_leap: <pattern>
#   %motif:count: <rectus occurrences>
#   %motif:abc: <rectus excerpt>
#   %motif:inversion_count: <inversion occurrences>
#   %motif:abc:inversion_instance: <inversion excerpt>   (only when count > 0)
# Same recipe as the multilen4to10 / trans full fine-tunes (lengths 4-10,
# 21 crops/piece, bias4, 2 synthetic + 5 real epochs, LR from config), so the
# result is directly comparable to those and to the LoRA variant. USE_LORA is
# off, so NAME has no _lora16 suffix and the LoRA checkpoint is untouched.
#
# GPU: RTX Pro 6000 (Blackwell, sm_120). The notagen env's torch 2.3/cu11.8 has
# no Blackwell kernels, so this runs in notagen-bw (torch 2.9.1+cu128, same
# transformers/peft/numpy pins otherwise), built 2026-09-14 on xtmp.

set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate /usr/xtmp/cy232/envs/notagen-bw

# Fail fast if the GPU/torch pairing is wrong, before loading 500k files.
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0), 'archs', torch.cuda.get_arch_list())
x = torch.randn(256, 256, device='cuda'); y = (x @ x).sum().item()
assert y == y, 'CUDA matmul produced NaN'
print('cuda matmul ok')
PY

export EXP_TAG="multilen4to10_inv_v1_bias4_syn2real5"
export USE_LORA=0
export WANDB_CACHE_DIR=/usr/xtmp/cy232/accompgen/wandb_cache

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
