#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Fine-tune on the transformation-aware multi-length dataset (built 2026-08-31):
# motif identification folds inversion / retrograde / retrograde-inversion into
# the first-seen canonical motif, and the %motif:abc preamble carries an example
# of each transformation that occurs ("... I: <ex> R: <ex> RI: <ex>").
# Same recipe as the multilen4to10 run otherwise: lengths 4-10, 21 crops/piece,
# bias4, 2 synthetic + 5 real epochs. New tag so the previous multilen4to10
# checkpoint is untouched. 4-day walltime avoids the 48h 2-job split.

set -eo pipefail

eval "$(mamba shell hook --shell bash)"
mamba activate notagen

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export EXP_TAG="multilen4to10_trans_v1_bias4_syn2real5"
export WANDB_CACHE_DIR=/usr/xtmp/cy232/accompgen/wandb_cache

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
