#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Train on the full-Irishman dataset built by run_build_irishfull.sh (~213k real
# Irish + Lieder + ~650k synthetic crops). Same recipe as the irish50k runs
# (bias4, matched-eval fixes, curriculum), except the epoch schedule:
#
#   1 synthetic + 1 real epoch. At the measured ~6.7 it/s a synthetic epoch on
#   ~650k pieces is ~27h, so the 50k schedule (4+3) cannot fit a 48h walltime.
#   In total training tokens, 1 epoch x 4.3x-data ~= the 50k run's 4 epochs, so
#   this is the token-budget-matched configuration rather than a truncation.
#   Expected wall: ~27h synthetic + ~9h real + eval passes ~= 37h.

set -eo pipefail

eval "$(mamba shell hook --shell bash)"
mamba activate notagen
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export DATA_TRAIN_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_train.jsonl"
export DATA_EVAL_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_eval.jsonl"
export EXP_TAG="1motif_v1_bias4_cropsyn_irishfull"
export NUM_EPOCHS_SYNTHETIC=1
export NUM_EPOCHS_REAL=1

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
