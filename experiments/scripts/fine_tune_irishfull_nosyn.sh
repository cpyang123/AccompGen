#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Full-Irishman long run: 4 synthetic + 4 real epochs (vs the 1+1 token-matched
# run tagged ..._irishfull). At measured rates (~24.5h/synthetic epoch,
# ~8.6h/real epoch) this needs ~133h ~= 5.5 days; the partition allows 90 days
# and its 4-day DEFAULT is too short, so an explicit 8-day limit is set.
# New _4x4 tag so the 1+1 checkpoint is preserved for comparison.

set -eo pipefail

eval "$(mamba shell hook --shell bash)"
mamba activate notagen
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export DATA_TRAIN_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_train.jsonl"
export DATA_EVAL_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_eval.jsonl"
export EXP_TAG="1motif_v1_bias4_cropsyn_irishfull_nosyn"
export NUM_EPOCHS_SYNTHETIC=0
export NUM_EPOCHS_REAL=1

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
