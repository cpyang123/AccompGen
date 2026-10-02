#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Ablation "- synthetic phase", schedule-matched to the 4x4 full run: 0
# synthetic + 4 real epochs, bias 4. The earlier ..._irishfull_nosyn
# checkpoint trained only 1 real epoch and so conflates the removed phase with
# a shorter schedule; this run holds the real-epoch count at the full model's
# 4. At ~8.6h/real epoch: ~35h + eval passes, so a 3-day limit.

set -eo pipefail

eval "$(mamba shell hook --shell bash)"
mamba activate notagen
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export DATA_TRAIN_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_train.jsonl"
export DATA_EVAL_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_eval.jsonl"
export EXP_TAG="1motif_v1_bias4_irishfull_nosyn04"
export NUM_EPOCHS_SYNTHETIC=0
export NUM_EPOCHS_REAL=4

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
