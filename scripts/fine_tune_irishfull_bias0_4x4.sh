#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=8-00:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Ablation "- attention bias": identical to the 4x4 full run
# (fine_tune_irishfull_4x4.sh: 4 synthetic + 4 real epochs, full-Irishman
# corpus) except MOTIF_ATTENTION_BIAS=0 at training, so the motif prompt line
# is present but receives no attention emphasis. Matched eval/inference for
# this checkpoint also uses bias 0 (motif_check_noreal_abl_bias0_*_4x4).

set -eo pipefail

eval "$(mamba shell hook --shell bash)"
mamba activate notagen
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export DATA_TRAIN_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_train.jsonl"
export DATA_EVAL_INDEX_PATH="../data/abcfiles_processed_v1_irishfull_eval.jsonl"
export EXP_TAG="1motif_v1_bias0_cropsyn_irishfull_4x4"
export MOTIF_ATTENTION_BIAS=0
export NUM_EPOCHS_SYNTHETIC=4
export NUM_EPOCHS_REAL=4

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
