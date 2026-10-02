#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# v2 ablation on the 10k-subset corpus (all Lieder + 10k Irishman = the
# default v1 index, 19k real + 57k synthetic rows): bias=0,
# 4 synthetic + 4 real epochs. Schedule-matched to the 4x4 protocol.
set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate notagen
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
export DATA_TRAIN_INDEX_PATH="../data/abcfiles_processed_v1_train.jsonl"
export DATA_EVAL_INDEX_PATH="../data/abcfiles_processed_v1_eval.jsonl"
export EXP_TAG="1motif_v1_bias0_cropsyn_irish10k_44"
export MOTIF_ATTENTION_BIAS=0
export NUM_EPOCHS_SYNTHETIC=4
export NUM_EPOCHS_REAL=4
cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
