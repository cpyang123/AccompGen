#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# irish50k retrain (r2) on the Jun-30 rebuilt data: same corpus/indices as run 1,
# but the Jun-30 build refreshed the Lieder %motif labels with the fixed extractor
# (run 1 trained before that refresh, so its 1,217 Lieder files carried old-logic
# labels). Everything else identical to fine_tune_irish50k.sh: matched-eval fixes,
# bias4, 4 synthetic + 3 real epochs. New _r2 tag so run 1's checkpoint (the one
# all existing motif-check results reference) is preserved.

set -eo pipefail

eval "$(mamba shell hook --shell bash)"
mamba activate notagen
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export DATA_TRAIN_INDEX_PATH="../data/abcfiles_processed_v1_irish50k_train.jsonl"
export DATA_EVAL_INDEX_PATH="../data/abcfiles_processed_v1_irish50k_eval.jsonl"
export EXP_TAG="1motif_v1_bias4_cropsyn_irish50k_evalfix_r2"
export NUM_EPOCHS_SYNTHETIC=4
export NUM_EPOCHS_REAL=3

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
