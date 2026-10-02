#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Train on the isolated 50k-Irish dataset built by run_build_irish50k.sh. Uses the
# same curriculum + matched-eval fixes (synthetic-only eval; bias-on eval in the real
# phase) as the irish20k_evalfix run, just pointed at the irish50k indices via env.
#
# Leaner epoch schedule: the synthetic set is ~2.5x larger than the 20k build, so 10
# synthetic epochs would exceed walltime. Synthetic eval bottomed ~epoch 3 and the real
# phase overfits after ~2 epochs (see irish20k logs), so 4 synthetic + 3 real epochs is
# both walltime-safe (~28h) and consistent with where eval actually bottoms.

set -eo pipefail

eval "$(mamba shell hook --shell bash)"
mamba activate notagen
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export DATA_TRAIN_INDEX_PATH="../data/abcfiles_processed_v1_irish50k_train.jsonl"
export DATA_EVAL_INDEX_PATH="../data/abcfiles_processed_v1_irish50k_eval.jsonl"
export EXP_TAG="1motif_v1_bias4_cropsyn_irish50k_evalfix"
export NUM_EPOCHS_SYNTHETIC=4
export NUM_EPOCHS_REAL=3

cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
