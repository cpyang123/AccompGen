#!/bin/bash
#SBATCH -J distill
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err

# Stage 1 patch-encoder feature distillation.
#   sbatch scripts/distill.sh            # precompute teacher feature cache, then train (default)
#   sbatch scripts/distill.sh precompute # only build the /usr/xtmp feature cache
#   sbatch scripts/distill.sh train      # only train the student from an existing cache
# Precompute is resumable: already-cached samples are skipped.

STAGE="${1:-both}"

mamba activate notagen

cd distillation/
CUDA_VISIBLE_DEVICES=0 python distill_patch_encoder.py --stage "$STAGE"
