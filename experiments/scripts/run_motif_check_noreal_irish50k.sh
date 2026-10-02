#!/bin/bash
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/motifcheck_noreal_irish50k-%j.out
#SBATCH --error=slurm_logs/motifcheck_noreal_irish50k-%j.err
#
# Motif-conditioning check (noreal regime: abstract %motif:v1 only) on the irish50k
# epoch-5 checkpoint. Same motif (0,3,-1,-1), bias4, 50 pieces, Schubert Art Song as
# the irish20k noreal run, so hit-rates are directly comparable. Uses the fixed
# motif/extract.py backbone.
set -eo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate notagen
cd notebook/
jupyter nbconvert --to notebook --execute --allow-errors \
    --ExecutePreprocessor.timeout=4800 \
    --output /usr/xtmp/cy232/accompgen/motifcheck_out/motif_check_noreal_irish50k_executed_${SLURM_JOB_ID}.ipynb \
    motif_check_noreal_irish50k.ipynb
echo "EXECUTED_NB=/usr/xtmp/cy232/accompgen/motifcheck_out/motif_check_noreal_irish50k_executed_${SLURM_JOB_ID}.ipynb"
