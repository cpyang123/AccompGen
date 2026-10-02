#!/bin/bash
# Submit the full ablation-table batch (see notebook/make_ablation_notebooks.py):
#  - 3 training runs (bias0 4+4, nosyn04, bias0+nosyn04)
#  - immediate evals: norecipe (prompt-only), nosyn01 (existing ckpt), base_backbone
#  - evals for the new checkpoints, chained with --dependency=afterok
set -eo pipefail
cd "$(dirname "$0")/.."

MOTIFS=("" _top1 _top2 _top3 _top4 _top5 _mid1 _mid2 _mid3 _m0333 _m0n3)

T_BIAS0=$(sbatch --parsable scripts/fine_tune_irishfull_bias0_4x4.sh)
T_NOSYN=$(sbatch --parsable scripts/fine_tune_irishfull_nosyn04.sh)
T_BOTH=$(sbatch --parsable scripts/fine_tune_irishfull_bias0_nosyn04.sh)
echo "training: bias0=$T_BIAS0 nosyn04=$T_NOSYN both=$T_BOTH"

for m in "${MOTIFS[@]}"; do
    sbatch --parsable scripts/motif_check/run_motif_check_noreal_abl_norecipe${m}_4x4.sh
    sbatch --parsable scripts/motif_check/run_motif_check_noreal_abl_nosyn01${m}_4x4.sh
    sbatch --parsable --dependency=afterok:$T_BIAS0 scripts/motif_check/run_motif_check_noreal_abl_bias0${m}_4x4.sh
    sbatch --parsable --dependency=afterok:$T_NOSYN scripts/motif_check/run_motif_check_noreal_abl_nosyn04${m}_4x4.sh
    sbatch --parsable --dependency=afterok:$T_BOTH scripts/motif_check/run_motif_check_noreal_abl_both${m}_4x4.sh
done
sbatch --parsable scripts/motif_check/run_motif_check_noreal_abl_base_backbone_4x4.sh
