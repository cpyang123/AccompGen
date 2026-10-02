#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/.."
T=$(sbatch --parsable scripts/fine_tune_irishfull_bias0_nosyn010.sh)
echo "training: $T"
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_both10ep_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_both10ep_top1_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_both10ep_top2_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_both10ep_top3_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_both10ep_top4_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_both10ep_top5_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_both10ep_mid1_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_both10ep_mid2_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_both10ep_mid3_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_both10ep_m0333_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_both10ep_m0n3_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r001.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r002.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r004.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r005.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r007.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r008.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r009.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r010.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r011.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r012.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r013.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r018.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r020.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r022.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r023.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r028.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r038.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r063.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r072.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r087.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r103.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r116.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r119.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_r121.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_both10ep_base.sh
