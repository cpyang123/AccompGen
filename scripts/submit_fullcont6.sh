#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/.."
T=$(sbatch --parsable scripts/fine_tune_irishfull_4x4_cont6.sh)
echo "training: $T"
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_fullcont6_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_fullcont6_top1_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_fullcont6_top2_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_fullcont6_top3_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_fullcont6_top4_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_fullcont6_top5_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_fullcont6_mid1_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_fullcont6_mid2_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_fullcont6_mid3_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_fullcont6_m0333_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_abl_fullcont6_m0n3_4x4.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r001.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r002.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r004.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r005.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r007.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r008.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r009.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r010.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r011.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r012.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r013.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r018.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r020.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r022.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r023.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r028.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r038.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r063.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r072.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r087.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r103.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r116.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r119.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_r121.sh
sbatch --parsable --dependency=afterok:$T scripts/motif_check/run_motif_check_ablv2_fullcont6_base.sh
