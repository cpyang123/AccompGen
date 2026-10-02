#!/bin/bash
# v3 matched-bias ablation matrix (make_ablation_v3.py).
set -eo pipefail
cd "$(dirname "$0")/.."
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r001.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r002.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r004.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r005.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r007.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r008.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r009.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r010.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r011.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r012.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r013.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r018.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r020.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r022.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r023.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r028.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r038.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r063.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r072.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r087.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r103.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r116.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r119.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_r121.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full_base.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r001.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r002.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r004.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r005.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r007.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r008.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r009.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r010.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r011.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r012.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r013.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r018.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r020.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r022.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r023.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r028.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r038.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r063.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r072.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r087.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r103.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r116.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r119.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_r121.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias0_base.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r001.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r002.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r004.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r005.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r007.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r008.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r009.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r010.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r011.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r012.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r013.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r018.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r020.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r022.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r023.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r028.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r038.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r063.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r072.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r087.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r103.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r116.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r119.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_r121.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn_base.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r001.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r002.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r004.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r005.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r007.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r008.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r009.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r010.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r011.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r012.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r013.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r018.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r020.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r022.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r023.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r028.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r038.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r063.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r072.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r087.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r103.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r116.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r119.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_r121.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both_base.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r001.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r002.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r004.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r005.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r007.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r008.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r009.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r010.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r011.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r012.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r013.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r018.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r020.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r022.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r023.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r028.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r038.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r063.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r072.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r087.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r103.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r116.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r119.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_r121.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_full10k_base.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r001.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r002.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r004.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r005.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r007.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r008.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r009.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r010.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r011.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r012.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r013.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r018.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r020.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r022.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r023.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r028.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r038.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r063.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r072.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r087.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r103.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r116.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r119.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_r121.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_bias010k_base.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r001.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r002.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r004.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r005.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r007.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r008.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r009.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r010.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r011.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r012.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r013.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r018.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r020.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r022.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r023.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r028.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r038.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r063.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r072.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r087.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r103.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r116.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r119.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_r121.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_nosyn10k_base.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r001.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r002.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r004.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r005.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r007.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r008.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r009.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r010.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r011.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r012.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r013.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r018.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r020.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r022.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r023.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r028.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r038.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r063.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r072.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r087.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r103.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r116.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r119.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_r121.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv3_both10k_base.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r001.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r002.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r004.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r005.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r007.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r008.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r009.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r010.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r011.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r012.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r013.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r018.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r020.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r022.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r023.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r028.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r038.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r063.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r072.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r087.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r103.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r116.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r119.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_r121.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_base.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_v1.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_v1_top1.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_v1_top2.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_v1_top3.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_v1_top4.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_v1_top5.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_v1_mid1.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_v1_mid2.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_v1_mid3.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_v1_m0333.sh
sbatch --parsable --dependency=afterany:12279620 scripts/motif_check/run_motif_check_ablv3_both10ep_v1_m0n3.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r001.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r002.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r004.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r005.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r007.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r008.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r009.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r010.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r011.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r012.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r013.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r018.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r020.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r022.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r023.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r028.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r038.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r063.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r072.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r087.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r103.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r116.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r119.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_r121.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_base.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_v1.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_v1_top1.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_v1_top2.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_v1_top3.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_v1_top4.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_v1_top5.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_v1_mid1.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_v1_mid2.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_v1_mid3.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_v1_m0333.sh
sbatch --parsable --dependency=afterany:12279673 scripts/motif_check/run_motif_check_ablv3_fullcont6_v1_m0n3.sh
