#!/bin/bash
# Submit the v2 ablation batch (written by make_ablation_v2.py).
set -eo pipefail
cd "$(dirname "$0")/.."
T10KFULL=$(sbatch --parsable scripts/fine_tune_irish10k_abl_44.sh)
T10KBIAS0=$(sbatch --parsable scripts/fine_tune_irish10k_abl_bias0_44.sh)
T10KNOSYN=$(sbatch --parsable scripts/fine_tune_irish10k_abl_nosyn04.sh)
T10KBOTH=$(sbatch --parsable scripts/fine_tune_irish10k_abl_bias0_nosyn04.sh)
BIAS0FULL=12243138  # running full-data bias0 training
echo "10k trainings: $T10KFULL $T10KBIAS0 $T10KNOSYN $T10KBOTH"
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r001.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r002.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r004.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r005.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r007.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r008.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r009.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r010.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r011.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r012.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r013.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r018.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r020.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r022.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r023.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r028.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r038.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r063.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r072.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r087.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r103.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r116.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r119.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_r121.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_full_base.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r001.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r002.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r004.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r005.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r007.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r008.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r009.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r010.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r011.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r012.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r013.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r018.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r020.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r022.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r023.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r028.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r038.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r063.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r072.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r087.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r103.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r116.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r119.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_r121.sh
sbatch --parsable --dependency=afterok:$BIAS0FULL scripts/motif_check/run_motif_check_ablv2_bias0_base.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r001.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r002.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r004.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r005.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r007.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r008.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r009.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r010.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r011.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r012.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r013.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r018.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r020.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r022.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r023.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r028.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r038.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r063.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r072.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r087.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r103.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r116.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r119.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_r121.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_nosyn_base.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r001.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r002.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r004.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r005.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r007.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r008.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r009.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r010.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r011.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r012.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r013.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r018.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r020.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r022.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r023.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r028.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r038.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r063.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r072.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r087.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r103.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r116.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r119.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_r121.sh
sbatch --parsable scripts/motif_check/run_motif_check_ablv2_both_base.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r001.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r002.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r004.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r005.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r007.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r008.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r009.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r010.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r011.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r012.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r013.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r018.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r020.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r022.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r023.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r028.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r038.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r063.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r072.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r087.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r103.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r116.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r119.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_r121.sh
sbatch --parsable --dependency=afterok:$T10KFULL scripts/motif_check/run_motif_check_ablv2_full10k_base.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r001.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r002.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r004.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r005.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r007.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r008.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r009.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r010.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r011.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r012.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r013.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r018.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r020.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r022.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r023.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r028.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r038.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r063.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r072.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r087.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r103.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r116.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r119.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_r121.sh
sbatch --parsable --dependency=afterok:$T10KBIAS0 scripts/motif_check/run_motif_check_ablv2_bias010k_base.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r001.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r002.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r004.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r005.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r007.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r008.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r009.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r010.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r011.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r012.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r013.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r018.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r020.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r022.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r023.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r028.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r038.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r063.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r072.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r087.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r103.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r116.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r119.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_r121.sh
sbatch --parsable --dependency=afterok:$T10KNOSYN scripts/motif_check/run_motif_check_ablv2_nosyn10k_base.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r001.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r002.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r004.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r005.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r007.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r008.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r009.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r010.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r011.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r012.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r013.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r018.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r020.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r022.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r023.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r028.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r038.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r063.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r072.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r087.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r103.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r116.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r119.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_r121.sh
sbatch --parsable --dependency=afterok:$T10KBOTH scripts/motif_check/run_motif_check_ablv2_both10k_base.sh
