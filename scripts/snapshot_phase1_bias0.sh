#!/bin/bash
# Snapshot the bias0_4x4 run's end-of-phase-1 (synthetic) checkpoint before the
# real phase overwrites it. Phase 1 trains with use_motif_weights=False, so
# this checkpoint is bias-independent: it is THE curriculum phase-1 artifact
# any future real-phase variant (e.g. the beta=2/6/8 ladder) can fork from via
# SKIP_SYNTHETIC_PHASE=1 + LOAD_FROM_CHECKPOINT=1, saving ~4 days each.
#
# Trigger: train-gen.py prints "Loading best synthetic checkpoint" to the slurm
# .out at the phase boundary; the weights file is then untouched until the
# first real-epoch save ~8.6h later. Poll every 10 min, copy once, exit.
set -u
OUT=slurm_logs/slurm-fine_tune_irishfull_bias0_4x4.sh-12243138.out
W=/usr/xtmp/cy232/accompgen/weights/weights_notagen_1motif_v1_bias0_cropsyn_irishfull_4x4_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1.pth
DEST=/usr/xtmp/cy232/accompgen/weights/weights_notagen_1motif_v1_bias0_cropsyn_irishfull_4x4_PHASE1_SYNTHETIC_BEST.pth
DEADLINE=$(( $(date +%s) + 5*24*3600 ))
cd "$(dirname "$0")/.."
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if grep -q "Loading best synthetic checkpoint" "$OUT" 2>/dev/null; then
        cp "$W" "$DEST"
        echo "$(date): phase-1 checkpoint snapshotted to $DEST" >> slurm_logs/phase1_snapshot.log
        exit 0
    fi
    sleep 600
done
echo "$(date): watcher deadline reached without phase transition" >> slurm_logs/phase1_snapshot.log
