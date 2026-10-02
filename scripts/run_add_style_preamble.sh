#!/bin/bash
#SBATCH -p compsci
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#
# Add the %Period/%Composer/%Art Song preamble above the %motif: lines in every
# processed corpus, IN PLACE (no regeneration). Idempotent: re-running is a
# no-op, and the change is reversible by stripping the leading three % lines.
#
# Real dirs are patched first (fast, and they are what training indexes point
# at); synthetic crops follow. The repo Lieder dir is shared by every build, so
# it is listed once.

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

eval "$(mamba shell hook --shell bash)" 2>/dev/null || \
  source /home/users/cy232/miniforge3/etc/profile.d/conda.sh
mamba activate notagen 2>/dev/null || conda activate notagen

X=/usr/xtmp/cy232/accompgen

echo "==================== REAL corpora ===================="
python data/add_style_preamble.py --workers 16 \
    --real data/abcfiles_processed_v1 \
    --real $X/irishman/abcfiles_processed_v1 \
    --real $X/irish1k/abcfiles_processed_v1 \
    --real $X/irish50k/abcfiles_processed_v1 \
    --real $X/irishfull/abcfiles_processed_v1

echo "==================== SYNTHETIC crops ===================="
# --real is still passed: it supplies the Lieder bar index used to recover the
# composer of Lieder-derived crops.
python data/add_style_preamble.py --workers 16 \
    --real data/abcfiles_processed_v1 \
    --synthetic $X/synthetic_motifs_v1 \
    --synthetic $X/irish1k/synthetic_motifs_v1 \
    --synthetic $X/irish50k/synthetic_motifs_v1 \
    --synthetic $X/irishfull/synthetic_motifs_v1

echo "==================== VERIFY ===================="
# NB: no `ls | head` here. Under `set -o pipefail` the SIGPIPE that head sends
# to ls on a 650k-file directory makes the pipeline non-zero and kills the job
# *after* all the work is done, which then strands anything chained afterok.
python - <<'PY'
import os, glob
for d in ['data/abcfiles_processed_v1/C',
          '/usr/xtmp/cy232/accompgen/irishfull/abcfiles_processed_v1/C',
          '/usr/xtmp/cy232/accompgen/synthetic_motifs_v1/C',
          '/usr/xtmp/cy232/accompgen/irishfull/synthetic_motifs_v1/C']:
    with os.scandir(d) as it:
        f = next((e.path for e in it if e.name.endswith('.abc')), None)
    if f:
        head = open(f, encoding='utf-8').read().split('\n')[:4]
        print(f'--- {f}')
        print('\n'.join(head))
# the whole point of this pass: nothing may exceed the 128-char byte alphabet
bad = 0
for pat in ['data/abcfiles_processed_v1/*/*.abc',
            '/usr/xtmp/cy232/accompgen/*/abcfiles_processed_v1/*/*.abc']:
    for p in glob.iglob(pat):
        with open(p, 'rb') as fh:
            if any(b > 126 for b in fh.read(200)):
                bad += 1
                if bad < 4:
                    print('NON-ASCII:', p)
print(f'real-corpus files with non-ASCII in first 200 bytes: {bad}')
PY
echo "==================== done ===================="
