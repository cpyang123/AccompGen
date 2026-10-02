#!/usr/bin/env python3
"""Build per-length stratified motif sets for the multi-length containment eval.

Mirrors the v2 scheme (notebook/motif_sets_v2.json, seed 20260731): rank motifs
by piece-annotation frequency, then sample 8 from the top-10 ranks, 8 from ranks
11-30, and 8 from the tail (rank 31+) — but per motif LENGTH 4..10, ranked on
the CURRENT (transformation-folded) real training headers, so one common motif
set serves both the trans model and the multilen baseline.

Ranking source: the %motif:v1 header of every real _len<L> file in reference key
C (Lieder + Irishman processed dirs) — "piece-annotation ranking", as in v2.

Output: notebook/motif_sets_multilen.json
    {"len4": {"A_top10": [...], "B_r11to30": [...], "C_tail": [...],
              "ranks": {pattern: rank}, "n_pieces": N, "n_distinct": D}, ...}
"""
import glob
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_DIRS = [
    '/usr/xtmp/cy232/accompgen/lieder/abcfiles_processed_v1',
    '/usr/xtmp/cy232/accompgen/irishman/abcfiles_processed_v1',
]
KEY = 'C'
LENGTHS = range(4, 11)
SEED = 20260908


def header_pattern(path):
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('%motif:v1:'):
                body = line.split(':', 3)[-1]
                try:
                    return ','.join(str(int(x)) for x in body.split(','))
                except ValueError:
                    return None
            if line.startswith('[V:'):
                return None
    return None


def main():
    rng = random.Random(SEED)
    out = {}
    for L in LENGTHS:
        counts = Counter()
        n_pieces = 0
        for d in REAL_DIRS:
            for path in glob.iglob(os.path.join(d, KEY, f'*_len{L}_{KEY}.abc')):
                pat = header_pattern(path)
                if pat is None:
                    continue
                n_pieces += 1
                counts[pat] += 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ranks = {pat: i + 1 for i, (pat, _) in enumerate(ranked)}
        top10 = [p for p, _ in ranked[:10]]
        mid = [p for p, _ in ranked[10:30]]
        tail = [p for p, _ in ranked[30:]]
        # Sample within each stratum (rank order kept for readability).
        A = sorted(rng.sample(top10, min(8, len(top10))), key=ranks.get)
        B = sorted(rng.sample(mid, min(8, len(mid))), key=ranks.get)
        C = sorted(rng.sample(tail, min(8, len(tail))), key=ranks.get)
        sel = A + B + C
        out[f'len{L}'] = {
            'A_top10': A, 'B_r11to30': B, 'C_tail': C,
            'ranks': {p: ranks[p] for p in sel},
            'counts': {p: counts[p] for p in sel},
            'n_pieces': n_pieces, 'n_distinct': len(ranked),
        }
        print(f'len{L}: {n_pieces} pieces, {len(ranked)} distinct patterns; '
              f'sampled {len(sel)} (tail pool {len(tail)})')

    out['_meta'] = {'seed': SEED, 'key': KEY, 'scheme': '8 of top-10 + 8 of r11-30 + 8 of tail',
                    'source': 'transformation-folded %motif:v1 headers, real _len files, key C'}
    dst = os.path.join(HERE, 'motif_sets_multilen.json')
    with open(dst, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', dst)


if __name__ == '__main__':
    main()
