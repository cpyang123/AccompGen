#!/usr/bin/env python3
"""Build per-length stratified RHYTHM motif sets for the rhythm-conditioned containment eval.

Same scheme as make_motif_sets_multilen.py (8 of top-10 ranks + 8 of ranks 11-30 + 8 of the
tail, per length 4..10), but ranking the `%motif:v1:rhythm:` header of every real _len<L>
file in reference key C of the ISOLATED rhythm dataset (rhythm/{lieder,irishman}). Stratum
sizes match the melodic sets so the eval can pair melodic prompt i with rhythm prompt i.

Output: notebook/rhythm_sets_multilen.json
    {"len4": {"A_top10": [...], "B_r11to30": [...], "C_tail": [...],
              "ranks": {pattern: rank}, "counts": {...}, "n_pieces": N, "n_distinct": D}, ...}
"""
import glob
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from motif import rhythm as rx  # noqa: E402

REAL_DIRS = [
    '/usr/xtmp/cy232/accompgen/rhythm/lieder/abcfiles_processed_v1',
    '/usr/xtmp/cy232/accompgen/rhythm/irishman/abcfiles_processed_v1',
]
KEY = 'C'
LENGTHS = range(4, 11)
SEED = 20260928
PREFIX = '%motif:v1:rhythm:'


def header_rhythm(path):
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.startswith(PREFIX):
                toks = [t.strip() for t in line[len(PREFIX):].split(',') if t.strip()]
                try:
                    for t in toks:
                        rx.parse_ratio(t)
                except (ValueError, ZeroDivisionError):
                    return None
                return ','.join(toks)
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
                pat = header_rhythm(path)
                if pat is None:
                    continue
                n_pieces += 1
                counts[pat] += 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ranks = {pat: i + 1 for i, (pat, _) in enumerate(ranked)}
        top10 = [p for p, _ in ranked[:10]]
        mid = [p for p, _ in ranked[10:30]]
        tail = [p for p, _ in ranked[30:]]
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
        print(f'len{L}: {n_pieces} pieces, {len(ranked)} distinct rhythm patterns; '
              f'sampled {len(sel)} (tail pool {len(tail)})')
        print('   A:', A[:3], '... C:', C[:2])
    out['_meta'] = {'seed': SEED, 'key': KEY, 'scheme': '8 of top-10 + 8 of r11-30 + 8 of tail',
                    'source': '%motif:v1:rhythm headers, real _len files of the rhythm dataset, key C'}
    dst = os.path.join(HERE, 'rhythm_sets_multilen.json')
    with open(dst, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', dst)


if __name__ == '__main__':
    main()
