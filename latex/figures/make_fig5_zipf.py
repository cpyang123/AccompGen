#!/usr/bin/env python3
"""Figure 5: rank-frequency (Zipf) plot of motif annotations over the FULL
corpus (all Lieder + full Irishman). Reads motif_ranking_full.txt, built from
one reference key -- the annotation is transposition-invariant, so per-key
counts give the identical ranking; matches the exploratory notebook's
universe (210 patterns over 217,146 pieces).

Shaded rank bands mark the head (1-10), mid (11-40), and tail (41-end)
frequency regimes targeted by the evaluation sampling."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

counts, names = [], []
with open(os.path.join(HERE, 'motif_ranking_full.txt')) as fh:
    for line in fh:
        n, pat = line.split(None, 1)
        counts.append(int(n))
        names.append(pat.strip())
ranks = list(range(1, len(counts) + 1))
total = sum(counts)

# Palette slots 1-3 (validated fixed order); light band fills, text-labeled.
BANDS = [(1, 10, '#2a78d6', 'A: head\n(ranks 1–10)'),
         (11, 40, '#eb6834', 'B: mid\n(11–40)'),
         (41, len(counts), '#1baf7a', f'C: tail\n(41–{len(counts)})')]
INK, MUTED = '#333333', '#666666'

plt.rcParams.update({'font.size': 15, 'axes.labelsize': 17,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14})
fig, ax = plt.subplots(figsize=(8.6, 5.0))

for lo, hi, color, label in BANDS:
    ax.axvspan(lo * 0.97, hi * 1.03, color=color, alpha=0.10, lw=0)
    ax.text((lo * hi) ** 0.5, 2.1, label, ha='center', va='bottom',
            color=color, fontsize=15, fontweight='bold', linespacing=1.2)

ax.plot(ranks, counts, lw=2, color=INK, zorder=3)
ax.scatter(ranks, counts, s=12, color=INK, zorder=4)

for pat, xf, yf in (('0,-1,-1,-1', 1.13, 1.05), ('0,1,1,1', 1.13, 0.62)):
    r = names.index(pat) + 1
    ax.annotate(pat, (r, counts[r - 1]),
                xytext=(r * xf, counts[r - 1] * yf),
                fontsize=13, color=MUTED, family='monospace', ha='left')

share = (counts[0] + counts[1]) / total * 100
ax.annotate(f'top 2 patterns:\n{share:.0f}% of all pieces',
            (4.2, counts[0] * 0.55), fontsize=13, color=MUTED, ha='left')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('motif rank (by pieces annotated with the pattern)')
ax.set_ylabel('training pieces')
ax.set_xlim(0.9, len(counts) * 1.15)
ax.set_ylim(0.8, 1.6e5)
ax.grid(True, which='major', lw=0.5, color='#dddddd', zorder=0)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_5_motif_zipf.png'), dpi=200)
fig.savefig(os.path.join(HERE, 'fig_5_motif_zipf.pdf'))
print(f'{len(counts)} patterns over {total} pieces; top-2 share {share:.1f}%')
