#!/usr/bin/env python3
"""Generate Figure 1 (data pipeline) as an icon-based SVG.

Layout: Combined Corpus -> Data Preprocessing & Augmentation (raw ABC -> V:1
-> 12 key transpositions) -> Motif Extraction (single piece -> top-3 motifs)
-> Real Dataset (Phase 2) + Synthetic Dataset (Phase 1).
"""
import math
import os

W, H = 1180, 580

INK = '#334155'
MUTED = '#64748b'
LIGHT = '#cbd5e1'
BLUE = '#2563eb'
BLUE_BG = '#eff6ff'
PURPLE = '#7c3aed'
PURPLE_BG = '#f5f3ff'
GREEN = '#059669'
AMBER = '#d97706'
AMBER_BG = '#fef3c7'
FLOW = '#b8c9c2'
SANS = 'Helvetica, Arial, sans-serif'
MONO = 'DejaVu Sans Mono, Menlo, monospace'

parts = []
def add(s):
    parts.append(s)

def text(x, y, s, size=13, color=INK, weight='normal', anchor='start',
         font=SANS, style='normal'):
    s = s.replace('&', '&amp;').replace('<', '&lt;')
    add(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
        f'fill="{color}" font-weight="{weight}" text-anchor="{anchor}" '
        f'font-style="{style}">{s}</text>')

def rrect(x, y, w, h, r, fill, stroke='none', sw=0, dash=None, opacity=1):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} '
        f'opacity="{opacity}"/>')

def line(x1, y1, x2, y2, color, sw=2, dash=None, cap='round'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}" stroke-linecap="{cap}"{d}/>')

def arrowhead(x, y, dx, dy, size, color):
    n = math.hypot(dx, dy) or 1
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    x1 = x - ux * size + px * size * 0.55
    y1 = y - uy * size + py * size * 0.55
    x2 = x - ux * size - px * size * 0.55
    y2 = y - uy * size - py * size * 0.55
    add(f'<polygon points="{x},{y} {x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f}" '
        f'fill="{color}"/>')

def arrow(x1, y1, x2, y2, color=MUTED, sw=2.2, head=9):
    n = math.hypot(x2 - x1, y2 - y1) or 1
    ux, uy = (x2 - x1) / n, (y2 - y1) / n
    line(x1, y1, x2 - ux * head * 0.8, y2 - uy * head * 0.8, color, sw)
    arrowhead(x2, y2, x2 - x1, y2 - y1, head, color)

def flow_arrow(p0, c1, c2, p3, sw=13, color=FLOW, head=24):
    add(f'<path d="M {p0[0]},{p0[1]} C {c1[0]},{c1[1]} {c2[0]},{c2[1]} '
        f'{p3[0]},{p3[1]}" fill="none" stroke="{color}" stroke-width="{sw}" '
        f'stroke-linecap="round" opacity="0.85"/>')
    arrowhead(p3[0], p3[1], p3[0] - c2[0], p3[1] - c2[1], head, color)

def file_icon(x, y, w, h, stroke, fill='#ffffff', fold=12, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<path d="M {x},{y} L {x+w-fold},{y} L {x+w},{y+fold} L {x+w},{y+h} '
        f'L {x},{y+h} Z" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
        f'{d} stroke-linejoin="round"/>')
    add(f'<path d="M {x+w-fold},{y} L {x+w-fold},{y+fold} L {x+w},{y+fold}" '
        f'fill="none" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linejoin="round"/>')

def text_bars(x, y, w, rows, color=LIGHT, gap=8, bh=4):
    for i, frac in enumerate(rows):
        rrect(x, y + i * gap, w * frac, bh, 2, color)

def staff(x, y, w, color='#9ca3af', n=5, sp=4.5, sw=1):
    for i in range(n):
        line(x, y + i * sp, x + w, y + i * sp, color, sw, cap='butt')
    return y + 2 * sp  # middle line

def noteheads(x0, ymid, degrees, dx=16, step=2.6, color=INK, r=3.4,
              stems=True, hl=None):
    xs = []
    for i, d in enumerate(degrees):
        cx = x0 + i * dx
        cy = ymid - d * step
        xs.append((cx, cy))
    if hl is not None:
        lo, hi = hl
        x_a = xs[lo][0] - 7
        x_b = xs[hi][0] + 7
        rrect(x_a, ymid - 14, x_b - x_a, 28, 6, AMBER_BG, opacity=0.9)
    for cx, cy in xs:
        if stems:
            line(cx + r - 0.6, cy - 1, cx + r - 0.6, cy - 13, color, 1.4)
        add(f'<ellipse cx="{cx}" cy="{cy}" rx="{r}" ry="{r*0.78}" '
            f'fill="{color}" transform="rotate(-18 {cx} {cy})"/>')

def cylinder(cx, y, rx, h, ry, fill='#e2e8f0', stroke=INK):
    add(f'<path d="M {cx-rx},{y+ry} A {rx},{ry} 0 0 1 {cx+rx},{y+ry} '
        f'L {cx+rx},{y+h} A {rx},{ry} 0 0 1 {cx-rx},{y+h} Z" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    add(f'<ellipse cx="{cx}" cy="{y+ry}" rx="{rx}" ry="{ry}" fill="#f1f5f9" '
        f'stroke="{stroke}" stroke-width="2"/>')
    for frac in (0.45, 0.72):
        yy = y + h * frac
        add(f'<path d="M {cx-rx},{yy} A {rx},{ry} 0 0 0 {cx+rx},{yy}" '
        f'fill="none" stroke="{stroke}" stroke-width="1.2" opacity="0.5"/>')

def pill(x, y, w, h, label, color, bg):
    rrect(x, y, w, h, h / 2, bg, color, 1.5)
    text(x + w / 2, y + h / 2 + 4, label, 12, color, 'bold', 'middle')

def key_badge(cx, cy, label, color=GREEN):
    add(f'<circle cx="{cx}" cy="{cy}" r="10" fill="#ffffff" stroke="{color}" '
        f'stroke-width="1.8"/>')
    text(cx, cy + 3.6, label, 10, color, 'bold', 'middle')

def magnifier(cx, cy, r, color=AMBER):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" '
        f'fill-opacity="0.55" stroke="{color}" stroke-width="3"/>')
    ux = r * 0.7071
    line(cx + ux, cy + ux, cx + ux + r * 0.9, cy + ux + r * 0.9, color, 4.5)

# ---------------------------------------------------------------- background
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

# ---------------------------------------------------------------- flow arrows
flow_arrow((470, 152), (620, 58), (720, 58), (842, 108))
flow_arrow((470, 438), (620, 532), (720, 542), (842, 468))
# extraction -> datasets (amber, motif annotations)
add(f'<path d="M 802,232 C 828,210 826,190 844,172" fill="none" '
    f'stroke="{AMBER}" stroke-width="2.4"/>')
arrowhead(846, 170, 18, -18, 9, AMBER)
add(f'<path d="M 802,358 C 828,382 826,402 844,420" fill="none" '
    f'stroke="{AMBER}" stroke-width="2.4"/>')
arrowhead(846, 422, 18, 18, 9, AMBER)

# ---------------------------------------------------------------- corpus
cylinder(80, 248, 46, 84, 13)
text(80, 362, 'Combined', 14, INK, 'bold', 'middle')
text(80, 379, 'Corpus', 14, INK, 'bold', 'middle')
arrow(130, 290, 166, 290, MUTED, 2.6, 10)

# ------------------------------------------------- preprocessing box
rrect(170, 150, 320, 290, 16, '#f8fafc', GREEN, 2.2)
text(330, 177, 'Data Preprocessing & Augmentation', 14.5, GREEN, 'bold',
     'middle')

# raw ABC file
file_icon(192, 210, 60, 78, INK)
text_bars(200, 224, 44, [1.0, 0.75, 0.9, 0.6, 0.85, 0.7])
text(222, 302, 'ABC', 11, MUTED, 'bold', 'middle')
text(222, 318, 'raw file', 11, MUTED, 'normal', 'middle')

arrow(258, 250, 292, 250, MUTED, 2.2, 9)

# V:1-only file
file_icon(296, 210, 60, 78, INK)
text_bars(304, 224, 44, [1.0, 0.75])
rrect(304, 240, 44, 5, 2, AMBER)  # highlighted melody line
text_bars(304, 252, 44, [0.85, 0.7, 0.6])
text(326, 302, 'V:1', 11, AMBER, 'bold', 'middle')
text(326, 318, 'melody only', 11, MUTED, 'normal', 'middle')

# fan-out arrows to transposed stack
arrow(360, 238, 398, 222, MUTED, 1.8, 8)
arrow(360, 250, 398, 288, MUTED, 1.8, 8)
arrow(360, 262, 398, 375, MUTED, 1.8, 8)

for (yy, key) in ((200, 'C'), (262, 'D'), (352, 'Bb')):
    file_icon(404, yy, 48, 58, INK, sw=1.8, fold=10)
    text_bars(411, yy + 12, 34, [1.0, 0.7, 0.85, 0.6])
    key_badge(452, yy + 2, key)
text(428, 342, '. . .', 13, MUTED, 'bold', 'middle')
text(340, 430, '12 key transpositions', 12, MUTED, 'italic', 'middle')

arrow(492, 295, 536, 295, MUTED, 2.6, 10)

# ------------------------------------------------- motif extraction box
rrect(540, 150, 260, 290, 16, '#f8fafc', AMBER, 2.2)
text(670, 178, 'Motif Extraction', 15, AMBER, 'bold', 'middle')
text(670, 196, 'top-3 recurring interval patterns', 11, MUTED, 'italic',
     'middle')

file_icon(556, 240, 54, 70, INK)
text_bars(564, 254, 38, [1.0, 0.7, 0.9, 0.65, 0.8])
magnifier(604, 300, 13)
text(583, 336, 'a single', 11, MUTED, 'normal', 'middle')
text(583, 350, 'piece', 11, MUTED, 'normal', 'middle')

MOTIFS = [
    ('(0, +1, +1, +1)', [0, 1, 2, 3]),
    ('(0, +1, -1, -2)', [0, 1, 0, -2]),
    ('(0, -1, -1, -1)', [0, -1, -2, -3]),
]
chip_y = [205, 283, 361]
for (label, degs), cy in zip(MOTIFS, chip_y):
    rrect(636, cy, 150, 64, 10, '#ffffff', '#e5d3a8', 1.6)
    mid = staff(650, cy + 12, 122)
    noteheads(662, mid, degs, dx=26, color=INK)
    text(711, cy + 56, label, 11.5, AMBER, 'bold', 'middle', MONO)
for cy in chip_y:
    arrow(618, 275 + (cy - 283) * 0.28, 632, cy + 32, MUTED, 1.6, 7)

# ------------------------------------------------- real dataset card
rrect(850, 40, 305, 215, 18, BLUE_BG, BLUE, 2.4)
text(872, 70, 'Real Dataset', 16, BLUE, 'bold')
pill(1058, 52, 80, 24, 'Phase 2', BLUE, '#ffffff')

rrect(868, 84, 270, 138, 8, '#ffffff', LIGHT, 1.4)
rrect(876, 92, 254, 15, 3, AMBER_BG)
text(880, 103, '%motif:v1:step_skip_leap: 0,+1,+1,+1', 9.5, INK, 'normal',
     'start', MONO)
rrect(876, 110, 254, 15, 3, AMBER_BG)
text(880, 121, '%motif:abc: B2 c2 d2 e2', 9.5, INK, 'normal', 'start', MONO)
text_bars(880, 134, 90, [0.5, 0.75, 0.4], color='#d3dce6', gap=7)
mid = staff(880, 162, 240)
noteheads(892, mid, [0, 1, 2, 3, 1, -2, -1, 0, 2], dx=26, hl=(0, 3))
mid2 = staff(880, 196, 240)
noteheads(892, mid2, [2, 1, 0, -1, 1, 3, 2, 0, -1], dx=26)
text(1002, 246, 'full pieces + motif annotations', 12, BLUE, 'italic',
     'middle')

# ------------------------------------------------- synthetic dataset card
rrect(850, 305, 305, 235, 18, PURPLE_BG, PURPLE, 2.4)
text(872, 335, 'Synthetic Dataset', 16, PURPLE, 'bold')
pill(1058, 317, 80, 24, 'Phase 1', PURPLE, '#ffffff')

for i, (dx_, dy_) in enumerate(((0, 0), (52, 22), (104, 44))):
    x0, y0 = 868 + dx_, 350 + dy_
    file_icon(x0, y0, 158, 118, PURPLE, fill='#ffffff', fold=14, sw=1.8,
              dash='5,4')
    rrect(x0 + 8, y0 + 10, 142, 13, 3, AMBER_BG)
    text(x0 + 12, y0 + 20, '%motif: ...', 9, INK, 'normal', 'start', MONO)
    text_bars(x0 + 8, y0 + 30, 60, [0.6, 0.4], color='#d3dce6', gap=6)
    mid = staff(x0 + 10, y0 + 52, 138)
    degs = MOTIFS[i][1]
    noteheads(x0 + 22, mid, degs + [degs[-1] + 1, degs[-1] - 1], dx=22,
              hl=(0, 3))
    ymid3 = staff(x0 + 10, y0 + 88, 138)
    noteheads(x0 + 22, ymid3, [1, 0, 2, 1, -1, 0], dx=22)
text(1002, 532, 'one ±10-bar crop per motif', 12, PURPLE, 'italic', 'middle')

# ------------------------------------------------- curriculum arrow
arrow(1002, 300, 1002, 262, MUTED, 2, 9)
text(1012, 285, 'curriculum', 11, MUTED, 'italic')

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
       f'width="{W}" height="{H}">' + '\n'.join(parts) + '</svg>')

out_dir = os.path.dirname(os.path.abspath(__file__))
svg_path = os.path.join(out_dir, 'fig1_pipeline.svg')
with open(svg_path, 'w') as f:
    f.write(svg)
print('wrote', svg_path)

import cairosvg
cairosvg.svg2png(url=svg_path, write_to=os.path.join(out_dir, 'fig1_pipeline.png'),
                 output_width=W * 2)
cairosvg.svg2pdf(url=svg_path, write_to=os.path.join(out_dir, 'fig1_pipeline.pdf'))
print('wrote png + pdf')
