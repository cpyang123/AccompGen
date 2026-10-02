#!/usr/bin/env python3
"""Generate Figure 2 (training: curriculum + motif attention bias) and
Figure 3 (step/skip/leap interval encoding) in the same style as Figure 1."""
import math
import os

INK = '#334155'
MUTED = '#64748b'
LIGHT = '#cbd5e1'
BLUE = '#2563eb'
BLUE_BG = '#eff6ff'
PURPLE = '#7c3aed'
PURPLE_BG = '#f5f3ff'
GREEN = '#059669'
GREEN_BG = '#ecfdf5'
AMBER = '#d97706'
AMBER_BG = '#fef3c7'
SANS = 'Helvetica, Arial, sans-serif'
MONO = 'DejaVu Sans Mono, Menlo, monospace'
GLYPH = 'DejaVu Sans, sans-serif'  # has ♯ ♭ − ± (the Helvetica substitute does not)


class Fig:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.parts = [f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>']

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=13, color=INK, weight='normal',
             anchor='start', font=SANS, style='normal'):
        s = s.replace('&', '&amp;').replace('<', '&lt;')
        self.add(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
                 f'fill="{color}" font-weight="{weight}" text-anchor="{anchor}" '
                 f'font-style="{style}">{s}</text>')

    def tspans(self, x, y, chunks, size=13, weight='normal', anchor='start',
               font=MONO):
        spans = ''.join(
            f'<tspan fill="{c}" font-weight="{w}">{t.replace("&", "&amp;").replace("<", "&lt;")}</tspan>'
            for (t, c, w) in chunks)
        self.add(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
                 f'font-weight="{weight}" text-anchor="{anchor}">{spans}</text>')

    def rrect(self, x, y, w, h, r, fill, stroke='none', sw=0, dash=None, opacity=1):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} '
                 f'opacity="{opacity}"/>')

    def line(self, x1, y1, x2, y2, color, sw=2, dash=None, cap='round'):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                 f'stroke-width="{sw}" stroke-linecap="{cap}"{d}/>')

    def arrowhead(self, x, y, dx, dy, size, color):
        n = math.hypot(dx, dy) or 1
        ux, uy = dx / n, dy / n
        px, py = -uy, ux
        pts = (f'{x},{y} {x - ux*size + px*size*0.55:.1f},{y - uy*size + py*size*0.55:.1f} '
               f'{x - ux*size - px*size*0.55:.1f},{y - uy*size - py*size*0.55:.1f}')
        self.add(f'<polygon points="{pts}" fill="{color}"/>')

    def arrow(self, x1, y1, x2, y2, color=MUTED, sw=2.2, head=9, dash=None):
        n = math.hypot(x2 - x1, y2 - y1) or 1
        ux, uy = (x2 - x1) / n, (y2 - y1) / n
        self.line(x1, y1, x2 - ux * head * 0.8, y2 - uy * head * 0.8, color, sw,
                  dash=dash)
        self.arrowhead(x2, y2, x2 - x1, y2 - y1, head, color)

    def arc(self, x1, y1, x2, y2, apex_y, color, sw=2.2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        xm = (x1 + x2) / 2
        self.add(f'<path d="M {x1},{y1} Q {xm},{apex_y} {x2},{y2}" fill="none" '
                 f'stroke="{color}" stroke-width="{sw}"{d} stroke-linecap="round"/>')

    def pill(self, x, y, w, h, label, color, bg, size=12):
        self.rrect(x, y, w, h, h / 2, bg, color, 1.5)
        self.text(x + w / 2, y + h / 2 + size * 0.36, label, size, color, 'bold',
                  'middle')

    def file_icon(self, x, y, w, h, stroke, fill='#ffffff', fold=10, sw=1.8,
                  dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<path d="M {x},{y} L {x+w-fold},{y} L {x+w},{y+fold} '
                 f'L {x+w},{y+h} L {x},{y+h} Z" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}"{d} stroke-linejoin="round"/>')
        self.add(f'<path d="M {x+w-fold},{y} L {x+w-fold},{y+fold} L {x+w},{y+fold}" '
                 f'fill="none" stroke="{stroke}" stroke-width="{sw}" '
                 f'stroke-linejoin="round"/>')

    def text_bars(self, x, y, w, rows, color=LIGHT, gap=7, bh=3.5):
        for i, frac in enumerate(rows):
            self.rrect(x, y + i * gap, w * frac, bh, 2, color)

    def nn_icon(self, cx, cy, color=INK, sp=13):
        cols = [(-sp, (-sp, 0, sp)), (0, (-sp * 1.4, -sp * 0.47, sp * 0.47, sp * 1.4)),
                (sp, (-sp, 0, sp))]
        pts = []
        for dx, ys in cols:
            pts.append([(cx + dx, cy + dy) for dy in ys])
        for a in pts[0]:
            for b in pts[1]:
                self.line(a[0], a[1], b[0], b[1], LIGHT, 1)
        for a in pts[1]:
            for b in pts[2]:
                self.line(a[0], a[1], b[0], b[1], LIGHT, 1)
        for col in pts:
            for (x, y) in col:
                self.add(f'<circle cx="{x}" cy="{y}" r="3.4" fill="{color}"/>')

    def flag(self, x, y, color=GREEN, h=30):
        self.line(x, y, x, y - h, color, 2.5)
        self.add(f'<path d="M {x},{y-h} L {x+22},{y-h+6} L {x},{y-h+12} Z" '
                 f'fill="{color}"/>')

    def notehead(self, cx, cy, color=INK, r=4.2, stem=14):
        if stem:
            self.line(cx + r - 0.6, cy - 1, cx + r - 0.6, cy - stem, color, 1.5)
        self.add(f'<ellipse cx="{cx}" cy="{cy}" rx="{r}" ry="{r*0.78}" '
                 f'fill="{color}" transform="rotate(-18 {cx} {cy})"/>')

    def svg(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}">' + '\n'.join(self.parts) + '</svg>')


# =====================================================================
# Figure 2: training process — curriculum + motif attention bias
# =====================================================================
def build_fig2():
    F = Fig(1180, 650)

    # ---------------- dataset mini-cards feeding the phases
    F.rrect(252, 36, 180, 70, 12, PURPLE_BG, PURPLE, 2)
    for i, dx in enumerate((0, 11, 22)):
        F.file_icon(264 + dx, 52 + i * 4, 28, 38, PURPLE, dash='4,3', fold=7,
                    sw=1.4)
    F.text(368, 66, 'Synthetic', 13, PURPLE, 'bold', 'middle')
    F.text(368, 82, 'motif crops', 11, MUTED, 'normal', 'middle')
    F.arrow(342, 110, 342, 162, PURPLE, 2.4, 9)

    F.rrect(588, 36, 180, 70, 12, BLUE_BG, BLUE, 2)
    F.file_icon(606, 48, 36, 46, BLUE, fold=8, sw=1.6)
    F.text_bars(612, 56, 24, [1.0, 0.7, 0.9, 0.6])
    F.text(700, 66, 'Real', 13, BLUE, 'bold', 'middle')
    F.text(700, 82, 'full pieces', 11, MUTED, 'normal', 'middle')
    F.arrow(678, 110, 678, 162, BLUE, 2.4, 9)

    # ---------------- curriculum spine
    F.rrect(30, 196, 150, 72, 12, '#f8fafc', INK, 2)
    F.nn_icon(74, 232)
    F.text(139, 226, 'NotaGen', 13, INK, 'bold', 'middle')
    F.text(139, 243, 'pretrained', 11, MUTED, 'normal', 'middle')
    F.arrow(182, 232, 228, 232, MUTED, 2.6, 10)

    F.rrect(232, 166, 220, 130, 14, PURPLE_BG, PURPLE, 2.4)
    F.text(342, 194, 'Phase 1', 15, PURPLE, 'bold', 'middle')
    F.text(342, 212, 'fine-tune on synthetic', 12, INK, 'normal', 'middle')
    F.pill(262, 224, 160, 24, 'motif bias OFF', MUTED, '#ffffff')
    F.text(342, 280, 'early stopping on eval loss', 11, MUTED, 'italic',
           'middle')

    F.arrow(456, 232, 492, 232, MUTED, 2.6, 10)
    F.flag(510, 246, GREEN)
    F.text(510, 268, 'best-eval', 11, GREEN, 'bold', 'middle')
    F.text(510, 282, 'checkpoint', 11, GREEN, 'bold', 'middle')
    F.arrow(528, 232, 564, 232, MUTED, 2.6, 10)

    F.rrect(568, 166, 220, 130, 14, BLUE_BG, BLUE, 2.4)
    F.text(678, 194, 'Phase 2', 15, BLUE, 'bold', 'middle')
    F.text(678, 212, 'fine-tune on real', 12, INK, 'normal', 'middle')
    F.pill(598, 224, 160, 24, 'motif bias ON (+4)', AMBER, AMBER_BG)
    F.text(678, 280, 'early stopping on eval loss', 11, MUTED, 'italic',
           'middle')

    F.arrow(792, 232, 828, 232, MUTED, 2.6, 10)
    F.flag(846, 246, GREEN)
    F.text(846, 268, 'best-eval', 11, GREEN, 'bold', 'middle')
    F.text(846, 282, 'checkpoint', 11, GREEN, 'bold', 'middle')
    F.arrow(864, 232, 900, 232, MUTED, 2.6, 10)

    F.rrect(904, 196, 216, 72, 12, GREEN_BG, GREEN, 2.2)
    F.nn_icon(946, 232, GREEN)
    F.text(1028, 226, 'motif-conditioned', 13, GREEN, 'bold', 'middle')
    F.text(1028, 243, 'model', 13, GREEN, 'bold', 'middle')
    F.pill(932, 276, 160, 24, 'inference: bias ON', AMBER, '#ffffff')
    F.text(560, 318, 'eval matched to each phase’s data', 11, MUTED,
           'italic', 'middle')

    # ---------------- motif attention bias panel
    F.rrect(40, 350, 1100, 272, 16, '#fffdf5', AMBER, 2.4)
    F.text(66, 382, 'Motif attention bias', 16, AMBER, 'bold')
    F.text(250, 382, '— every generated token is pulled toward the motif',
           12, MUTED, 'italic')

    # token row
    sq, gap = 34, 7
    x0, ty = 80, 470
    groups = [(3, AMBER_BG, AMBER), (2, '#f1f5f9', LIGHT), (6, '#ffffff', INK)]
    xs = []
    x = x0
    for n, fill, stroke in groups:
        for _ in range(n):
            F.rrect(x, ty, sq, sq, 6, fill, stroke, 1.6)
            xs.append(x + sq / 2)
            x += sq + gap
    qx = xs[-1]
    F.rrect(qx - sq / 2, ty, sq, sq, 6, GREEN_BG, GREEN, 2.4)  # query square
    # mini note glyphs in tunebody squares
    for cx in xs[5:-1]:
        F.notehead(cx - 2, ty + sq / 2 + 4, INK, 3.4, 11)
    F.text((xs[0] + xs[2]) / 2, ty + sq + 20, '%motif: 0,+1,+1,+1', 11, AMBER,
           'bold', 'middle', MONO)
    F.text((xs[3] + xs[4]) / 2, ty + sq + 20, 'header', 11, MUTED, 'normal',
           'middle')
    F.text((xs[5] + xs[9]) / 2, ty + sq + 20, 'generated so far', 11, MUTED,
           'normal', 'middle')
    F.text(qx, ty + sq + 20, 'query', 11, GREEN, 'bold', 'middle')

    # attention arcs from query to keys
    for cx in xs[:3]:
        F.arc(qx, ty - 4, cx, ty - 4, ty - 4 - (qx - cx) * 0.22, AMBER, 2.6)
        F.arrowhead(cx, ty - 4, -1, 3, 7, AMBER)
    for cx in xs[3:-1]:
        F.arc(qx, ty - 4, cx, ty - 4, ty - 4 - (qx - cx) * 0.22, LIGHT, 1.2)
    bx = (xs[1] + qx) / 2
    F.add(f'<circle cx="{bx}" cy="398" r="15" fill="{AMBER}"/>')
    F.text(bx, 403, '+4', 12.5, '#ffffff', 'bold', 'middle')

    # attention-weight bar chart
    F.line(620, 540, 620, 400, '#e2e8f0', 1.2, cap='butt')
    cx0, base = 668, 528
    F.text(668, 412, 'attention weights', 12.5, INK, 'bold')
    heights = [64, 56, 60, 14, 12, 18, 14, 16, 12, 15, 13]
    for i, h in enumerate(heights):
        color = AMBER if i < 3 else LIGHT
        F.rrect(cx0 + i * 18, base - h, 12, h, 2, color)
    F.line(cx0 - 6, base, cx0 + len(heights) * 18, base, MUTED, 1.4, cap='butt')
    F.text(cx0 + 27, base + 18, 'motif', 10.5, AMBER, 'bold', 'middle')
    F.text(cx0 + 140, base + 18, 'other positions', 10.5, MUTED, 'normal',
           'middle')

    # formula
    F.line(910, 540, 910, 400, '#e2e8f0', 1.2, cap='butt')
    F.text(1022, 430, 'Attention =', 13.5, INK, 'bold', 'middle')
    F.text(1022, 456, 'softmax( QKᵀ/√d + b·1[motif] )', 12,
           INK, 'normal', 'middle', MONO)
    F.text(1022, 486, 'b = +4 on motif key positions,', 12, MUTED, 'normal',
           'middle')
    F.text(1022, 503, 'all layers, before softmax', 12, MUTED, 'normal',
           'middle')
    F.text(1022, 530, 'b=+4 optimal; +6/+8 collapse', 11, MUTED, 'italic',
           'middle')

    F.text(590, 600, 'Bias applied in Phase 2 training, matched eval, and '
           'inference — the loss itself stays unweighted cross-entropy.',
           12, MUTED, 'italic', 'middle')
    return F


# =====================================================================
# Figure 3: step / skip / leap interval encoding
# =====================================================================
def build_fig3():
    F = Fig(1560, 620)

    # ---------------- three class columns, 3 staff examples each
    # degrees relative to bottom staff line E4 = 0 (one diatonic letter = 1)
    cols = [
        ('step', '±1', '2nd', GREEN, [
            ('F', 'G', None, 1, 2, '+1', '2nd'),
            ('B', 'A', None, 4, 3, '−1', '2nd'),
            ('F', 'G♯', '♯', 1, 2, '+1', 'aug 2nd'),
        ]),
        ('skip', '±2', '3rd', BLUE, [
            ('E', 'G', None, 0, 2, '+2', '3rd'),
            ('C', 'A', None, 5, 3, '−2', '3rd'),
            ('F', 'A♭', '♭', 1, 3, '+2', 'min 3rd'),
        ]),
        ('leap', '±3', '4th +', PURPLE, [
            ('E', 'A', None, 0, 3, '+3', '4th'),
            ('G', 'D', None, 2, 6, '+3', '5th'),
            ('E', 'G', None, 7, 2, '−3', '6th'),
        ]),
    ]
    x0, colw, sp = 34, 330, 10
    for ci, (title, tok, ival_desc, col, exs) in enumerate(cols):
        cx = x0 + ci * colw
        center = cx + 145
        if ci:
            F.line(cx - 20, 36, cx - 20, 520, '#e2e8f0', 1.4, cap='butt')
        F.text(center, 60, title, 32, col, 'bold', 'middle')
        F.text(center - 6, 92, tok, 20, col, 'bold', 'end', MONO)
        F.text(center + 6, 92, '=  ' + ival_desc, 19, MUTED, 'normal',
               'start', GLYPH)

        for ei, (n1, n2, acc, d1, d2, token, ival) in enumerate(exs):
            ey = 152 + ei * 128
            for i in range(5):
                F.line(cx, ey + i * sp, cx + 190, ey + i * sp, '#9ca3af',
                       1.1, cap='butt')
            bot = ey + 4 * sp
            for x, d, nm in ((cx + 58, d1, n1), (cx + 132, d2, n2)):
                yy = bot - d * sp / 2
                if acc and nm[-1:] in '♯♭' and len(nm) > 1:
                    F.text(x - 16, yy + 6, acc, 19, INK, 'bold', 'middle',
                           GLYPH)
                F.notehead(x, yy, INK, 6.4, -20 if d >= 5 else 20)
                F.text(x, bot + 30, nm, 16, INK, 'bold', 'middle', GLYPH)
            F.arc(cx + 66, ey - 8, cx + 124, ey - 8, ey - 36, col, 3)
            F.text(cx + 246, ey + 18, token, 28, col, 'bold', 'middle', MONO)
            F.text(cx + 246, ey + 44, ival, 14.5, MUTED, 'italic', 'middle')

    # ---------------- footnotes
    F.text(529, 562, '+ up      − down', 18, INK, 'normal', 'middle', GLYPH)
    F.text(529, 592, 'letter-name distance, not semitones', 15.5, MUTED,
           'italic', 'middle')

    # ---------------- right panel: example ABC training file
    fx, fy, fw = 1130, 76, 400
    F.text(fx + fw / 2, 52, 'training file', 21, INK, 'bold', 'middle')
    F.file_icon(fx, fy, fw, 430, INK, fold=22, sw=2.4)

    # motif prompt block (highlighted)
    F.rrect(fx + 14, fy + 22, fw - 28, 118, 10, AMBER_BG)
    F.text(fx + 28, fy + 46, '%motif:v1:step_skip_leap:', 14, MUTED,
           'normal', 'start', MONO)
    F.tspans(fx + 44, fy + 82, [
        ('0', INK, 'bold'), (', ', MUTED, 'normal'),
        ('+1', GREEN, 'bold'), (', ', MUTED, 'normal'),
        ('+2', BLUE, 'bold'), (', ', MUTED, 'normal'),
        ('−3', PURPLE, 'bold'),
    ], size=27, anchor='start')
    F.text(fx + 28, fy + 118, '%motif:abc:', 14, MUTED, 'normal', 'start',
           MONO)
    F.tspans(fx + 138, fy + 120, [
        ('C2 ', INK, 'bold'), ('D2 ', GREEN, 'bold'),
        ('F2 ', BLUE, 'bold'), ('C2', PURPLE, 'bold'),
    ], size=19, anchor='start')

    F.text(fx + 28, fy + 168, 'K:C  M:4/4', 14, MUTED, 'normal', 'start',
           MONO)

    body = [
        ([('[V:1] ', MUTED, 'normal'), ('G2 A2 B2 G2 |', INK, 'normal')],
         False),
        ([('[V:1] ', MUTED, 'normal'), ('C2 ', INK, 'bold'),
          ('D2 ', GREEN, 'bold'), ('F2 ', BLUE, 'bold'),
          ('C2 ', PURPLE, 'bold'), ('|', MUTED, 'normal')], True),
        ([('[V:1] ', MUTED, 'normal'), ('E2 D2 C2 z2 |', INK, 'normal')],
         False),
        ([('[V:1] ', MUTED, 'normal'), ('E2 ', INK, 'bold'),
          ('F2 ', GREEN, 'bold'), ('A2 ', BLUE, 'bold'),
          ('E2 ', PURPLE, 'bold'), ('|', MUTED, 'normal')], True),
        ([('[V:1] ', MUTED, 'normal'), ('G2 F2 E2 D2 |', INK, 'normal')],
         False),
        ([('[V:1] ', MUTED, 'normal'), ('C6 z2 |]', INK, 'normal')], False),
    ]
    hl_ys = []
    for i, (chunks, hl) in enumerate(body):
        y = fy + 202 + i * 36
        if hl:
            hl_ys.append(y)
            F.rrect(fx + 14, y - 20, fw - 28, 30, 6, AMBER_BG)
        F.tspans(fx + 28, y, chunks, size=19, anchor='start')

    # callout labels (left of the file) — one word each
    F.text(fx - 22, fy + 86, 'motif', 19, AMBER, 'bold', 'end')
    F.arrow(fx - 16, fy + 80, fx - 2, fy + 80, AMBER, 2.2, 8)
    F.text(fx - 22, hl_ys[0] + 1, 'realized', 19, AMBER, 'bold', 'end')
    F.arrow(fx - 16, hl_ys[0] - 5, fx - 2, hl_ys[0] - 5, AMBER, 2.2, 8)
    F.text(fx - 22, hl_ys[1] + 1, 'transposed', 19, AMBER, 'bold', 'end')
    F.arrow(fx - 16, hl_ys[1] - 5, fx - 2, hl_ys[1] - 5, AMBER, 2.2, 8)
    return F


out = os.path.dirname(os.path.abspath(__file__))
import cairosvg
for name, fig in (('fig2_training', build_fig2()), ('fig3_encoding', build_fig3())):
    p = os.path.join(out, name + '.svg')
    with open(p, 'w') as f:
        f.write(fig.svg())
    cairosvg.svg2png(url=p, write_to=os.path.join(out, name + '.png'),
                     output_width=fig.w * 2)
    cairosvg.svg2pdf(url=p, write_to=os.path.join(out, name + '.pdf'))
    print('wrote', name)
