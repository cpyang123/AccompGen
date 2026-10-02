#!/usr/bin/env python3
"""Generate motif_check_styles_mid1_4x4.ipynb: the full 4x4 model conditioned
on the fixed mid1 motif (0,1,-1,1), swept over 10 period/composer/
instrumentation prompts (20 pieces each) to test whether motif adoption
depends on the style prompt. Clones the irishfull_mid1_4x4 notebook, replaces
the generation loop, and appends a per-style summary cell. Styles are
validated against gradio/prompts.txt."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

STYLES = [
    # All Art Song (the model is melody-only); vary period/composer.
    ("Romantic", "Schubert, Franz", "Art Song"),   # reference: all prior evals
    ("Classical", "Beethoven, Ludwig van", "Art Song"),
    ("Classical", "Paradis, Maria Theresia von", "Art Song"),
    ("Classical", "Reichardt, Louise", "Art Song"),
    ("Romantic", "Brahms, Johannes", "Art Song"),
    ("Romantic", "Schumann, Robert", "Art Song"),
    ("Romantic", "Hensel, Fanny", "Art Song"),
    ("Romantic", "Debussy, Claude", "Art Song"),
    ("Romantic", "Faure, Gabriel", "Art Song"),
    ("Romantic", "Wolf, Hugo", "Art Song"),
]

known = {tuple(l.rstrip('\n').split('_')) for l in
         open(HERE.parent / 'gradio' / 'prompts.txt') if l.strip()}
for s in STYLES:
    assert s in known, f'not in prompts.txt: {s}'

LOOP_SRC = '''\
# Style sweep: fixed mid1 motif, 10 period/composer/instrumentation prompts,
# NUM_PER_STYLE pieces each -- does motif adoption depend on the style prompt?
def sanitize(name: str) -> str:
    """Convert to safe filename (remove spaces, commas, etc.)."""
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')

STYLES = {styles}
NUM_PER_STYLE = 20

motif_line = "%motif:v1:step_skip_leap: 0,1,-1,1 \\n"
USE_REALIZED_MOTIF = False

interval_mode, target_patterns = parse_motif_prompt(motif_line)
print(f"Motif mode: {{interval_mode}}")
print(f"Target patterns: {{target_patterns}}")

OUT_DIR = Path("motif_check_styles_mid1_4x4")
OUT_DIR.mkdir(exist_ok=True)

results = []
idx = 0
for (period, composer, instrumentation) in STYLES:
    style = f"{{sanitize(period)}}_{{sanitize(composer)}}_{{sanitize(instrumentation)}}"
    for i in range(NUM_PER_STYLE):
        print(f"\\n========== [{{style}}] PIECE {{i+1}}/{{NUM_PER_STYLE}} "
              f"(global {{idx+1}}/{{len(STYLES)*NUM_PER_STYLE}}) ==========")
        stem = f"motif_check_{{style}}_{{i:02d}}"
        abc_path = OUT_DIR / f"{{stem}}.abc"
        try:
            result = inference_patch(period, composer, instrumentation, [motif_line])
            abc_lines = result.splitlines()
            abc_lines = [line + '\\n' for line in abc_lines if line.strip()]
            abc_lines = rest_unreduce(abc_lines)
            abc_text = ''.join(abc_lines)
            with open(abc_path, "w", encoding="utf-8") as f:
                f.write(abc_text)
            !python abc2xml.py -o {{OUT_DIR}} {{abc_path}}
            hits = piece_contains_motif(abc_text, target_patterns, interval_mode)
            contains = bool(hits)
            results.append({{
                "index": idx, "style": style, "file": str(abc_path),
                "contains_motif": contains, "hits": hits, "error": None,
                "raw_text": globals().get("_LAST_GENERATED_RAW", ""),
            }})
            print(f"\\n[piece {{idx:03d}}] {{style}} contains_motif={{contains}}  hits={{hits}}")
        except Exception as e:
            print(f"[piece {{idx:03d}}] {{style}} generation failed: {{e}}")
            results.append({{
                "index": idx, "style": style, "file": str(abc_path),
                "contains_motif": False, "hits": {{}}, "error": str(e),
                "raw_text": globals().get("_LAST_GENERATED_RAW", ""),
            }})
        idx += 1
'''.format(styles=json.dumps([list(s) for s in STYLES], indent=4))

STYLE_SUMMARY_SRC = '''\
# === Per-style breakdown: containment / faithfulness by prompt style ===
from collections import defaultdict
by_style = defaultdict(lambda: {"n": 0, "contain": 0, "emit": 0, "faith": 0, "hits": 0})
for r, a in zip(results, realization_analyses):
    if r.get('error') is not None or a is None:
        continue
    s = by_style[r['style']]
    s["n"] += 1
    s["contain"] += bool(r['contains_motif'])
    s["emit"] += a['realization'] is not None
    s["faith"] += bool(a['fully_faithful'])
    s["hits"] += sum(c for v in r['hits'].values() for c in v.values())
print(f"{'style':52s} {'n':>3s} {'contain':>9s} {'faith':>8s} {'hits/pc':>8s} {'emit':>7s}")
for style, s in by_style.items():
    print(f"{style:52s} {s['n']:3d} {100*s['contain']/max(1,s['n']):8.1f}% "
          f"{100*s['faith']/max(1,s['n']):7.1f}% {s['hits']/max(1,s['n']):8.2f} {s['emit']:3d}/{s['n']}")
'''

nb = json.loads((HERE / 'motif_check_noreal_irishfull_mid1_4x4.ipynb').read_text())
gen_cells = [i for i, c in enumerate(nb['cells'])
             if c['cell_type'] == 'code' and 'NUM_PIECES' in ''.join(c['source'])
             and 'inference_patch(' in ''.join(c['source'])]
assert len(gen_cells) == 1, gen_cells
nb['cells'][gen_cells[0]]['source'] = LOOP_SRC.splitlines(keepends=True)
nb['cells'].append({'cell_type': 'code', 'execution_count': None,
                    'metadata': {}, 'outputs': [],
                    'source': STYLE_SUMMARY_SRC.splitlines(keepends=True)})
out = HERE / 'motif_check_styles_mid1_4x4.ipynb'
out.write_text(json.dumps(nb, indent=1))
print('wrote', out)
