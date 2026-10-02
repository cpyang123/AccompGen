import json, re, sys, os
sys.path.insert(0, '/home/users/cy232/Music_Research/AccompGen')
from motif import extract as ex

SITE_DIR = os.path.dirname(os.path.abspath(__file__))

def blank(m):  # equal-length space padding preserves offsets
    return ' ' * len(m.group(0))

def motif_spans(abc, target):
    # 1) pad-out inline junk exactly where strip_voice_and_text would delete it
    padded = re.sub(r'\".*?\"', blank, abc)
    padded = re.sub(r'\[V:[^\]]*\]', blank, padded)
    padded = re.sub(r'![^!\n]*!', blank, padded)
    padded = re.sub(r'\+[^+\n]*\+', blank, padded)
    # 2) keep only music lines, tracking each kept char's original offset
    kept, mapping = [], []
    off = 0
    for line in padded.split('\n'):
        s = line.strip()
        keep = bool(s) and not s.startswith('%') and not re.match(r'^[A-Za-z]:', s)
        if keep:
            kept.append(line)
            mapping.extend(range(off, off + len(line)))
            mapping.append(-1)  # the joining newline
        off += len(line) + 1
    stripped = '\n'.join(kept)
    if kept: mapping.pop()

    key = ex.detect_key_field(abc)
    pitches, bars, chars, s2, _ = ex.abc_to_pitches_with_bars(stripped, key=key)
    assert s2 == stripped, "strip must be a no-op on pre-stripped input"
    # cross-check against the checker's own full pipeline
    ref = ex.abc_to_pitches_with_bars(abc)[0]
    assert [int(p) for p in pitches] == [int(p) for p in ref], "pitch mismatch vs backbone"

    counts, occ = ex.count_interval_motifs(pitches, bars, window_notes=len(target),
                                           interval_mode='step_skip_leap')
    spans = []
    for o in occ.get(tuple(target), []):
        a = mapping[chars[o['start_note']][0]]
        b = mapping[chars[o['end_note']][1] - 1] + 1
        spans.append([a, b])
    # merge overlaps for clean display
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return counts.get(tuple(target), 0), merged

data = json.load(open(os.path.join(SITE_DIR, 'examples.json')))
for e in data['examples']:
    target = [int(x) for x in e['motif'].split(',')]
    n, spans = motif_spans(e['abc'], target)
    e['spans'] = spans
    flag = 'OK ' if n == e['body_hits'] else '** MISMATCH'
    print(f"{e['id']:10s} windows={n:3d} recorded={e['body_hits']:3d} merged_spans={len(spans)} {flag}")
    for a, b in spans[:2]:
        print('   e.g.', repr(e['abc'][a:b]))
json.dump(data, open(os.path.join(SITE_DIR, 'examples.json'), 'w'), indent=1)

# emit the site artifacts: data.js and per-example abc/ downloads
with open(os.path.join(SITE_DIR, 'data.js'), 'w') as f:
    f.write('window.MOTIGEN_DATA = ' + json.dumps(data, indent=1) + ';\n')
os.makedirs(os.path.join(SITE_DIR, 'abc'), exist_ok=True)
for e in data['examples']:
    with open(os.path.join(SITE_DIR, 'abc', e['id'] + '.abc'), 'w') as f:
        f.write(e['abc'])
print(f"wrote data.js and abc/ for {len(data['examples'])} examples")
