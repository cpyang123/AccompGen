"""Convert the HuggingFace `sander-wood/irishman` dataset into pipeline-ready ABC.

The Irishman dataset (~216k Irish folk tunes) is distributed as train.json /
validation.json, each a list of {"control code": ..., "abc notation": ...}. The
"abc notation" is *monophonic* ABC with only X:/L:/M:/K: headers and no %%score or
V: voice structure.

data/2_data_preprocess.py (via abctoolkit) requires voice structure:
  * extract_metadata_and_tunebody() splits header/body at the LAST bare `V:1` line,
  * extract_global_and_local_metadata() splits global/local metadata at the FIRST
    `V:` line (so the header must contain a V: declaration).
So a monophonic tune needs TWO `V:1` lines: one header declaration (kept in the
metadata, satisfies extract_global_and_local_metadata) and one tunebody marker
(the split point for extract_metadata_and_tunebody). We emit:

    X:1
    %%score 1
    L:1/8
    M:4/4
    K:<key>
    V:1          <- voice declaration (local metadata)
    V:1          <- tunebody marker
    <tunebody...>

This mirrors the shape of the Lieder ABC that 2_data_preprocess.py already
consumes, just single-voice. X:/Q: etc. are stripped by the pipeline anyway.

Quota: the raw JSON and the converted ABC both live on scratch (/usr/xtmp), since
216k small files would blow the home-dir quota (same reason the synthetic V:1 copy
lives on scratch -- see data/generate_synthetic_motifs.py).

Run (in the `notagen` env, though this script has no heavy deps):
    python data/irishman_to_abc.py
"""

import json
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR = os.environ.get('IRISH_RAW_DIR', '/usr/xtmp/cy232/accompgen/irishman/raw')
# Override OUT_DIR (IRISH_OUT_DIR) to write a separate ABC set without clobbering an
# existing one (e.g. a 50k build alongside the 20k set).
OUT_DIR = os.environ.get('IRISH_OUT_DIR', '/usr/xtmp/cy232/accompgen/irishman/abc')
# validation first so a small smoke-test run (MAX_TUNES small) covers it.
SOURCES = ['validation.json', 'train.json']

# Cap the number of tunes converted. The real Lieder V:1 set is ~1.5k pieces;
# all 216k Irishman tunes (x15 key transpositions in preprocessing) would both
# dwarf Lieder (~145x) and take ~38h to preprocess. 20k keeps Irish ~15x Lieder.
# Override with IRISH_MAX_TUNES; None = convert everything.
MAX_TUNES = int(os.environ['IRISH_MAX_TUNES']) if os.environ.get('IRISH_MAX_TUNES') else 20000


def normalize_tune(abc: str):
    """Return pipeline-ready ABC text for one Irishman tune, or None if unusable."""
    lines = abc.split('\n')

    # The K: line terminates the ABC header; everything after it is the tunebody.
    k_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith('K:'):
            k_idx = i
            break
    if k_idx is None:
        return None

    header, body = lines[:k_idx + 1], lines[k_idx + 1:]

    l_line, m_line, q_lines, key_val = None, None, [], None
    for ln in header:
        if ln.startswith('K:'):
            key_val = ln[2:].strip()
        elif ln.startswith('L:'):
            l_line = ln.strip()
        elif ln.startswith('M:'):
            m_line = ln.strip()
        elif ln.startswith('Q:'):
            q_lines.append(ln.strip())

    # K:none / empty -> C (matches 2_data_preprocess.py's none->C handling, but done
    # here so transpose sees a real key and the 'none' tunes aren't dropped).
    if not key_val or key_val.lower() == 'none':
        key_val = 'C'

    body = [b for b in body if b.strip() != '']
    if not body:
        return None

    out = ['X:1', '%%score 1', l_line or 'L:1/8', m_line or 'M:4/4']
    out += q_lines
    out.append('K:' + key_val)
    out.append('V:1')   # voice declaration -> stays in metadata
    out.append('V:1')   # tunebody marker   -> header/body split point
    out += body
    return '\n'.join(out) + '\n'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = skipped = 0

    for src in SOURCES:
        path = os.path.join(RAW_DIR, src)
        if not os.path.exists(path):
            print(f'  (missing {path}, skipping)')
            continue
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        print(f'{src}: {len(data)} tunes')

        for entry in data:
            if MAX_TUNES is not None and written >= MAX_TUNES:
                print(f'Reached MAX_TUNES={MAX_TUNES}.')
                print(f'Wrote {written} ABC files to {OUT_DIR} (skipped {skipped}).')
                return

            abc = entry.get('abc notation') or entry.get('abc') or ''
            norm = normalize_tune(abc)
            if norm is None:
                skipped += 1
                continue
            name = f'irish_{written}'
            with open(os.path.join(OUT_DIR, name + '.abc'), 'w', encoding='utf-8') as w:
                w.write(norm)
            written += 1

    print(f'Wrote {written} ABC files to {OUT_DIR} (skipped {skipped}).')


if __name__ == '__main__':
    main()
