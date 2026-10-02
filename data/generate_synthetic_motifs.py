"""Generate synthetic motif training data by cropping real pieces around their motif.

Why this exists
---------------
The previous generator emitted files that contained ONLY the two %motif comment
lines (no tunebody, no K:/Q:/M:/L: header). That does not resemble real training
data at all -- the model never saw the motif inside real musical context -- so the
synthetic phase was near useless.

Instead, for every real processed piece we now:
  1. Locate the conditioning motif (the %motif:abc snippet) inside the V:1 tunebody.
  2. Crop a +-WINDOW_RADIUS-bar phrase around it.
  3. Write that crop with the FULL metadata header (%%score / L: / Q: / M: / K: /
     V:) and the per-bar tunebody -- i.e. exactly the shape of real data, just
     shorter and centred on the motif's musical context.

Transposition augmentation
--------------------------
Real data is already preprocessed into all 15 keys
(.../abcfiles_processed_v1/<key>/<name>_<key>.abc): every diatonic transposition
(down a 2nd, up a 5th, ...) already exists with a correctly spelled key signature
AND a correctly transposed %motif:abc snippet. We read all 15 per-key files and
write the cropped excerpt for each key, so the synthetic set carries the same
transposition augmentation as real data -- for free and without re-introducing the
extreme-key spelling bugs that a home-grown fragment transposer would.

The data loader (finetune/train-gen.py) additionally samples a key within +-3
semitones at load time, reading <folder>/<des_key>/<name>_<des_key>.abc. This is
why every synthetic piece MUST exist in all 15 key folders, which we guarantee by
emitting one cropped file per key.

The motif's bar position is identical across all transpositions (transposition
changes pitches, not rhythm or bar count), so we locate the window once on a clean
reference key (C) and apply the same bar range to every key's file.
"""

import gc
import json
import os
import sys
import random
import re
import shutil
import time

# Ensure project root is on sys.path so `import motif` works when running from data/.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# The motif analysis comes from the single backbone (motif/extract.py): same
# enharmonic-aware step/skip/leap classification used by data/2_data_preprocess.py.
from motif import get_best_motifs_per_length, get_best_rhythm_motifs_per_length
from motif import extract as mx
from motif import rhythm as rx

# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

ALL_KEYS = ['Cb', 'C', 'C#', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
REFERENCE_KEY = 'C'   # clean key (no enharmonic spelling issues) used to locate the motif

SYNTHETIC_MARKER = 'synthetic_motifs'

# ---------------------------------------------------------------------------
# Algorithmic motif realization
# ---------------------------------------------------------------------------
# This generator builds synthetic data by cropping real phrases (it no longer
# realizes motifs from scratch). These two names are retained because the motif
# checker's realization fallback imports them: notebook/motif_helpers.py uses
# pattern_to_abc() to synthesize a %motif:abc snippet for an abstract pattern that
# has NO real training realization (e.g. rare/unseen motifs in the eval prompt).

# A representative starting note for pattern_to_abc() for each key.
KEY_TO_START_NOTE = {
    'Cb': 'C', 'C': 'C', 'C#': 'c',
    'Db': 'D', 'D': 'D',
    'Eb': 'E', 'E': 'E',
    'F': 'F', 'F#': 'f',
    'Gb': 'G', 'G': 'G',
    'Ab': 'A', 'A': 'A',
    'Bb': 'B,', 'B': 'B',
}


def pattern_to_abc(pattern, start_note='C', duration_mode='uniform'):
    """Realize a step/skip/leap pattern as an ABC fragment.

    pattern      : list of ints (0, +-1, +-2, +-3 = reference, step, skip, leap)
    start_note   : one of the entries in the diatonic note array below
    duration_mode: 'uniform' (no explicit durations) or 'varied'
    """
    notes = [
        "C,", "D,", "E,", "F,", "G,", "A,", "B,",
        "C", "D", "E", "F", "G", "A", "B",
        "c", "d", "e", "f", "g", "a", "b",
        "c'", "d'", "e'", "f'", "g'", "a'", "b'",
    ]
    durations = ['', '/', '2', '3']

    try:
        current_idx = notes.index(start_note)
    except ValueError:
        current_idx = 14  # fall back to middle 'c'

    first_dur = random.choice(durations) if duration_mode == 'varied' else ''
    abc_notes = [notes[current_idx] + first_dur]

    for val in pattern[1:]:
        if val == 1:
            step_diff = 1
        elif val == -1:
            step_diff = -1
        elif val == 2:
            step_diff = 2
        elif val == -2:
            step_diff = -2
        elif val == 3:
            step_diff = random.choice([3, 4, 5])
        elif val == -3:
            step_diff = random.choice([-3, -4, -5])
        else:
            step_diff = 0

        current_idx += step_diff
        # Bounce at boundaries instead of clipping (avoids repeated notes)
        if current_idx < 0:
            current_idx = abs(current_idx)
        elif current_idx >= len(notes):
            overshoot = current_idx - (len(notes) - 1)
            current_idx = (len(notes) - 1) - overshoot
        current_idx = max(0, min(current_idx, len(notes) - 1))

        # Occasional accidental (10%, avoids C and F to keep it simple)
        acc = ''
        base = notes[current_idx].replace("'", '').replace(',', '')
        if random.random() < 0.1 and base not in ('C', 'F'):
            acc = random.choice(['^', '_'])

        dur = random.choice(durations) if duration_mode == 'varied' else ''
        abc_notes.append(acc + notes[current_idx] + dur)

    return ''.join(abc_notes)

# ---------------------------------------------------------------------------
# Note tokenisation (for locating the motif inside the tunebody)
# ---------------------------------------------------------------------------

# optional accidentals, note letter, optional octave marks, optional duration
_NOTE_RE = re.compile(r"([_^=]*)([A-Ga-g])([,']*)(\d*/*\d*)")


def _note_tokens(text: str) -> list:
    """Return the pitch tokens (accidental+letter+octave, duration ignored) of `text`.

    Quoted annotations, !..! / +..+ decorations and inline [V:]/[r:] markers are
    stripped first so only real notes remain. Duration is ignored: we only need to
    locate the motif by pitch sequence, and the snippet and tunebody come from the
    same source text so the pitch spelling already matches within a key.
    """
    text = re.sub(r'"[^"]*"', '', text)        # quoted text annotations
    text = re.sub(r'![^!\n]*!', '', text)      # !f! !ped! ... decorations
    text = re.sub(r'\+[^+\n]*\+', '', text)    # legacy +..+ decorations
    text = re.sub(r'\[V:[^\]]*\]', '', text)   # inline voice markers
    text = re.sub(r'\[r:[^\]]*\]', '', text)   # stream position markers
    return [m.group(1) + m.group(2) + m.group(3) for m in _NOTE_RE.finditer(text)]


def locate_motif_window(motif_abc_content: str, tunebody_lines: list, radius: int):
    """Return (lo, hi) inclusive bar indices of a +-radius window around the motif.

    `motif_abc_content` is the %motif:abc snippet text (prefix already stripped,
    wrapped continuation lines already joined). The motif is located as the first
    contiguous run of tunebody notes whose pitch sequence matches the snippet. Falls
    back to matching just the first 4 motif notes if the full match fails (e.g. rare
    spelling drift). Returns None if the motif cannot be located at all.
    """
    # With TOP_N_MOTIFS>1 the snippet holds several motifs joined by ' ; '; locate on
    # the first one (the most frequent), which is enough to centre the window. A
    # motif segment may also carry transformation examples ("... I: <inv> R: <retro>
    # RI: <ri>"); keep only the canonical part before the first transform label.
    first_motif = motif_abc_content.split(' ; ')[0]
    first_motif = re.split(r'\s(?:RI|I|R):\s', first_motif)[0]
    motif_tokens = _note_tokens(first_motif)
    if not motif_tokens:
        return None

    flat_tokens = []   # every tunebody note, flattened
    bar_of = []        # bar index of each flattened note
    for bar_idx, line in enumerate(tunebody_lines):
        for tok in _note_tokens(line):
            flat_tokens.append(tok)
            bar_of.append(bar_idx)

    def _find(seq):
        m = len(seq)
        for start in range(0, len(flat_tokens) - m + 1):
            if flat_tokens[start:start + m] == seq:
                return start, start + m - 1
        return None

    hit = _find(motif_tokens)
    if hit is None and len(motif_tokens) > 4:
        hit = _find(motif_tokens[:4])   # relaxed fallback
    if hit is None:
        return None

    first_bar = bar_of[hit[0]]
    last_bar = bar_of[hit[1]]
    lo = max(0, first_bar - radius)
    hi = min(len(tunebody_lines) - 1, last_bar + radius)
    return lo, hi


def locate_motif_span(motif_snippet: str, tunebody_lines: list):
    """Return the (first_line, last_line) bar span of `motif_snippet` in the tunebody.

    Bars are tunebody lines (one bar per line in the processed format). The snippet
    is matched as a contiguous run of note tokens, with the same first-4-notes
    fallback as `locate_motif_window`. Returns None if it cannot be located.
    """
    motif_tokens = _note_tokens(motif_snippet)
    if not motif_tokens:
        return None
    flat_tokens, bar_of = [], []
    for bar_idx, line in enumerate(tunebody_lines):
        for tok in _note_tokens(line):
            flat_tokens.append(tok)
            bar_of.append(bar_idx)

    def _find(seq):
        m = len(seq)
        for start in range(0, len(flat_tokens) - m + 1):
            if flat_tokens[start:start + m] == seq:
                return start, start + m - 1
        return None

    hit = _find(motif_tokens)
    if hit is None and len(motif_tokens) > 4:
        hit = _find(motif_tokens[:4])
    if hit is None:
        return None
    return bar_of[hit[0]], bar_of[hit[1]]


def _bar_content(line: str) -> str:
    """Strip inline [V:..]/[r:..] markers and a trailing barline from a tunebody line,
    leaving just the bar's note text (matching the %motif:abc snippet style)."""
    t = re.sub(r'\[V:[^\]]*\]', '', line)
    t = re.sub(r'\[r:[^\]]*\]', '', t)
    t = t.strip()
    if t.endswith('|'):
        t = t[:-1].strip()
    return t


def motif_snippet_for_key(tunebody_lines: list, first_line: int, last_line: int) -> str:
    """Rebuild a %motif:abc snippet from the motif's bars in a given key's tunebody.

    Because all keys are transpositions with identical bar structure, the motif sits
    in the same line span in every key; reconstructing the snippet from that key's
    own bars yields the correctly transposed spelling and guarantees the snippet
    actually appears in the cropped excerpt.
    """
    bars = [_bar_content(tunebody_lines[i]) for i in range(first_line, last_line + 1)]
    return ' | '.join(b for b in bars if b)


# ---------------------------------------------------------------------------
# Parse a processed .abc file into its sections
# ---------------------------------------------------------------------------

def _is_metadata_field(ln: str) -> bool:
    """True for a real ABC metadata field line (%%score / L: / Q: / M: / K: / V: ...).

    Deliberately excludes %motif: comments and the wrapped continuation lines of a
    %motif:abc snippet (which start with a space, a note, or a bracket).
    """
    return (ln.startswith('%%')
            or ln.startswith('score')
            or re.match(r'^[A-Za-z]:', ln) is not None)


def parse_processed_file(path: str):
    """Split a processed file into (motif_block, motif_abc_content, metadata, tunebody).

    Lines are kept verbatim (no trailing newline). The %motif:abc snippet can wrap
    across several physical lines (the snippet text carries embedded newlines from
    the source), so the motif block absorbs those continuation lines, and
    `motif_abc_content` is the full snippet text (used to locate the motif).
      - motif_block      : the %motif: comment lines plus wrapped continuations
      - motif_abc_content: the joined %motif:abc snippet text ('' if none)
      - metadata         : the header fields (%%score / L: / Q: / M: / K: / V:)
      - tunebody         : the [V:..] bar lines
    Output order is motif_block + metadata + tunebody, matching real data.
    Returns None if the file has no tunebody.
    """
    with open(path, 'r', encoding='utf-8') as fh:
        lines = [ln.rstrip('\n') for ln in fh if ln.strip() != '']

    tunebody_start = None
    for i, ln in enumerate(lines):
        if ln.startswith('[V:'):
            tunebody_start = i
            break
    if tunebody_start is None:
        return None

    header = lines[:tunebody_start]
    tunebody_lines = lines[tunebody_start:]

    motif_block = []
    metadata_lines = []
    motif_abc_parts = []
    in_motif_abc = False      # inside a wrapped %motif:abc: snippet (the melodic one)
    in_other_snippet = False  # inside a wrapped %motif:rhythm:abc: snippet
    for ln in header:
        if ln.startswith('%motif:abc: '):
            motif_block.append(ln)
            motif_abc_parts.append(ln.split('%motif:abc:', 1)[-1])
            in_motif_abc, in_other_snippet = True, False
        elif ln.startswith('%motif:rhythm:abc: '):
            motif_block.append(ln)
            in_motif_abc, in_other_snippet = False, True
        elif ln.startswith('%motif:'):
            motif_block.append(ln)
            in_motif_abc = in_other_snippet = False
        elif _is_metadata_field(ln):
            metadata_lines.append(ln)
            in_motif_abc = in_other_snippet = False
        elif in_motif_abc:
            # wrapped continuation of the %motif:abc snippet
            motif_block.append(ln)
            motif_abc_parts.append(ln)
        elif in_other_snippet:
            motif_block.append(ln)
        else:
            metadata_lines.append(ln)

    motif_abc_content = ' '.join(part.strip() for part in motif_abc_parts).strip()
    return motif_block, motif_abc_content, metadata_lines, tunebody_lines


# ---------------------------------------------------------------------------
# JSONL refresh helpers
# ---------------------------------------------------------------------------

def refresh_jsonl(jsonl_path: str, new_entries: list):
    """Replace all synthetic entries in `jsonl_path` with `new_entries`.

    Real-data entries (lines whose path does not contain SYNTHETIC_MARKER) are
    preserved as-is. If the file does not exist, it is created.
    """
    real_lines = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if SYNTHETIC_MARKER not in entry.get('path', ''):
                        real_lines.append(line)
                except json.JSONDecodeError:
                    pass  # drop malformed lines

    with open(jsonl_path, 'w', encoding='utf-8') as fh:
        for line in real_lines:
            fh.write(line + '\n')
        for entry in new_entries:
            fh.write(json.dumps(entry) + '\n')

    return len(real_lines), len(new_entries)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Keep in sync with VOICE1_ONLY in data/2_data_preprocess.py: when True, read
    # the V:1-only dataset and register synthetic entries in its *_v1.jsonl indices.
    VOICE1_ONLY = True
    suffix = '_v1' if VOICE1_ONLY else ''

    data_dir = os.path.dirname(os.path.abspath(__file__))
    # Primary real-data source; overridable because the multi-length Lieder build
    # lives on scratch (home quota), see scripts/run_multilen_data.sh.
    processed_dir = os.environ.get(
        'SYNTH_PRIMARY_SOURCE', os.path.join(data_dir, 'abcfiles_processed' + suffix))
    # The synthetic copies exceed the home-dir quota; keep every copy on scratch.
    # '_multilen' keeps this length-4..10 set separate from the original length-4
    # set (this folder is wiped and recreated on every run).
    output_dir = '/usr/xtmp/cy232/accompgen/synthetic_motifs_multilen' + suffix
    real_index_path = os.path.join(data_dir, 'abcfiles_processed' + suffix + '.jsonl')
    train_index_path = os.path.join(data_dir, 'abcfiles_processed' + suffix + '_train.jsonl')
    eval_index_path = os.path.join(data_dir, 'abcfiles_processed' + suffix + '_eval.jsonl')

    # Env overrides so an isolated build (e.g. a 50k-Irish set) writes its synthetic
    # crops and index entries to separate locations without clobbering the default set.
    # SYNTH_OUTPUT_DIR: where synthetic .abc files are written (wiped+recreated).
    # SYNTH_{REAL,TRAIN,EVAL}_INDEX: the jsonl indices to read real keys from / refresh.
    output_dir = os.environ.get('SYNTH_OUTPUT_DIR', output_dir)
    real_index_path = os.environ.get('SYNTH_REAL_INDEX', real_index_path)
    train_index_path = os.environ.get('SYNTH_TRAIN_INDEX', train_index_path)
    eval_index_path = os.environ.get('SYNTH_EVAL_INDEX', eval_index_path)

    # ---- Parameters --------------------------------------------------------
    WINDOW_RADIUS = 5      # bars kept on each side of the motif
    TOP_N_MOTIFS = 3       # per source piece AND motif length: one synthetic piece per top motif
    # Motif lengths to cover: TOP_N_MOTIFS crops per piece for EACH length. Keep in
    # sync with MOTIF_LENGTHS in data/2_data_preprocess.py.
    MOTIF_LENGTHS = list(range(4, 11))
    EVAL_SPLIT = 0.1       # fraction of synthetic pieces routed to the eval index
    _max_pieces_env = os.environ.get('SYNTH_MAX_PIECES')
    MAX_PIECES = int(_max_pieces_env) if _max_pieces_env else None  # cap pieces per source for a quick run; None = all
    SEED = 0
    INTERVAL_MODE = 'step_skip_leap'
    # Source corpora to crop from. Each is a processed (15-key augmented) folder laid
    # out as <dir>/<key>/<piece>_<key>.abc. The Irishman corpus is augmented onto
    # scratch (see scripts/run_preprocess_irishman.sh); include it when present. Override the
    # extra sources with a ':'-separated SYNTH_EXTRA_SOURCES env var if needed.
    sources = [processed_dir]
    if VOICE1_ONLY:
        extra_env = os.environ.get('SYNTH_EXTRA_SOURCES')
        extra = (extra_env.split(':') if extra_env
                 else ['/usr/xtmp/cy232/accompgen/irishman/abcfiles_processed_v1'])
        for d in extra:
            if d and os.path.isdir(os.path.join(d, REFERENCE_KEY)):
                sources.append(d)
            elif d:
                print(f'Note: skipping missing source {d}')
    # ------------------------------------------------------------------------

    random.seed(SEED)

    # Map piece basename -> original key (for the jsonl `key` field, so synthetic
    # data follows the same key distribution as real data at load time). Index
    # basenames carry a _len<L> suffix in the multi-length layout while the folder
    # scan below may keep a differently-suffixed (or unsuffixed) variant of the
    # same piece, so both sides are keyed by the _len-stripped base name (all
    # length variants of a piece share its key).
    name_to_key = {}
    if os.path.exists(real_index_path):
        with open(real_index_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                base = re.sub(r'_len\d+$', '', os.path.basename(entry.get('path', '')))
                name_to_key[base] = entry.get('key', REFERENCE_KEY)

    ref_suffix = f'_{REFERENCE_KEY}.abc'

    # Wipe and recreate the synthetic output folder so stale files are removed.
    if os.path.exists(output_dir):
        print(f'Clearing old synthetic data in {output_dir} ...')
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    synthetic_data = []
    synth_idx = 0
    n_no_motif = 0
    n_unlocated = 0

    # Disable automatic garbage collection during the generation loop. The results
    # list accumulates hundreds of thousands of dicts; with automatic GC enabled,
    # recurring generation-2 collections rescan all of those live objects and the
    # per-piece cost degrades progressively (observed: ~21 pieces/s at the start of
    # a full-Irishman run decaying to ~1.2 pieces/s by 500k pieces). The loop
    # allocates almost no reference cycles, so bounded manual collects suffice.
    gc.disable()
    _progress_t0 = time.time()
    _progress_last = _progress_t0
    _sources_seen = 0

    for src_dir in sources:
        ref_dir = os.path.join(src_dir, REFERENCE_KEY)
        if not os.path.isdir(ref_dir):
            print(f'Reference key dir not found, skipping source: {ref_dir}')
            continue
        raw_names = sorted(
            fname[:-len(ref_suffix)]
            for fname in os.listdir(ref_dir)
            if fname.endswith(ref_suffix)
        )
        # Multi-length real datasets carry several files per piece (piece_len4 ..
        # piece_len10) that share one tunebody; keep a single variant per piece so
        # each source tunebody is cropped once, not once per length variant.
        variant_of = {}
        for name in raw_names:
            variant_of.setdefault(re.sub(r'_len\d+$', '', name), name)
        piece_names = sorted(variant_of.values())
        if MAX_PIECES is not None:
            piece_names = piece_names[:MAX_PIECES]
        print(f'Source {src_dir}: {len(piece_names)} pieces in reference key {REFERENCE_KEY}.')

        for piece in piece_names:
            _sources_seen += 1
            if _sources_seen % 5000 == 0:
                gc.collect()
                now = time.time()
                rate = 5000 / max(now - _progress_last, 1e-9)
                print(f'  ...{_sources_seen} sources, {synth_idx} synthetic pieces, '
                      f'{rate:.1f} sources/s (elapsed {(now - _progress_t0)/60:.0f} min)',
                      flush=True)
                _progress_last = now
            # Find the top motifs on the clean reference key, using the shared backbone.
            ref_path = os.path.join(ref_dir, piece + ref_suffix)
            parsed = parse_processed_file(ref_path)
            if parsed is None:
                continue
            _motif_block, _ref_abc, _meta, ref_tunebody = parsed
            ref_v1_text = '\n'.join(ref_tunebody)
            motifs_by_len = get_best_motifs_per_length(
                ref_v1_text, window_range=(MOTIF_LENGTHS[0], MOTIF_LENGTHS[-1] + 1),
                interval_mode=INTERVAL_MODE, key=REFERENCE_KEY, fold_forms=('I',))
            if not motifs_by_len:
                n_no_motif += 1
                continue
            # Meter / unit length for the rhythmic motif of each crop (identical
            # across keys: transposition changes pitches, not durations).
            _meta_text = '\n'.join(_meta)
            crop_meter = rx.detect_meter_field(_meta_text)
            crop_unit = rx.detect_unit_length_field(_meta_text)

            # Cache each key's parsed (metadata, tunebody) once for this piece.
            key_parsed = {}
            for key in ALL_KEYS:
                pk = parse_processed_file(os.path.join(src_dir, key, f'{piece}_{key}.abc'))
                if pk is not None:
                    key_parsed[key] = (pk[2], pk[3])  # (metadata, tunebody)

            # Emit one synthetic piece per top motif of EACH length, cropped to
            # +-WINDOW_RADIUS bars around its own occurrence and carrying only
            # that single motif.
            for motif_len in sorted(motifs_by_len):
                for motif in motifs_by_len[motif_len][:TOP_N_MOTIFS]:
                    span = locate_motif_span(motif['abc'], ref_tunebody)
                    if span is None:
                        n_unlocated += 1
                        continue
                    mfl, mll = span
                    lo = max(0, mfl - WINDOW_RADIUS)
                    hi = mll + WINDOW_RADIUS
                    hi_ref = min(hi, len(ref_tunebody) - 1)
                    pat_str = ','.join(str(x) for x in motif['pattern'])

                    # The header must describe the CROP, not the whole piece: recount
                    # the motif and its inversion inside the cropped bars, and take the
                    # inversion instance from the first inversion occurrence within them.
                    crop_text = '\n'.join(ref_tunebody[lo:hi_ref + 1])
                    c_pitches, c_bars, c_spans, c_s, _ = mx.abc_to_pitches_with_bars(crop_text, key=REFERENCE_KEY)
                    pat = tuple(int(x) for x in motif['pattern'])
                    inv_pat = mx._pattern_transforms(pat)[1]
                    n_rectus = n_inv = 0
                    inv_span = None   # (first_line, last_line) of the inversion instance
                    if len(c_pitches) >= motif_len:
                        c_counts, c_occ = mx.count_interval_motifs(
                            c_pitches, c_bars, window_notes=motif_len, stride_notes=1,
                            interval_mode=INTERVAL_MODE, fold_transformations=False)
                        n_rectus = c_counts.get(pat, 0)
                        n_inv = c_counts.get(inv_pat, 0)
                        if n_inv:
                            o = c_occ[inv_pat][0]
                            # stripped text keeps one physical line per tunebody line,
                            # so the newline count before a note is its crop line index
                            l0 = lo + c_s.count('\n', 0, c_spans[o['start_note']][0])
                            l1 = lo + c_s.count('\n', 0, c_spans[o['end_note']][0])
                            inv_span = (l0, l1)
                    # The rhythmic motif of the same length, found INSIDE the crop
                    # (the header describes the crop). Its first occurrence is
                    # located as a line span so each key rebuilds its own snippet.
                    r_by_len = get_best_rhythm_motifs_per_length(
                        crop_text, window_range=(motif_len, motif_len + 1),
                        meter=crop_meter, unit_length=crop_unit)
                    r_top = r_by_len.get(motif_len)
                    if n_rectus == 0 or not r_top:
                        n_unlocated += 1   # crop does not contain the motif as counted; skip
                        continue
                    r = r_top[0]
                    r_pat_str = ','.join(r['pattern'])
                    r_span = (lo + r['start_line'], lo + r['end_line'])

                    name = f'synth_{synth_idx}'
                    wrote_any = False
                    for key, (k_meta, k_tunebody) in key_parsed.items():
                        if mll >= len(k_tunebody):  # bar counts match across keys; guard anyway
                            continue
                        k_hi = min(hi, len(k_tunebody) - 1)
                        k_snippet = motif_snippet_for_key(k_tunebody, mfl, mll)
                        k_rsnippet = ''
                        if r_span[1] < len(k_tunebody):
                            k_rsnippet = motif_snippet_for_key(k_tunebody, r_span[0], r_span[1])
                        if not k_snippet or not k_rsnippet:
                            continue
                        motif_block = [
                            f'%motif:v1:{INTERVAL_MODE}: {pat_str} ',
                            f'%motif:v1:rhythm: {r_pat_str} ',
                            f'%motif:count: {n_rectus} ',
                            f'%motif:abc: {k_snippet} ',
                            f'%motif:rhythm:count: {r["count"]} ',
                            f'%motif:rhythm:abc: {k_rsnippet} ',
                            f'%motif:inversion_count: {n_inv} ',
                        ]
                        if inv_span is not None and inv_span[1] < len(k_tunebody):
                            k_inv = motif_snippet_for_key(k_tunebody, inv_span[0], inv_span[1])
                            if k_inv:
                                motif_block.append(f'%motif:abc:inversion_instance: {k_inv} ')
                        out_lines = motif_block + k_meta + k_tunebody[lo:k_hi + 1]

                        key_folder = os.path.join(output_dir, key)
                        os.makedirs(key_folder, exist_ok=True)
                        with open(os.path.join(key_folder, f'{name}_{key}.abc'), 'w', encoding='utf-8') as fh:
                            fh.write('\n'.join(out_lines) + '\n')
                        wrote_any = True

                    if wrote_any:
                        synthetic_data.append({
                            'path': os.path.join(output_dir, name),
                            'key': name_to_key.get(re.sub(r'_len\d+$', '', piece), REFERENCE_KEY),
                        })
                        synth_idx += 1

    gc.enable()
    gc.collect()

    print(f'Generated {synth_idx} synthetic pieces '
          f'({synth_idx * len(ALL_KEYS)} key-specific files).')
    if n_no_motif:
        print(f'  Skipped {n_no_motif} pieces with no extractable motif.')
    if n_unlocated:
        print(f'  Skipped {n_unlocated} motif(s) that could not be located.')

    # Split train/eval at the PIECE level (all 15 key files of a piece are in the
    # same split) and refresh the JSONL indices.
    random.shuffle(synthetic_data)
    split_idx = int(EVAL_SPLIT * len(synthetic_data))
    eval_data = synthetic_data[:split_idx]
    train_data = synthetic_data[split_idx:]

    real_n, syn_n = refresh_jsonl(train_index_path, train_data)
    print(f'Train index: kept {real_n} real entries, wrote {syn_n} synthetic entries.')

    real_n, syn_n = refresh_jsonl(eval_index_path, eval_data)
    print(f'Eval  index: kept {real_n} real entries, wrote {syn_n} synthetic entries.')

    print('Done.')


if __name__ == '__main__':
    main()
