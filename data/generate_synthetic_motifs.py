"""Generate synthetic motif training data from real motifs.

Approach:
  1. Sample real motifs of various lengths from abcfiles_processed/.
  2. Augment each motif:
       a. Perturb rhythmic durations (pitches preserved).
       b. Fresh re-realizations from the same step/skip/leap pattern.
  3. Transpose every variant to all 15 keys (diatonic note-letter shift).
  4. Output files contain ONLY the two %%motif comment lines.
"""

import json
import os
import random
import re

# ---------------------------------------------------------------------------
# Diatonic transposition helpers
# ---------------------------------------------------------------------------

DIATONIC_NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# Diatonic distance (in letter-steps) of each key's tonic from C.
# We use only the letter, ignoring chromatic alterations (Db == D letter-wise).
KEY_DIATONIC_OFFSET = {
    'Cb': 0, 'C': 0, 'C#': 0,
    'Db': 1, 'D': 1,
    'Eb': 2, 'E': 2,
    'F': 3, 'F#': 3,
    'Gb': 4, 'G': 4,
    'Ab': 5, 'A': 5,
    'Bb': 6, 'B': 6,
}

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

ALL_KEYS = ['Cb', 'C', 'C#', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']


def _note_to_diatonic_idx(letter: str, octave_marks: str) -> int:
    """Map an ABC note to an integer diatonic index.

    C=0, D=1, ..., B=6, c=7, ..., b=13, c'=14, C,=-7, etc.
    """
    base = DIATONIC_NOTES.index(letter.upper())
    if letter.islower():
        base += 7
    for ch in octave_marks:
        if ch == "'":
            base += 7
        elif ch == ',':
            base -= 7
    return base


def _diatonic_idx_to_note(idx: int) -> str:
    """Convert a diatonic index back to an ABC note string (no accidental, no duration)."""
    letter = DIATONIC_NOTES[((idx % 7) + 7) % 7]
    octave_num = idx // 7          # Python floor-div handles negatives correctly
    if octave_num >= 1:
        return letter.lower() + "'" * (octave_num - 1)
    elif octave_num == 0:
        return letter
    else:
        return letter + ',' * (-octave_num)


# Regex: optional accidentals, note letter, optional octave marks, optional duration
_NOTE_RE = re.compile(r'([_^=]*)([A-Ga-g])([,\']*)(\d*/*\d*)')


def transpose_abc_fragment(abc_str: str, diatonic_steps: int) -> str:
    """Shift every note letter in an ABC fragment by `diatonic_steps`.

    When steps != 0: accidentals are stripped since they won't be valid after
    a diatonic shift.  When steps == 0: the fragment is returned unchanged.
    Everything else (barlines, slurs, rests, text annotations) is kept.
    """
    if diatonic_steps == 0:
        return abc_str

    def _replace(m):
        _acc, letter, oct_marks, dur = m.groups()
        new_idx = _note_to_diatonic_idx(letter, oct_marks) + diatonic_steps
        return _diatonic_idx_to_note(new_idx) + dur

    return _NOTE_RE.sub(_replace, abc_str)


# ---------------------------------------------------------------------------
# Duration perturbation
# ---------------------------------------------------------------------------

SIMPLE_DURATIONS = ['', '2', '/', '/2', '3', '4', '3/2']


def perturb_abc_fragment(abc_str: str) -> str:
    """Randomise the duration of every note while keeping all pitches intact.

    Non-note tokens (barlines, slurs, text annotations, rests) are untouched.
    """
    def _replace(m):
        acc, letter, oct_marks, _dur = m.groups()
        return acc + letter + oct_marks + random.choice(SIMPLE_DURATIONS)

    return _NOTE_RE.sub(_replace, abc_str)


# ---------------------------------------------------------------------------
# Fresh ABC realization from a step/skip/leap pattern
# ---------------------------------------------------------------------------

def pattern_to_abc(pattern, start_note='C', duration_mode='uniform'):
    """Realize a step/skip/leap pattern as an ABC fragment.

    pattern  : list of ints (0, ±1, ±2, ±3 = reference, step, skip, leap)
    start_note: one of the entries in the diatonic note array below
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
# Load real motifs
# ---------------------------------------------------------------------------

def load_real_motifs(processed_dir: str, min_len: int = 3, max_len: int = 8):
    """Yield (key, pattern_list, abc_str) for every individual motif in the
    processed corpus.  key is the folder name (e.g. 'C', 'Bb').
    """
    motifs = []
    for key in ALL_KEYS:
        key_dir = os.path.join(processed_dir, key)
        if not os.path.isdir(key_dir):
            continue
        for fname in os.listdir(key_dir):
            if not fname.endswith('.abc'):
                continue
            fpath = os.path.join(key_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as fh:
                    content = fh.read()
            except OSError:
                continue

            pattern_line = abc_line = None
            for line in content.splitlines():
                if line.startswith('%motif:v1:step_skip_leap:'):
                    pattern_line = line[len('%motif:v1:step_skip_leap:'):].strip()
                elif line.startswith('%motif:abc:'):
                    abc_line = line[len('%motif:abc:'):].strip()

            if not pattern_line or not abc_line:
                continue

            pat_parts = [p.strip() for p in pattern_line.split(';')]
            abc_parts = [a.strip() for a in abc_line.split(';')]
            if len(pat_parts) != len(abc_parts):
                continue

            for pat_str, abc_str in zip(pat_parts, abc_parts):
                try:
                    pat = [int(x) for x in pat_str.split(',')]
                except ValueError:
                    continue
                if min_len <= len(pat) <= max_len and abc_str:
                    motifs.append((key, pat, abc_str.strip()))

    return motifs


# ---------------------------------------------------------------------------
# Write one set of 15 key-variant files
# ---------------------------------------------------------------------------

def write_key_variants(output_dir: str, name: str, pattern: list, base_abc: str,
                       base_key: str, mode: str = 'transpose'):
    """Write 15 files (one per key) for a single motif variant.

    mode='transpose' : diatonically transpose `base_abc` to each key.
    mode='rerealiz'  : re-realize `pattern` from each key's start note.

    Returns list of (path_without_key, key) for the JSONL index.
    """
    pat_str = ','.join(map(str, pattern))
    base_offset = KEY_DIATONIC_OFFSET[base_key]
    records = []

    for key in ALL_KEYS:
        if mode == 'transpose':
            steps = KEY_DIATONIC_OFFSET[key] - base_offset
            abc_content = transpose_abc_fragment(base_abc, steps)
        else:  # rerealiz
            start_note = KEY_TO_START_NOTE[key]
            dur_mode = random.choice(['uniform', 'varied'])
            abc_content = pattern_to_abc(pattern, start_note=start_note, duration_mode=dur_mode)

        key_folder = os.path.join(output_dir, key)
        os.makedirs(key_folder, exist_ok=True)
        filepath = os.path.join(key_folder, f'{name}_{key}.abc')

        with open(filepath, 'w', encoding='utf-8') as fh:
            fh.write(f'%motif:v1:step_skip_leap: {pat_str}\n')
            fh.write(f'%motif:abc: {abc_content}\n')

    base_path = os.path.join(output_dir, name)
    records.append({'path': base_path, 'key': base_key})
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(data_dir, 'abcfiles_processed')
    output_dir = os.path.join(data_dir, 'synthetic_motifs')
    train_index_path = os.path.join(data_dir, 'abcfiles_processed_train.jsonl')
    eval_index_path = os.path.join(data_dir, 'abcfiles_processed_eval.jsonl')

    os.makedirs(output_dir, exist_ok=True)

    # ---- Parameters --------------------------------------------------------
    MAX_REAL_MOTIFS = 2000   # cap how many real motifs we use
    N_PERTURB = 2            # rhythm-perturbed variants per real motif
    N_REREALIZ = 2           # fresh pattern re-realizations per real motif
    # ------------------------------------------------------------------------

    print('Loading real motifs from processed data...')
    all_motifs = load_real_motifs(processed_dir, min_len=3, max_len=8)
    print(f'  Found {len(all_motifs)} individual real motifs.')

    random.shuffle(all_motifs)
    sampled = all_motifs[:MAX_REAL_MOTIFS]
    print(f'  Using {len(sampled)} motifs for augmentation.')

    synthetic_data = []
    motif_idx = 0

    for orig_key, pattern, original_abc in sampled:

        # 1. Direct diatonic transpositions of the original realization
        name = f'synth_{motif_idx}'
        records = write_key_variants(output_dir, name, pattern, original_abc,
                                     orig_key, mode='transpose')
        synthetic_data.extend(records)
        motif_idx += 1

        # 2. Duration-perturbed variants (rhythm only, same pitches)
        for _ in range(N_PERTURB):
            perturbed = perturb_abc_fragment(original_abc)
            name = f'synth_{motif_idx}'
            records = write_key_variants(output_dir, name, pattern, perturbed,
                                         orig_key, mode='transpose')
            synthetic_data.extend(records)
            motif_idx += 1

        # 3. Fresh re-realizations from the same abstract pattern
        for _ in range(N_REREALIZ):
            name = f'synth_{motif_idx}'
            records = write_key_variants(output_dir, name, pattern, original_abc,
                                         orig_key, mode='rerealiz')
            synthetic_data.extend(records)
            motif_idx += 1

    total_files = motif_idx * len(ALL_KEYS)
    print(f'Generated {motif_idx} motif variants → {total_files} key-specific files.')

    # Split 90/10 train/eval
    random.shuffle(synthetic_data)
    split_idx = int(0.9 * len(synthetic_data))
    train_data = synthetic_data[:split_idx]
    eval_data = synthetic_data[split_idx:]

    mode = 'a' if os.path.exists(train_index_path) else 'w'
    with open(train_index_path, mode, encoding='utf-8') as fh:
        for d in train_data:
            fh.write(json.dumps(d) + '\n')
    print(f'Wrote {len(train_data)} entries to train index ({mode}).')

    mode = 'a' if os.path.exists(eval_index_path) else 'w'
    with open(eval_index_path, mode, encoding='utf-8') as fh:
        for d in eval_data:
            fh.write(json.dumps(d) + '\n')
    print(f'Wrote {len(eval_data)} entries to eval index ({mode}).')

    print('Done.')


if __name__ == '__main__':
    main()
