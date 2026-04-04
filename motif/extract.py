import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Literal

NOTE_BASE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
IntervalMode = Literal["chromatic", "diatonic", "step_skip_leap", "contour"]


def strip_voice_and_text(abc: str) -> str:
    """Remove lyrics/metadata/voice markers so pitch parsing sees mostly note tokens."""
    # remove quoted annotations
    abc = re.sub(r'\".*?\"', '', abc)
    # remove inline voice fields like [V:1]
    abc = re.sub(r'\[V:[^\]]*\]', '', abc)
    # drop comment + metadata/lyrics lines (e.g., K:, M:, w:)
    kept_lines: List[str] = []
    for line in abc.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith('%'):
            continue
        if re.match(r'^[A-Za-z]:', s):
            # information field (including w:/W: lyrics)
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)

def abc_to_pitches_with_bars(abc: str) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """
    Returns:
      pitches: list of absolute pitches (semitones)
      note_to_bar: list mapping each pitch to its bar index
      note_to_char_idx: list mapping each pitch to its (start, end) character indices in `s`
    """
    s = strip_voice_and_text(abc)
    pitches: List[int] = []
    note_to_bar: List[int] = []
    note_to_char_idx: List[Tuple[int, int]] = []

    i = 0
    n = len(s)
    bar_idx = 0

    while i < n:
        ch = s[i]

        # barlines increment bar index
        if ch == '|':
            bar_idx += 1
            i += 1
            continue

        # skip structural characters
        if ch in '{}[]() \n\t\r':
            i += 1
            continue

        # rests
        if ch == 'z':
            i += 1
            while i < n and s[i] in '0123456789/':
                i += 1
            continue

        # accidentals
        acc = 0
        while i < n and s[i] in '^_=':
            if s[i] == '^':
                acc += 1
            elif s[i] == '_':
                acc -= 1
            i += 1

        if i >= n:
            break

        ch = s[i]
        if ch in 'ABCDEFGabcdefg':
            start_i = i - abs(acc)  # accidentals happen before the note
            base = NOTE_BASE[ch.upper()]
            octave = 12 if ch.islower() else 0
            pitch = base + octave + acc
            i += 1

            # octave marks
            while i < n and s[i] in "',":
                pitch += 12 if s[i] == "'" else -12
                i += 1

            # skip duration tokens
            while i < n and s[i] in '0123456789/':
                i += 1
            
            end_i = i

            pitches.append(pitch)
            note_to_bar.append(bar_idx)
            note_to_char_idx.append((start_i, end_i))

        else:
            i += 1

    bar_starts = {0: 0}
    current_bar = 0
    for j, ch in enumerate(s):
        if ch == '|':
            current_bar += 1
            bar_starts[current_bar] = j + 1
    
    return pitches, note_to_bar, note_to_char_idx, s, bar_starts

def motif_pattern_relative_to_first(window: List[int], no_prepetition: bool = True) -> Tuple[int, ...]:
    """Encode a window as successive intervals relative to the previous note.

    First element is 0 (reference), then each subsequent entry is the difference
    from the immediately preceding note in the (optionally de-duplicated) window.
    """
    if not window:
        return tuple()

    # Optionally remove immediate repetitions (pre-duplication).
    modified_window: List[int] = [window[0]]
    if no_prepetition:
        prev_note = window[0]
        for p in window[1:]:
            if p == prev_note:
                continue
            modified_window.append(p)
            prev_note = p
    else:
        modified_window = list(window)

    if not modified_window:
        return tuple()

    # First entry is 0, then each is the interval to the previous note.
    intervals: List[int] = [0]
    for i in range(1, len(modified_window)):
        intervals.append(modified_window[i] - modified_window[i - 1])

    return tuple(intervals)

def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)

def _chrom_to_diatonic_number(semitones: int) -> int:
    """
    Map chromatic interval size to diatonic interval number (quality-free).
    Uses nearest/common mapping:
      0->0
      1/2->1 (2nd)
      3/4->2 (3rd)
      5->3 (4th)
      6/7->4 (5th)   (tritone grouped with 5th here; tweak if you want)
      8/9->5 (6th)
      10/11->6 (7th)
      12->7 (8ve)
    and repeats every 12 with octave increments.
    """
    sgn = _sign(semitones)
    a = abs(semitones)
    octs, rem = divmod(a, 12)

    if rem == 0:
        base = 0
    elif rem in (1, 2):
        base = 1
    elif rem in (3, 4):
        base = 2
    elif rem == 5:
        base = 3
    elif rem in (6, 7):
        base = 4
    elif rem in (8, 9):
        base = 5
    else:  # 10,11
        base = 6

    return sgn * (base + 7 * octs)

def _coarsen_pattern(pat_chromatic: Tuple[int, ...], interval_mode: IntervalMode) -> Tuple[int, ...]:
    """
    Convert chromatic (semitone) offsets-from-first into a coarser signature.
    """
    if interval_mode == "chromatic":
        return pat_chromatic

    if interval_mode == "diatonic":
        return tuple(_chrom_to_diatonic_number(x) for x in pat_chromatic)

    if interval_mode == "step_skip_leap":
        # first convert to diatonic numbers, then bucket by size
        diat = [_chrom_to_diatonic_number(x) for x in pat_chromatic]
        out = [0]
        for d in diat[1:]:
            ad = abs(d)
            if ad == 0:
                cls = 0
            elif ad == 1:
                cls = 1      # step (2nd)
            elif ad == 2:
                cls = 2      # skip (3rd)
            else:
                cls = 3      # leap (4th+)
            out.append(_sign(d) * cls)
        return tuple(out)

    if interval_mode == "contour":
        return tuple(_sign(x) for x in pat_chromatic)

    raise ValueError(f"Unknown interval_mode: {interval_mode}")

def count_interval_motifs(
    pitches: List[int],
    note_to_bar: List[int],
    window_notes: int,
    stride_notes: int = 1,
    interval_mode: IntervalMode = "chromatic",
    bar_numbering: Literal["0-based", "1-based"] = "1-based",
) -> Tuple[Counter, Dict[Tuple[int, ...], List[dict]]]:
    """
    Like count_interval_motifs, but removes same-note repetitions before windowing.
    For each run of repeated notes, only the first is kept; the rest are replaced with a rest (sentinel -1).
    Windows containing any rests are skipped.
    """
    if window_notes <= 0:
        raise ValueError("window_notes must be >= 1")
    if stride_notes <= 0:
        raise ValueError("stride_notes must be >= 1")
    if len(pitches) != len(note_to_bar):
        raise ValueError("pitches and note_to_bar must have same length")

    # Preprocess: replace repeated notes with rests (-1)
    processed_pitches = []
    processed_note_to_bar = []
    prev_pitch = None
    for pitch, bar in zip(pitches, note_to_bar):
        if pitch == prev_pitch:
            processed_pitches.append(-1)  # rest
        else:
            processed_pitches.append(pitch)
            prev_pitch = pitch
        processed_note_to_bar.append(bar)

    counts = Counter()
    occurrences: Dict[Tuple[int, ...], List[dict]] = defaultdict(list)

    n = len(processed_pitches)
    bar_offset = 1 if bar_numbering == "1-based" else 0

    for start in range(0, n - window_notes + 1, stride_notes):
        window = processed_pitches[start:start + window_notes]
        window_bars = processed_note_to_bar[start:start + window_notes]
        if -1 in window:
            continue  # skip windows containing rests
        pat_chromatic = motif_pattern_relative_to_first(window)
        pat = _coarsen_pattern(pat_chromatic, interval_mode)

        counts[pat] += 1
        occurrences[pat].append({
            "start_note": start,
            "start_bar": window_bars[0] + bar_offset,
            "chromatic_pattern": pat_chromatic if interval_mode != "chromatic" else None
        })

    return counts, occurrences

def find_top_motif(
    abc: str,
    window_notes: int,
    stride_notes: int = 1,
    min_count: int = 2,
    interval_mode: IntervalMode = "chromatic",
    bar_numbering: Literal["0-based", "1-based"] = "1-based",
):
    pitches, note_to_bar, note_to_char_idx, s, bar_starts = abc_to_pitches_with_bars(abc)
    counts, occurrences = count_interval_motifs(
        pitches,
        note_to_bar,
        window_notes=window_notes,
        stride_notes=stride_notes,
        interval_mode=interval_mode,
        bar_numbering=bar_numbering,
    )

    # augment occurrences with original abc string snippet
    for pat in occurrences:
        for occ in occurrences[pat]:
            start_note_idx = occ["start_note"]
            end_note_idx = start_note_idx + window_notes - 1
            
            # Find the bars containing the motif
            start_bar = note_to_bar[start_note_idx]
            end_bar = note_to_bar[end_note_idx]
            
            # Map bars back to character indices in original string `s`
            char_start = bar_starts.get(start_bar, 0)
            
            # The end character is the start of the next bar (which is the | character),
            # or the end of the string if it's the last bar.
            # Actually, to include the bar contents, we go up to the | of the end_bar,
            # which means up to bar_starts[end_bar + 1] - 1.
            if end_bar + 1 in bar_starts:
                char_end = bar_starts[end_bar + 1] - 1
            else:
                char_end = len(s)
                
            # If we want to include the surrounding barlines, we could adjust this,
            # but usually the bar content itself is what's requested. 
            # We'll include the bar content. Let's make sure we strip any trailing spaces.
            occ["abc_snippet"] = s[char_start:char_end].strip()

    return {
        "total_notes": len(pitches),
        "interval_mode": interval_mode,
        "window_notes": window_notes,
        "stride_notes": stride_notes,
        }, {pat: {
        "count": c,
        "occurrences": occurrences.get(pat, []),
    } for pat, c in counts.most_common(3)}

def get_best_motifs(
    abc_voice: str,
    window_range: Tuple[int, int] = (4, 11),
    interval_mode: IntervalMode = "contour",
    min_count: int = 2
) -> List[Dict]:
    
    best = {
        "window_notes": None,
        "max_count": 0,
        "score": float("-inf"),
        "top_motifs": [],   # list of motif dicts
        "meta": None,
    }

    for w in range(window_range[0], window_range[1]):
        res = find_top_motif(
            abc_voice,
            interval_mode=interval_mode,
            window_notes=w,
            stride_notes=1,
            min_count=min_count,
        )

        meta, motifs = res

        if not motifs:  
            continue

        max_count = max(v["count"] for v in motifs.values())
        score = max_count

        if score > best["score"]:
            # Extract patterns and their abc snippets for top-3
            top_motifs = []
            for pat, info in motifs.items():
                abc_snippet = ""
                if info["occurrences"]:
                    abc_snippet = info["occurrences"][0].get("abc_snippet", "")
                top_motifs.append({
                    "pattern": pat,
                    "abc": abc_snippet
                })
            
            best.update(
                window_notes=w,
                max_count=max_count,
                score=score,
                top_motifs=top_motifs,
                meta=meta,
            )
            
    return best["top_motifs"]
