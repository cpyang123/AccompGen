import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Literal

NOTE_BASE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
# Diatonic staff position of each letter within an octave (C=0 .. B=6). This is
# the letter-name index, independent of accidentals: it is what tells a 2nd from
# a 3rd. Two notes a fixed number of semitones apart can be different diatonic
# intervals (e.g. an augmented 2nd C-D# and a minor 3rd C-Eb are both 3 semitones
# but span 1 and 2 letter-steps respectively), so semitone count alone cannot
# classify step/skip/leap correctly.
DIA_BASE = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6}
IntervalMode = Literal["chromatic", "diatonic", "step_skip_leap", "contour"]


class _DiaInt(int):
    """A semitone value that also remembers a parallel diatonic value in `.dia`.

    For a *pitch*, `.dia` is its absolute staff position (letter index + 7*octave).
    For an *interval* (pitch difference), `.dia` is the signed number of letter-steps
    it spans (0 = unison, 1 = 2nd, 2 = 3rd, ...). Subtraction propagates `.dia`, so an
    interval pattern derived from these keeps the letter-name distance alongside the
    semitone distance. This lets `_coarsen_pattern` honour enharmonic spelling — e.g.
    an augmented 2nd (3 semitones, 1 letter-step) is classified as a step (2nd), not
    a skip (3rd).

    It subclasses `int`, so it behaves exactly as its semitone value for equality,
    hashing, ordering and arithmetic — all existing semitone-based code (de-duplication
    of repeated notes, chromatic patterns, Counter keys) keeps working unchanged. When
    diatonic info is unavailable (`.dia is None`) callers fall back to the semitone
    heuristic.
    """

    def __new__(cls, value: int, dia: Optional[int] = None) -> "_DiaInt":
        obj = super().__new__(cls, value)
        obj.dia = dia
        return obj

    def __sub__(self, other):
        sem = int(self) - int(other)
        self_dia, other_dia = self.dia, getattr(other, "dia", None)
        dia = self_dia - other_dia if (self_dia is not None and other_dia is not None) else None
        return _DiaInt(sem, dia)

    def __rsub__(self, other):
        sem = int(other) - int(self)
        self_dia, other_dia = self.dia, getattr(other, "dia", None)
        dia = other_dia - self_dia if (self_dia is not None and other_dia is not None) else None
        return _DiaInt(sem, dia)

# Order in which sharps/flats are added to a key signature (circle of fifths).
SHARP_ORDER = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
FLAT_ORDER = ['B', 'E', 'A', 'D', 'G', 'C', 'F']
# Position of each natural letter on the circle of fifths (C = 0).
_BASE_FIFTHS = {'F': -1, 'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5}
# Offset (in fifths) of each church mode relative to ionian/major on the same tonic.
_MODE_OFFSET = {
    'ion': 0, 'maj': 0,
    'lyd': 1,
    'mix': -1,
    'dor': -2,
    'aeo': -3,
    'phr': -4,
    'loc': -5,
}


def _mode_offset(mode_raw: str) -> int:
    """Map an ABC mode token (e.g. 'minor', 'dor', 'mix') to a fifths offset."""
    if not mode_raw:
        return 0
    mode_raw = mode_raw.lower()
    if mode_raw in ('m', 'min', 'minor'):
        return -3
    for prefix, off in _MODE_OFFSET.items():
        if mode_raw.startswith(prefix):
            return off
    return 0  # unknown token (e.g. a clef like 'treble') -> treat as major


def parse_key_signature(key_field: Optional[str]) -> Dict[str, int]:
    """Map a K: field (e.g. "C#", "Bb", "Am", "F# minor") to per-letter accidentals.

    Returns a dict like {'F': 1, 'C': 1}: letter names raised (+1) or lowered (-1)
    by the key signature, applied across all octaves. Inline accidentals on a note
    override this. An empty/None/"none" field yields no key signature.
    """
    if not key_field:
        return {}
    s = key_field.strip()
    if not s or s.lower().startswith('none'):
        return {}
    m = re.match(r'^([A-Ga-g])([#b]?)\s*([A-Za-z]*)', s)
    if not m:
        return {}
    letter = m.group(1).upper()
    acc = m.group(2)
    mode_raw = m.group(3)

    fifths = _BASE_FIFTHS[letter]
    if acc == '#':
        fifths += 7
    elif acc == 'b':
        fifths -= 7
    fifths += _mode_offset(mode_raw)

    sig: Dict[str, int] = {}
    if fifths > 0:
        for k in range(min(fifths, 7)):
            sig[SHARP_ORDER[k]] = 1
    elif fifths < 0:
        for k in range(min(-fifths, 7)):
            sig[FLAT_ORDER[k]] = -1
    return sig


def detect_key_field(abc: str) -> Optional[str]:
    """Return the value of the first K: information field in `abc`, if any."""
    for line in abc.splitlines():
        m = re.match(r'^\s*K:\s*(.+)$', line)
        if m:
            return m.group(1).strip()
    return None


def strip_voice_and_text(abc: str) -> str:
    """Remove lyrics/metadata/voice markers so pitch parsing sees mostly note tokens."""
    # remove quoted annotations
    abc = re.sub(r'\".*?\"', '', abc)
    # remove inline voice fields like [V:1]
    abc = re.sub(r'\[V:[^\]]*\]', '', abc)
    # remove decorations such as !f! !p! !ped! !trill! (and the legacy +...+ form).
    # Their inner letters (e.g. the f in !f!, or p/e/d in !ped!) would otherwise be
    # mis-parsed as notes; being key-independent, they also break transposition
    # invariance.
    abc = re.sub(r'![^!\n]*!', '', abc)
    abc = re.sub(r'\+[^+\n]*\+', '', abc)
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

def abc_to_pitches_with_bars(
    abc: str,
    key: Optional[str] = None,
) -> Tuple[List[int], List[int], List[Tuple[int, int]], str, Dict[int, int]]:
    """
    Parse a single ABC voice into absolute semitone pitches.

    The ABC key signature is applied to notes that lack an inline accidental, and
    inline accidentals persist for the rest of the bar for the same pitch (letter +
    octave), per the ABC standard. Inline accidentals are absolute and override the
    key signature. Honouring the key signature is what makes interval-based motifs
    transposition-invariant: without it, notes altered only by the key signature
    (not by an inline accidental) are mis-read as naturals, so different keys yield
    different motifs for the same piece.

    `key` may be a key name (e.g. "C#", "Bb", "Am"). If None, the key is taken from
    a K: field in `abc` (if present); otherwise no key signature is applied.

    Returns:
      pitches: list of absolute pitches (semitones)
      note_to_bar: list mapping each pitch to its bar index
      note_to_char_idx: list mapping each pitch to its (start, end) character indices in `s`
      s: the stripped ABC string the indices refer to
      bar_starts: mapping bar index -> char offset of the bar's first note region
    """
    key_field = key if key is not None else detect_key_field(abc)
    key_sig = parse_key_signature(key_field)

    s = strip_voice_and_text(abc)
    pitches: List[int] = []
    note_to_bar: List[int] = []
    note_to_char_idx: List[Tuple[int, int]] = []

    i = 0
    n = len(s)
    bar_idx = 0
    # within-bar accidental memory: (letter, octave) -> accidental (semitones)
    bar_accidentals: Dict[Tuple[str, int], int] = {}

    while i < n:
        ch = s[i]

        # barlines increment bar index and reset within-bar accidentals
        if ch == '|':
            bar_idx += 1
            bar_accidentals.clear()
            i += 1
            continue

        # inline information fields such as [K:G] or [M:6/8]: skip them whole so
        # their contents are never read as notes (a bare '[' skip would parse the
        # G of [K:G] as a melody pitch), and apply inline key changes so notes
        # after the change resolve against the new key signature.
        if ch == '[':
            m_field = re.match(r'\[([A-Za-z]):([^\]\n]*)\]', s[i:])
            if m_field:
                if m_field.group(1).upper() == 'K':
                    key_sig = parse_key_signature(m_field.group(2))
                    bar_accidentals.clear()
                i += m_field.end()
                continue
            i += 1  # chord/voltas bracket: skip the bracket only
            continue

        # skip structural characters
        if ch in '{}]() \n\t\r':
            i += 1
            continue

        # rests
        if ch == 'z':
            i += 1
            while i < n and s[i] in '0123456789/':
                i += 1
            continue

        tok_start = i

        # inline accidentals (absolute; '=' is an explicit natural)
        acc = 0
        has_inline = False
        while i < n and s[i] in '^_=':
            has_inline = True
            if s[i] == '^':
                acc += 1
            elif s[i] == '_':
                acc -= 1
            i += 1

        if i >= n:
            break

        ch = s[i]
        if ch in 'ABCDEFGabcdefg':
            letter = ch.upper()
            base = NOTE_BASE[letter]
            octave = 1 if ch.islower() else 0
            i += 1

            # octave marks
            while i < n and s[i] in "',":
                octave += 1 if s[i] == "'" else -1
                i += 1

            # resolve accidental: inline (absolute) > within-bar memory > key signature
            if has_inline:
                eff_acc = acc
                bar_accidentals[(letter, octave)] = acc
            elif (letter, octave) in bar_accidentals:
                eff_acc = bar_accidentals[(letter, octave)]
            else:
                eff_acc = key_sig.get(letter, 0)

            # Carry the diatonic staff position (letter index + 7*octave) so that
            # interval coarsening can distinguish enharmonically-equal intervals.
            # The accidental never changes the staff position.
            dia = DIA_BASE[letter] + 7 * octave
            pitch = _DiaInt(base + 12 * octave + eff_acc, dia)

            # skip duration tokens
            while i < n and s[i] in '0123456789/':
                i += 1

            end_i = i

            pitches.append(pitch)
            note_to_bar.append(bar_idx)
            note_to_char_idx.append((tok_start, end_i))

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

    # First entry is 0, then each is the interval to the previous note. The
    # reference 0 carries a diatonic value of 0; subtraction of _DiaInt pitches
    # propagates the diatonic (letter-step) distance into the later intervals.
    intervals: List[int] = [_DiaInt(0, 0)]
    for i in range(1, len(modified_window)):
        intervals.append(modified_window[i] - modified_window[i - 1])

    return tuple(intervals)

def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)

# Order matters: it is the label-priority used when a degenerate pattern equals
# several of its own transforms (e.g. (0,1,-1) is its own retrograde).
TRANSFORM_LABELS = ("original", "I", "R", "RI")


def _pattern_transforms(pat: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    """The (original, inversion, retrograde, retrograde-inversion) forms of `pat`.

    `pat` is a successive-interval pattern (0, i1, ..., ik). Inversion negates
    every interval; retrograde plays the notes backwards, which reverses AND
    negates the interval tail; retrograde-inversion therefore just reverses it.
    Entries are normalized to plain ints (a `_DiaInt` compares equal to its int
    value, so this changes nothing for counting/hashing).
    """
    tail = tuple(int(x) for x in pat[1:])
    orig = (0,) + tail
    inv = (0,) + tuple(-x for x in tail)
    retro = (0,) + tuple(-x for x in reversed(tail))
    retro_inv = (0,) + tuple(reversed(tail))
    return orig, inv, retro, retro_inv

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

def _diatonic_number(interval: int) -> int:
    """Signed diatonic interval number for one interval entry.

    Prefers the true letter-step distance carried on a `_DiaInt` (so an augmented
    2nd counts as a 2nd, not a 3rd). Falls back to the semitone-based heuristic when
    no diatonic info is available (e.g. an interval built from plain ints).

    A chromatic alteration of the same scale degree (an augmented/diminished unison,
    e.g. F->F#: nonzero semitones but zero letter-steps) is counted as a step (a 2nd,
    ±1), not as "no motion". The two notes are genuinely distinct pitches, and a
    plain-int semitone of 1 already maps to a 2nd via `_chrom_to_diatonic_number`, so
    this keeps the `.dia` path consistent. A returned 0 therefore means the same note
    (identical pitch) — and those are dropped by the upstream repetition collapse, so
    an interior 0 should not occur for distinct successive notes.
    """
    dia = getattr(interval, "dia", None)
    if dia is not None:
        if dia == 0 and int(interval) != 0:
            return _sign(int(interval))
        return dia
    return _chrom_to_diatonic_number(int(interval))


def _coarsen_pattern(pat_chromatic: Tuple[int, ...], interval_mode: IntervalMode) -> Tuple[int, ...]:
    """
    Convert chromatic (semitone) offsets-from-first into a coarser signature.
    """
    if interval_mode == "chromatic":
        return pat_chromatic

    if interval_mode == "diatonic":
        return tuple(_diatonic_number(x) for x in pat_chromatic)

    if interval_mode == "step_skip_leap":
        # Classify by diatonic (letter-step) distance so enharmonic spelling is
        # honoured: an augmented 2nd is a step, a diminished 3rd is a skip, etc.
        out = [0]
        for x in pat_chromatic[1:]:
            d = _diatonic_number(x)
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
    fold_transformations: bool = True,
    fold_forms: Tuple[str, ...] = ("I", "R", "RI"),
) -> Tuple[Counter, Dict[Tuple[int, ...], List[dict]]]:
    """
    Count interval motifs over the melodic line, collapsing immediate note
    repetitions before windowing.

    With `fold_transformations` (default), the inversion (I), retrograde (R) and
    retrograde-inversion (RI) of a motif count as occurrences of that motif. The
    four forms make up one closed orbit (applying I/R/RI to any member reaches
    the other three), so whichever member appears FIRST in scan order becomes
    the orbit's canonical pattern: every later member is counted under it and
    its occurrence dict is labelled with the "transform" ("original"/"I"/"R"/
    "RI") that maps the canonical form to the observed one. The reverse never
    happens — an earlier motif is never re-attributed to a later transform.
    `fold_forms` restricts which transformations fold (e.g. ("I",) folds only the
    inversion; retrograde forms then count as separate motifs).

    For each run of the same pitch, only the first note is kept; the rest are
    dropped, and the sliding window runs over this collapsed sequence. This honours
    the `no_prepetition` intent: a motif is a contour of *distinct* successive notes,
    so a realization that repeats a note within the motif span (e.g. a tied or
    restruck note, "G c c B A") still counts as containing the motif "G c B A". The
    earlier behaviour (insert a rest sentinel for each repeat and skip any window
    touching it) made such occurrences false misses, so the same motif present
    "somewhere" in the body went uncounted.

    `window_notes` therefore counts *distinct* successive notes, not raw notes; a
    window may span more than `window_notes` raw notes when repeats fall inside it.

    Occurrence dicts report `start_note`/`end_note` as indices into the *original*
    `pitches`/`note_to_bar` (the collapsed→original mapping is tracked), so callers
    can map a hit back to its position and bar in the source.
    """
    if window_notes <= 0:
        raise ValueError("window_notes must be >= 1")
    if stride_notes <= 0:
        raise ValueError("stride_notes must be >= 1")
    if len(pitches) != len(note_to_bar):
        raise ValueError("pitches and note_to_bar must have same length")

    # Collapse runs of the same pitch, keeping the first note of each run. Track the
    # original index of each kept note so occurrences map back to the source.
    comp_pitches: List[int] = []
    comp_bars: List[int] = []
    comp_orig_idx: List[int] = []
    for orig_idx, (pitch, bar) in enumerate(zip(pitches, note_to_bar)):
        if comp_pitches and pitch == comp_pitches[-1]:
            continue  # drop immediate repetition
        comp_pitches.append(pitch)
        comp_bars.append(bar)
        comp_orig_idx.append(orig_idx)

    counts = Counter()
    occurrences: Dict[Tuple[int, ...], List[dict]] = defaultdict(list)
    # orbit invariant (min of the four forms) -> canonical (first-seen) pattern
    canon_by_orbit: Dict[Tuple[int, ...], Tuple[int, ...]] = {}

    n = len(comp_pitches)
    bar_offset = 1 if bar_numbering == "1-based" else 0

    for start in range(0, n - window_notes + 1, stride_notes):
        window = comp_pitches[start:start + window_notes]
        window_bars = comp_bars[start:start + window_notes]
        pat_chromatic = motif_pattern_relative_to_first(window)
        pat = _coarsen_pattern(pat_chromatic, interval_mode)

        if fold_transformations:
            forms = _pattern_transforms(pat)  # forms[0] is pat normalized to plain ints
            active = [f for lbl, f in zip(TRANSFORM_LABELS, forms)
                      if lbl == "original" or lbl in fold_forms]
            canon = canon_by_orbit.setdefault(min(active), forms[0])
            transform = next(
                lbl for lbl, form in zip(TRANSFORM_LABELS, _pattern_transforms(canon))
                if form == forms[0] and (lbl == "original" or lbl in fold_forms)
            )
            pat = canon
        else:
            transform = "original"

        counts[pat] += 1
        occurrences[pat].append({
            "start_note": comp_orig_idx[start],
            "end_note": comp_orig_idx[start + window_notes - 1],
            "start_bar": window_bars[0] + bar_offset,
            "transform": transform,
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
    key: Optional[str] = None,
    fold_transformations: bool = True,
    fold_forms: Tuple[str, ...] = ("I", "R", "RI"),
):
    pitches, note_to_bar, note_to_char_idx, s, bar_starts = abc_to_pitches_with_bars(abc, key=key)
    counts, occurrences = count_interval_motifs(
        pitches,
        note_to_bar,
        window_notes=window_notes,
        stride_notes=stride_notes,
        interval_mode=interval_mode,
        bar_numbering=bar_numbering,
        fold_transformations=fold_transformations,
        fold_forms=fold_forms,
    )

    # augment occurrences with original abc string snippet
    for pat in occurrences:
        for occ in occurrences[pat]:
            start_note_idx = occ["start_note"]
            # end_note is reported by count_interval_motifs as an index into the
            # original pitch list; recomputing it from window_notes would be wrong
            # because a window may span collapsed note repetitions.
            end_note_idx = occ["end_note"]

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

def get_best_motifs_per_length(
    abc_voice: str,
    window_range: Tuple[int, int] = (4, 11),
    interval_mode: IntervalMode = "contour",
    min_count: int = 2,
    key: Optional[str] = None,
    fold_forms: Tuple[str, ...] = ("I", "R", "RI"),
) -> Dict[int, List[Dict]]:
    """Top motifs for EVERY window length in `window_range`, not just the best one.

    Returns {window_notes: [motif dicts]} where each motif dict carries the
    abstract "pattern", its first realization "abc", its occurrence "count",
    "transforms": {label: abc snippet} holding the first realization of each
    folded transformation that actually occurs in the voice, and "count_by_form":
    {label: n} splitting "count" into original vs transformed occurrences. The
    "pattern" and "abc" are always the canonical (first-seen) form; transformed
    occurrences fold into its count (see count_interval_motifs).
    A length is absent from the result when the voice is too short to fit a
    single window of that length. One call costs the same as get_best_motifs()
    (which already sweeps all lengths internally but keeps only the top scorer).
    """
    per_length: Dict[int, List[Dict]] = {}

    for w in range(window_range[0], window_range[1]):
        meta, motifs = find_top_motif(
            abc_voice,
            interval_mode=interval_mode,
            window_notes=w,
            stride_notes=1,
            min_count=min_count,
            key=key,
            fold_forms=fold_forms,
        )
        if not motifs:
            continue

        top_motifs = []
        for pat, info in motifs.items():
            # occurrences[0] is the canonical (first-seen) form by construction,
            # so "abc" is always an original-form realization.
            abc_snippet = ""
            if info["occurrences"]:
                abc_snippet = info["occurrences"][0].get("abc_snippet", "")
            transform_abcs: Dict[str, str] = {}
            count_by_form: Counter = Counter()
            for occ in info["occurrences"]:
                lbl = occ.get("transform", "original")
                count_by_form[lbl] += 1
                if lbl != "original" and lbl not in transform_abcs:
                    snip = occ.get("abc_snippet", "")
                    if snip:
                        transform_abcs[lbl] = snip
            top_motifs.append({
                "pattern": pat,
                "abc": abc_snippet,
                "count": info["count"],
                "count_by_form": dict(count_by_form),
                "transforms": transform_abcs,
            })
        per_length[w] = top_motifs

    return per_length


def get_best_motifs(
    abc_voice: str,
    window_range: Tuple[int, int] = (4, 11),
    interval_mode: IntervalMode = "contour",
    min_count: int = 2,
    key: Optional[str] = None,
) -> List[Dict]:
    """Top motifs of the single best-scoring window length (highest max count;
    ties go to the shorter window). Kept as the length-agnostic entry point."""
    per_length = get_best_motifs_per_length(
        abc_voice, window_range=window_range, interval_mode=interval_mode,
        min_count=min_count, key=key,
    )

    best_motifs: List[Dict] = []
    best_score = float("-inf")
    for w in sorted(per_length):
        max_count = max(m["count"] for m in per_length[w])
        if max_count > best_score:
            best_score = max_count
            best_motifs = per_length[w]

    return best_motifs
