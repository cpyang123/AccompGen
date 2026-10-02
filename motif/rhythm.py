"""Rhythmic motif extraction: the duration-ratio counterpart of extract.py.

A rhythmic motif is a window of successive note durations, each expressed as a
ratio to the beat unit implied by the piece's *initial* time signature, taken
literally from its denominator (4/4 and 3/4 -> quarter, 6/8 -> eighth, 2/2 ->
half). Under 4/4 the run "eighth eighth quarter half" is the pattern
('1/2', '1/2', '1', '2'). Rests are events too, written as their ratio with a
'z' suffix ('1z' = quarter rest under 4/4): they sit *inside* a window but do
not count toward its length, so a length-L window holds exactly L sounding
notes plus whatever rests fall between them. A rest before the first or after
the last note of a window is outside it.

Differences from the melodic backbone (extract.py), all deliberate:
  - no collapsing of repeated values (equal successive durations are the point),
  - no transformation folding (inversion has no rhythmic meaning),
  - ties merge into one event of the played duration, a chord is one event,
    grace notes are dropped, tuplets and broken rhythms are resolved,
  - ratios are exact fractions ('1/3', '3/2'), never decimals.

Mid-piece meter changes ([M:...]) are ignored: the initial meter rules for the
whole piece. Inline [L:...] changes ARE honoured, because they change what a
duration token means rather than how it is measured.

Header lines emitted by the data pipeline (mirroring the melodic block):
    %motif:v1:rhythm: 1/2,1/2,1,2
    %motif:rhythm:count: <occurrences>
    %motif:rhythm:abc: <first realization, bar text>
"""

import re
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .extract import strip_voice_and_text

Meter = Tuple[int, int]

# ---------------------------------------------------------------------------
# Header fields: M: (meter) and L: (unit note length)
# ---------------------------------------------------------------------------

def parse_meter_field(value: Optional[str]) -> Optional[Meter]:
    """'6/8' -> (6, 8); 'C' -> (4, 4); 'C|' -> (2, 2); '(3+2)/8' -> (5, 8).
    'none'/'free'/empty -> None (no meter)."""
    if value is None:
        return None
    v = value.strip()
    if not v or v.lower() in ('none', 'free'):
        return None
    if v == 'C':
        return (4, 4)
    if v == 'C|':
        return (2, 2)
    m = re.match(r'^\(?([\d+\s]+)\)?\s*/\s*(\d+)', v)
    if not m:
        return None
    num = sum(int(x) for x in re.findall(r'\d+', m.group(1)))
    den = int(m.group(2))
    if num <= 0 or den <= 0:
        return None
    return (num, den)


def parse_length_field(value: Optional[str]) -> Optional[Fraction]:
    """'1/8' -> Fraction(1, 8). None/invalid -> None."""
    if value is None:
        return None
    m = re.match(r'^\s*(\d+)\s*/\s*(\d+)', value)
    if not m or int(m.group(2)) == 0:
        return None
    return Fraction(int(m.group(1)), int(m.group(2)))


def detect_meter_field(abc: str) -> Optional[Meter]:
    """The first M: header line in `abc` (None if absent or unparseable)."""
    for line in abc.splitlines():
        m = re.match(r'^\s*M:\s*(.*)$', line)
        if m:
            return parse_meter_field(m.group(1))
    return None


def detect_unit_length_field(abc: str) -> Optional[Fraction]:
    """The first L: header line in `abc` (None if absent)."""
    for line in abc.splitlines():
        m = re.match(r'^\s*L:\s*(.*)$', line)
        if m:
            return parse_length_field(m.group(1))
    return None


def default_unit_length(meter: Optional[Meter]) -> Fraction:
    """ABC standard default for a missing L: field: 1/16 when the meter is
    below 0.75, else 1/8 (also 1/8 with no meter)."""
    if meter is not None and Fraction(meter[0], meter[1]) < Fraction(3, 4):
        return Fraction(1, 16)
    return Fraction(1, 8)


def beat_unit(meter: Optional[Meter], unit_length: Fraction) -> Fraction:
    """The note value that counts as 1: 1/denominator of the meter. With no
    meter (M:none / missing) the L: unit note length stands in."""
    if meter is None:
        return unit_length
    return Fraction(1, meter[1])


def _is_compound(meter: Optional[Meter]) -> bool:
    return meter is not None and meter[0] > 3 and meter[0] % 3 == 0


def _tuplet_time(p: int, compound: bool) -> int:
    """q of an ABC '(p' tuplet without an explicit ':q' (p notes in the time of q)."""
    if p in (2, 4, 8):
        return {2: 3, 4: 3, 8: 6}[p]
    if p == 3:
        return 2
    if p == 6:
        return 4
    if p in (5, 7, 9):
        return 2 if compound else 3
    return 2


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def format_ratio(dur: Fraction, is_rest: bool = False) -> str:
    """Fraction(1, 2) -> '1/2'; Fraction(2) -> '2'; rests get a 'z' suffix."""
    tok = str(dur.numerator) if dur.denominator == 1 else f"{dur.numerator}/{dur.denominator}"
    return tok + 'z' if is_rest else tok


def parse_ratio(tok: str) -> Tuple[Fraction, bool]:
    """Inverse of format_ratio: '1/2z' -> (Fraction(1, 2), True)."""
    is_rest = tok.endswith('z')
    if is_rest:
        tok = tok[:-1]
    return Fraction(tok), is_rest


_DUR_RE = re.compile(r'(\d*)(/*)(\d*)')


def _parse_duration(s: str, i: int) -> Tuple[Fraction, int]:
    """Duration multiplier written at s[i:] ('', '2', '/', '//', '3/2', '/4')
    in units of L:. Returns (multiplier, index after it)."""
    m = _DUR_RE.match(s, i)
    a, slashes, c = m.group(1), m.group(2), m.group(3)
    num = int(a) if a else 1
    if slashes:
        den = int(c) if c else 2 ** len(slashes)
    else:
        den = 1
    if den == 0:
        den = 1
    return Fraction(num, den), m.end()


_CHORD_NOTE_RE = re.compile(r"[_^=]*([A-Ga-g][,']*)(\d*/*\d*)")


# ---------------------------------------------------------------------------
# Parsing a voice into rhythm events
# ---------------------------------------------------------------------------

def abc_to_rhythm_events(
    abc: str,
    unit_length: Optional[Fraction] = None,
    meter: Optional[Meter] = None,
) -> Tuple[List[dict], str, Dict[int, int]]:
    """Parse one ABC voice into a list of rhythm events.

    Each event is a dict:
      dur     : Fraction, duration in beats (see beat_unit)
      rest    : bool
      bar     : bar index (0-based) where the event starts
      end_bar : bar index where it ends (differs from `bar` only for a tie that
                crosses a barline)
      span    : (start, end) char indices into `s` (end exclusive)

    `meter`/`unit_length` default to the first M:/L: lines found in `abc`; a
    missing L: falls back to the ABC default for the meter. The stripped string
    `s` and the bar -> char offset map `bar_starts` are returned as in
    extract.abc_to_pitches_with_bars so callers can cut bar snippets.
    """
    if meter is None:
        meter = detect_meter_field(abc)
    if unit_length is None:
        unit_length = detect_unit_length_field(abc) or default_unit_length(meter)
    beat = beat_unit(meter, unit_length)
    compound = _is_compound(meter)
    bar_len_beats = Fraction(meter[0], meter[1]) / beat if meter is not None else Fraction(1)

    s = strip_voice_and_text(abc)
    n = len(s)
    events: List[dict] = []

    i = 0
    bar_idx = 0
    tuplet_left = 0
    tuplet_factor = Fraction(1)
    broken_next = Fraction(1)      # factor pending for the next event
    tie_pending = False            # a '-' was seen: the next sounding event merges
    last_event = -1                # index of the last event (broken-rhythm target)
    last_sounding = -1             # index of the last sounding event (tie target)

    def emit(dur_beats: Fraction, is_rest: bool, span: Tuple[int, int], ident):
        nonlocal tuplet_left, broken_next, tie_pending, last_event, last_sounding
        dur = dur_beats
        if tuplet_left > 0:
            dur *= tuplet_factor
            tuplet_left -= 1
        dur *= broken_next
        broken_next = Fraction(1)
        if (tie_pending and not is_rest and last_sounding >= 0
                and events[last_sounding]['ident'] == ident):
            prev = events[last_sounding]
            prev['dur'] += dur
            prev['span'] = (prev['span'][0], span[1])
            prev['end_bar'] = bar_idx
            tie_pending = False
            last_event = last_sounding
            return
        tie_pending = False
        events.append({
            'dur': dur, 'rest': is_rest, 'bar': bar_idx, 'end_bar': bar_idx,
            'span': span, 'ident': ident,
        })
        last_event = len(events) - 1
        if not is_rest:
            last_sounding = last_event

    while i < n:
        ch = s[i]

        if ch == '|':
            bar_idx += 1
            i += 1
            continue

        if ch == '[':
            m_field = re.match(r'\[([A-Za-z]):([^\]\n]*)\]', s[i:])
            if m_field:
                if m_field.group(1) == 'L':
                    new_unit = parse_length_field(m_field.group(2))
                    if new_unit:
                        unit_length = new_unit
                # [M:...] deliberately ignored: the initial meter rules.
                i += m_field.end()
                continue
            if i + 1 < n and (s[i + 1] == '|' or s[i + 1].isdigit()):
                # '[|' barline or '[1' / '[2' volta bracket
                i += 1
                while i < n and s[i].isdigit():
                    i += 1
                continue
            j = s.find(']', i + 1)
            nl = s.find('\n', i + 1)
            if j == -1 or (nl != -1 and nl < j):
                i += 1
                continue
            inner = s[i + 1:j]
            notes = _CHORD_NOTE_RE.findall(inner)
            if not notes:
                i = j + 1
                continue
            first_dur_tok = notes[0][1]
            inner_dur, _ = _parse_duration(first_dur_tok, 0)
            outer_dur, k = _parse_duration(s, j + 1)
            tok_start = i
            i = k
            ident = ('chord', tuple(sorted(nt[0] for nt in notes)))
            emit(inner_dur * outer_dur * unit_length / beat, False, (tok_start, i), ident)
            continue

        if ch == '{':
            j = s.find('}', i + 1)
            i = n if j == -1 else j + 1
            continue

        if ch == '(':
            m_tup = re.match(r'\((\d+)(?::(\d*))?(?::(\d*))?', s[i:])
            if m_tup:
                p = int(m_tup.group(1))
                q = int(m_tup.group(2)) if m_tup.group(2) else _tuplet_time(p, compound)
                r = int(m_tup.group(3)) if m_tup.group(3) else p
                if p > 0:
                    tuplet_left = r
                    tuplet_factor = Fraction(q, p)
                i += m_tup.end()
                continue
            i += 1  # slur
            continue

        if ch in '><':
            run = 0
            while i < n and s[i] == ch:
                run += 1
                i += 1
            long_f = 2 - Fraction(1, 2 ** run)
            short_f = Fraction(1, 2 ** run)
            prev_f, next_f = (long_f, short_f) if ch == '>' else (short_f, long_f)
            if last_event >= 0:
                events[last_event]['dur'] *= prev_f
            broken_next = next_f
            continue

        if ch == '-':
            tie_pending = True
            i += 1
            continue

        if ch in 'zx':
            tok_start = i
            dur_units, i = _parse_duration(s, i + 1)
            emit(dur_units * unit_length / beat, True, (tok_start, i), None)
            continue

        if ch in 'ZX':
            tok_start = i
            m_bars = re.match(r'\d*', s[i + 1:])
            n_bars = int(m_bars.group(0)) if m_bars.group(0) else 1
            i += 1 + m_bars.end()
            emit(bar_len_beats * n_bars, True, (tok_start, i), None)
            continue

        if ch == 'y':
            _, i = _parse_duration(s, i + 1)
            continue

        tok_start = i
        while i < n and s[i] in '^_=':
            i += 1
        if i < n and s[i] in 'ABCDEFGabcdefg':
            letter = s[i]
            i += 1
            octave = ''
            while i < n and s[i] in "',":
                octave += s[i]
                i += 1
            dur_units, i = _parse_duration(s, i)
            emit(dur_units * unit_length / beat, False, (tok_start, i), (letter, octave))
            continue

        i = max(i, tok_start) + 1

    bar_starts = {0: 0}
    current_bar = 0
    for j, c in enumerate(s):
        if c == '|':
            current_bar += 1
            bar_starts[current_bar] = j + 1

    return events, s, bar_starts


# ---------------------------------------------------------------------------
# Windowing and counting
# ---------------------------------------------------------------------------

def event_token(ev: dict) -> str:
    return format_ratio(ev['dur'], ev['rest'])


def count_rhythm_motifs(
    events: List[dict],
    window_notes: int,
    stride_notes: int = 1,
    bar_numbering: str = "1-based",
) -> Tuple[Counter, Dict[Tuple[str, ...], List[dict]]]:
    """Slide a window of `window_notes` SOUNDING notes over `events`.

    Rests between the first and last note of a window are part of its pattern
    but never count toward `window_notes`; rests outside are excluded. No
    collapsing, no transformation folding: every window is counted as is.
    Occurrences report `start_event`/`end_event` as indices into `events`.
    """
    if window_notes <= 0:
        raise ValueError("window_notes must be >= 1")
    if stride_notes <= 0:
        raise ValueError("stride_notes must be >= 1")

    note_idx = [k for k, ev in enumerate(events) if not ev['rest']]
    counts: Counter = Counter()
    occurrences: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    bar_offset = 1 if bar_numbering == "1-based" else 0

    for start in range(0, len(note_idx) - window_notes + 1, stride_notes):
        a = note_idx[start]
        b = note_idx[start + window_notes - 1]
        pat = tuple(event_token(ev) for ev in events[a:b + 1])
        counts[pat] += 1
        occurrences[pat].append({
            'start_event': a,
            'end_event': b,
            'start_bar': events[a]['bar'] + bar_offset,
        })
    return counts, occurrences


def _bar_snippet(s: str, bar_starts: Dict[int, int], start_bar: int, end_bar: int) -> str:
    """Text of bars start_bar..end_bar (0-based) of the stripped string `s`,
    cut exactly as extract.find_top_motif cuts melodic snippets."""
    char_start = bar_starts.get(start_bar, 0)
    if end_bar + 1 in bar_starts:
        char_end = bar_starts[end_bar + 1] - 1
    else:
        char_end = len(s)
    return s[char_start:char_end].strip()


def _annotate_occurrences(occurrences, events, s, bar_starts):
    for occs in occurrences.values():
        for occ in occs:
            ev_a = events[occ['start_event']]
            ev_b = events[occ['end_event']]
            occ['abc_snippet'] = _bar_snippet(s, bar_starts, ev_a['bar'], ev_b['end_bar'])
            # Line indices into the stripped text: the processed tunebody keeps one
            # bar per physical line, so these map straight onto tunebody lines.
            occ['start_line'] = s.count('\n', 0, ev_a['span'][0])
            occ['end_line'] = s.count('\n', 0, ev_b['span'][1])


def find_top_rhythm_motif(
    abc: str,
    window_notes: int,
    stride_notes: int = 1,
    unit_length: Optional[Fraction] = None,
    meter: Optional[Meter] = None,
    bar_numbering: str = "1-based",
    top_n: int = 3,
):
    """Rhythmic counterpart of extract.find_top_motif: (meta, {pattern: {count,
    occurrences}}) for the `top_n` most frequent patterns of one length."""
    events, s, bar_starts = abc_to_rhythm_events(abc, unit_length=unit_length, meter=meter)
    counts, occurrences = count_rhythm_motifs(
        events, window_notes=window_notes, stride_notes=stride_notes, bar_numbering=bar_numbering)
    _annotate_occurrences(occurrences, events, s, bar_starts)
    meta = {
        'total_events': len(events),
        'total_notes': sum(1 for ev in events if not ev['rest']),
        'window_notes': window_notes,
        'stride_notes': stride_notes,
    }
    return meta, {pat: {'count': c, 'occurrences': occurrences.get(pat, [])}
                  for pat, c in counts.most_common(top_n)}


def get_best_rhythm_motifs_per_length(
    abc_voice: str,
    window_range: Tuple[int, int] = (4, 11),
    unit_length: Optional[Fraction] = None,
    meter: Optional[Meter] = None,
    bar_numbering: str = "1-based",
    top_n: int = 3,
) -> Dict[int, List[Dict]]:
    """Top rhythmic motifs for EVERY window length in `window_range`.

    Returns {window_notes: [motif dicts]}, most frequent first (ties go to the
    first-seen pattern, as for melodic motifs). Each dict carries the abstract
    "pattern" (tuple of ratio tokens), its first realization "abc" (bar text),
    the occurrence "count", and the first occurrence's "start_line"/"end_line"
    in the stripped voice text. A length is absent when the voice holds fewer
    sounding notes than that length. The voice is parsed once for all lengths.
    """
    events, s, bar_starts = abc_to_rhythm_events(abc_voice, unit_length=unit_length, meter=meter)
    per_length: Dict[int, List[Dict]] = {}
    for w in range(window_range[0], window_range[1]):
        counts, occurrences = count_rhythm_motifs(events, window_notes=w, bar_numbering=bar_numbering)
        if not counts:
            continue
        _annotate_occurrences(occurrences, events, s, bar_starts)
        top: List[Dict] = []
        for pat, c in counts.most_common(top_n):
            first = occurrences[pat][0]
            top.append({
                'pattern': pat,
                'abc': first['abc_snippet'],
                'count': c,
                'start_line': first['start_line'],
                'end_line': first['end_line'],
            })
        per_length[w] = top
    return per_length
