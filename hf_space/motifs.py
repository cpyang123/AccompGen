"""Validate motif input and encode the training-time conditioning format."""
from fractions import Fraction
import re

DEFAULT_NOTES = [["C4", "1"], ["G4", "1/2"], ["F4", "1/2"], ["E4", "2"]]
PITCH = re.compile(r"^([A-Ga-g])([#b]?)([0-8])$")
LETTERS = "CDEFGAB"
SEMITONES = dict(zip(LETTERS, (0, 2, 4, 5, 7, 9, 11)))


def abstract_pattern(text):
    if not isinstance(text, str) or len(text) > 120:
        raise ValueError("Enter 4–10 comma-separated interval classes.")
    fields = text.strip().split(",")
    if not 4 <= len(fields) <= 10 or any(not re.fullmatch(r"[+-]?[0-3]", f.strip()) for f in fields):
        raise ValueError("Use 4–10 values: an initial 0, then +1/−1 (step), +2/−2 (skip), or +3/−3 (leap).")
    values = [int(f.strip()) for f in fields]
    if values[0] != 0 or 0 in values[1:]:
        raise ValueError("Start with 0. Subsequent intervals must be ±1, ±2, or ±3; repeated notes are collapsed during training.")
    return ",".join(map(str, values))


def normalize_concrete_rows(rows):
    """Keep barline rows separate from pitched notes in the two-column API."""
    if not isinstance(rows, (list, tuple)):
        raise ValueError("Enter one pitch and duration per row.")
    normalized = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("Each row must contain a pitch and duration, or a barline.")
        pitch, beats = [str(value).strip() if value is not None else '' for value in row]
        if not pitch and not beats:
            continue
        if pitch == '|':
            if beats:
                raise ValueError("A barline has no duration.")
            if not normalized or normalized[-1][0] == '|':
                raise ValueError("Put barlines after notes, without empty measures.")
        normalized.append([pitch, beats])
    return normalized


def concrete_motif(rows):
    rows = normalize_concrete_rows(rows)
    if not 4 <= sum(row[0] != '|' for row in rows) <= 10:
        raise ValueError("Enter 4–10 notes; barlines do not count as notes.")
    notes, pitches, positions = [], [], []
    accidentals = {}
    for i, row in enumerate(rows, 1):
        if row[0] == '|':
            notes.append('|')
            accidentals.clear()
            continue
        if len(row) != 2:
            raise ValueError(f"Row {i}: supply a pitch and duration.")
        match = PITCH.fullmatch(str(row[0]).strip())
        if not match:
            raise ValueError(f"Row {i}: use a pitch such as C4, F#4, or Bb3.")
        letter, accidental, octave = match.groups()
        letter, octave = letter.upper(), int(octave)
        try:
            duration = Fraction(str(row[1]).strip())
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"Row {i}: duration must be a number or fraction of a quarter-note beat.") from None
        if not Fraction(1, 8) <= duration <= 8 or duration.denominator not in (1, 2, 3, 4, 6, 8):
            raise ValueError(f"Row {i}: use a duration from 1/8 to 8 beats (for example 1/2, 1, or 2).")
        midi = 12 * (octave + 1) + SEMITONES[letter] + {"": 0, "#": 1, "b": -1}[accidental]
        if not 12 <= midi <= 119:
            raise ValueError(f"Row {i}: choose a pitch between C0 and B8.")
        if pitches and midi == pitches[-1]:
            raise ValueError("Adjacent repeated pitches are collapsed by this model. Combine their durations into one note.")
        pitches.append(midi)
        positions.append(7 * octave + LETTERS.index(letter))
        abc_pitch = letter + "," * (4 - octave) if octave <= 4 else letter.lower() + "'" * (octave - 5)
        # Concrete prompts pin K:C. Naturals need a sign only to cancel a
        # prior accidental on this pitch in the same measure.
        pitch_key = (letter, octave)
        prefix = {"#": "^", "b": "_"}.get(accidental, "=" if accidentals.get(pitch_key) else "")
        accidentals[pitch_key] = accidental
        abc_pitch = prefix + abc_pitch
        units = duration * 2  # concrete prompts pin L:1/8
        suffix = "" if units == 1 else str(units)
        notes.append(abc_pitch + suffix)
    pattern = [0]
    for i in range(1, len(positions)):
        distance = positions[i] - positions[i - 1]
        if distance == 0:
            distance = 1 if pitches[i] > pitches[i - 1] else -1
        pattern.append((1 if distance > 0 else -1) * min(abs(distance), 3))
    return ",".join(map(str, pattern)), " ".join(notes)


def rhythm_pattern(text, length):
    """Validate a rhythmic motif: `length` comma-separated durations, each a ratio to the beat
    (the training definition: beat = 1 / the meter denominator, so under 4/4 "1" is a quarter
    and "1/2" an eighth). Returns the canonical header spelling, or None when blank."""
    if text is None or not str(text).strip():
        return None
    if not isinstance(text, str) or len(text) > 160:
        raise ValueError("Enter one duration per motif note, comma-separated (for example 1,1/2,1/2,2).")
    tokens = [t.strip() for t in text.split(",")]
    if len(tokens) != length:
        raise ValueError(f"The rhythm needs exactly {length} durations, one per motif note (got {len(tokens)}).")
    out = []
    for token in tokens:
        if not re.fullmatch(r"\d+(/\d+)?", token):
            raise ValueError(f"Rhythm value {token!r}: use a whole number or fraction of the beat such as 1, 1/2, or 3/2; rests are not supported.")
        try:
            value = Fraction(token)
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"Rhythm value {token!r} is not a valid duration.") from None
        if not Fraction(1, 8) <= value <= 8 or value.denominator not in (1, 2, 3, 4, 6, 8):
            raise ValueError(f"Rhythm value {token!r}: use durations from 1/8 to 8 beats (for example 1/2, 1, or 3/2).")
        out.append(str(value))
    return ",".join(out)


def concrete_rhythm(rows):
    """The rhythmic motif implied by a concrete melody: concrete prompts pin M:4/4, whose beat
    is the quarter, so every entered quarter-note duration is already its beat ratio."""
    return ",".join(str(Fraction(str(row[1]).strip())) for row in normalize_concrete_rows(rows) if row[0] != "|")


def build_prompt(mode, abstract, rows, style, occurrences=3, rhythm=""):
    """Training-order header: contour, rhythm, count, abc, rhythm count, rhythm abc, inversion.

    Returns (prompt, pattern, snippet, rhythm): `rhythm` is the `%motif:v1:rhythm:` value the
    prompt carries (None when an abstract motif is sent without one; the model then writes its
    own rhythm line)."""
    if not 1 <= int(occurrences) <= 8:
        raise ValueError("Motif occurrences must be between 1 and 8.")
    if mode == "Abstract motif":
        pattern, snippet = abstract_pattern(abstract), None
        rhythm = rhythm_pattern(rhythm, len(pattern.split(",")))
    elif mode == "Concrete notes":
        pattern, snippet = concrete_motif(rows)
        rhythm = concrete_rhythm(rows)
    else:
        raise ValueError("Choose abstract motif or concrete notes.")
    prompt = "".join(f"%{field}\n" for field in style)
    prompt += f"%motif:v1:step_skip_leap: {pattern} \n"
    if rhythm is not None:
        prompt += f"%motif:v1:rhythm: {rhythm} \n"
    if snippet is not None:
        prompt += f"%motif:count: {int(occurrences)} \n%motif:abc: {snippet} \n"
        prompt += f"%motif:rhythm:count: {int(occurrences)} \n%motif:rhythm:abc: {snippet} \n"
        prompt += "%motif:inversion_count: 0 \n"
        prompt += "%%score 1\nL:1/8\nM:4/4\nK:C\nV:1\n"
    return prompt, pattern, snippet, rhythm
