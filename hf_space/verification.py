"""Check the actual score melody, retaining exact ABC positions for highlighting."""
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

import re

from motifs import PITCH, SEMITONES, abstract_pattern, concrete_motif, normalize_concrete_rows, rhythm_pattern

ROOT = Path(__file__).resolve().parent


@dataclass
class MotifMatch:
    occurrences: list
    kind: str

    @property
    def count(self):
        return len(self.occurrences)

    @property
    def spans(self):
        return sorted({tuple(span) for occurrence in self.occurrences for span in occurrence['spans']})

    @property
    def bars(self):
        return sorted({occurrence['bar'] for occurrence in self.occurrences})


def _parse_score(abc, *, analysis=False):
    try:
        result = subprocess.run(['node', str(ROOT / 'parse_melody.js')] + (['--analysis'] if analysis else []), input=abc,
                                capture_output=True, text=True, timeout=8)
    except subprocess.TimeoutExpired:
        raise ValueError('The generated notation took too long to check.') from None
    if result.returncode:
        raise ValueError('The generated melody could not be parsed for motif verification.')
    return json.loads(result.stdout)


def parse_melody(abc):
    return _parse_score(abc)


def analyze_score(abc):
    parsed = _parse_score(abc, analysis=True)
    # abcjs rounds timings; recover standard rhythmic fractions so equivalent
    # tuplets/ties do not become new note types due to floating-point noise.
    note_types = {(int(pitch), Fraction(str(beats)).limit_denominator(1920))
                  for pitch, beats in parsed['notes']}
    return parsed['events'], len(note_types)


def interval_pattern(events):
    """Training's successive diatonic step/skip/leap classes (no folding)."""
    result = [0]
    for previous, current in zip(events, events[1:]):
        distance = current['diatonic'] - previous['diatonic']
        if not distance and current['midi'] != previous['midi']:
            distance = 1 if current['midi'] > previous['midi'] else -1
        result.append((1 if distance > 0 else -1 if distance < 0 else 0) * min(abs(distance), 3))
    return tuple(result)


def beat_in_quarters(abc):
    """Quarter-note length of one beat under the score's first M: line (training's rule:
    beat = 1 / meter denominator; C = 4/4, C| = 2/2; no M: -> 4/4)."""
    match = re.search(r'^M:\s*(\S+)', abc, re.M)
    meter = match.group(1) if match else '4/4'
    if meter == 'C':
        meter = '4/4'
    elif meter == 'C|':
        meter = '2/2'
    match = re.fullmatch(r'\d+/(\d+)', meter)
    denominator = int(match.group(1)) if match and int(match.group(1)) else 4
    return Fraction(4, denominator)


def find_motif(abc, mode, pattern, rows=None, *, events=None, rhythm=None):
    if events is None:
        events = parse_melody(abc)
    target_beats = None
    if mode == 'Abstract motif':
        target = tuple(map(int, abstract_pattern(pattern).split(',')))
        rhythm = rhythm_pattern(rhythm, len(target))
        if rhythm is not None:
            beat = beat_in_quarters(abc)
            target_beats = [float(Fraction(token) * beat) for token in rhythm.split(',')]
        melody = []
        # Match the training semantics: ignore rests and collapse consecutive
        # repeated pitches, including tied or restruck notes.
        for event in events:
            if event.get('rest'):
                continue
            if event.get('barrier'):
                melody.append(event)
                continue
            if melody and melody[-1].get('midi') == event['midi']:
                melody[-1]['spans'].extend(event['spans'])
            else:
                melody.append(dict(event, spans=list(event['spans'])))
        kind = 'contour and rhythm' if target_beats else 'contour'
    elif mode == 'Concrete notes':
        concrete_motif(rows)  # Same validation as the input/prompt path.
        target = []
        for row in normalize_concrete_rows(rows):
            if row[0] == '|':
                continue
            if not any(str(value or '').strip() for value in row):
                continue
            letter, accidental, octave = PITCH.fullmatch(str(row[0]).strip()).groups()
            midi = 12 * (int(octave) + 1) + SEMITONES[letter.upper()] + {'': 0, '#': 1, 'b': -1}[accidental]
            target.append((midi, float(Fraction(str(row[1]).strip()))))
        melody, kind = events, 'exact pitches and rhythm'
    else:
        raise ValueError('Unknown motif input method.')

    occurrences = []
    for start in range(len(melody) - len(target) + 1):
        window = melody[start:start + len(target)]
        if any(event.get('rest') or event.get('barrier') for event in window):
            continue
        if mode == 'Abstract motif':
            matches = interval_pattern(window) == target
            if matches and target_beats:
                matches = all(math.isclose(event['beats'], beats, rel_tol=1e-7, abs_tol=1e-5)
                              for event, beats in zip(window, target_beats))
        else:
            # abcjs rounds MIDI event times to six decimal places in whole-note
            # units; allow that rounding for tuplets, not musical rhythm changes.
            matches = all(event['midi'] == midi and math.isclose(event['beats'], beats, rel_tol=1e-7, abs_tol=1e-5)
                          for event, (midi, beats) in zip(window, target))
        if matches:
            occurrences.append({'bar': window[0]['bar'],
                                'spans': [span for event in window for span in event['spans']]})
    return MotifMatch(occurrences, kind)
