"""Chord playback through the real ABC → MusicXML → WAV export path."""
from pathlib import Path
import sys
import wave

import numpy as np
from music21 import converter, note, stream, tie
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outputs import export_result, playback_events


def export(body, headers=''):
    result = export_result(f'L:1/8\nM:4/4\nK:C\n{headers}V:1\n[V:1]{body}\n', 120)
    assert not result[-1], result[-1]
    return result, [event[:3] for event in playback_events(converter.parse(result[3]), 120)]


def test_abc_chord_plays_all_three_pitches_simultaneously_in_wav():
    result, events = export('[CEG]4 z4|')
    assert events == [(0, 1, 60), (0, 1, 64), (0, 1, 67)]
    with wave.open(result[4]) as wav:
        rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(float)
    # Each independent fundamental must be audible in the same time window.
    segment = audio[int(.08 * rate):int(.75 * rate)]
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    frequencies = np.fft.rfftfreq(len(segment), 1 / rate)
    for midi in [60, 64, 67]:
        hz = 440 * 2 ** ((midi - 69) / 12)
        assert spectrum[np.abs(frequencies - hz) < 3].max() > .3 * spectrum.max()


@pytest.mark.parametrize('body,expected', [
    ('[CEG]2-[CEG]2 z4|', [(0, 1, 60), (0, 1, 64), (0, 1, 67)]),
    ('[C-EG]2 [CFA]2 z4|', [(0, 1, 60), (0, .5, 64), (0, .5, 67), (.5, .5, 65), (.5, .5, 69)]),
    ('[CEG]2 [CEG]2 z4|', [(0, .5, 60), (0, .5, 64), (0, .5, 67), (.5, .5, 60), (.5, .5, 64), (.5, .5, 67)]),
])
def test_chord_ties_and_repeated_attacks(body, expected):
    _, events = export(body)
    assert events == expected


def test_multiple_abc_voices_retain_chords_and_simultaneous_bass():
    _, events = export('[CEG]4 z4|\n[V:2]C,8|', 'V:2\n')
    assert sorted(events) == sorted([(0, 1, 60), (0, 1, 64), (0, 1, 67), (0, 2, 48)])


def test_unison_ties_in_different_voices_do_not_merge():
    score = stream.Score()
    part = stream.Part()
    measure = stream.Measure()
    for voice_id in ['1', '2']:
        voice = stream.Voice(id=voice_id)
        for offset, kind in [(0, 'start'), (1, 'stop')]:
            n = note.Note('C4', quarterLength=1)
            n.tie = tie.Tie(kind)
            voice.insert(offset, n)
        measure.insert(0, voice)
    part.append(measure)
    score.insert(0, part)
    assert playback_events(score, 120) == [(0, 1, 60, 1.0), (0, 1, 60, 1.0)]


def test_chord_labels_add_minor_and_seventh_accompaniment_at_label_offsets():
    result, _ = export('"Am"C2 E2 "G7"G2 B2|')
    events = playback_events(converter.parse(result[3]), 120)
    assert [event for event in events if event[3] == 0.5] == [
        (0, 1, 45, .5), (0, 1, 48, .5), (0, 1, 52, .5),
        (1, 1, 43, .5), (1, 1, 47, .5), (1, 1, 50, .5), (1, 1, 53, .5)]
    assert len([event for event in events if event[3] == 1]) == 4
    with wave.open(result[4]) as wav:
        rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(float)
    for start, midis in [(0, [45, 48, 52]), (1, [43, 47, 50, 53])]:
        segment = audio[int((start+.08)*rate):int((start+.45)*rate)]
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
        frequencies = np.fft.rfftfreq(len(segment), 1/rate)
        for midi in midis:
            hz = 440 * 2 ** ((midi-69)/12)
            assert spectrum[np.abs(frequencies-hz) < 4].max() > .15*spectrum.max()


def test_slash_chord_preserves_bass_and_chromatic_notes():
    result, _ = export('"Bb/D"C8|')
    events = playback_events(converter.parse(result[3]), 120)
    assert [event[2] for event in events if event[3] == .5] == [50, 53, 58]


def test_no_chord_label_stops_accompaniment_without_stopping_melody():
    result, _ = export('"C"C2 D2 "N.C."E2 F2|')
    events = playback_events(converter.parse(result[3]), 120)
    assert all(start+length <= 1 for start, length, _, gain in events if gain == .5)
    assert any(start >= 1 and gain == 1 for start, _, _, gain in events)


def test_labels_and_written_chords_play_together_and_plain_text_is_not_a_chord():
    result, _ = export('"Am"[CEG]4 "gently"[DFA]4|')
    events = playback_events(converter.parse(result[3]), 120)
    assert len([event for event in events if event[3] == 1]) == 6
    assert len([event for event in events if event[3] == .5]) == 3
    assert all(event[1] == 2 for event in events if event[3] == .5)


def test_chord_labels_over_rests_can_be_played():
    result, events = export('"Am"z8|')
    assert sorted(event[2] for event in events) == [45, 48, 52]
    assert result[4]


def test_repeated_labels_on_multiple_staves_are_not_doubled():
    result, _ = export('"Am"C8|\n[V:2]"Am"E8|', 'V:2\n')
    events = playback_events(converter.parse(result[3]), 120)
    assert len([event for event in events if event[3] == .5]) == 3
    assert len([event for event in events if event[3] == 1]) == 2
