from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import GenerationUpdate
from motifs import DEFAULT_NOTES
from outputs import clean_abc, score_html
from retries import generate_verified
from verification import analyze_score, find_motif, parse_melody


def score(body, key='C', headers=''):
    return f'X:1\n{headers}L:1/8\nM:4/4\nK:{key}\nV:1\n[V:1]{body}\n'


def test_metadata_is_never_a_match():
    abc = score('C D E F|', headers='%motif:abc: C G F E\nT:C G F E\n')
    assert not find_motif(abc, 'Abstract motif', '0,3,-1,-1').count


def test_contour_collapses_repetitions_and_highlights_all_actual_notes():
    abc = score('C G G F E|C c B A|')
    match = find_motif(abc, 'Abstract motif', '0,3,-1,-1')
    assert match.count == 2
    assert match.bars == [1, 2]
    highlighted = ''.join(abc[a:b] for a, b in match.spans)
    assert 'G G' in highlighted and 'V:' not in highlighted and 'X:' not in highlighted
    assert len(match.occurrences[0]['spans']) == 5


def test_inversion_alone_does_not_satisfy_requested_contour():
    assert not find_motif(score('c G A B|'), 'Abstract motif', '0,3,-1,-1').count


@pytest.mark.parametrize('body,key,expected', [('C2 G F E4|', 'C', 1), ('C2 G F E4|', 'G', 0),
                                            ('C2 G =F E4|', 'G', 1), ('C2 G F E2|', 'C', 0),
                                            ('C2 G z F E4|', 'C', 0), ('c2 g f e4|', 'C', 0)])
def test_concrete_requires_exact_absolute_pitches_and_rhythm(body, key, expected):
    assert find_motif(score(body, key), 'Concrete notes', '0,3,-1,-1', DEFAULT_NOTES).count == expected


def test_concrete_ties_count_as_one_note_with_combined_duration():
    abc = score('C-C G F E4|')
    match = find_motif(abc, 'Concrete notes', '0,3,-1,-1', DEFAULT_NOTES)
    assert match.count == 1 and len(match.spans) == 5


def test_tuplets_use_sounding_duration():
    rows = [['C4', '1/3'], ['D4', '1/3'], ['E4', '1/3'], ['F4', '1']]
    assert find_motif(score('(3CDE F2|'), 'Concrete notes', '0,1,1,1', rows).count == 1


def test_inline_key_change_and_accidental_memory():
    rows = [['C4', '1'], ['G4', '1/2'], ['F#4', '1/2'], ['E4', '2']]
    assert find_motif(score('[K:G]C2 G F E4|'), 'Concrete notes', '0,3,-1,-1', rows).count == 1
    rows = [['F#4', '1/2'], ['G4', '1/2'], ['F#4', '1/2'], ['E4', '1/2']]
    assert find_motif(score('^F G F E|'), 'Concrete notes', '0,1,-1,-1', rows).count == 1


def test_voices_are_not_concatenated_into_false_motif():
    abc = 'X:1\nL:1/8\nM:4/4\nK:C\nV:1\nV:2\n[V:1]C G|\n[V:2]F E|\n'
    assert not find_motif(abc, 'Abstract motif', '0,3,-1,-1').count


def test_chord_tones_do_not_form_a_fictitious_melody():
    assert not find_motif(score('[CGFE]8|'), 'Abstract motif', '0,3,-1,-1').count


def test_partial_chord_ties_cannot_skip_notes_into_a_false_match():
    # The upper melody is C G A F E. The tied lower C must not hide the A
    # and cause C G F E to be accepted as a contiguous contour.
    body = 'C [C-G] [CA] F E|'
    assert not find_motif(score(body), 'Abstract motif', '0,3,-1,-1').count
    match = find_motif(score(body + 'C G F E|'), 'Abstract motif', '0,3,-1,-1')
    assert match.count == 1 and match.bars == [2]


def test_export_offsets_match_rendered_source():
    raw = '%motif:abc: C G F E\nL:1/8\nM:4/4\nK:C\nV:1\n[r:0/1][V:1]C G F E|\n'
    abc = clean_abc(raw)
    match = find_motif(abc, 'Abstract motif', '0,3,-1,-1')
    assert [abc[a:b].strip() for a, b in match.spans] == ['C', 'G', 'F', 'E']
    html = score_html(abc, match)
    assert 'data-motif-match' in html and 'const matches = 1' in html


def test_retry_rejects_miss_changes_seed_then_accepts_match():
    calls = []
    def attempt(**kwargs):
        calls.append(kwargs)
        body = 'C D E F|' if len(calls) == 1 else 'C G F E|'
        yield GenerationUpdate(score(body), 3, 0.1, 'Complete.')
    updates = list(generate_verified(attempt, 'prompt', 1.2, 42, 8, 'Abstract motif', '0,3,-1,-1', DEFAULT_NOTES, 100))
    assert [call['seed'] for call in calls] == list(range(42, 54))
    assert any(update.stage == 'rejected' for update in updates)
    assert updates[-1].stage == 'verified' and updates[-1].match.count == 1
    assert updates[-1].attempt == 2 and updates[-1].seed == 43


def test_exhaustion_never_presents_an_unverified_composition():
    def attempt(**kwargs):
        yield GenerationUpdate(score('C D E F|'), 3, 0.1, 'Complete.')
    updates = list(generate_verified(attempt, 'prompt', 1.2, 42, 8, 'Abstract motif', '0,3,-1,-1', DEFAULT_NOTES, 100, max_attempts=3))
    assert updates[-1].stage == 'exhausted' and updates[-1].attempt == 3
    assert not updates[-1].text and all(update.match is None for update in updates)


def test_cancelling_generator_prevents_more_attempts():
    calls = []
    def attempt(**kwargs):
        calls.append(kwargs['seed'])
        yield GenerationUpdate(score('C D E F|'), 3, 0.1, 'Complete.')
    updates = generate_verified(attempt, 'prompt', 1.2, 42, 8, 'Abstract motif', '0,3,-1,-1', DEFAULT_NOTES, 100)
    for update in updates:
        if update.stage == 'rejected':
            updates.close()
            break
    assert calls == [42]


def test_concrete_contour_only_match_is_rejected_and_retried():
    calls = []
    def attempt(**kwargs):
        calls.append(kwargs['seed'])
        yield GenerationUpdate(score('c2 g f e4|' if len(calls) == 1 else 'C2 G F E4|'), 3, 0.1, 'Complete.')
    result = list(generate_verified(attempt, 'prompt', 1.2, 42, 8, 'Concrete notes', '0,3,-1,-1', DEFAULT_NOTES, 100))[-1]
    assert calls == list(range(42, 54)) and result.stage == 'verified' and result.match.kind == 'exact pitches and rhythm'


@pytest.mark.parametrize('mode', ['Abstract motif', 'Concrete notes'])
def test_reported_short_score_is_rejected_even_with_matching_preamble(mode):
    from motifs import build_prompt
    prompt, pattern, _, _ = build_prompt(mode, '0,3,-1,-1', DEFAULT_NOTES,
                                         ('Romantic', 'Schubert, Franz', 'Art Song'))
    sample = '''X:1
T:MotiGen composition
Q:1/4=100
%%score 1
L:1/4
M:4/4
K:G
V:1
[V:1]D|
[V:1]G G G G|
[V:1]G B, B, B,|
'''
    calls = []
    def attempt(**kwargs):
        calls.append(kwargs)
        yield GenerationUpdate(prompt + sample, 3, .1, 'Complete.')
    updates = list(generate_verified(attempt, prompt, 1.2, 42, 8, mode,
                                     pattern, DEFAULT_NOTES, 100, max_attempts=2))
    assert len(calls) == 2 and all(call['prompt'] == prompt for call in calls)
    assert updates[-1].stage == 'exhausted'
    assert updates[-1].text == '' and updates[-1].match is None


def test_concrete_barline_markers_do_not_add_notes_or_rests_to_verification():
    rows = DEFAULT_NOTES[:2] + [['|', '']] + DEFAULT_NOTES[2:] + [['|', '']]
    match = find_motif(score('C2 G|F E4|'), 'Concrete notes', '0,3,-1,-1', rows)
    assert match.count == 1 and len(match.spans) == 4
    assert not find_motif(score('C2 G|z F E4|'), 'Concrete notes', '0,3,-1,-1', rows).count


def test_all_twelve_are_ranked_even_after_first_match_and_last_can_win():
    calls = []
    bodies = ['C G F E|'] * 12
    bodies[1] = 'C G F E|D2 A4 B/2 c3|'
    # More note types, but no requested motif: never eligible.
    bodies[2] = "C D E F G A B c d e f g a b c' d'|"
    bodies[11] = 'C G F E|D2 A4 B/2 c3 d e f|'
    def attempt(**kwargs):
        calls.append(kwargs)
        yield GenerationUpdate(score(bodies[len(calls)-1]), 3, .1, 'Complete.')
    updates = list(generate_verified(attempt, 'prompt', 1.2, 42, 8, 'Abstract motif', '0,3,-1,-1', DEFAULT_NOTES, 100))
    result = updates[-1]
    assert len(calls) == 12 and result.evaluated == 12 and result.eligible == 11
    assert result.attempt == 12 and result.seed == 53
    assert result.note_types == 11 and result.text == score(bodies[-1])
    assert sum(u.stage == 'verified' for u in updates) == 1
    assert all(u.stage != 'verified' for u in updates[:-1])
    assert all(0 < call['max_seconds'] <= 100 for call in calls)


def test_complexity_ties_keep_earliest_matching_candidate():
    def attempt(**kwargs):
        yield GenerationUpdate(score('C G F E|'), 3, .1, 'Complete.')
    result = list(generate_verified(attempt, 'prompt', 1.2, 2**32-1, 8, 'Abstract motif', '0,3,-1,-1', DEFAULT_NOTES, 100))[-1]
    assert result.attempt == 1 and result.seed == 2**32-1
    assert result.note_types == 4 and result.evaluated == result.eligible == 12


def test_variety_counts_pitch_duration_pairs_and_not_metadata_or_repetitions():
    assert analyze_score(score('C G F E|'))[1] == 4
    assert analyze_score(score('C G F E|C G F E|'))[1] == 4
    assert analyze_score(score('C G F E|C2|'))[1] == 5
    assert analyze_score(score('C G F E|c|'))[1] == 5
    assert analyze_score(score('C G F E|', headers='%motif:abc: D2 A3 B4\n'))[1] == 4
    assert analyze_score(score('"Am"C G F E|z2|'))[1] == 4


def test_variety_includes_all_chord_tones_and_other_voices():
    assert analyze_score(score('[CEG]2 [CEG]2|'))[1] == 3
    abc='X:1\nL:1/8\nM:4/4\nK:C\nV:1\nV:2\n[V:1]C G F E|\n[V:2]D2 A2|\n'
    assert analyze_score(abc)[1] == 6


def test_variety_resolves_enharmonics_and_tied_durations():
    assert analyze_score(score('^C _D|'))[1] == 1
    assert analyze_score(score('C-C C2|'))[1] == 1
    assert analyze_score(score('(3CDE (3CDE|'))[1] == 3


def test_cancel_after_eligible_candidate_does_not_finish_or_generate_more():
    calls=[]
    def attempt(**kwargs):
        calls.append(kwargs['seed'])
        yield GenerationUpdate(score('C G F E|'), 3, .1, 'Complete.')
    updates=generate_verified(attempt, 'prompt', 1.2, 42, 8, 'Abstract motif', '0,3,-1,-1', DEFAULT_NOTES, 100)
    for update in updates:
        assert update.stage != 'verified'
        if update.stage == 'candidate':
            updates.close()
            break
    assert calls == [42]


def test_time_budget_never_presents_partial_batch_as_best_of_twelve(monkeypatch):
    import retries
    now=[0.0]
    monkeypatch.setattr(retries.time, 'monotonic', lambda: now[0])
    def attempt(**kwargs):
        now[0] = 106
        yield GenerationUpdate(score('C G F E|'), 3, .1, 'Complete.')
    updates=[]
    with pytest.raises(ValueError, match='1/12 candidates'):
        for update in generate_verified(attempt, 'prompt', 1.2, 42, 8, 'Abstract motif', '0,3,-1,-1', DEFAULT_NOTES, 100):
            updates.append(update)
    assert all(u.stage != 'verified' for u in updates)


@pytest.mark.parametrize('mode', ['Abstract motif', 'Concrete notes'])
def test_more_motif_occurrences_beat_greater_note_variety(mode):
    motif = 'C2 G F E4|'
    bodies = [motif + 'D2 A4 B/2 c3 d e f|'] * 12
    bodies[6] = motif * 3
    calls = []
    def attempt(**kwargs):
        calls.append(kwargs['seed'])
        yield GenerationUpdate(score(bodies[len(calls)-1]), 3, .1, 'Complete.')
    result = list(generate_verified(attempt, 'prompt', 1.2, 42, 8, mode,
                                   '0,3,-1,-1', DEFAULT_NOTES, 100))[-1]
    assert calls == list(range(42,54))
    assert result.attempt == 7 and result.seed == 48 and result.evaluated == 12
    assert result.match.count == 3 and result.note_types == 4
    assert result.text == score(bodies[6])
    assert analyze_score(score(bodies[-1]))[1] > result.note_types


@pytest.mark.parametrize('mode', ['Abstract motif', 'Concrete notes'])
def test_equal_motif_counts_prefer_variety_then_earlier_candidate(mode):
    motif = 'C2 G F E4|'
    bodies = [motif * 2] * 12
    bodies[4] += 'D2 A4 B/2 c3 d e f|'
    bodies[9] = bodies[4]
    calls = []
    def attempt(**kwargs):
        calls.append(kwargs['seed'])
        yield GenerationUpdate(score(bodies[len(calls)-1]), 3, .1, 'Complete.')
    result = list(generate_verified(attempt, 'prompt', 1.2, 42, 8, mode,
                                   '0,3,-1,-1', DEFAULT_NOTES, 100))[-1]
    assert len(calls) == 12 and result.attempt == 5
    assert result.match.count == 2 and result.note_types > 4
    assert result.text == score(bodies[4])


def test_abstract_rhythm_requires_matching_durations_under_the_score_meter():
    # 4/4, L:1/8: C2 G F E4 = quarter, eighth, eighth, half -> beat ratios 1,1/2,1/2,2
    abc = score('C2 G F E4|C2 G2 F2 E2|')
    assert find_motif(abc, 'Abstract motif', '0,3,-1,-1').count == 2
    match = find_motif(abc, 'Abstract motif', '0,3,-1,-1', rhythm='1,1/2,1/2,2')
    assert match.count == 1 and match.bars == [1] and match.kind == 'contour and rhythm'
    assert find_motif(abc, 'Abstract motif', '0,3,-1,-1', rhythm='1,1,1,1').bars == [2]
    assert not find_motif(abc, 'Abstract motif', '0,3,-1,-1', rhythm='2,1,1,1').count


def test_abstract_rhythm_beat_is_the_meter_denominator():
    # 6/8: an eighth is one beat, so C G F E3 (eighths + dotted quarter) is 1,1,1,3.
    abc = 'X:1\nL:1/8\nM:6/8\nK:C\nV:1\n[V:1]C G F E3|\n'
    assert find_motif(abc, 'Abstract motif', '0,3,-1,-1', rhythm='1,1,1,3').count == 1
    assert not find_motif(abc, 'Abstract motif', '0,3,-1,-1', rhythm='1/2,1/2,1/2,3/2').count
    # 2/2: the half note is the beat.
    abc = 'X:1\nL:1/8\nM:2/2\nK:C\nV:1\n[V:1]C4 G2 F E|\n'
    assert find_motif(abc, 'Abstract motif', '0,3,-1,-1', rhythm='1,1/2,1/4,1/4').count == 1


def test_retry_rejects_contour_only_match_when_rhythm_is_requested():
    miss = score('C2 G2 F2 E2|')          # contour only
    hit = score('C2 G F E4|')             # contour and rhythm
    outputs = iter([miss, hit])
    def attempt(**kwargs):
        yield GenerationUpdate(next(outputs), 1, 0.1)
    updates = list(generate_verified(attempt, '', 1.0, 5, 8, 'Abstract motif', '0,3,-1,-1', DEFAULT_NOTES, 100,
                                     rhythm='1,1/2,1/2,2', max_attempts=2))
    stages = [u.stage for u in updates]
    assert stages.count('rejected') == 1 and stages.count('candidate') == 1 and stages[-1] == 'verified'
    assert updates[-1].match.kind == 'contour and rhythm' and updates[-1].seed == 6
