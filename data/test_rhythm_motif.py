"""Hand-checked cases for motif/rhythm.py (run: python data/test_rhythm_motif.py).

Every case pins one rule of the rhythmic-motif definition: beat = 1/meter
denominator, rests inside a window carry a 'z' and do not count toward the
length, no collapsing, ties merge, chords are one event, grace notes vanish,
tuplets and broken rhythms resolve to exact fractions.
"""
import os
import sys
from fractions import Fraction

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from motif import rhythm as rx  # noqa: E402


def tokens(abc, **kw):
    events, _, _ = rx.abc_to_rhythm_events(abc, **kw)
    return [rx.event_token(e) for e in events]


def top(abc, L, **kw):
    per_len = rx.get_best_rhythm_motifs_per_length(abc, window_range=(L, L + 1), **kw)
    return per_len.get(L)


def check(name, got, want):
    if got != want:
        print(f'FAIL {name}\n  want {want}\n  got  {got}')
        sys.exit(1)
    print(f'ok   {name}')


# --- beat unit -------------------------------------------------------------
check('4/4 L:1/8: eighth eighth quarter half',
      tokens('M:4/4\nL:1/8\nK:C\nC D E2 F4|'), ['1/2', '1/2', '1', '2'])
check('4/4 L:1/4: same rhythm, different unit',
      tokens('M:4/4\nL:1/4\nK:C\nC/ D/ E F2|'), ['1/2', '1/2', '1', '2'])
check('6/8: denominator is the beat, so an eighth is 1',
      tokens('M:6/8\nL:1/8\nK:C\ngeg c\'ba|gag e3|'),
      ['1', '1', '1', '1', '1', '1', '1', '1', '1', '3'])
check('2/2: half note is the beat',
      tokens('M:2/2\nL:1/8\nK:C\nC4 D2 E F|'), ['1', '1/2', '1/4', '1/4'])
check('3/4 with explicit meter/unit args (no header lines in the voice)',
      tokens('C2 D2 E2|', meter=(3, 4), unit_length=Fraction(1, 8)), ['1', '1', '1'])
check('no M: field: the L: unit is the beat',
      tokens('L:1/8\nK:C\nC D E2|'), ['1', '1', '2'])
check('no M: and no L: -> ABC default unit 1/8 is the beat',
      tokens('K:C\nC D E2|'), ['1', '1', '2'])
check('M:C is 4/4 and M:C| is 2/2',
      (tokens('M:C\nL:1/8\nC2|'), tokens('M:C|\nL:1/8\nC2|')), (['1'], ['1/2']))
check('mid-piece [M:] is ignored; [L:] is honoured',
      tokens('M:4/4\nL:1/8\nK:C\nC2 D2|[M:3/4]E2 F2|[L:1/16]G4 A4|'),
      ['1', '1', '1', '1', '1', '1'])

# --- rests -----------------------------------------------------------------
check('rest carries a z suffix',
      tokens('M:4/4\nL:1/4\nK:C\nC z D z/ E|'), ['1', '1z', '1', '1/2z', '1'])
check('multi-bar rest Z2 = two bars of rest',
      tokens('M:3/4\nL:1/4\nK:C\nC D E|Z2|F|'), ['1', '1', '1', '6z', '1'])
check('interior rest is in the window but not counted: length-2 window over C z D',
      [m['pattern'] for m in top('M:4/4\nL:1/4\nK:C\nC z D|', 2)], [('1', '1z', '1')])
check('edge rests are outside the window',
      [m['pattern'] for m in top('M:4/4\nL:1/4\nK:C\nz C D E F z|', 4)], [('1', '1', '1', '1')])

# --- ties, chords, grace notes, tuplets, broken rhythm ---------------------
check('tie merges into the played duration',
      tokens('M:4/4\nL:1/8\nK:C\nC2-C2 D E F|'), ['2', '1/2', '1/2', '1/2'])
check('tie across a barline merges too',
      tokens('M:4/4\nL:1/8\nK:C\nC4 D2 E2-|E2 F6|'), ['2', '1', '2', '3'])
check('a "-" between different pitches is not a tie',
      tokens('M:4/4\nL:1/8\nK:C\nC2-D2|'), ['1', '1'])
check('chord is one event; inner and outer duration multiply',
      tokens('M:4/4\nL:1/8\nK:C\n[CEG]2 [C2E2] [CEG]|'), ['1', '1', '1/2'])
check('tied chord merges',
      tokens('M:4/4\nL:1/8\nK:C\n[CE]2-[CE]2 D|'), ['2', '1/2'])
check('grace notes are dropped',
      tokens('M:4/4\nL:1/8\nK:C\n{ab}C D {g}E F|'), ['1/2', '1/2', '1/2', '1/2'])
check('triplet: three eighths in the time of two',
      tokens('M:4/4\nL:1/8\nK:C\n(3CDE F|'), ['1/3', '1/3', '1/3', '1/2'])
check('explicit tuplet spec (5:4:5)',
      tokens('M:4/4\nL:1/16\nK:C\n(5:4:5CDEFG A|'), ['1/5', '1/5', '1/5', '1/5', '1/5', '1/4'])
check('duplet in 6/8 (2 in the time of 3)',
      tokens('M:6/8\nL:1/8\nK:C\n(2CD E|'), ['3/2', '3/2', '1'])
check('broken rhythm > and <',
      tokens('M:4/4\nL:1/8\nK:C\nC>D E<F|'), ['3/4', '1/4', '1/4', '3/4'])
check('double broken rhythm >>',
      tokens('M:4/4\nL:1/8\nK:C\nC>>D|'), ['7/8', '1/8'])
check('broken rhythm applies to a rest too',
      tokens('M:4/4\nL:1/8\nK:C\nC>z|'), ['3/4', '1/4z'])

# --- structure that must not become durations ------------------------------
check('voltas, repeats, thick barlines and decorations leave no stray events',
      tokens('M:4/4\nL:1/4\nK:C\n|:C D |1 E F :|2 [|E2 |] .G ~A y2 !f! "Am" B|'),
      ['1', '1', '1', '1', '2', '1', '1', '1'])
check('processed [V:1] bar lines parse like plain text',
      tokens('[V:1]C D E2|\n[V:1]F4|', meter=(4, 4), unit_length=Fraction(1, 8)),
      ['1/2', '1/2', '1', '2'])

# --- counting and realization ---------------------------------------------
piece = ('M:4/4\nL:1/8\nK:C\n'
         'C D E2 F4|G A B2 c4|C2 D2 E2 F2|G A B2 c4|')
m3 = top(piece, 3)
check('top length-3 motif is the most frequent window, first-seen on ties',
      (m3[0]['pattern'], m3[0]['count']), (('1/2', '1/2', '1'), 3))
check('realization is the bar text of the first occurrence',
      m3[0]['abc'], 'C D E2 F4')
check('no collapsing: equal successive durations are kept',
      [m['pattern'] for m in top('M:4/4\nL:1/8\nK:C\nC2 D2 E2 F2|', 4)],
      [('1', '1', '1', '1')])
check('length absent when the voice has too few notes',
      top('M:4/4\nL:1/8\nK:C\nC D E|', 4), None)
m_rest = top('M:4/4\nL:1/8\nK:C\nC z D E2|F z G A2|', 3)
check('windows with an interior rest count together across bars',
      (m_rest[0]['pattern'], m_rest[0]['count']), (('1/2', '1/2z', '1/2', '1'), 2))
check('start/end line of the first occurrence (one bar per line)',
      (m_rest[0]['start_line'], m_rest[0]['end_line']), (0, 0))
m_span = top('M:4/4\nL:1/8\nK:C\nC D E2 F2-|\nF2 G A B|', 5)
check('a tie across the barline extends the realization to the next line',
      (m_span[0]['abc'], m_span[0]['end_line']), ('C D E2 F2-|\nF2 G A B', 1))
check('token round trip', rx.parse_ratio('3/2z'), (Fraction(3, 2), True))

print('PASS: all rhythm motif cases')
