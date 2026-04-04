import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from motif.extract import abc_to_pitches_with_bars
abc = "X:1\nT:Sample\nM:4/4\nK:C\nV:1\nC D E F | G A B c | C D E F | G A B c | C D E F |\n"
pitches, note_to_bar, _, _, bar_starts = abc_to_pitches_with_bars(abc)
for i, p in enumerate(pitches):
    print(f"note {i}: pitch {p}, bar {note_to_bar[i]}")
print("bar_starts:", bar_starts)
