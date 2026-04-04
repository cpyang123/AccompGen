import os
import sys

# Ensure project root is on sys.path so `import motif` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from motif import get_best_motifs

sample_abc = """X:1
T:Sample
M:4/4
K:C
V:1
C D E F | G A B c | C D E F | G A B c | C D E F |
"""

motifs = get_best_motifs(sample_abc, interval_mode="step_skip_leap")
print(motifs)
