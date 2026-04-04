import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from motif.extract import find_top_motif
abc = "X:1\nT:Sample\nM:4/4\nK:C\nV:1\nC D E F | G A B c | C D E F | G A B c | C D E F |\n"
res = find_top_motif(abc, 4, interval_mode="step_skip_leap")
print(res[1])
