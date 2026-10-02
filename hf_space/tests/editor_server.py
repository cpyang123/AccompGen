"""Deterministic model substitute for visual-editor browser integration checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from engine import GenerationUpdate
from motifs import concrete_motif
from outputs import OUTPUT_ROOT
from retries import generate_verified


def fixture_stream(prompt, temperature, seed, bars, mode, pattern, notes, tempo):
    if mode == 'Concrete notes':
        _, snippet = concrete_motif(notes)
    else:
        positions = [28]
        for move in map(int, pattern.split(',')[1:]):
            positions.append(positions[-1] + move)
        rows = [[f'{"CDEFGAB"[p % 7]}{p // 7}', '1'] for p in positions]
        _, snippet = concrete_motif(rows)
    calls = 0
    def attempt(**kwargs):
        nonlocal calls
        calls += 1
        text = 'L:1/8\nM:4/4\nK:C\nV:1\n[r:0/0][V:1]' + snippet + ('\n' if snippet.rstrip().endswith('|') else '|\n')
        if calls == 7:
            text += "[V:1]" + snippet + ("\n" if snippet.rstrip().endswith("|") else "|\n")
        if calls == 12:
            text += '[V:1]B,/2 c3/2 d2 e4|\n'
        yield GenerationUpdate(text, 4, .1, 'Complete.')
    yield from generate_verified(attempt, prompt, temperature, seed, bars, mode, pattern, notes, tempo)


app.stream_model = fixture_stream
app.build_demo().launch(server_name='127.0.0.1', server_port=7869,
                        allowed_paths=[str(OUTPUT_ROOT)], show_error=True)
