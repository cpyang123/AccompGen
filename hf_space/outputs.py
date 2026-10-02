"""Isolated ABC/MusicXML export, score preview, and local piano-like synthesis."""
import base64
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(tempfile.gettempdir()) / "motigen-outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)


def clean_abc(raw, tempo=100):
    lines = []
    for line in raw.splitlines(keepends=True):
        if line.startswith("[r:") and not line.endswith("\n"):
            # Retain only completed measures if stopped in the middle of a patch.
            last_bar = line.rfind("|")
            if last_bar < 0:
                continue
            line = line[:last_bar + 1]
        line = re.sub(r"\[r:[^\]]*\]", "", line).strip()
        # Motif comments document the conditioning; notation parsers ignore them.
        # Keep filtering unrelated comments/directives such as abc-include.
        if not line or (line.startswith("%") and not line.startswith(("%%score", "%motif:"))):
            continue
        if line.startswith(("X:", "T:", "Q:", "I:")):
            continue
        line = "".join(c for c in line if c.isprintable() or c == "\t")
        lines.append(line)
    if not any("[V:" in s for s in lines) or not any(s.startswith("K:") for s in lines):
        raise ValueError("The model did not finish a playable measure. Try Generate again or choose a longer length.")
    # A stopped excerpt must not carry an unclosed repeat into playback.
    return f"X:1\nT:MotiGen composition\nQ:1/4={int(tempo)}\n" + "\n".join(lines) + "\n"


def score_html(abc, motif_match=None):
    encoded = base64.b64encode(abc.encode()).decode()
    library = (ROOT / "assets/abcjs-basic-min.js").read_text()
    document = """<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1">
    <style>body{margin:0;padding:18px;background:#fffdf7;color:#302a24;font:14px system-ui}svg{max-width:100%}#error{color:#98422d}
    .motif-note,.motif-note *{fill:#95652a!important;stroke:#95652a!important}#score text{font-family:Georgia,"Times New Roman",serif}</style>
    </head><body><div id="score"></div><p id="error"></p><script>""" + library + "</script><script>" + f"""
    const spans = {json.dumps(motif_match.spans if motif_match else [])};
    const matches = {motif_match.count if motif_match else 0};
    try {{
      const tune = ABCJS.renderAbc('score', atob('{encoded}'), {{responsive:'resize', staffwidth:760, wrap:{{minSpacing:1.8,maxSpacing:2.7,preferredMeasuresPerLine:4}}, add_classes:true}})[0];
      if (matches) {{
        (tune.lines || []).forEach(line => (line.staff || []).forEach(staff =>
          (staff.voices || []).forEach(voice => voice.forEach(note => {{
            if (note.el_type !== 'note' || note.rest || note.startChar == null) return;
            if (!spans.some(([a, b]) => note.startChar < b && note.endChar > a)) return;
            if (note.abselem && note.abselem.elemset) note.abselem.elemset.forEach(element => {{
              if (element.classList) {{ element.classList.add('motif-note'); element.setAttribute('data-motif-match', 'true'); }}
            }});
          }}))));
      }}
    }}
    catch(e) {{ document.getElementById('error').textContent='The score could not be rendered. ABC and XML remain available.'; }}
    </script></body></html>"""
    return '<iframe title="Generated sheet music" sandbox="allow-scripts" style="width:100%;height:430px;border:0;border-radius:3px;background:#fffdf7" srcdoc="' + html.escape(document, quote=True) + '"></iframe>'


def playback_events(score, tempo):
    """Return (start, seconds, MIDI pitch, gain) for notes and chord labels."""
    from music21 import chord, expressions, harmony, note, stream
    import math

    raw, harmonies, part_ends = [], {}, {}
    def visit(container, base, part_id, voice_id='1'):
        for element in container:
            start = base + float(element.offset)
            if isinstance(element, stream.Stream):
                voice = str(element.id) if isinstance(element, stream.Voice) else voice_id
                visit(element, start, part_id, voice)
            elif isinstance(element, harmony.ChordSymbol):
                # Symbols have zero duration in MusicXML. Sustain a modest lower
                # voicing until the next label (or the end of this part).
                pitches = sorted(p.midi for p in element.pitches)
                if pitches:
                    while pitches[0] < 43:
                        pitches = [p + 12 for p in pitches]
                    while pitches[0] > 54:
                        pitches = [p - 12 for p in pitches]
                harmonies.setdefault(part_id, {})[start] = pitches
            elif (isinstance(element, expressions.TextExpression)
                  and re.sub(r'[\s.\-]', '', element.content).lower() in ('nc', 'nochord')):
                harmonies.setdefault(part_id, {})[start] = []
            elif isinstance(element, (note.Note, chord.Chord)):
                notes = element.notes if isinstance(element, chord.Chord) else [element]
                for member in notes:
                    length = float(member.quarterLength)
                    if length <= 0:
                        continue
                    if len(raw) >= 5000:
                        raise ValueError("The generated score is too large to play in the demo.")
                    key = (part_id, voice_id, member.pitch.midi)
                    raw.append((start, length, member.pitch.midi, key,
                                member.tie.type if member.tie else None))

    parts = list(score.parts) if isinstance(score, stream.Score) else []
    if parts:
        for part_id, part in enumerate(parts):
            part_ends[part_id] = float(part.offset) + float(part.highestTime)
            visit(part, float(part.offset), part_id)
    else:
        part_ends[0] = float(score.highestTime)
        visit(score, 0.0, 0)
    events, pending = [], {}
    for start, length, midi, key, tie in sorted(raw, key=lambda event: event[0]):
        previous = pending.get(key)
        if (tie in ('stop', 'continue') and previous is not None
                and math.isclose(events[previous][0] + events[previous][1], start, abs_tol=1e-7)):
            index = previous
            events[index][1] += length
        else:
            index = len(events)
            events.append([start, length, midi])
        if tie in ('start', 'continue'):
            pending[key] = index
        else:
            pending.pop(key, None)
    # Repeated harmony labels on multiple staves must not double the volume.
    accompaniment = set()
    for part_id, track in harmonies.items():
        changes = sorted(track.items())
        for index, (start, pitches) in enumerate(changes):
            end = changes[index + 1][0] if index + 1 < len(changes) else part_ends[part_id]
            if end > start:
                accompaniment.update((start, end - start, midi) for midi in pitches)
            if len(events) + len(accompaniment) > 5000:
                raise ValueError("The generated score is too large to play in the demo.")
    seconds_per_beat = 60.0 / tempo
    return ([(start * seconds_per_beat, length * seconds_per_beat, midi, 1.0)
             for start, length, midi in events]
            + [(start * seconds_per_beat, length * seconds_per_beat, midi, 0.5)
               for start, length, midi in sorted(accompaniment)])


def synthesize(xml_path, wav_path, tempo):
    from music21 import converter
    events = playback_events(converter.parse(str(xml_path)), tempo)
    if not events:
        raise ValueError("No playable notes were found in the generated score.")
    duration = max(start + length for start, length, _, _ in events) + 0.3
    if duration > 300:
        raise ValueError("Playback is limited to five minutes; download the score to play the full piece.")
    rate = 22050
    audio = np.zeros(int(duration * rate) + 1, dtype=np.float32)
    for start, length, midi, gain in events:
        size = max(1, int((length + 0.12) * rate))
        t = np.arange(size, dtype=np.float32) / rate
        freq = 440 * 2 ** ((midi - 69) / 12)
        tone = sum((1 / harmonic ** 2) * np.sin(2 * np.pi * freq * harmonic * t)
                   for harmonic in range(1, 6) if freq * harmonic < rate / 2)
        envelope = np.minimum(t / 0.008, 1) * np.exp(-1.4 * t / max(length, 0.1))
        envelope *= np.clip((length + 0.12 - t) / 0.12, 0, 1)
        offset = int(start * rate)
        count = min(size, len(audio) - offset)
        audio[offset:offset + count] += (tone * envelope)[:count] * 0.2 * gain
    peak = float(np.max(np.abs(audio)))
    pcm = (audio / max(peak, 0.01) * 0.88 * 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


def export_result(raw, tempo=100, motif_match=None):
    abc = clean_abc(raw, tempo)
    # Gradio cleans its cached copies; also expire our original request files.
    cutoff = time.time() - 3600
    for old in OUTPUT_ROOT.glob("composition-*"):
        try:
            if old.is_dir() and old.stat().st_mtime < cutoff:
                shutil.rmtree(old)
        except FileNotFoundError:
            pass
    directory = Path(tempfile.mkdtemp(prefix="composition-", dir=OUTPUT_ROOT))
    abc_path, xml_path, wav_path = [directory / f"composition.{ext}" for ext in ("abc", "xml", "wav")]
    abc_path.write_text(abc)
    warnings = []
    xml, audio = "", None
    try:
        result = subprocess.run([sys.executable, str(ROOT / "vendor/abc2xml.py"), "-o", str(directory), str(abc_path)],
                                capture_output=True, text=True, timeout=30)
        if result.returncode or not xml_path.exists():
            raise ValueError("The generated ABC could not be converted to MusicXML.")
        xml = xml_path.read_text()
        document = ET.fromstring(xml)
        if document.tag != "score-partwise" or not (document.findall(".//note/pitch") or document.findall(".//harmony/root")):
            raise ValueError("The model produced no exportable notes or chord labels.")
    except (ValueError, subprocess.TimeoutExpired, ET.ParseError) as error:
        warnings.append(str(error) + " The ABC is available; try another generation for XML and audio.")
        xml_path = None
    if xml_path:
        try:
            synthesize(xml_path, wav_path, tempo)
            audio = str(wav_path)
        except Exception as error:
            warnings.append(f"Playback could not be prepared ({type(error).__name__}). Both score downloads are available.")
    return abc, xml, str(abc_path), str(xml_path) if xml_path else None, audio, score_html(abc, motif_match), " ".join(warnings)
