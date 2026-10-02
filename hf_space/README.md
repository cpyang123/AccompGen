---
title: MotiGen
emoji: 🎼
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: 3.10.13
app_file: app.py
pinned: false
license: mit
short_description: Motif-guided music with playback, ABC and MusicXML exports
---

# MotiGen

Interactive music generation from an abstract interval motif (4–10 notes) or a
note-by-note melody with pitches and durations. The model output streams live;
completed compositions have sheet music, synthesized playback, ABC source,
MusicXML source, and both downloadable files. Every result is checked for the
input motif; matching notes are highlighted in amber in the score.

Composition styles are restricted to the checkpoint’s Art Song presets. The
dropdown and generation API enforce the same choices; Schubert remains the default.

The interface uses a classical score-inspired design: ivory paper, ink-colored
controls, muted brass accents, serif headings, and fine rules. The editor and
score preview share the palette, with responsive layouts and visible keyboard
focus. Detailed instructions stay under collapsed Controls sections. Successful results
show the score and playback without status or match-count banners. Candidate,
seed, verification, and prompt information is available under collapsed
Generation details; only brief progress messages and actionable failures remain
in the main view.

**Rhythm.** An abstract motif may carry an optional rhythm: one duration per
motif note, comma-separated, as ratios to the beat (`1,1/2,1/2,2`). The beat is
the meter's bottom number, exactly as in training: under 4/4 `1` is a quarter,
under 6/8 an eighth. It is sent as the `%motif:v1:rhythm:` header line right after
the contour line; when it is blank the model writes its own rhythm line. With a
rhythm given, verification requires the contour *and* those durations (scaled to
the generated score's meter); rests are not supported in the rhythm. Concrete
notes always carry their rhythm (their quarter-note durations under the pinned
4/4), which is also sent as the rhythm block; their verification is unchanged
(exact pitches and durations). The generation API takes the rhythm as a trailing
optional string argument, empty by default.

The abstract editor lets you drag contour points vertically, with live
Up/Down and Step/Skip/Leap labels. Use +/− to choose 4–10 notes; arrow keys
also move focused points. Neighboring points cannot represent repeated notes. The concrete editor
uses bundled VexFlow 5.0.0 with Bravura engraving for the treble staff (C3–C6),
clef, notes, accidentals, dots, ledger lines, and bar lines. Its fonts are bundled
so notation requires no external CDN. Drag duration symbols onto the staff, move notes
vertically to change pitch, or move them horizontally to reorder. Select a note
to change its duration, dot, or accidental. Tap-to-place, pitch selectors, arrow
keys, Undo, and staff-scroll buttons provide alternatives to dragging. Clicking or
dragging a note focuses it for arrow-key editing. Extra instructions live under
collapsed Controls sections. Hear motif
previews the entered notes locally before generation. Drag **Barline** onto a gap,
or select the tool and tap between notes; click a barline to remove it. Undo also
restores barlines. These divisions are included in the concrete motif prompt,
without adding pauses or counting toward the 4–10 note limit. The existing
generation API still accepts the same abstract string and concrete pitch/duration
table; optional `["|", ""]` rows mark barlines after notes. Bar placement guides
the model's phrasing; motif verification continues to require exact pitches and
durations across measure boundaries.

The demo uses the latest saved AccompGen checkpoint as of September 28, 2026:
`weights_notagen_multilen4to10_rhythm_v1_bias4_syn2real5_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1.pth`.
It was saved September 24 (real phase, epoch 3; best validation checkpoint, eval loss
0.142). This "rhythm" checkpoint was fine-tuned on headers that carry a rhythmic motif
next to the melodic one, so prompts may condition on a rhythm as well (see below).
The previous demo checkpoint (`..._inv_v1_...`, September 16) can be restored with
`prepare.py --checkpoint`.
`checkpoint.json` records the exact source, architecture, and exported SHA-256.
The exported safetensors uses the same float16 precision as GPU inference and is
compressed losslessly with gzip to fit the private Space's repository size limit.
Optimizer state, training data, and credentials are excluded.

Abstract intervals start with 0, followed by ±1 (second), ±2 (third), or ±3
(fourth or larger). Concrete rows use scientific pitch notation and quarter-note
beats. Concrete prompts include the trained `%motif:abc:` field, and use C major,
4/4, L:1/8. Sharps and flats are explicit; natural signs appear only when
canceling an earlier accidental on the same pitch within a measure. Ordinary
natural notes have no redundant `=` prefix. Abstract verification checks successive step/skip/leap intervals,
collapsing immediate repeated pitches and ignoring rests as in training. It does
not accept an inversion in place of the requested contour. Concrete verification
requires the entered absolute pitches and durations, with ties combined; rests or
extra notes interrupt an exact concrete match. The melody is the first voice;
chords contribute their upper melodic note. Matches are found in actual score
notes, never in the prompt's motif comments. The ABC display and download retain
`%motif:` comments so the conditioning metadata remains inspectable. The
expandable **Model prompt used** section shows the exact server-built prefill,
including the concrete ABC motif when supplied; generated metadata in the score
is not a substitute for that prompt record. The requested occurrence count still
guides the model, while acceptance requires at least one verified occurrence.

Generation uses request-local patch and character KV caches, with the original
motif attention bias of +4 on every `%motif:` patch, including the concrete
realization notes. The training weight 3 is a flag (`weight > 1`), not an
additional multiplier. Concrete prompt tokenization follows the training
`Patchilizer`: the `%motif:abc:` tag is patched separately, and its notes use
the training bar-splitting rules. Treating the whole realization line as ordinary
metadata gives the model different patch boundaries and has been corrected.
Tests check the actual attention masks and cached/full-context equivalence.
Requests are serialized. Every request generates all
12 candidates using consecutive seeds (starting seed + candidate - 1), even if
an early candidate contains the motif. Candidates without at least one verified
occurrence are discarded. Eligible pieces are ranked first by the number of
verified motif occurrences, then by distinct (pitch, sounding duration) pairs
across all written voices and chord members. More motif occurrences take priority
over note variety.
Repeated identical notes, rests, chord labels, and motif comments do not increase
the score. Enharmonic pitches are equivalent; tied durations are combined. Ties
in both motif count and variety favor the earlier candidate. This is a simple variety heuristic, not
a measure of musical quality. The result identifies the winning candidate, seed,
variety count, and number of eligible candidates.

The 105-second generation budget is shared across remaining candidates, with up
to 800 patches and 8/16/32 completed measures per candidate. A time limit or Stop
before all 12 finish leaves no selected result. If none of the 12 contains the
motif, all result/export panels remain empty. Only the winner is exported and
highlighted, after the full batch has been evaluated.
Playback is synthesized from the exported XML without an external soundfont.
Every written chord tone and simultaneous voice is played, with ties resolved
individually per pitch and voice. Chord labels such as `"Am"`, `"G7"`, and
`"Bb/D"` add a lower-register block-chord accompaniment at half the gain of
written notes, sustained until the next label or the part's end. `N.C.` stops
that accompaniment. Duplicate labels across staves are not played twice.
The ABC and MusicXML exports preserve the written score and chord labels; the
additional accompaniment voicing is a playback arrangement. Invalid generated notation
retains the ABC and reports conversion errors instead of providing broken files.

## Local use

```bash
python -m pip install -r requirements.txt
# Install Node.js (used to verify ABC with the same parser as the score renderer).
python prepare.py                     # latest mtime in AccompGen's weights directory
python app.py                         # http://localhost:7860
```

Alternatively, `python prepare.py --checkpoint /path/to/checkpoint.pth` pins a
specific source. Set `ACCOMPGEN_CHECKPOINT` to use a different local weights file
with the same architecture. CPU works for testing; use a GPU for interactive speed.

```bash
python -m pip install pytest
python -m pytest tests -q
```

For the visual editor integration checks, install Playwright and Chromium, then
run `python tests/browser_editor.py`. This uses a deterministic model substitute
to check that visual edits reach generation and verification without loading the
checkpoint; hosted model checks are recorded separately in `validation/`.

The sample site’s walking cat crosses the bottom of the screen every three
minutes, alternating directions. Its lane follows the visible screen edge,
including inside a tall embedded Space while the parent page scrolls or resizes.
It never intercepts clicks and respects reduced
motion preferences and hidden browser tabs.

## Hugging Face deployment

```bash
hf auth login
python deploy.py --repo cpyang/motigen
```

This uploads only the explicitly listed app assets and exported checkpoint to a
new public Gradio Space using ZeroGPU. Deployment aborts if the repository name
already exists, before uploading any files. Use `--private` for a private Space.
It does not modify other Spaces or request
paid dedicated hardware. The [ZeroGPU runtime](https://huggingface.co/docs/hub/spaces-zerogpu)
allocates GPU time per request; visitors may encounter queueing or quota limits.
Keep the `spaces` import before torch and the model initialization outside the
decorated function. The model loads at startup on Spaces, once per local process.

Vendored `notagen_core` comes from this AccompGen checkout; `abc2xml.py` retains
Willem G. Vree's LGPL notice, and abcjs 6.4.3 retains its MIT notice. Update these
copies deliberately if the upstream inference architecture changes.
VexFlow 5.0.0 is distributed under MIT (`assets/VEXFLOW-LICENSE`); its embedded
Bravura and Academico fonts use SIL OFL 1.1 (`assets/BRAVURA-LICENSE` and
`assets/ACADEMICO-LICENSE`).
