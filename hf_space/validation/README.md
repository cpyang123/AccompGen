# Validation — September 17, 2026

- All 86 unit/integration checks passed (`python -m pytest tests -q`).
- The exported September 16 checkpoint loaded with strict state-dictionary matching.
- Patch and character caching matched the original uncached model's probabilities
  within floating-point tolerances on a small instance of the same architecture.
- Full-checkpoint local generations succeeded for both abstract and concrete motifs.
- Chromium rendered the generated sheet music; synthesized audio loaded and played.
- Both download buttons delivered nonempty files (`composition.abc` and
  `composition.xml`); MusicXML conversion and pitched notes were checked.
- No JavaScript page errors occurred. The 390px mobile viewport had a 390px document
  width (no horizontal overflow).
- Deployed to the newly created **private** Hugging Face Space
  [cpyang/motigen](https://huggingface.co/spaces/cpyang/motigen), commit
  `468908397ecef6fb683584da5d580428688c3dcc`. Other repositories were not modified.
- To fit the 1 GB private Space repository limit, model weights are exported as
  float16 safetensors compressed with gzip (952,941,836 bytes). Every weight was
  checked for exact equality with the original checkpoint converted to the same
  float16 precision already used for GPU inference.
- The Space built successfully and reached `RUNNING` on `zero-a10g` hardware.
- Motif verification checks the score melody, excluding prompt metadata. Abstract
  input checks the requested contour; concrete input checks exact pitches and
  durations. Regression tests cover keys, accidentals, ties, tuplets, multiple
  voices, partial chord ties, and highlights at the actual note positions.
- Deterministic browser fixtures confirmed that both modes reject the first
  candidate, accept a matching second candidate, and highlight its notes.
- Hosted API verification found four matches for abstract input `0,3,-1,-1`,
  seed 0, eight measures, with valid exports and audio. The concrete default motif,
  seed 42, eight measures, failed all 12 attempts; the app correctly retried and
  returned no accepted composition, downloads, or audio. These checks used
  commit `86c62951a4f97b65f14346d40afb4c62094451d0` before the partial chord tie fix.
- Hosted details are recorded in [deployment.json](deployment.json).
- The final deployment reached `RUNNING` and its authenticated hosted page
  rendered verified abstract generation in Chromium:
  one match, six highlighted SVG elements, and successful audio playback with no
  JavaScript errors; see [highlighted hosted screenshot](highlighted-hosted.png).
- Automatic retries stop after 12 attempts or the 105-second generation budget;
  unmatched candidates are never accepted as finished compositions.

See [desktop screenshot](desktop.png) and [mobile screenshot](mobile.png).

## Visual motif editors

- The abstract editor now uses draggable points with live direction and interval labels.
- The concrete editor supports palette-to-staff dragging, pitch movement, note
  reordering, duration changes, dotted notes, accidentals, and local audio preview.
- Browser integration checks passed with a deterministic model substitute: both
  editors' actual serialized input reached generation and motif verification;
  accepted results retained highlighted notation and export controls.
- Undo, clearing, the ten-note limit, keyboard pitch changes, touch placement,
  horizontal staff navigation, and the 390px mobile layout were checked.
- All 48 existing unit/integration tests still pass. The visual editor browser
  checks can be reproduced with `python tests/browser_editor.py`.
- Local screenshots: [abstract editor](abstract-editor.png),
  [concrete editor](concrete-editor.png), and [mobile](editor-mobile.png).
- The visual editor deployment reached `RUNNING` at commit
  `cef4f8324876b188d7d6f6c12a4380939b074507`. Hosted Chromium checks confirmed
  abstract generation with a verified match, six highlighted SVG elements, and
  working playback. A note dragged onto the concrete staff was included unchanged
  in the generation request. That concrete request exhausted 12 attempts without
  an exact match and correctly returned no accepted piece. No JavaScript errors
  occurred. See [hosted results](visual-editors-hosted.json),
  [hosted abstract editor](abstract-editor-hosted.png), and
  [hosted concrete editor](concrete-editor-hosted.png).

## Direct contour editing and walking cat

- Abstract dropdowns were replaced with draggable contour points and +/− length
  controls. Mouse, real touch dragging, keyboard changes, and generation using
  the edited contour passed the browser integration checks.
- Removed the duplicate pitch/duration button row below the staff. Direct note
  selection, dragging, keyboard editing, and concrete generation still pass.
- Reused the SVG cat and gait from `AccompGen/website/index.html`, scheduled every
  180,000 ms. Browser checks cover one timer, alternating directions, no overlapping
  cats, cleanup, pointer transparency, no horizontal overflow, and reduced motion.
- Deployed commit `617394ded4485940ae4515cacf995b472259194e` reached RUNNING.
  Hosted dragging produced contour `0,-2,3,-1`, which reached the model unchanged
  and generated one verified match on attempt 1. The staff selected notes directly;
  there were no abstract dropdowns or duplicate note buttons. The cat's timer was
  registered at 180,000 ms and its live animation moved 105.87 pixels during the
  check. No JavaScript errors occurred. See [hosted results](drag-contour-cat-hosted.json)
  and [screenshot](drag-contour-cat-hosted.png).

## Chord playback

- All 60 unit/integration tests pass, including 12 playback checks through actual
  ABC → MusicXML → WAV export. Spectral checks confirm simultaneous written chord
  tones and the accompaniment tones before and after a chord-label change.
- Written chords retain every tone, simultaneous voices stay independent, and
  ties are merged per pitch/voice, including ties through changing chords.
- Chord labels supply quieter accompaniment until the next label or the part's
  end. Tests cover minor and seventh chords, chromatic slash-chord basses,
  `N.C.`, labels over rests, generic text annotations, and duplicate labels on
  multiple staves. Source ABC and XML remain notation exports, without inserting
  the synthesized accompaniment as extra score notes.
- Playback deployment `17537f24f095a9ab6d4061b106a347242045cae4` reached
  RUNNING. A hosted Chopin/Keyboard generation verified the motif on attempt 1
  and exported a valid 14.1-second WAV containing 18 scheduled notes. Its duration
  matched the generated score. This sampled piece did not contain chords or chord
  labels; those cases were tested through the real local export/audio path above.
  Hosted results: [chord-playback-hosted.json](chord-playback-hosted.json).

## Prefill and exported motif metadata

- Confirmed that both abstract and concrete prompts reach `CachedDecoder.encode`
  intact, with motif attention flags on their conditioning lines. The export
  cleanup had been removing all `%motif:` comments after generation.
- ABC display/download now preserve motif comments, while include directives
  remain filtered. Export cleanup is idempotent and motif metadata is not counted
  as a melodic match. The UI shows the exact server-built prompt in an expandable
  `Model prompt used` section, including on unsuccessful retry attempts.
- All 65 unit/integration tests pass. The supplied short D/G/B score is rejected
  in both modes for the default motif, even with a matching motif preamble. The
  original input and seed of that historical example have not been supplied.
- Desktop/touch browser checks pass, including viewing the exact abstract and
  concrete prompts alongside generation, verification, and highlighted notation.
- Deployment `8bc0baea60c417695d8aa980249efce6b4ba4880` reached RUNNING.
  Hosted abstract generation preserved the exact prompt in all seven sampled
  generation updates, accepted a verified melody, and retained motif metadata in
  both the ABC display and download. Concrete generation retained its exact notes
  in all 57 sampled updates over 12 attempts; those attempts did not contain an
  exact match and correctly produced no accepted score. The prompt remained
  inspectable in both modes. See [hosted prefill checks](prefill-hosted.json).

## Staff barlines

- Desktop drag and tap placement, trailing barlines, click/keyboard removal,
  Undo/Clear/Reset, deletion before a measure boundary, and mobile touch placement
  passed in the real browser editor.
- Barline markers survive the two-column generation payload and `%motif:abc:`
  prefill, without contributing to the 4–10 note count. Invalid leading,
  consecutive, or duration-bearing markers are rejected.
- Verification accepts exact pitches/durations across bar boundaries and still
  rejects intervening rests. The editor preview schedules notes without barline
  pauses. The deterministic generation fixture renders and highlights the result.
- Private hosted commit `9accdd09a95ff4f0d022196667a7b9c13bacafb7`
  reached RUNNING. A real browser entered two barlines, submitted the exact
  marker rows, and confirmed `%motif:abc: =C2 =G | =F =E4 |` in the server's
  model prompt. No JavaScript errors occurred. This sampled concrete request
  exhausted 12 attempts without an exact match; no unverified output was accepted.
  See `barlines-hosted.json` and `barlines-hosted.png`.

## Keyboard focus, concise editor, and viewport cat

- Reproduced the hosted keyboard failure: clicking a note left focus on the mode
  tab, so arrow presses never reached the note handler. Selection and dragging
  now focus the selected note; rerenders preserve note focus.
- Browser regressions click a note and press all four arrows without calling
  `focus()` from the test. Drag-then-keyboard and touch-tab switching also pass.
- Detailed instructions are collapsed under Controls. The introduction,
  selection readout, section labels, and ready status are shorter; redundant
  footer guidance is removed.
- The cat uses the visible intersection of the app with the top-level viewport,
  rather than assuming an embedded frame fits the screen. Cross-origin fixture
  checks passed with a 2400px frame, 180px host header, scroll offsets 0/350/850,
  and desktop-to-mobile resizing. The lane remained at y=screen height−40.
- The original three-minute schedule, alternating direction, pointer transparency,
  animation cleanup, and reduced-motion checks still pass.
- Hosted commit `075a516a6e004056e35216420e9a74152fcec4d7` reached
  RUNNING. A real browser verified click-then-arrow editing and collapsed help.
  The hosted app's cat stayed at screen y=900 in standalone and a controlled
  cross-origin embed at scroll offsets 0/350/850, then y=700 after mobile resize.
  No JavaScript errors occurred. See `keyboard-viewport-hosted.json`,
  `keyboard-clean-hosted.png`, and `cat-viewport-hosted.png`.

## Twelve-candidate variety selection

- All twelve consecutive-seed candidates are generated before choosing a result.
  At least one motif occurrence is required. The eligible piece with the most
  distinct (MIDI pitch, sounding duration) pairs wins; equal scores favor the
  earlier candidate. All written voices and chord members contribute.
- Regression checks cover a final-candidate winner, a more varied ineligible
  candidate, early matches without early exit, deterministic ties, no-match
  batches, cancellation, and a time limit without a partial-batch winner.
- Note variety tests cover repeated notes, duration/octave differences, chord
  members, multiple voices, enharmonics, ties, tuplets, metadata, rests, and chord
  labels. Only actual written sounding notes contribute to the score.
- The deterministic browser fixture makes candidate 12 the best in both input
  modes. Both select that candidate and produce highlighted sheet music, exports,
  and playback. Other desktop, mobile, barline, keyboard, and cat checks pass.
- The latest Chromium requires its local-network check disabled for the synthetic
  cross-origin local-frame test fixture; production browser settings are unchanged.
- Hosted commit `303453caed5e0e40c20d05dce2d227d13521cc92` reached RUNNING.
  Both modes generated all 12 candidates. Each complete candidate was captured
  and independently reverified and rescored locally. The abstract batch chose
  candidate 5 with 18 note types and 4 motif matches; candidate 6 tied on variety,
  so the earlier candidate correctly won. Exports and highlighted source matched
  candidate 5. The sampled concrete batch had no exact motif matches and accepted
  no output. Full candidate rankings are in `ranking-hosted.json`.

## Concrete prefill natural signs

- Ordinary natural notes no longer get an explicit `=` in the `K:C` concrete
  prefill: the default motif is `C2 G F E4`. Sharps/flats stay explicit; a natural
  appears only to cancel an earlier accidental on that pitch in the same measure.
- Tests verify omission, required cancellation, barline resets, separate octave
  accidental state, and exact pitches when reparsed. Browser generation confirms
  the edited motif with barlines reaches the model prompt without redundant signs.
- All 82 tests and the full browser suite passed before deployment.
- Hosted commit `29c673de826ed7e9976c5ed2c11f9a83737c8c75` reached RUNNING.
  The live concrete request generated all 12 candidates; every captured candidate
  started with the exact expected prefill including `%motif:abc: C2 G F E4`.
  Independent local checks found no exact motif matches in this sampled batch,
  and the demo correctly returned no accepted composition or exports. See
  `ranking-hosted-natural.json` for all candidate variety scores and the prefill.

## Classical visual design

- Ivory paper surfaces, ink controls, muted brass accents, serif headings,
  fine rules, and a small SVG music emblem replace the previous green theme.
  The contour, staff, playback panel, and generated score share the palette.
- Kept Controls collapsed and the existing concise copy. Desktop uses a fine
  column divider; mobile stacks the editor and score without horizontal overflow.
- The complete browser suite passed: desktop/touch/keyboard editors, barlines,
  twelve-candidate selection in both modes, score highlights, audio, and cat
  positioning. Two focused export/highlight regression tests also passed.
- Reviewed `classical-desktop.png`, `classical-staff.png`, `classical-mobile.png`,
  `classical-dark-preference.png`, and the generated-score screenshot. The paper
  design remains readable under a dark browser preference.
- Hosted commit `21e15adce830803486a96085a9c7c87d421b0e9e` reached RUNNING.
  Live browser checks confirmed the theme, mobile layout, keyboard editing,
  barlines, and readable paper styling with dark preference. A real generation
  selected candidate 5/12 with 18 note types and 4 highlighted motif matches.
  The rendered score used the ivory background; no JavaScript errors occurred.
  See `classical-hosted.json`, `classical-hosted.png`,
  `classical-mobile-hosted.png`, and `classical-score-hosted.png`.

## Art Song style restriction

- The composition dropdown and API share an allowlist containing only the 37
  presets whose genre field is exactly Art Song. Schubert remains the default.
- Checked the actual built Gradio dropdown configuration against that allowlist
  and confirmed a Keyboard-style API input is rejected before model generation.
- Hosted commit `c730f511cd389272a469cf8f427194af93897997` reached RUNNING.
  The live Gradio configuration exposes exactly 37 Art Song choices under
  “Art Song style,” with Schubert selected by default. See `art-song-hosted.json`.

## Favor motif recurrence

- All 12 candidates are still generated. Ranking now compares verified motif
  occurrence count first, distinct pitch-duration pairs second, then prefers
  the earlier candidate on complete ties. Progress and final status show the
  verified occurrence counts.
- Both-mode regressions confirm a less varied piece with more motif occurrences
  beats a more varied one with fewer. Equal-count candidates favor variety, then
  earlier order. All 86 backend tests pass.
- Browser fixtures now make candidate 7 contain more occurrences and candidate
  12 more note types. Both input modes select candidate 7 and correctly render,
  highlight, export, and play it. The full desktop/touch editor suite passed.
- Hosted commit `996055f1e287bd6915e4a13cf8bc7acdca37e182` reached RUNNING.
  A live batch generated all 12 candidates. Independently rescoring every
  candidate confirmed the winner had 4 motif occurrences and 12 note types;
  other eligible pieces had only 1–2 occurrences. The accepted ABC, downloads,
  audio, and highlighted score belonged to that winner. See
  `occurrence-ranking-hosted.json` for the complete comparison.

## Uncluttered result view

- Successful results have no status banner or score match-count legend. The
  motif notes still highlight; the result is the score, playback, and downloads.
- Candidate/count/seed/verification information and the exact prompt are under
  collapsed Generation details. Progress is just Composing/Preparing score;
  failure and conversion-warning messages remain visible.
- The full desktop/touch browser suite confirmed hidden success status, absent
  score legend, expandable stats/prompt, and preserved generation, highlights,
  audio, exports, barlines, keyboard editing, and cat behavior. Two focused
  export/highlight checks also passed.
- Hosted commit `468908397ecef6fb683584da5d580428688c3dcc` reached RUNNING.
  A real generated result had no success status or score legend, with diagnostics
  collapsed. All 18 highlighted SVG elements remained; candidate information and
  exact prompt were available when expanded. Mobile had no overflow and the page
  reported no JavaScript errors. See `clean-results-hosted.json` and screenshots.

## VexFlow and realization conditioning

- VexFlow 5.0.0 now engraves the editable staff, palette, notes, clef,
  accidentals, dots, ledger lines, and bar lines using bundled Bravura/Academico.
  Dragging, tapping, keyboard editing, undo, length limits and barline editing
  passed the desktop/touch browser suite. Font rendering and hitboxes were
  visually inspected; success panels remain uncluttered.
- Found and corrected a concrete prompt encoding mismatch: training patches
  `%motif:abc:` separately from its ABC realization, with training bar splits.
  Previously the entire line was patched as ordinary metadata. Compared actual
  token IDs and per-patch flags with the training Patchilizer for 12 cases
  (ordinary/inversion tags, bar lines, repeats, accidentals, and long motifs);
  every corrected case matches. See `realization-encoding-audit.json`.
- 89 unit tests pass. Tests inspect the additive masks received by attention
  layers, proving +4 is applied to realization patches and changes encoder
  output, and compare cached decoding against full-context decoding. Training's
  weight 3 only marks motif patches; it does not multiply the +4 bias.
- Live comparison of the same default concrete motif, Art Song style,
  temperature 1.2, 8-measure target and seeds 0–11: **0/12 eligible before,
  7/12 after**. The winner has two exact motif occurrences (candidate 11),
  ahead of more varied single-occurrence candidates. Independently verified
  all candidates, the winning ABC, available ABC/XML/audio exports and score
  highlighting. See `encoding-before.json` and `encoding-after.json`. This
  is a single controlled batch, not a general success-rate claim.
- Deployed commit `983cc6994384c9d7b925f331c455eb8141f6bb5f` reached RUNNING.
