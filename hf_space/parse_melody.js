// Use the same ABC parser as the displayed score, including keys and tuplets.
const fs = require('fs');
const ABCJS = require('./assets/abcjs-basic-min.js');

try {
  const abc = fs.readFileSync(0, 'utf8');
  const tune = ABCJS.parseOnly(abc)[0];
  if (!tune) throw new Error('No score found');
  tune.setUpAudio({});
  const events = [];
  let bar = 1;
  let notesInBar = false;
  // The checkpoint conditions V:1: inspect the first melodic voice, never
  // concatenate simultaneous voices into a fictitious melodic sequence.
  for (const line of tune.lines) {
    const voice = line.staff && line.staff[0] && line.staff[0].voices[0];
    for (const element of voice || []) {
      if (element.el_type === 'bar') {
        if (notesInBar) bar++;
        notesInBar = false;
        continue;
      }
      if (element.el_type !== 'note') continue;
      notesInBar = true;
      if (element.rest) {
        events.push({rest: true});
        continue;
      }
      const pitches = element.pitches || [];
      if (!pitches.length) continue;
      // A chord contributes its upper melodic note, not sequential chord tones.
      const index = pitches.reduce((best, p, i) => p.pitch > pitches[best].pitch ? i : best, 0);
      const written = pitches[index];
      const previous = events[events.length - 1];
      if (written.endTie && previous && !previous.rest && previous.tieOut && previous.diatonic === written.pitch) {
        previous.spans.push([element.startChar, element.endChar]);
        previous.tieOut = Boolean(written.startTie);
        continue; // MIDI duration on the first note already includes the tie.
      }
      const sounding = (element.midiPitches || [])[index];
      if (!sounding || !Number.isFinite(sounding.pitch)) {
        events.push({barrier: true}); // Never match across an unknown chord tone.
        continue;
      }
      events.push({
        midi: sounding.pitch, diatonic: written.pitch,
        beats: sounding.duration * 4, bar,
        spans: [[element.startChar, element.endChar]],
        tieOut: Boolean(written.startTie),
      });
    }
  }
  if (process.argv.includes('--analysis')) {
    // Rank written sounding notes across all staves/voices, including chord
    // members. Labels and motif comments never create artificial note types.
    const notes = [];
    for (const line of tune.lines) for (const staff of line.staff || []) {
      for (const voice of staff.voices || []) for (const element of voice) {
        if (element.el_type !== 'note' || element.rest) continue;
        for (const note of element.midiPitches || []) {
          if (Number.isFinite(note.pitch) && Number.isFinite(note.duration) && note.duration > 0)
            notes.push([note.pitch, note.duration * 4]);
        }
      }
    }
    process.stdout.write(JSON.stringify({events, notes}));
  } else process.stdout.write(JSON.stringify(events));
} catch (error) {
  process.stderr.write(String(error.message || error));
  process.exitCode = 1;
}
