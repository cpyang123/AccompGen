"""Generate every candidate, then rank only motif-containing compositions."""
from dataclasses import dataclass, replace
import time

from outputs import clean_abc
from verification import MotifMatch, analyze_score, find_motif

MAX_ATTEMPTS = 12
MAX_SECONDS = 105


@dataclass
class VerifiedUpdate:
    text: str
    attempt: int
    seed: int
    stage: str
    message: str
    match: MotifMatch | None = None
    note_types: int = 0
    evaluated: int = 0
    eligible: int = 0


def generate_verified(generate_attempt, prompt, temperature, seed, bars, mode, pattern, notes, tempo,
                      *, rhythm=None, max_attempts=MAX_ATTEMPTS, max_seconds=MAX_SECONDS):
    started = time.monotonic()
    best = None
    eligible = 0
    active_seed = seed
    for attempt in range(1, max_attempts + 1):
        remaining = max_seconds - (time.monotonic() - started)
        if remaining <= 1:
            # Never label a partial search as the best of all twelve candidates.
            raise ValueError(f'Time limit reached after {attempt - 1}/{max_attempts} candidates. '
                             'No winner was selected. Try fewer measures and generate again.')
        active_seed = (seed + attempt - 1) % (2**32)
        yield VerifiedUpdate('', attempt, active_seed, 'generating',
                             f'Candidate {attempt}/{max_attempts} · composing…')
        final = None
        # Share the remaining GPU budget among every remaining candidate,
        # reserving time for CPU parsing and verification between generations.
        budget = max(.1, remaining / (max_attempts - attempt + 1) - .25)
        for update in generate_attempt(prompt=prompt, temperature=temperature, seed=active_seed,
                                       max_bars=bars, max_seconds=min(100, budget)):
            final = update
            yield VerifiedUpdate(update.text, attempt, active_seed, 'generating',
                                 f'Candidate {attempt}/{max_attempts} · composing · {update.elapsed:.1f}s')
        yield VerifiedUpdate(final.text if final else '', attempt, active_seed, 'checking',
                             f'Candidate {attempt}/{max_attempts} · checking motif and note variety…')
        try:
            if final is None:
                raise ValueError('The model returned no music.')
            abc = clean_abc(final.text, tempo)
            events, note_types = analyze_score(abc)
            match = find_motif(abc, mode, pattern, notes, events=events, rhythm=rhythm)
        except ValueError:
            match, note_types = None, 0
        if match and match.count:
            eligible += 1
            candidate = VerifiedUpdate(final.text, attempt, active_seed, 'candidate',
                                       f'Candidate {attempt}/{max_attempts} · {match.count} motif matches · {note_types} note types',
                                       match, note_types, attempt, eligible)
            # Prefer motif recurrence, then variety; keep the earlier candidate
            # when both measures tie. Counts come from the score, not metadata.
            if best is None or (candidate.match.count, candidate.note_types) > (best.match.count, best.note_types):
                best = candidate
            candidate.message += f' · best: {best.match.count} matches / {best.note_types} types'
            yield candidate
        else:
            yield VerifiedUpdate(final.text if final else '', attempt, active_seed, 'rejected',
                                 f'Candidate {attempt}/{max_attempts} · no motif match',
                                 evaluated=attempt, eligible=eligible)
    if best is not None:
        message = (f'Selected candidate {best.attempt}/{max_attempts} · {best.match.count} motif match{"es" if best.match.count != 1 else ""}'
                   f' · {best.note_types} note types')
        yield replace(best, stage='verified', evaluated=max_attempts, eligible=eligible, message=message)
    else:
        yield VerifiedUpdate('', max_attempts, active_seed, 'exhausted',
                             f'No verified match in {max_attempts} candidates. No composition was accepted. '
                             'Try more measures or a different style, then Generate again.',
                             evaluated=max_attempts)
