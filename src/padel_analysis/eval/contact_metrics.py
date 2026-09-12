"""What can honestly be said about detected contacts, and what cannot.

The shot annotation marks racket hits only, as intervals rather than instants, and
those intervals cover 48.2 percent of the annotated frames on the women's evaluation
slice. A detector drawing its instants at random scores 0.480 precision there; the
real one scores 0.492. Precision against this annotation therefore carries almost no
information, and reporting it as a performance would be a lie by omission.

Two things replace it. `chance_precision` puts the random control beside every figure
so a reader sees what the number is worth. `bounces_between` counts detected contacts
between two annotated shots and is checked against the physics of the game: a padel
ball bounces nought, once or twice between two racket hits, so a detector claiming
five is over-firing whatever its recall says.
"""

import itertools
import math
import random
from collections.abc import Sequence

from .shots import ShotEvent


def _covered_frames(events: Sequence[ShotEvent]) -> set[int]:
    return {f for e in events for f in range(e.start_frame, e.end_frame + 1)}


def interval_coverage(frames: Sequence[int], events: Sequence[ShotEvent]) -> float:
    """Share of `frames` that sit inside an annotated shot.

    This is the floor any precision figure must be read against: a detector firing
    at random reaches it without knowing anything.
    """
    if not frames:
        return math.nan
    covered = _covered_frames(events)
    return sum(1 for f in frames if f in covered) / len(frames)


def bounces_between(
    contacts: Sequence[int], events: Sequence[ShotEvent], max_gap: int = 120
) -> dict[int, int]:
    """How many contacts fall between two consecutive shots, tallied.

    Args:
        contacts: detected contact frames.
        events: the annotated shots, in order.
        max_gap: longest gap still counted as one rally. Beyond it the point ended
            and the ball was carried back by hand, which is not play.

    Returns:
        A count of pairs by number of contacts between them - `{1: 33, 2: 19}` reads
        as thirty-three exchanges with one bounce and nineteen with two.
    """
    tally: dict[int, int] = {}
    for first, second in itertools.pairwise(events):
        gap = second.start_frame - first.end_frame
        if gap <= 0 or gap > max_gap:
            continue
        between = sum(1 for c in contacts if first.end_frame < c < second.start_frame)
        tally[between] = tally.get(between, 0) + 1
    return tally


def chance_precision(
    frames: Sequence[int],
    events: Sequence[ShotEvent],
    count: int,
    draws: int = 200,
    seed: int = 0,
) -> float:
    """Precision a detector would reach by drawing `count` frames at random.

    Reported beside the real precision. Where the two agree, the detector has shown
    nothing, however good the number looks on its own.
    """
    if count <= 0 or count > len(frames):
        return math.nan
    covered = _covered_frames(events)
    generator = random.Random(seed)
    pool = list(frames)
    total = 0.0
    for _ in range(draws):
        picks = generator.sample(pool, count)
        total += sum(1 for p in picks if p in covered) / count
    return total / draws
