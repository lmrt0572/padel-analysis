"""What can be said about detected contacts against the shot annotation.

The annotation marks racket hits as intervals covering about half the frames, so
precision against it is reported beside what a random draw would score. The count
of bounces between two shots is checked against the game instead.
"""

import itertools
import math
import random
from collections.abc import Sequence

from .shots import ShotEvent


def _covered_frames(events: Sequence[ShotEvent]) -> set[int]:
    return {f for e in events for f in range(e.start_frame, e.end_frame + 1)}


def interval_coverage(frames: Sequence[int], events: Sequence[ShotEvent]) -> float:
    """Return the share of `frames` that sit inside an annotated shot."""
    if not frames:
        return math.nan
    covered = _covered_frames(events)
    return sum(1 for f in frames if f in covered) / len(frames)


def bounces_between(
    contacts: Sequence[int], events: Sequence[ShotEvent], max_gap: int = 120
) -> dict[int, int]:
    """Return how many contacts fall between two consecutive shots, tallied.

    Args:
        contacts: detected contact frames.
        events: the annotated shots, in order.
        max_gap: longest gap still counted as one rally.

    Returns:
        A count of pairs by number of contacts between them.
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
    """Return the precision a detector would reach by drawing `count` frames at random."""
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
