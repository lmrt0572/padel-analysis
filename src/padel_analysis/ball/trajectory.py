"""Following the ball across frames, rather than judging each frame alone.

The detection stage returns a ranked list and refuses to decide: the ball is the
sixth candidate of seventy-eight, or the second of nineteen once the moving limbs
are demoted. Picking it out of that list is what this module does, and it does it
with the only thing a still frame cannot offer - continuity.

A segment grows from two consecutive positions. At each step the expected position
is the one a constant velocity would give, and the nearest candidate is taken if it
falls inside a gate. The gate is measured, not chosen: over free play the distance
between the real position and that prediction has a median of 5.4 px and a ninetieth
percentile of 28.5 px, so thirty pixels covers ninety-one percent of it.

Nothing here assumes a smooth trajectory over a whole rally. A padel ball bounces
off the floor, the glass and the mesh, and a global smoother would impose a
continuity the data does not have. Segments simply stop where the ball stops
behaving, which is also where the contacts are.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .candidates import Candidate

Point = tuple[float, float]


@dataclass(frozen=True)
class Segment:
    """One stretch over which the ball was followed without losing it."""

    positions: dict[int, Point]
    confirmed: set[int] = field(default_factory=set)

    @property
    def start(self) -> int:
        return min(self.positions)

    @property
    def stop(self) -> int:
        return max(self.positions)

    @property
    def length(self) -> int:
        return self.stop - self.start + 1


def _nearest(
    candidates: Sequence[Candidate], target: Point, gate: float
) -> Candidate | None:
    best, best_distance = None, gate
    for candidate in candidates:
        distance = math.hypot(candidate.x - target[0], candidate.y - target[1])
        if distance <= best_distance:
            best, best_distance = candidate, distance
    return best


def _extend(
    candidates: dict[int, Sequence[Candidate]],
    positions: dict[int, Point],
    confirmed: set[int],
    direction: int,
    gate: float,
    max_misses: int,
) -> None:
    """Walk one way from the seed, in place, until the ball stops behaving."""
    frame = max(positions) if direction > 0 else min(positions)
    misses = 0
    while misses <= max_misses:
        following = frame + direction
        here = positions[frame]
        behind = positions.get(frame - direction)
        target = (
            here
            if behind is None
            else (2 * here[0] - behind[0], 2 * here[1] - behind[1])
        )

        found = _nearest(candidates.get(following, []), target, gate)
        if found is None:
            positions[following] = target
            misses += 1
        else:
            positions[following] = (found.x, found.y)
            confirmed.add(following)
            misses = 0
        frame = following

    for extra in [f for f in positions if f not in confirmed]:
        beyond = extra > max(confirmed) if direction > 0 else extra < min(confirmed)
        if beyond:
            del positions[extra]


def grow(
    candidates: dict[int, Sequence[Candidate]],
    first: int,
    second: int,
    gate: float = 30.0,
    max_step: float = 60.0,
    max_misses: int = 2,
) -> Segment | None:
    """Grow a segment both ways from two consecutive frames.

    Args:
        candidates: ranked candidates per frame.
        first, second: the two consecutive frames the seed is taken from.
        gate: how far the real position may sit from the constant-velocity
            prediction. Thirty pixels covers ninety-one percent of free play.
        max_step: how far the ball may travel between the two seed frames. The
            ninety-fifth percentile of the real displacement is 52.7 px.
        max_misses: how many frames in a row may go unconfirmed before the segment
            ends.
    """
    if not candidates.get(first) or not candidates.get(second):
        return None

    start, follow = candidates[first][0], candidates[second][0]
    if math.hypot(follow.x - start.x, follow.y - start.y) > max_step:
        return None

    positions: dict[int, Point] = {
        first: (start.x, start.y),
        second: (follow.x, follow.y),
    }
    confirmed = {first, second}

    _extend(candidates, positions, confirmed, +1, gate, max_misses)
    _extend(candidates, positions, confirmed, -1, gate, max_misses)
    return Segment(positions=positions, confirmed=confirmed)


def build_segments(
    candidates: dict[int, Sequence[Candidate]],
    gate: float = 30.0,
    max_step: float = 60.0,
    max_misses: int = 2,
    min_length: int = 5,
) -> list[Segment]:
    """Seed on every pair of consecutive frames and keep what survives.

    Two segments may claim the same frame. The longer one wins: it survived more
    continuity constraints, so it is the better explanation of what was seen.

    Args:
        candidates: ranked candidates per frame.
        gate, max_step, max_misses: passed through to `grow`.
        min_length: a segment shorter than this is noise, not a trajectory.
    """
    frames = sorted(candidates)
    grown: list[Segment] = []
    for frame in frames:
        if frame + 1 not in candidates:
            continue
        segment = grow(candidates, frame, frame + 1, gate, max_step, max_misses)
        if segment is not None and segment.length >= min_length:
            grown.append(segment)

    grown.sort(key=lambda s: s.length, reverse=True)
    taken: set[int] = set()
    kept: list[Segment] = []
    for segment in grown:
        if taken.isdisjoint(segment.positions):
            kept.append(segment)
            taken.update(segment.positions)

    kept.sort(key=lambda s: s.start)
    return kept
