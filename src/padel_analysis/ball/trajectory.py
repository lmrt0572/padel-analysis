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
from itertools import pairwise

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

    @property
    def speed(self) -> float:
        """Median distance travelled between two frames, in pixels.

        A ball covers 14.4 px a frame in median. A limb that escaped its box, or a
        moving advertising board, covers far less - and it is that difference, not
        the length of the track, that tells them apart.
        """
        ordered = [self.positions[f] for f in sorted(self.positions)]
        steps = [
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in pairwise(ordered)
        ]
        if not steps:
            return 0.0
        steps.sort()
        middle = len(steps) // 2
        if len(steps) % 2:
            return steps[middle]
        return (steps[middle - 1] + steps[middle]) / 2


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
    start: Candidate | None = None,
    follow: Candidate | None = None,
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
        start, follow: the two candidates to seed on. Defaulting to the best-scored
            one would seed on the ball only half the time - it is the second of the
            list in median - so the caller is expected to try several.
    """
    if not candidates.get(first) or not candidates.get(second):
        return None

    start = candidates[first][0] if start is None else start
    follow = candidates[second][0] if follow is None else follow
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
    max_length: int = 60,
    min_speed: float = 6.0,
    seeds_per_frame: int = 3,
) -> list[Segment]:
    """Seed on every pair of consecutive frames and keep what physics allows.

    Two bounds separate a ball from everything else that moves smoothly, and both
    are measured on 814 annotated arcs rather than chosen.

    A ball touches something every fifteen frames in median, and never goes more
    than sixty-four without: a cap at sixty keeps 99.8 percent of real arcs and
    refuses the long, placid tracks that a limb or an advertising board produces.
    Ranking by length alone does the opposite of what is wanted here - the ball's
    arcs are the short ones.

    A ball also travels 14.4 px a frame in median. A floor at six keeps 85 percent
    of real arcs and refuses what crawls.

    Two segments may still claim the same frame; the longer one wins, having
    survived more continuity constraints.

    Args:
        candidates: ranked candidates per frame.
        gate, max_step, max_misses: passed through to `grow`.
        min_length: a segment shorter than this is noise, not a trajectory.
        max_length: a segment longer than this is not a ball.
        min_speed: median pixels per frame a segment must cover.
        seeds_per_frame: how many of the best candidates each seed frame offers.
            The ball is the second of the list in median, so trying only the best
            one would start half the segments on something else.
    """
    frames = sorted(candidates)
    grown: list[Segment] = []
    for frame in frames:
        if frame + 1 not in candidates:
            continue
        for start in candidates[frame][:seeds_per_frame]:
            for follow in candidates[frame + 1][:seeds_per_frame]:
                segment = grow(
                    candidates, frame, frame + 1, gate, max_step, max_misses,
                    start=start, follow=follow,
                )
                if segment is None:
                    continue
                if not min_length <= segment.length <= max_length:
                    continue
                if segment.speed < min_speed:
                    continue
                grown.append(segment)

    grown.sort(key=lambda s: s.speed, reverse=True)
    taken: set[int] = set()
    kept: list[Segment] = []
    for segment in grown:
        if taken.isdisjoint(segment.positions):
            kept.append(segment)
            taken.update(segment.positions)

    kept.sort(key=lambda s: s.start)
    return kept


def positions_of(
    segments: Sequence[Segment], start: int, stop: int
) -> dict[int, Point | None]:
    """One position per frame of [start, stop], None where no segment covers it.

    This is the shape `ball_score` expects: a frame the trajectory never reached is
    a miss, not a wrong answer.
    """
    found: dict[int, Point] = {}
    for segment in segments:
        found.update(segment.positions)
    return {frame: found.get(frame) for frame in range(start, stop + 1)}
