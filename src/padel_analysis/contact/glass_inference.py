"""Wall contacts the picture does not show, inferred from how long the ball took.

A ball struck from one half bounces in the other, and the player there strikes it
back. It reached the bounce at a speed the strike and the bounce give, and a bounce
keeps well over half of it. When the player then stood so close to the bounce that the
ball, at that pace, would have reached them several times over, it went somewhere
first: to the wall along its heading. That wall is often invisible from a broadcast
camera, the rebound moving the ball by a few pixels.

Where the contact happened is left to the network: the frame, between the bounce and
the strike, where it found a wall most likely, even below its own threshold. When that
frame is the bounce itself, the bounce was the wall.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..geometry.court import Court

FPS = 30.0
SLOWEST = 0.35
"""The pace at which the ball would have reached the player straight from the bounce,
as a share of its speed on arrival, below which it cannot have gone straight. Measured
over 151 bounces followed by a strike on the same side: straight shots sit from about
0.4 to 0.9, detours below 0.35."""
DETOUR = 0.9
"""The same share for the path by the wall, above which that path is too long to have
been flown in the time: the ball did not go by the wall either."""
HORIZON = 90
"""Strike, bounce and strike more than three seconds apart are not one exchange."""
COURT = Court()


@dataclass(frozen=True)
class Touch:
    """A contact as the inference needs it: when, what, which half, and where.

    `place` is the bounce on the floor for a bounce, the striker's feet for a strike.
    """

    frame: int
    kind: str
    side: str | None
    place: tuple[float, float] | None


@dataclass(frozen=True)
class InferredWall:
    """A wall contact to add at `frame`, or, when `replaces_bounce`, the bounce at
    `bounce` to read as a wall."""

    frame: int
    bounce: int
    replaces_bounce: bool


def wall_ahead(point: tuple[float, float], heading: tuple[float, float],
               court: Court = COURT) -> tuple[float, tuple[float, float]]:
    """How far the first wall is from `point` along the unit `heading`, and where."""
    reach = math.inf
    if abs(heading[0]) > 1e-9:
        wall = court.half_width if heading[0] > 0 else -court.half_width
        reach = min(reach, (wall - point[0]) / heading[0])
    if abs(heading[1]) > 1e-9:
        wall = court.half_length if heading[1] > 0 else -court.half_length
        reach = min(reach, (wall - point[1]) / heading[1])
    reach = max(reach, 0.0)
    return reach, (point[0] + reach * heading[0], point[1] + reach * heading[1])


def inferred_walls(touches: Sequence[Touch], wall_probability: np.ndarray,
                   start: int) -> list[InferredWall]:
    """The wall contacts implied by the pace of each bounce followed by a strike.

    Args:
        touches: the contacts found, in time order.
        wall_probability: per frame from `start`, the network's probability of a wall.
    """
    found = []
    for index, bounce in enumerate(touches):
        if bounce.kind != "sol" or bounce.place is None:
            continue
        strike = _strike_before(touches, index)
        answer = _next_strike_or_bounce(touches, index)
        if strike is None or answer is None or answer.kind != "raquette":
            continue
        if strike.side == bounce.side or answer.side != bounce.side:
            continue
        between = [t for t in touches if bounce.frame < t.frame < answer.frame]
        if any(t.kind in ("verre", "grillage") for t in between):
            continue
        if answer.frame - bounce.frame > HORIZON:
            continue
        travelled = math.dist(strike.place, bounce.place)
        if travelled < 0.5:
            continue
        speed = travelled / ((bounce.frame - strike.frame) / FPS)
        heading = ((bounce.place[0] - strike.place[0]) / travelled,
                   (bounce.place[1] - strike.place[1]) / travelled)
        reach, wall = wall_ahead(bounce.place, heading)
        flown = speed * (answer.frame - bounce.frame) / FPS
        straight = math.dist(bounce.place, answer.place) / flown
        by_wall = (reach + math.dist(wall, answer.place)) / flown
        if straight >= SLOWEST or by_wall >= DETOUR:
            continue
        low = max(bounce.frame - 3 - start, 0)
        high = min(answer.frame - 3 - start, len(wall_probability))
        if high <= low:
            continue
        frame = start + low + int(np.argmax(wall_probability[low:high]))
        found.append(InferredWall(frame, bounce.frame, abs(frame - bounce.frame) <= 3))
    return found


def _strike_before(touches: Sequence[Touch], index: int) -> Touch | None:
    bounce = touches[index]
    for touch in reversed(touches[:index]):
        if touch.kind == "raquette":
            if touch.place is None or not 5 <= bounce.frame - touch.frame <= HORIZON:
                return None
            return touch
    return None


def _next_strike_or_bounce(touches: Sequence[Touch], index: int) -> Touch | None:
    for touch in touches[index + 1:]:
        if touch.kind in ("raquette", "sol"):
            return touch if touch.place is not None else None
    return None
