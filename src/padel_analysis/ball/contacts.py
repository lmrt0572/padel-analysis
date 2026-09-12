"""Turning a ball path into the instants where something hit the ball.

The stage before answers on every frame, so there are no gaps to read a contact
from - the criterion has to be the shape of the path, not its holes.

What marks a contact is a change of direction. Measured in pixels it is not
comparable between a lob and a smash, so the turn is divided by the speed that
produced it: a 40 px deviation is a sharp bend at 5 px/frame and nothing at 30.
That ratio is what this module thresholds.
"""

import math

Point = tuple[float, float]


def velocities(
    path: dict[int, Point | None], frame: int, span: int
) -> tuple[Point, Point] | None:
    """Mean velocity entering and leaving `frame`, or None if either is unknown.

    Args:
        path: one position per frame, or None where the ball is not held.
        frame: the frame to look at.
        span: how many frames on each side the velocity is measured over. Two was
            measured best: one frame is inside the annotation noise of a 10 px ball,
            and four smooths the bend away.
    """
    before = path.get(frame - span)
    here = path.get(frame)
    after = path.get(frame + span)
    if before is None or here is None or after is None:
        return None
    incoming = (here[0] - before[0], here[1] - before[1])
    outgoing = (after[0] - here[0], after[1] - here[1])
    return incoming, outgoing


def turn_of(incoming: Point, outgoing: Point) -> float:
    """How far the velocity changed, in pixels."""
    return math.hypot(outgoing[0] - incoming[0], outgoing[1] - incoming[1])


def sharpness_of(incoming: Point, outgoing: Point) -> float:
    """The same turn divided by the speed that produced it.

    Scale-free, so one threshold serves a slow ball and a fast one. A full reversal
    at constant speed scores 1.0 whatever that speed is.
    """
    speed = math.hypot(*incoming) + math.hypot(*outgoing)
    if speed <= 0.0:
        return 0.0
    return turn_of(incoming, outgoing) / speed
