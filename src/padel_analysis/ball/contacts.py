"""Turning a ball path into the instants where something hit the ball.

The stage before answers on every frame, so there are no gaps to read a contact
from - the criterion has to be the shape of the path, not its holes.

What marks a contact is a change of direction. Measured in pixels it is not
comparable between a lob and a smash, so the turn is divided by the speed that
produced it: a 40 px deviation is a sharp bend at 5 px/frame and nothing at 30.
That ratio is what this module thresholds.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Contact:
    """One instant where the ball changed direction, and how."""

    frame: int
    incoming: Point
    outgoing: Point
    turn: float
    sharpness: float


def find_contacts(
    path: dict[int, Point | None],
    span: int = 2,
    sharpness: float = 0.5,
    floor: float = 25.0,
    ceiling: float = 300.0,
    suppression: int = 5,
    cuts: Sequence[int] = (),
) -> list[Contact]:
    """The frames where the path bends sharply enough to be a contact.

    Args:
        path: one position per frame, as `best_path` returns.
        span: frames each side used to measure velocity.
        sharpness: least turn-over-speed ratio to accept. 0.5 was swept.
        floor: least turn in pixels. Below it the bend is annotation noise rather
            than a contact, whatever the ratio says.
        ceiling: most turn in pixels. Above it the path jumped further than a ball
            can travel, so the bend is a tracking error and not a contact. Measured:
            the reconstructed path reaches 512 px at the ninety-fifth percentile of
            speed where the annotated ball reaches 186.
        suppression: least distance between two kept contacts. A real rally cannot
            place two contacts closer, so a burst of frames around one bend must
            yield one contact and not five.
        cuts: frames where the broadcast splices two clips together. The velocity
            window straddles a splice for `span` frames each side, so that whole
            window is refused - a splice is not a contact, and without this every
            one of them would produce one.
    """
    scored: list[Contact] = []
    for frame in sorted(path):
        if any(abs(frame - cut) <= span for cut in cuts):
            continue
        pair = velocities(path, frame, span)
        if pair is None:
            continue
        incoming, outgoing = pair
        ratio = sharpness_of(incoming, outgoing)
        if ratio < sharpness:
            continue
        bend = turn_of(incoming, outgoing)
        if bend < floor or bend > ceiling:
            continue
        scored.append(Contact(frame, incoming, outgoing, bend, ratio))

    kept: list[Contact] = []
    # The sharpest first, and at equal sharpness the widest: around a bounce both
    # flanks are as sharp as the vertex, only the amplitude separates them.
    for contact in sorted(scored, key=lambda c: (c.sharpness, c.turn), reverse=True):
        if any(abs(contact.frame - k.frame) <= suppression for k in kept):
            continue
        kept.append(contact)
    return sorted(kept, key=lambda c: c.frame)
