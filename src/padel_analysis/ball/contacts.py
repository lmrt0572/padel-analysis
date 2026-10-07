"""Turning a ball path into the instants where something hit the ball.

A contact is a change of direction. The turn is divided by the speed that produced
it, so one threshold serves a lob and a smash.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

Point = tuple[float, float]


def velocities(
    path: dict[int, Point | None], frame: int, span: int
) -> tuple[Point, Point] | None:
    """Return the mean velocity entering and leaving `frame`, or None if either is unknown.

    Args:
        path: one position per frame, or None where the ball is not held.
        frame: the frame to look at.
        span: how many frames on each side the velocity is measured over.
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
    """Return how far the velocity changed, in pixels."""
    return math.hypot(outgoing[0] - incoming[0], outgoing[1] - incoming[1])


def sharpness_of(incoming: Point, outgoing: Point) -> float:
    """Return the turn divided by the speed that produced it.

    A full reversal at constant speed scores 1.0 whatever that speed is.
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
    """Return the frames where the path bends sharply enough to be a contact.

    Args:
        path: one position per frame, as `best_path` returns.
        span: frames each side used to measure velocity.
        sharpness: least turn-over-speed ratio to accept.
        floor: least turn in pixels; below it the bend is noise.
        ceiling: most turn in pixels; above it the path jumped and the bend is a
            tracking error.
        suppression: least distance between two kept contacts.
        cuts: frames where the broadcast splices two clips; the velocity window
            around each is refused.
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
    # sharpest first, then widest: both flanks of a bounce are as sharp as its vertex
    for contact in sorted(scored, key=lambda c: (c.sharpness, c.turn), reverse=True):
        if any(abs(contact.frame - k.frame) <= suppression for k in kept):
            continue
        kept.append(contact)
    return sorted(kept, key=lambda c: c.frame)
