"""Drawing the ball, its recent trail, and what each contact hit.

The ball is drawn from the path chosen over the whole sequence, never from a
per-frame detection: that path is what the measurements describe, so the video shows
the same thing the numbers do.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from ..contact.surfaces import RACKET, Verdict

Point = tuple[float, float]

LABEL_COLOURS: dict[str, tuple[int, int, int]] = {
    "SOL": (80, 200, 80),
    "VITRE": (255, 200, 60),
    "GRILLAGE": (160, 160, 160),
    "FILET": (80, 220, 255),
    "RAQUETTE": (60, 60, 240),
}
BALL = (0, 255, 255)
TRAIL = (0, 190, 255)


@dataclass(frozen=True)
class ContactEvent:
    """One contact, labelled, with where to draw it on the frame and on the minimap."""

    frame: int
    label: str
    pixel: Point
    court_xy: Point | None


def contact_label(verdict: Verdict) -> str | None:
    """What to write on screen for a verdict, or None when no surface was admissible.

    An undecided contact is left unlabelled rather than given a guess: the rule
    reports that it could not decide, and the video must not claim otherwise.
    """
    if verdict.surface is None:
        return None
    if verdict.surface == RACKET:
        return "RAQUETTE"
    if verdict.surface == "floor":
        return "SOL"
    if verdict.surface == "net":
        return "FILET"
    return "GRILLAGE" if verdict.material == "grillage" else "VITRE"


def trail(path: dict[int, Point | None], frame: int, length: int = 12) -> list[Point]:
    """The ball's last positions up to and including `frame`, oldest first."""
    return [
        point
        for f in range(frame - length + 1, frame + 1)
        if (point := path.get(f)) is not None
    ]


def visible_events(
    events: Sequence[ContactEvent], frame: int, hold: int = 20
) -> list[ContactEvent]:
    """Contacts that happened in the last `hold` frames, the only ones still labelled."""
    return [e for e in events if e.frame <= frame < e.frame + hold]


def draw_ball(
    frame: np.ndarray, points: Sequence[Point], events: Sequence[ContactEvent]
) -> np.ndarray:
    """A copy of `frame` with the trail, the ball, and a label at each recent contact."""
    canvas = frame.copy()
    for older, newer in itertools.pairwise(points):
        cv2.line(canvas, _pixel(older), _pixel(newer), TRAIL, 2)
    if points:
        cv2.circle(canvas, _pixel(points[-1]), 7, BALL, 2)

    for event in events:
        colour = LABEL_COLOURS.get(event.label, (255, 255, 255))
        x, y = _pixel(event.pixel)
        cv2.circle(canvas, (x, y), 20, colour, 3)
        position = (x + 26, max(30, y - 26))
        cv2.putText(canvas, event.label, position, cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 0), 5)
        cv2.putText(canvas, event.label, position, cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    colour, 2)
    return canvas


def _pixel(point: Point) -> tuple[int, int]:
    return round(point[0]), round(point[1])
