"""Drawing the ball, its recent trail, and what each contact hit.

The ball is drawn from the path chosen over the whole sequence, never from a
per-frame detection: that path is what the measurements describe, so the video shows
the same thing the numbers do.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..contact.surfaces import RACKET, Verdict

if TYPE_CHECKING:
    from .court_zones import Zone

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
    """One contact: its label gives the ring colour, its zone lights the minimap."""

    frame: int
    label: str
    pixel: Point
    zone: "Zone | None" = None
    box: np.ndarray | None = None  # la bbox du frappeur, pour une raquette
    point: np.ndarray | None = None  # sur la surface touchee, en metres du court


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
    """A copy of `frame` with the trail and the ball.

    Contacts are not marked on the ball: a surface contact lights its zone of the
    court, and a racket contact lights the player who struck.
    """
    canvas = frame.copy()
    for older, newer in itertools.pairwise(points):
        cv2.line(canvas, _pixel(older), _pixel(newer), TRAIL, 2)
    if points:
        cv2.circle(canvas, _pixel(points[-1]), 7, BALL, 2)
    return canvas


def hitter_box(ball: Point, people: Sequence) -> np.ndarray | None:
    """The box of the player whose visible wrist is nearest the ball, if any has one."""
    best, nearest = None, float("inf")
    for person in people:
        for wrist in (person.keypoints[9], person.keypoints[10]):
            if wrist[2] <= 0.3:
                continue
            distance = float(np.hypot(wrist[0] - ball[0], wrist[1] - ball[1]))
            if distance < nearest:
                best, nearest = person.bbox, distance
    return best


def following_box(box: np.ndarray, people: Sequence, jump: float = 120.0) -> np.ndarray:
    """The same player's box on a later frame, found by the nearest centre.

    A strike lasts a few frames and the striker keeps moving. Lighting the box of the
    contact frame would leave a bright rectangle behind them; this follows them. `jump`
    caps how far the box may move, so a missed detection falls back on the last known
    box rather than jumping to somebody else.
    """
    centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    best, nearest = box, jump
    for person in people:
        other = ((person.bbox[0] + person.bbox[2]) / 2, (person.bbox[1] + person.bbox[3]) / 2)
        distance = float(np.hypot(other[0] - centre[0], other[1] - centre[1]))
        if distance < nearest:
            best, nearest = person.bbox, distance
    return best


def draw_hitter(
    frame: np.ndarray, box: np.ndarray, strength: float, colour=(255, 255, 255)
) -> np.ndarray:
    """A copy of `frame` with the striking player's box lit, fading with `strength`."""
    x1, y1, x2, y2 = (round(float(v)) for v in box)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    alpha = 0.35 * max(0.0, min(1.0, strength))
    lit = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)
    cv2.rectangle(lit, (x1, y1), (x2, y2), colour, 2 + round(4 * strength))
    return lit


def _pixel(point: Point) -> tuple[int, int]:
    return round(point[0]), round(point[1])
