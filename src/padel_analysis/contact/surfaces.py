"""Deciding what a ball bounced off, at the instant it did.

A racket if a wrist is nearby. Otherwise the ray through the ball is intersected
with the surfaces of the court and the physically possible ones are kept; glass or
mesh follows from the impact height.
"""

import math
from dataclasses import dataclass

import numpy as np

from ..geometry.camera import CameraPose, Surface

RACKET = "raquette"
Point = tuple[float, float]


@dataclass(frozen=True)
class Verdict:
    """What a contact was, and how much the geometry had to choose."""

    surface: str | None
    point: np.ndarray | None
    material: str | None
    candidates: int
    """How many surfaces the ray could admissibly have met; more than one means the
    geometry alone did not settle it."""


def classify(
    ball: Point,
    wrists: list[Point],
    pose: CameraPose,
    surfaces: list[Surface],
    wrist_distance: float = 80.0,
    margin: float = 0.30,
    depth_cut: float | None = -7.5,
) -> Verdict:
    """Return what the ball hit at this contact.

    Args:
        ball: the ball's pixel on the contact frame.
        wrists: every visible wrist pixel on that frame.
        pose: the camera pose for this video.
        surfaces: the court's surfaces, floor first; the order breaks ties.
        wrist_distance: how close a wrist must be for the contact to be a racket.
        margin: slack in metres on each surface's extent, absorbing the pose error.
        depth_cut: court `y` beyond which a floor candidate is more likely a low
            contact on the near glass, so another admissible surface is preferred.
    """
    if wrists:
        nearest = min(math.dist(ball, wrist) for wrist in wrists)
        if nearest <= wrist_distance:
            return Verdict(RACKET, None, None, candidates=0)

    origin, direction = pose.ray(ball)
    admissible = [
        (surface, meeting)
        for surface in surfaces
        if (meeting := surface.intersect(origin, direction)) is not None
        and surface.contains(meeting, margin)
    ]
    if not admissible:
        return Verdict(None, None, None, candidates=0)

    surface, meeting = admissible[0]
    if (
        depth_cut is not None
        and surface.name == "floor"
        and len(admissible) > 1
        and meeting[1] <= depth_cut
    ):
        surface, meeting = admissible[1]

    return Verdict(
        surface=surface.name,
        point=meeting,
        material=surface.material_at(meeting),
        candidates=len(admissible),
    )
