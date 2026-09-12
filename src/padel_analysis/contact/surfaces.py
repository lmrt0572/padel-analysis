"""Deciding what a ball bounced off, at the instant it did.

Two questions in order. Was it a racket - which a wrist nearby answers, the ball
sitting 50 px from one in median when a shot is annotated, against 168 px otherwise.
Otherwise, which of the five surfaces of the court - answered by casting the ray
through the ball and keeping the intersections that are physically possible. For a
wall, glass or mesh then follows from the impact height, with no heuristic.

The middle step is what a ground homography could not do. It projects onto the floor,
so it misplaces anything above it: measured on real contacts, a third of them land
outside the court rectangle, some as far as 23 m down a 20 m court. A ray and a plane
do not have that problem, because at a contact - and only then - the ball is known to
be on a surface.
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
    """How many surfaces the ray could admissibly have met. More than one means the
    geometry alone did not settle it, which is reported rather than hidden - it is
    the quantity the annotation campaign exists to check."""


def classify(
    ball: Point,
    wrists: list[Point],
    pose: CameraPose,
    surfaces: list[Surface],
    wrist_distance: float = 80.0,
    margin: float = 0.30,
) -> Verdict:
    """What the ball hit at this contact.

    Args:
        ball: the ball's pixel on the contact frame.
        wrists: every visible wrist pixel on that frame, all players together.
        pose: the camera pose for this video.
        surfaces: the court's surfaces, floor first - the order breaks ties.
        wrist_distance: how close a wrist must be for the contact to be a racket.
        margin: slack in metres on each surface's extent, absorbing the pose error -
            8.6 px in median on the elevated control points, which is 13 cm near the
            camera and 56 cm at the far baseline. Metres and never pixels: the two
            are not comparable across a 4.3x depth asymmetry.
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
    return Verdict(
        surface=surface.name,
        point=meeting,
        material=surface.material_at(meeting),
        candidates=len(admissible),
    )
