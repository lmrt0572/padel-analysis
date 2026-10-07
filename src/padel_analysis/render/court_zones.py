"""The parts of the court a contact lights up, drawn on the court itself.

A contact position is known to within 13 cm near the camera and 56 cm at the far
baseline. Drawn as a point, an error of that size is plain to see; lighting the zone
it falls in tolerates it - and in padel what matters is which wall panel or which part
of the floor was hit, not the centimetre.

Each zone carries its corners in three dimensions, so the camera pose can draw it in
perspective on the video: a glass panel stands up from the ground, it does not lie on
it. Its two-dimensional footprint is kept for the minimap.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ..contact.surfaces import RACKET, Verdict
from ..geometry.camera import CameraPose
from ..geometry.court import Court

Point = tuple[float, float]
Corner = tuple[float, float, float]
Corner2 = tuple[float, float]

FLOOR = (80, 200, 80)
GLASS = (255, 200, 60)
MESH = (170, 170, 170)
NET = (80, 220, 255)
ZONE_OPACITY = 0.55
NEAR_WALL_OPACITY = 0.25
PATCH_SIZE = 1.5


@dataclass(frozen=True)
class Zone:
    """A named part of the court: its corners in metres, and its footprint seen from above."""

    name: str
    kind: str  # "area" on the floor, "line" for a wall seen from above
    points: tuple[Point, ...]
    colour: tuple[int, int, int]
    corners: tuple[Corner, ...] = ()
    opacity: float = ZONE_OPACITY


def zone_of(verdict: Verdict, court: Court) -> Zone | None:
    """The zone a contact lights, or None for a racket or an undecided contact."""
    if verdict.surface in (None, RACKET) or verdict.point is None:
        return None
    x, y = float(verdict.point[0]), float(verdict.point[1])
    w, length, s = court.half_width, court.half_length, court.service_line_distance
    half = "proche" if y < 0 else "eloigne"
    sign = -1.0 if y < 0 else 1.0
    l_signed, s_signed = sign * length, sign * s

    if verdict.surface == "floor":
        if abs(y) <= s:
            side = "droite" if x >= 0 else "gauche"
            edge = w if x >= 0 else -w
            footprint = ((0.0, 0.0), (edge, 0.0), (edge, s_signed), (0.0, s_signed))
            return Zone(f"sol_{half}_service_{side}", "area", footprint, FLOOR,
                        tuple((px, py, 0.0) for px, py in footprint))
        footprint = ((-w, s_signed), (w, s_signed), (w, l_signed), (-w, l_signed))
        return Zone(f"sol_{half}_fond", "area", footprint, FLOOR,
                    tuple((px, py, 0.0) for px, py in footprint))

    if verdict.surface == "net":
        top = court.net_height_posts
        return Zone("filet", "line", ((-w, 0.0), (w, 0.0)), NET,
                    ((-w, 0.0, 0.0), (w, 0.0, 0.0), (w, 0.0, top), (-w, 0.0, top)))

    material = verdict.material or "verre"
    colour = MESH if material == "grillage" else GLASS
    if verdict.surface.startswith("back_wall"):
        low, high = (
            (court.back_wall_glass_height, court.back_wall_total_height)
            if material == "grillage"
            else (0.0, court.back_wall_glass_height)
        )
        # The camera is just behind the near back wall: the whole wall, projected,
        # covers the entire lower half of the picture. It lights up in full, but
        # lightly, so that the players stay readable through it.
        opacity = NEAR_WALL_OPACITY if sign < 0 else ZONE_OPACITY
        return Zone(f"fond_{half}_{material}", "line", ((-w, l_signed), (w, l_signed)), colour,
                    ((-w, l_signed, low), (w, l_signed, low),
                     (w, l_signed, high), (-w, l_signed, high)), opacity)

    side = "droite" if verdict.surface.endswith("positive_x") else "gauche"
    wall_x = w if side == "droite" else -w
    top = court.side_wall_total_height
    glass_from = length - court.side_wall_glass_length
    if abs(y) >= glass_from:
        near, far = sign * glass_from, l_signed
        return Zone(f"cote_{side}_{half}_verre", "line", ((wall_x, near), (wall_x, far)), GLASS,
                    ((wall_x, near, 0.0), (wall_x, far, 0.0), (wall_x, far, top),
                     (wall_x, near, top)))
    return Zone(f"cote_{side}_grillage", "line", ((wall_x, -glass_from), (wall_x, glass_from)),
                MESH, ((wall_x, -glass_from, 0.0), (wall_x, glass_from, 0.0),
                       (wall_x, glass_from, top), (wall_x, -glass_from, top)))


def impact_patch(verdict: Verdict, court: Court, size: float = PATCH_SIZE) -> Zone | None:
    """A square of the surface itself, centred on where the ball hit it.

    Lighting a whole zone turns the position error into a change of zone: a bounce ten
    centimetres from the service line lights the entire back of the court, and the pose
    error reaches 56 cm at the far baseline. A patch moves with the error instead, and
    says where the ball landed rather than which box it belongs to.
    """
    if verdict.surface in (None, RACKET) or verdict.point is None:
        return None
    x, y, z = (float(v) for v in verdict.point)
    half = size / 2
    if verdict.surface == "floor":
        corners = _clamp_square((x, y), half, (-court.half_width, court.half_width),
                                (-court.half_length, court.half_length))
        footprint = tuple((a, b) for a, b in corners)
        return Zone("impact_sol", "area", footprint, FLOOR,
                    tuple((a, b, 0.0) for a, b in corners))

    if verdict.surface == "net":
        top = court.net_height_posts
        low, high = max(0.0, min(z, top) - half), min(top, max(z, 0.0) + half)
        left, right = _clamp_line(x, half, court.half_width)
        return Zone("impact_filet", "line", ((left, 0.0), (right, 0.0)), NET,
                    ((left, 0.0, low), (right, 0.0, low),
                     (right, 0.0, high), (left, 0.0, high)))

    colour = MESH if verdict.material == "grillage" else GLASS
    if verdict.surface.startswith("back_wall"):
        top = court.back_wall_total_height
        low, high = max(0.0, min(z, top) - half), min(top, max(z, 0.0) + half)
        left, right = _clamp_line(x, half, court.half_width)
        depth = court.half_length if y > 0 else -court.half_length
        half_name = "eloigne" if y > 0 else "proche"
        return Zone(f"impact_fond_{half_name}", "line", ((left, depth), (right, depth)), colour,
                    ((left, depth, low), (right, depth, low),
                     (right, depth, high), (left, depth, high)))

    top = court.side_wall_total_height
    low, high = max(0.0, min(z, top) - half), min(top, max(z, 0.0) + half)
    near, far = _clamp_line(y, half, court.half_length)
    wall_x = court.half_width if x > 0 else -court.half_width
    side = "droite" if x > 0 else "gauche"
    return Zone(f"impact_cote_{side}", "line", ((wall_x, near), (wall_x, far)), colour,
                ((wall_x, near, low), (wall_x, far, low),
                 (wall_x, far, high), (wall_x, near, high)))


def _clamp_line(centre: float, half: float, limit: float) -> tuple[float, float]:
    """The segment of `2 * half` around `centre`, kept inside the surface."""
    low = max(-limit, min(centre, limit) - half)
    high = min(limit, max(centre, -limit) + half)
    return low, high


def _clamp_square(centre: Point, half: float, x_limits, y_limits) -> tuple[Corner2, ...]:
    left, right = _clamp_line(centre[0], half, x_limits[1])
    near, far = _clamp_line(centre[1], half, y_limits[1])
    return ((left, near), (right, near), (right, far), (left, far))


def draw_zone(
    frame: np.ndarray, zone: Zone, pose: CameraPose, strength: float
) -> np.ndarray:
    """A copy of `frame` with the zone lit in perspective, as bright as `strength` allows.

    Args:
        strength: from 1 just after the contact down to 0, so the zone fades out
            rather than switching off.
    """
    corners = pose.project(np.array(zone.corners, dtype=np.float64))
    polygon = np.round(corners).astype(np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], zone.colour)
    alpha = zone.opacity * max(0.0, min(1.0, strength))
    lit = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)
    cv2.polylines(lit, [polygon], True, zone.colour, 2)
    return lit
