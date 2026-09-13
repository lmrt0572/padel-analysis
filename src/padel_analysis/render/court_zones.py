"""The areas of the court a contact can light up on the minimap.

A contact position is known to within 13 cm near the camera and 56 cm at the far
baseline. Drawn as a point, an error of that size is plain to see; lighting the zone
it falls in tolerates it - and in padel what matters is which wall panel or which part
of the floor was hit, not the centimetre.
"""

from dataclasses import dataclass

from ..contact.surfaces import RACKET, Verdict
from ..geometry.court import Court

Point = tuple[float, float]

FLOOR = (80, 200, 80)
GLASS = (255, 200, 60)
MESH = (170, 170, 170)
NET = (80, 220, 255)


@dataclass(frozen=True)
class Zone:
    """A named part of the court, as an area of floor or a stretch of wall seen from above."""

    name: str
    kind: str  # "area" ou "line"
    points: tuple[Point, ...]
    colour: tuple[int, int, int]


def zone_of(verdict: Verdict, court: Court) -> Zone | None:
    """The zone a contact lights, or None for a racket or an undecided contact."""
    if verdict.surface in (None, RACKET) or verdict.point is None:
        return None
    x, y = float(verdict.point[0]), float(verdict.point[1])
    w, l, s = court.half_width, court.half_length, court.service_line_distance
    half = "proche" if y < 0 else "eloigne"
    sign = -1.0 if y < 0 else 1.0

    if verdict.surface == "floor":
        if abs(y) <= s:
            side = "droite" if x >= 0 else "gauche"
            x0 = 0.0
            x1 = w if x >= 0 else -w
            return Zone(f"sol_{half}_service_{side}", "area",
                        ((x0, 0.0), (x1, 0.0), (x1, sign * s), (x0, sign * s)), FLOOR)
        return Zone(f"sol_{half}_fond", "area",
                    ((-w, sign * s), (w, sign * s), (w, sign * l), (-w, sign * l)), FLOOR)

    if verdict.surface == "net":
        return Zone("filet", "line", ((-w, 0.0), (w, 0.0)), NET)

    material = verdict.material or "verre"
    colour = MESH if material == "grillage" else GLASS
    if verdict.surface.startswith("back_wall"):
        return Zone(f"fond_{half}_{material}", "line",
                    ((-w, sign * l), (w, sign * l)), colour)

    side = "droite" if verdict.surface.endswith("positive_x") else "gauche"
    wall_x = w if side == "droite" else -w
    glass_from = l - court.side_wall_glass_length
    if abs(y) >= glass_from:
        return Zone(f"cote_{side}_{half}_verre", "line",
                    ((wall_x, sign * glass_from), (wall_x, sign * l)), GLASS)
    return Zone(f"cote_{side}_grillage", "line",
                ((wall_x, -glass_from), (wall_x, glass_from)), MESH)
