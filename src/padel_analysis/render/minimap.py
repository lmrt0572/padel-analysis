"""Top-down view of the court with the tracked players on it.

Positive `y` is drawn towards the top of the image, matching the broadcast camera.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..geometry.court import Court

if TYPE_CHECKING:
    from .court_zones import Zone

TEAM_COLOURS: dict[str, tuple[int, int, int]] = {
    "near_1": (60, 60, 230),
    "near_2": (90, 120, 240),
    "far_1": (230, 160, 60),
    "far_2": (240, 200, 120),
}
SURFACE = (110, 55, 28)
LINES = (235, 235, 235)
NET = (80, 220, 255)


class Minimap:
    """Schematic court with the player positions drawn on it."""

    def __init__(self, court: Court, width: int = 300, margin: int = 18) -> None:
        self._court = court
        self._width = width
        self._margin = margin
        self._scale = (width - 2 * margin) / court.width
        self._height = int(court.length * self._scale) + 2 * margin

    @property
    def size(self) -> tuple[int, int]:
        return self._width, self._height

    def to_pixels(self, court_xy: np.ndarray) -> tuple[int, int]:
        x, y = float(court_xy[0]), float(court_xy[1])
        px = self._margin + (x + self._court.half_width) * self._scale
        py = self._margin + (self._court.half_length - y) * self._scale
        return round(px), round(py)

    def draw(
        self,
        positions: dict[str, tuple[float, float]],
        lit: Sequence[tuple["Zone", float]] = (),
    ) -> np.ndarray:
        """Draw the players as filled discs, over the zones recent contacts have lit.

        Args:
            lit: zones with a strength in [0, 1], so that a zone can fade out.
        """
        court = self._court
        image = np.full((self._height, self._width, 3), 35, dtype=np.uint8)

        top_left = self.to_pixels(np.array([-court.half_width, court.half_length]))
        bottom_right = self.to_pixels(np.array([court.half_width, -court.half_length]))
        cv2.rectangle(image, top_left, bottom_right, SURFACE, -1)
        cv2.rectangle(image, top_left, bottom_right, LINES, 2)

        for y in (-court.service_line_distance, court.service_line_distance):
            cv2.line(
                image,
                self.to_pixels(np.array([-court.half_width, y])),
                self.to_pixels(np.array([court.half_width, y])),
                LINES,
                1,
            )
        cv2.line(
            image,
            self.to_pixels(np.array([0.0, -court.service_line_distance])),
            self.to_pixels(np.array([0.0, court.service_line_distance])),
            LINES,
            1,
        )
        cv2.line(
            image,
            self.to_pixels(np.array([-court.half_width, 0.0])),
            self.to_pixels(np.array([court.half_width, 0.0])),
            NET,
            2,
        )

        for zone, strength in lit:
            overlay = image.copy()
            pixels = np.array([self.to_pixels(np.array(p)) for p in zone.points], np.int32)
            if zone.kind == "area":
                cv2.fillPoly(overlay, [pixels], zone.colour)
            else:
                cv2.line(overlay, tuple(pixels[0]), tuple(pixels[1]), zone.colour, 7)
            alpha = 0.75 * max(0.0, min(1.0, strength))
            image = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)

        for name, (x, y) in positions.items():
            centre = self.to_pixels(np.array([x, y]))
            colour = TEAM_COLOURS.get(name, (200, 200, 200))
            cv2.circle(image, centre, 7, colour, -1)
            cv2.circle(image, centre, 7, (15, 15, 15), 1)
        return image
