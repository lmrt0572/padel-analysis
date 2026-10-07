"""Padel court reference frame and dimensions, after the FIP rules.

Origin at the centre of the net, at ground level; x along the net, y towards the
baselines, z upwards. All lengths in metres.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Court:
    """Dimensions of a regulation padel court."""

    # court rectangle
    length: float = 20.0
    width: float = 10.0
    # from the net to the service line, on each side
    service_line_distance: float = 6.95
    # net height at the centre and at the posts
    net_height_centre: float = 0.88
    net_height_posts: float = 0.92
    # back wall: 3 m of glass with 1 m of mesh above it
    back_wall_glass_height: float = 3.0
    back_wall_total_height: float = 4.0
    # side walls: glass over this length from each baseline, mesh between
    side_wall_glass_length: float = 4.1
    side_wall_total_height: float = 3.0

    @property
    def half_length(self) -> float:
        return self.length / 2

    @property
    def half_width(self) -> float:
        return self.width / 2

    def corners(self) -> np.ndarray:
        """Return the four court corners, shape (4, 2), starting from the near-left one."""
        return np.array(
            [
                [-self.half_width, -self.half_length],
                [self.half_width, -self.half_length],
                [self.half_width, self.half_length],
                [-self.half_width, self.half_length],
            ],
            dtype=np.float64,
        )

    def is_in_offensive_zone(self, points: np.ndarray, side: int) -> np.ndarray:
        """Return whether each point lies between the net and the service line of `side`.

        Args:
            points: array of shape (N, 2) in court coordinates.
            side: +1 for the positive-y half, -1 for the negative-y half.
        """
        y = points[:, 1] * side
        return (y > 0) & (y <= self.service_line_distance)
