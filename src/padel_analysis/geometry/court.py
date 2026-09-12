"""Padel court reference frame and dimensions.

Reference frame
---------------
Origin at the centre of the net, at ground level.
    x: along the net.         Range [-width/2, +width/2].
    y: towards the baselines. Range [-length/2, +length/2].
    z: upwards from the ground.
All lengths in metres.

Source
------
Constants follow the FIP Rules of Padel. Court rectangle, service line, net heights
and back wall composition are confirmed.

The side walls were left out at first, secondary sources disagreeing on whether they
are stepped. The rulebook allows two variants, and a manual annotation of the wall
panels on 2026-09-12 settled which one this court is: variant 2, "Crystal" - glass
over four metres from each baseline, mesh between them, and no step.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Court:
    """Dimensions of a regulation padel court."""

    # Court rectangle. FIP allows a 0.5% tolerance either way.
    length: float = 20.0
    width: float = 10.0
    # Distance from the net to the service line, on each side.
    service_line_distance: float = 6.95
    # Net height at the centre and at the posts.
    net_height_centre: float = 0.88
    net_height_posts: float = 0.92
    # Back wall: 3 m of tempered glass with 1 m of metallic mesh above it.
    back_wall_glass_height: float = 3.0
    back_wall_total_height: float = 4.0
    # Side walls: glass over this length from each baseline, mesh between the two.
    side_wall_glass_length: float = 4.1
    side_wall_total_height: float = 3.0

    @property
    def half_length(self) -> float:
        return self.length / 2

    @property
    def half_width(self) -> float:
        return self.width / 2

    def corners(self) -> np.ndarray:
        """The four court corners, starting from the near-left one.

        Returns an array of shape (4, 2) in court coordinates.
        """
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
        """Whether each point lies between the net and the service line of `side`.

        Args:
            points: array of shape (N, 2) in court coordinates.
            side: +1 for the positive-y half, -1 for the negative-y half.

        Returns:
            Boolean array of shape (N,).
        """
        y = points[:, 1] * side
        return (y > 0) & (y <= self.service_line_distance)
