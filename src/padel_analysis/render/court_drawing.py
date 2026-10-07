"""The court drawn from its own geometry, seen from the broadcast camera.

The picture is the court model projected by the calibrated camera, with no pixel of
the broadcast.
"""

import cv2
import numpy as np

from ..geometry.camera import CameraPose
from ..geometry.court import Court
from . import figure_style as style

LINE = (235, 235, 235)
GLASS_FILL = (230, 200, 140)
GLASS_EDGE = (240, 215, 170)
MESH_EDGE = (150, 150, 150)
NEAR_EDGE = (80, 80, 80)
COURT = Court()


def _bgr(colour: str) -> tuple[int, int, int]:
    return tuple(int(colour[i:i + 2], 16) for i in (5, 3, 1))


def court_backdrop(pose: CameraPose, size: tuple[int, int], court: Court = COURT) -> np.ndarray:
    """Return a dark picture of the court as the camera sees it."""
    width, height = size
    canvas = np.full((height, width, 3), _bgr(style.BACKGROUND), dtype=np.uint8)
    w, ln, s = court.half_width, court.half_length, court.service_line_distance

    def poly(points):
        return np.round(pose.project(np.array(points, dtype=np.float64))).astype(np.int32)

    def line(a, b, colour=LINE, thickness=2):
        cv2.polylines(canvas, [poly([a, b])], False, colour, thickness, cv2.LINE_AA)

    def panel(points, fill, edge, alpha):
        nonlocal canvas
        shape = poly(points)
        if fill is not None:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [shape], fill)
            canvas = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0)
        cv2.polylines(canvas, [shape], True, edge, 1, cv2.LINE_AA)

    def wall(far: bool):
        # the far wall and the sides go behind the floor; the near wall is drawn last,
        # more transparent
        y = ln if far else -ln
        alpha, edge = (0.16, GLASS_EDGE) if far else (0.05, NEAR_EDGE)
        panel([(-w, y, 0), (w, y, 0), (w, y, court.back_wall_glass_height),
               (-w, y, court.back_wall_glass_height)], GLASS_FILL, edge, alpha)
        panel([(-w, y, court.back_wall_glass_height), (w, y, court.back_wall_glass_height),
               (w, y, court.back_wall_total_height), (-w, y, court.back_wall_total_height)],
              None, MESH_EDGE if far else NEAR_EDGE, 0.0)

    floor = poly([(-w, -ln, 0), (w, -ln, 0), (w, ln, 0), (-w, ln, 0)])
    cv2.fillPoly(canvas, [floor], _bgr(style.COURT_BLUE))
    wall(far=True)
    glass_end = ln - court.side_wall_glass_length
    for x in (-w, w):
        top = court.side_wall_total_height
        for y0, y1 in ((-ln, -glass_end), (glass_end, ln)):
            panel([(x, y0, 0), (x, y1, 0), (x, y1, top), (x, y0, top)], GLASS_FILL,
                  GLASS_EDGE, 0.12)
        panel([(x, -glass_end, 0), (x, glass_end, 0), (x, glass_end, top),
               (x, -glass_end, top)], None, MESH_EDGE, 0.0)
    for y in (-ln, ln):
        line((-w, y, 0), (w, y, 0))
    for x in (-w, w):
        line((x, -ln, 0), (x, ln, 0))
    for y in (-s, s):
        line((-w, y, 0), (w, y, 0))
    line((0, -s, 0), (0, s, 0))
    panel([(-w, 0, 0), (w, 0, 0), (w, 0, court.net_height_posts),
           (-w, 0, court.net_height_posts)], (40, 40, 40), (200, 200, 200), 0.45)
    line((-w, 0, court.net_height_posts), (w, 0, court.net_height_posts), LINE, 2)
    wall(far=False)
    return canvas
