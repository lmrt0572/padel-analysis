"""Draws detections and identities onto a video frame.

This module performs no inference: it consumes what the pipeline already computed,
which is what allows rendering to be re-run without paying for detection again.
"""

import cv2
import numpy as np

from ..perception.pose_detector import PersonDetection
from .minimap import TEAM_COLOURS

SKELETON: tuple[tuple[int, int], ...] = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)
UNASSIGNED = (170, 170, 170)


def draw_people(
    frame: np.ndarray,
    detections: list[PersonDetection],
    assignment: dict[str, int],
    keypoint_threshold: float = 0.3,
) -> np.ndarray:
    """Return a copy of `frame` with boxes, skeletons and identifiers drawn."""
    canvas = frame.copy()
    name_of_index = {index: name for name, index in assignment.items()}

    for index, detection in enumerate(detections):
        name = name_of_index.get(index)
        colour = TEAM_COLOURS.get(name, UNASSIGNED) if name else UNASSIGNED

        x1, y1, x2, y2 = (round(float(v)) for v in detection.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

        points = detection.keypoints
        for a, b in SKELETON:
            if points[a, 2] >= keypoint_threshold and points[b, 2] >= keypoint_threshold:
                cv2.line(
                    canvas,
                    (int(points[a, 0]), int(points[a, 1])),
                    (int(points[b, 0]), int(points[b, 1])),
                    colour,
                    2,
                )

        if name:
            cv2.putText(canvas, name, (x1, max(14, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(canvas, name, (x1, max(14, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return canvas


def paste_minimap(
    frame: np.ndarray, minimap: np.ndarray, margin: int = 16
) -> np.ndarray:
    """Paste the minimap into the top-right corner, if it fits."""
    fh, fw = frame.shape[:2]
    mh, mw = minimap.shape[:2]
    if mh + 2 * margin > fh or mw + 2 * margin > fw:
        return frame
    canvas = frame.copy()
    canvas[margin : margin + mh, fw - mw - margin : fw - margin] = minimap
    return canvas
