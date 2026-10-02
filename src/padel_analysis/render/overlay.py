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
    labels: dict[str, str] | None = None,
    colours: dict[str, tuple[int, int, int]] | None = None,
    tracked_only: bool = False,
) -> np.ndarray:
    """Return a copy of `frame` with boxes, skeletons and identifiers drawn.

    Args:
        labels: text written above each slot's box; the slot name by default.
        colours: BGR colour per slot; the team colours by default.
        tracked_only: leave out the people the tracker holds in no slot - spectators,
            referee, ball boys.
    """
    canvas = frame.copy()
    name_of_index = {index: name for name, index in assignment.items()}
    palette = colours or TEAM_COLOURS

    for index, detection in enumerate(detections):
        slot = name_of_index.get(index)
        if tracked_only and slot is None:
            continue
        colour = palette.get(slot, UNASSIGNED) if slot else UNASSIGNED
        name = (labels or {}).get(slot, slot) if slot else None

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
