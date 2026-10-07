"""Pairing predicted boxes with annotated ones, so they can be compared.

The assignment is global rather than greedy.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(first: np.ndarray, second: np.ndarray) -> float:
    """Return the intersection over union of two xyxy boxes."""
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))

    width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
    intersection = width * height
    if intersection <= 0.0:
        return 0.0

    area_first = (float(first[2]) - float(first[0])) * (
        float(first[3]) - float(first[1])
    )
    area_second = (float(second[2]) - float(second[0])) * (
        float(second[3]) - float(second[1])
    )
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def match_by_iou(
    predicted: list[np.ndarray], annotated: list[np.ndarray], threshold: float = 0.5
) -> list[tuple[int, int]]:
    """Return (prediction index, annotation index) for every pair reaching `threshold`."""
    if not predicted or not annotated:
        return []

    overlaps = np.array(
        [[iou(p, a) for a in annotated] for p in predicted], dtype=np.float64
    )
    rows, cols = linear_sum_assignment(-overlaps)
    return [
        (int(r), int(c)) for r, c in zip(rows, cols) if overlaps[r, c] >= threshold
    ]
