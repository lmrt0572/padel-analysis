"""Colour signature of a player's kit, used to break positional ties in tracking.

The two teams of a padel match wear clearly distinct colours, which makes a coarse
hue histogram of the torso enough to tell partners apart when they cross.
"""

import cv2
import numpy as np

BINS = 8


def torso_histogram(frame: np.ndarray, bbox: np.ndarray, bins: int = BINS) -> np.ndarray:
    """Normalised hue histogram of the upper half of a bounding box.

    The upper half avoids the court surface showing between the legs, which would
    otherwise dominate the signature with the blue of the floor.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    x1, x2 = max(0, min(x1, width - 1)), max(1, min(x2, width))
    y1, y2 = max(0, min(y1, height - 1)), max(1, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(bins, dtype=np.float64)

    torso = frame[y1 : y1 + max(1, (y2 - y1) // 2), x1:x2]
    if torso.size == 0:
        return np.zeros(bins, dtype=np.float64)

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    total = histogram.sum()
    if total <= 0:
        return np.zeros(bins, dtype=np.float64)
    return (histogram / total).astype(np.float64)
