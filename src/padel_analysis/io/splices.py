"""Where the broadcast splices two stretches of play together.

At a splice the whole frame differs from the one before, where during play only the
players and the ball move.
"""

from collections.abc import Iterable

import cv2
import numpy as np

SPLICE = 4.0
"""Mean grey-level difference above which two frames belong to different stretches."""


def frame_changes(frames: Iterable[tuple[int, np.ndarray]]) -> dict[int, float]:
    """Return, for each frame, how much it differs from the one before (0 for the first)."""
    changes: dict[int, float] = {}
    previous = None
    for index, frame in frames:
        small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        changes[index] = 0.0 if previous is None else float(np.abs(small - previous).mean())
        previous = small
    return changes


def splices(changes: dict[int, float], threshold: float = SPLICE) -> list[int]:
    """Return the frames that open a new stretch of play."""
    return sorted(f for f, change in changes.items() if change > threshold)
