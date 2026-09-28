"""Where the broadcast splices two stretches of play together.

The video is a sequence of rallies with the dead time cut out, and at each splice the
picture changes at once: the whole frame differs from the one before, where during
play only the players and the ball move. The mean absolute difference between two
small greyscale copies of consecutive frames separates the two cleanly - measured on a
minute of the women's final, splices at 5.6 to 11.5 against a median of 0.5 in play.
"""

from collections.abc import Iterable

import cv2
import numpy as np

SPLICE = 4.0
"""Mean grey-level difference above which consecutive frames belong to two different
stretches of play. Play itself stays under about 2.3."""


def frame_changes(frames: Iterable[tuple[int, np.ndarray]]) -> dict[int, float]:
    """For each frame, how much it differs from the one before (0 for the first)."""
    changes: dict[int, float] = {}
    previous = None
    for index, frame in frames:
        small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        changes[index] = 0.0 if previous is None else float(np.abs(small - previous).mean())
        previous = small
    return changes


def splices(changes: dict[int, float], threshold: float = SPLICE) -> list[int]:
    """The frames that open a new stretch of play."""
    return sorted(f for f, change in changes.items() if change > threshold)
