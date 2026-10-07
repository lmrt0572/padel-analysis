"""Strategies for turning a detection into the point where the player meets the floor.

Switching between them is a configuration change, so the choice can be measured.
"""

from typing import Protocol

import numpy as np

from .pose_detector import PersonDetection


class GroundPointStrategy(Protocol):
    """A ground point in image pixels, and how much to trust it."""

    def __call__(self, detection: PersonDetection) -> tuple[np.ndarray, float]: ...


class AnkleMidpoint:
    """Midpoint of the two ankles, falling back on the bbox bottom when they are hidden.

    The fallback reports its low confidence.
    """

    def __init__(self, min_confidence: float = 0.3) -> None:
        self._min_confidence = min_confidence

    def __call__(self, detection: PersonDetection) -> tuple[np.ndarray, float]:
        confidence = detection.ankle_confidence()
        if confidence < self._min_confidence:
            return detection.bbox_bottom_centre(), confidence
        return detection.ankle_midpoint(), confidence


class BboxBottom:
    """Centre of the lower edge of the bounding box, the baseline strategy."""

    def __call__(self, detection: PersonDetection) -> tuple[np.ndarray, float]:
        return detection.bbox_bottom_centre(), detection.confidence
