"""Strategies for turning a detection into the point where the player meets the floor.

The two implementations exist so the choice can be settled by measurement rather
than opinion: switching between them is a configuration change, not a code branch.
"""

from typing import Protocol

import numpy as np

from .pose_detector import PersonDetection


class GroundPointStrategy(Protocol):
    """Returns a ground point in image pixels, and how much to trust it."""

    def __call__(self, detection: PersonDetection) -> tuple[np.ndarray, float]: ...


class AnkleMidpoint:
    """Midpoint of the two ankles, with a fallback when they are not visible.

    Padel players are frequently occluded by the mesh walls and by their partner.
    When both ankles fall below `min_confidence` the bbox bottom is used instead,
    and the low confidence is reported so downstream code can discount the point.
    """

    def __init__(self, min_confidence: float = 0.3) -> None:
        self._min_confidence = min_confidence

    def __call__(self, detection: PersonDetection) -> tuple[np.ndarray, float]:
        confidence = detection.ankle_confidence()
        if confidence < self._min_confidence:
            return detection.bbox_bottom_centre(), confidence
        return detection.ankle_midpoint(), confidence


class BboxBottom:
    """Centre of the lower edge of the bounding box.

    Cheaper, and the baseline the ankle strategy is measured against.
    """

    def __call__(self, detection: PersonDetection) -> tuple[np.ndarray, float]:
        return detection.bbox_bottom_centre(), detection.confidence
