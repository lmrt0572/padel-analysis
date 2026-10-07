"""Ball candidates for one frame, and the interface both detectors share.

This stage does not decide: it returns a ranked list and the trajectory stage picks
from it. Motion and network detectors sit behind the same interface, so comparing
them is a change of configuration.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    """A possible ball position, in image pixels."""

    x: float
    y: float
    score: float


class BallCandidates(Protocol):
    """Produce ranked candidates for one frame, given the frames it asks for."""

    def frames_needed(self, index: int) -> list[int]:
        """Return the frame numbers this detector needs to work on `index`."""
        ...

    def __call__(self, frames: dict[int, np.ndarray], index: int) -> list[Candidate]:
        ...


class MotionCandidates:
    """Blobs brighter than both temporal neighbours, sized like a ball."""

    def __init__(
        self,
        spacing: int = 2,
        min_peak: int = 20,
        min_area: int = 4,
        max_area: int = 900,
        max_candidates: int = 200,
    ) -> None:
        """Set the detector up.

        Args:
            spacing: how many frames away the two compared neighbours sit.
            min_peak: brightness a blob must gain over both neighbours.
            min_area, max_area: pixel area a blob must fall within.
            max_candidates: cap on the returned list, strongest kept.
        """
        self._spacing = spacing
        self._min_peak = min_peak
        self._min_area = min_area
        self._max_area = max_area
        self._max_candidates = max_candidates

    def frames_needed(self, index: int) -> list[int]:
        return [index - self._spacing, index, index + self._spacing]

    def __call__(self, frames: dict[int, np.ndarray], index: int) -> list[Candidate]:
        needed = self.frames_needed(index)
        if any(f not in frames for f in needed):
            return []

        grey = [
            cv2.cvtColor(frames[f], cv2.COLOR_BGR2GRAY).astype(np.int16)
            for f in needed
        ]
        previous, current, following = grey
        difference = np.minimum(current - previous, current - following)
        mask = (difference >= self._min_peak).astype(np.uint8)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        found: list[Candidate] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self._min_area or area > self._max_area:
                continue
            peak = float(difference[labels == label].max())
            x, y = centroids[label]
            found.append(Candidate(x=float(x), y=float(y), score=peak))

        found.sort(key=lambda c: c.score, reverse=True)
        return found[: self._max_candidates]


def demote_inside_boxes(
    candidates: Sequence[Candidate],
    boxes: Sequence[np.ndarray],
    factor: float = 0.25,
) -> list[Candidate]:
    """Push candidates sitting on a player down the ranking, without removing them.

    A ball passing in front of a player must stay reachable, so the score is
    multiplied rather than the candidate dropped.

    Args:
        candidates: ranked candidates, strongest first.
        boxes: player boxes as xyxy arrays.
        factor: what the score of a candidate inside a box is multiplied by.
    """
    adjusted: list[Candidate] = []
    for candidate in candidates:
        inside = any(
            b[0] <= candidate.x <= b[2] and b[1] <= candidate.y <= b[3] for b in boxes
        )
        adjusted.append(
            Candidate(candidate.x, candidate.y, candidate.score * factor)
            if inside
            else candidate
        )
    adjusted.sort(key=lambda c: c.score, reverse=True)
    return adjusted
