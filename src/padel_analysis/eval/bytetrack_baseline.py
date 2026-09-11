"""A plain ByteTrack, kept only as the baseline the constrained tracker is measured
against.

The production pipeline does not use it. What the comparison is meant to show is not
that ByteTrack loses players when partners cross - they hardly ever do, three times
in twenty-five minutes - but that nothing stops it from handing an identity to a
spectator, and the detector returns more than four people in 63 % of frames.

`supervision` is pinned below 0.31 because `sv.ByteTrack` is removed there, in favour
of a class in a separate `trackers` package. Pulling in a whole dependency for a
baseline that exists only to be compared against is not worth it; the pin keeps this
working and the deprecation warning keeps the situation visible.
"""

from dataclasses import dataclass

import numpy as np

from ..perception.pose_detector import PersonDetection


@dataclass(frozen=True)
class TrackedBox:
    """One box with the identifier ByteTrack gave it."""

    track_id: int
    bbox: np.ndarray  # (4,) x1 y1 x2 y2


class ByteTrackBaseline:
    """Wraps supervision's ByteTrack behind the same kind of plain interface."""

    def __init__(self) -> None:
        import supervision as sv

        self._sv = sv
        self._tracker = sv.ByteTrack()

    def update(self, detections: list[PersonDetection]) -> list[TrackedBox]:
        if not detections:
            return []

        sv = self._sv
        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        confidences = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.zeros(len(detections), dtype=int)

        tracked = self._tracker.update_with_detections(
            sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)
        )
        if tracked.tracker_id is None:
            return []
        return [
            TrackedBox(track_id=int(identifier), bbox=np.asarray(box, dtype=np.float64))
            for identifier, box in zip(tracked.tracker_id, tracked.xyxy)
        ]
