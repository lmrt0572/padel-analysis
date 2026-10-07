"""Person detection and pose estimation with a single Ultralytics model.

Nothing downstream of this module sees an Ultralytics object, so the model can be
replaced without touching tracking, analytics or rendering.
"""

from dataclasses import dataclass

import numpy as np

from .keypoints import LEFT_ANKLE, RIGHT_ANKLE

PERSON_CLASS = 0


@dataclass(frozen=True)
class PersonDetection:
    """One detected person on one frame, in image pixels."""

    bbox: np.ndarray  # (4,) x1 y1 x2 y2
    confidence: float
    keypoints: np.ndarray  # (17, 3) x, y, confidence, in COCO order

    def ankle_midpoint(self) -> np.ndarray:
        return (self.keypoints[LEFT_ANKLE, :2] + self.keypoints[RIGHT_ANKLE, :2]) / 2

    def ankle_confidence(self) -> float:
        """Return the weaker of the two ankle confidences."""
        return float(min(self.keypoints[LEFT_ANKLE, 2], self.keypoints[RIGHT_ANKLE, 2]))

    def bbox_bottom_centre(self) -> np.ndarray:
        x1, _, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, y2])


def detections_from_arrays(
    boxes: np.ndarray,
    scores: np.ndarray,
    keypoints: np.ndarray,
    min_confidence: float = 0.25,
) -> list[PersonDetection]:
    """Build detections from raw model output arrays.

    Args:
        boxes: (N, 4) in xyxy pixels.
        scores: (N,) detection confidences.
        keypoints: (N, 17, 3) in COCO order.
    """
    detections: list[PersonDetection] = []
    for box, score, points in zip(boxes, scores, keypoints):
        if float(score) < min_confidence:
            continue
        detections.append(
            PersonDetection(
                bbox=np.asarray(box, dtype=np.float64),
                confidence=float(score),
                keypoints=np.asarray(points, dtype=np.float64),
            )
        )
    return detections


class PoseDetector:
    """An Ultralytics pose model behind a plain-array interface."""

    def __init__(
        self,
        weights: str = "yolov8s-pose.pt",
        imgsz: int = 1600,
        min_confidence: float = 0.25,
        device: str = "cuda",
    ) -> None:
        """Load the model.

        Args:
            imgsz: inference resolution; below 1600 the far-court players are missed.
        """
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._imgsz = imgsz
        self._min_confidence = min_confidence
        self._device = device

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        result = self._model.predict(
            frame,
            imgsz=self._imgsz,
            classes=[PERSON_CLASS],
            device=self._device,
            verbose=False,
        )[0]
        if result.keypoints is None or result.boxes is None:
            return []
        return detections_from_arrays(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.keypoints.data.cpu().numpy(),
            min_confidence=self._min_confidence,
        )
