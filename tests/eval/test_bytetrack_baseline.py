import numpy as np
import pytest

pytest.importorskip("supervision")

from padel_analysis.eval.bytetrack_baseline import (
    ByteTrackBaseline,
    TrackedBox,
)
from padel_analysis.perception.pose_detector import PersonDetection


def _detection(x1, y1, x2, y2, confidence=0.9):
    return PersonDetection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
        confidence=confidence,
        keypoints=np.zeros((17, 3)),
    )


def test_a_detection_gets_a_track_identifier():
    baseline = ByteTrackBaseline()
    for _ in range(5):
        tracks = baseline.update([_detection(10, 10, 50, 120)])
    assert all(isinstance(t, TrackedBox) for t in tracks)


def test_a_stationary_person_keeps_the_same_identifier():
    baseline = ByteTrackBaseline()
    identifiers = []
    for _ in range(10):
        tracks = baseline.update([_detection(10, 10, 50, 120)])
        if tracks:
            identifiers.append(tracks[0].track_id)
    assert identifiers
    assert len(set(identifiers)) == 1


def test_bytetrack_does_not_limit_the_number_of_tracks():
    """The essential difference from the constrained tracker: nothing bounds it to four."""
    baseline = ByteTrackBaseline()
    detections = [_detection(120 * i, 10, 120 * i + 80, 220) for i in range(8)]
    for _ in range(6):
        tracks = baseline.update(detections)
    assert len(tracks) > 4


def test_no_detection_gives_no_track():
    baseline = ByteTrackBaseline()
    assert baseline.update([]) == []
