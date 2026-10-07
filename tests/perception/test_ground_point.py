import numpy as np
import pytest

from padel_analysis.perception.ground_point import AnkleMidpoint, BboxBottom
from padel_analysis.perception.keypoints import LEFT_ANKLE, RIGHT_ANKLE
from padel_analysis.perception.pose_detector import PersonDetection


def _detection(left_ankle, right_ankle, bbox=(10.0, 20.0, 50.0, 80.0)):
    keypoints = np.zeros((17, 3))
    keypoints[LEFT_ANKLE] = left_ankle
    keypoints[RIGHT_ANKLE] = right_ankle
    return PersonDetection(bbox=np.array(bbox), confidence=0.9, keypoints=keypoints)


def test_ankle_strategy_returns_the_midpoint():
    detection = _detection([100.0, 200.0, 0.9], [300.0, 400.0, 0.8])
    point, confidence = AnkleMidpoint()(detection)
    np.testing.assert_allclose(point, [200.0, 300.0])
    assert confidence == pytest.approx(0.8)


def test_ankle_strategy_falls_back_to_bbox_when_ankles_are_unreliable():
    detection = _detection([100.0, 200.0, 0.05], [300.0, 400.0, 0.05])
    point, confidence = AnkleMidpoint(min_confidence=0.3)(detection)
    np.testing.assert_allclose(point, [30.0, 80.0])
    assert confidence == pytest.approx(0.05)


def test_bbox_strategy_returns_the_bottom_centre():
    detection = _detection([100.0, 200.0, 0.9], [300.0, 400.0, 0.9])
    point, confidence = BboxBottom()(detection)
    np.testing.assert_allclose(point, [30.0, 80.0])
    assert confidence == pytest.approx(0.9)


def test_the_two_strategies_disagree_on_the_same_detection():
    """Justify the ablation: if they coincided, comparing them would mean nothing."""
    detection = _detection([100.0, 200.0, 0.9], [300.0, 400.0, 0.9])
    ankle, _ = AnkleMidpoint()(detection)
    bbox, _ = BboxBottom()(detection)
    assert not np.allclose(ankle, bbox)


def test_both_strategies_return_a_two_element_point():
    detection = _detection([100.0, 200.0, 0.9], [300.0, 400.0, 0.9])
    for strategy in (AnkleMidpoint(), BboxBottom()):
        point, _ = strategy(detection)
        assert point.shape == (2,)
