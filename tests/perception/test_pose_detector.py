import numpy as np
import pytest

from padel_analysis.perception.pose_detector import PersonDetection, detections_from_arrays


def test_detection_exposes_bbox_confidence_and_keypoints():
    boxes = np.array([[10.0, 20.0, 40.0, 60.0]])
    scores = np.array([0.9])
    keypoints = np.zeros((1, 17, 3))
    detections = detections_from_arrays(boxes, scores, keypoints)
    assert len(detections) == 1
    assert isinstance(detections[0], PersonDetection)
    np.testing.assert_allclose(detections[0].bbox, [10.0, 20.0, 40.0, 60.0])
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].keypoints.shape == (17, 3)


def test_low_confidence_detections_are_dropped():
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
    scores = np.array([0.9, 0.1])
    keypoints = np.zeros((2, 17, 3))
    detections = detections_from_arrays(boxes, scores, keypoints, min_confidence=0.5)
    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.9)


def test_ankle_midpoint_averages_both_ankles():
    from padel_analysis.perception.keypoints import LEFT_ANKLE, RIGHT_ANKLE

    keypoints = np.zeros((1, 17, 3))
    keypoints[0, LEFT_ANKLE] = [100.0, 200.0, 0.9]
    keypoints[0, RIGHT_ANKLE] = [300.0, 400.0, 0.8]
    detection = detections_from_arrays(
        np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([0.9]), keypoints
    )[0]
    np.testing.assert_allclose(detection.ankle_midpoint(), [200.0, 300.0])


def test_ankle_confidence_is_the_lower_of_the_two():
    from padel_analysis.perception.keypoints import LEFT_ANKLE, RIGHT_ANKLE

    keypoints = np.zeros((1, 17, 3))
    keypoints[0, LEFT_ANKLE] = [100.0, 200.0, 0.9]
    keypoints[0, RIGHT_ANKLE] = [300.0, 400.0, 0.2]
    detection = detections_from_arrays(
        np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([0.9]), keypoints
    )[0]
    assert detection.ankle_confidence() == pytest.approx(0.2)


def test_bbox_bottom_centre_is_the_middle_of_the_lower_edge():
    detection = detections_from_arrays(
        np.array([[10.0, 20.0, 50.0, 80.0]]), np.array([0.9]), np.zeros((1, 17, 3))
    )[0]
    np.testing.assert_allclose(detection.bbox_bottom_centre(), [30.0, 80.0])


def test_empty_input_returns_no_detections():
    detections = detections_from_arrays(
        np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 17, 3))
    )
    assert detections == []
