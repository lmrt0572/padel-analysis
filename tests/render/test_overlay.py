import numpy as np

from padel_analysis.perception.pose_detector import PersonDetection
from padel_analysis.render.overlay import draw_people, paste_minimap


def _detection(x1, y1, x2, y2):
    keypoints = np.zeros((17, 3))
    keypoints[:, 2] = 0.9
    keypoints[:, 0] = (x1 + x2) / 2
    keypoints[:, 1] = (y1 + y2) / 2
    return PersonDetection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
        confidence=0.9,
        keypoints=keypoints,
    )


def test_drawing_people_modifies_the_frame():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    drawn = draw_people(frame, [_detection(20, 30, 80, 150)], {"near_1": 0})
    assert not np.array_equal(frame, drawn)


def test_drawing_people_does_not_mutate_the_input():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    original = frame.copy()
    draw_people(frame, [_detection(20, 30, 80, 150)], {"near_1": 0})
    np.testing.assert_array_equal(frame, original)


def test_an_unassigned_detection_is_still_drawn():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    drawn = draw_people(frame, [_detection(20, 30, 80, 150)], {})
    assert not np.array_equal(frame, drawn)


def test_a_bbox_outside_the_frame_does_not_crash():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    drawn = draw_people(frame, [_detection(-50, -60, 10, 20)], {"near_1": 0})
    assert drawn.shape == frame.shape


def test_minimap_is_pasted_into_a_corner():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    minimap = np.full((60, 40, 3), 255, dtype=np.uint8)
    pasted = paste_minimap(frame, minimap, margin=5)
    assert pasted[10, 320 - 5 - 20, 0] == 255
    assert pasted[100, 10, 0] == 0


def test_an_oversized_minimap_is_skipped_rather_than_crashing():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    minimap = np.full((500, 500, 3), 255, dtype=np.uint8)
    pasted = paste_minimap(frame, minimap)
    np.testing.assert_array_equal(pasted, frame)
