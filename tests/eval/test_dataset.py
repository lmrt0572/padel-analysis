import json

import numpy as np
import pytest

from padel_analysis.eval.dataset import PoseAnnotations


@pytest.fixture
def annotation_file(tmp_path):
    """A minimal COCO file: 2 images, 2 people on the first."""
    keypoints_a = []
    keypoints_b = []
    for i in range(17):
        keypoints_a += [float(i), float(i + 100), 2.0]
        keypoints_b += [float(i + 500), float(i + 600), 2.0]
    payload = {
        "images": [
            {"id": 1, "width": 1920, "height": 1080, "file_name": "frame_000000.PNG"},
            {"id": 2, "width": 1920, "height": 1080, "file_name": "frame_000001.PNG"},
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "keypoints": keypoints_a, "num_keypoints": 17,
            },
            {
                "id": 2, "image_id": 1, "category_id": 1,
                "bbox": [50.0, 60.0, 70.0, 80.0],
                "keypoints": keypoints_b, "num_keypoints": 17,
            },
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frame_index_is_parsed_from_the_file_name(annotation_file):
    annotations = PoseAnnotations.load(annotation_file)
    assert sorted(annotations.frame_indices()) == [0, 1]


def test_a_frame_returns_one_entry_per_annotated_person(annotation_file):
    annotations = PoseAnnotations.load(annotation_file)
    people = annotations.for_frame(0)
    assert len(people) == 2


def test_a_frame_without_annotations_returns_an_empty_list(annotation_file):
    annotations = PoseAnnotations.load(annotation_file)
    assert annotations.for_frame(1) == []


def test_bbox_is_converted_from_xywh_to_xyxy(annotation_file):
    annotations = PoseAnnotations.load(annotation_file)
    person = annotations.for_frame(0)[0]
    np.testing.assert_allclose(person.bbox, [10.0, 20.0, 40.0, 60.0])


def test_keypoints_are_returned_in_coco_order(annotation_file):
    from padel_analysis.perception.keypoints import DATASET_KEYPOINTS, LEFT_ANKLE

    annotations = PoseAnnotations.load(annotation_file)
    person = annotations.for_frame(0)[0]
    assert person.keypoints.shape == (17, 3)
    # In the fixture row i is (i, i+100, 2) in DATASET order.
    expected_row = DATASET_KEYPOINTS.index("left_ankle")
    np.testing.assert_allclose(
        person.keypoints[LEFT_ANKLE], [expected_row, expected_row + 100, 2.0]
    )


def test_ankle_midpoint_uses_both_ankles(annotation_file):
    from padel_analysis.perception.keypoints import LEFT_ANKLE, RIGHT_ANKLE

    annotations = PoseAnnotations.load(annotation_file)
    person = annotations.for_frame(0)[0]
    expected = (person.keypoints[LEFT_ANKLE, :2] + person.keypoints[RIGHT_ANKLE, :2]) / 2
    np.testing.assert_allclose(person.ankle_midpoint(), expected)


def test_unparsable_file_name_is_rejected(tmp_path):
    payload = {
        "images": [{"id": 1, "width": 10, "height": 10, "file_name": "hello.png"}],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frame index"):
        PoseAnnotations.load(path)
