import json

import pytest

from padel_analysis.eval.ball_dataset import BallAnnotations


def _write(path, entries):
    """entries : {frame: [(x, y, w, h), ...]}"""
    images, annotations = [], []
    for i, frame in enumerate(sorted(entries)):
        images.append({"id": i, "file_name": f"frame_{frame:06d}.PNG"})
        for bbox in entries[frame]:
            annotations.append({"image_id": i, "category_id": 1, "bbox": list(bbox)})
    path.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "Ball"}, {"id": 2, "name": "Wall"}],
                "images": images,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_the_centre_comes_from_the_bounding_box(tmp_path):
    path = _write(tmp_path / "ball.json", {7: [(100.0, 200.0, 10.0, 10.0)]})
    balls = BallAnnotations.load(path)
    assert balls.for_frame(7) == [pytest.approx((105.0, 205.0))]


def test_a_frame_without_annotation_holds_no_ball(tmp_path):
    path = _write(tmp_path / "ball.json", {7: [(100.0, 200.0, 10.0, 10.0)]})
    balls = BallAnnotations.load(path)
    assert balls.for_frame(8) == []


def test_the_annotated_frames_are_listed(tmp_path):
    path = _write(
        tmp_path / "ball.json",
        {3: [(0.0, 0.0, 4.0, 4.0)], 9: [(0.0, 0.0, 4.0, 4.0)]},
    )
    assert BallAnnotations.load(path).frame_indices() == [3, 9]


def test_a_frame_carrying_two_balls_keeps_both(tmp_path):
    """Le dataset en contient un cas : le lecteur ne doit pas en perdre un."""
    path = _write(
        tmp_path / "ball.json",
        {4: [(0.0, 0.0, 10.0, 10.0), (500.0, 500.0, 10.0, 10.0)]},
    )
    assert len(BallAnnotations.load(path).for_frame(4)) == 2


def test_only_the_ball_category_is_read(tmp_path):
    """Wall, shot-event et les autres sont declares dans le fichier mais inutilises."""
    path = tmp_path / "ball.json"
    path.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "Ball"}, {"id": 2, "name": "Wall"}],
                "images": [{"id": 0, "file_name": "frame_000005.PNG"}],
                "annotations": [
                    {"image_id": 0, "category_id": 2, "bbox": [0, 0, 50, 50]},
                    {"image_id": 0, "category_id": 1, "bbox": [10, 10, 10, 10]},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert BallAnnotations.load(path).for_frame(5) == [pytest.approx((15.0, 15.0))]


def test_positions_come_back_as_an_array_when_asked(tmp_path):
    path = _write(tmp_path / "ball.json", {2: [(0.0, 0.0, 10.0, 10.0)]})
    centres = BallAnnotations.load(path).centres()
    assert centres[2] == pytest.approx((5.0, 5.0))
    assert isinstance(centres, dict)


def test_an_unreadable_file_name_is_refused(tmp_path):
    path = tmp_path / "ball.json"
    path.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "Ball"}],
                "images": [{"id": 0, "file_name": "no_digits_here.PNG"}],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frame index"):
        BallAnnotations.load(path)
