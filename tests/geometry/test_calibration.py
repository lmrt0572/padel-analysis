import json

import numpy as np
import pytest

from padel_analysis.geometry.calibration import (
    Calibration,
    CalibrationPoint,
    calibration_from_points,
)
from padel_analysis.geometry.court import Court

# Same correspondences as in test_projector, to stay consistent.
IMAGE_CORNERS = [(240.0, 940.0), (1680.0, 940.0), (1180.0, 330.0), (740.0, 330.0)]


def _corner_points() -> list[CalibrationPoint]:
    court = Court()
    return [
        CalibrationPoint(name=f"corner_{i}", court_xy=tuple(c), image_xy=img)
        for i, (c, img) in enumerate(zip(court.corners(), IMAGE_CORNERS))
    ]


def test_calibration_from_four_corners_builds_a_projector():
    calibration = calibration_from_points(_corner_points())
    centre = calibration.projector.court_to_image(np.array([[0.0, 0.0]]))
    assert centre.shape == (1, 2)


def test_calibration_requires_at_least_four_points():
    with pytest.raises(ValueError, match="at least 4"):
        calibration_from_points(_corner_points()[:3])


def test_control_points_are_excluded_from_the_fit_and_used_for_the_error():
    points = _corner_points()
    court = Court()
    # A fifth point, marked as a control point.
    extra_court = (0.0, court.service_line_distance)
    fitted = calibration_from_points(points)
    extra_image = tuple(fitted.projector.court_to_image(np.array([extra_court]))[0])
    points.append(
        CalibrationPoint("service_centre", extra_court, extra_image, is_control=True)
    )

    calibration = calibration_from_points(points)
    assert calibration.error.rmse_pixels == pytest.approx(0.0, abs=1e-6)
    assert calibration.n_control_points == 1


def test_error_is_none_when_no_control_point_is_provided():
    calibration = calibration_from_points(_corner_points())
    assert calibration.error is None
    assert calibration.n_control_points == 0


def test_calibration_roundtrips_through_json(tmp_path):
    calibration = calibration_from_points(_corner_points())
    path = tmp_path / "calibration.json"
    calibration.save(path)

    loaded = Calibration.load(path)
    original = calibration.projector.court_to_image(np.array([[1.0, 2.0]]))
    restored = loaded.projector.court_to_image(np.array([[1.0, 2.0]]))
    np.testing.assert_allclose(restored, original, atol=1e-9)


def test_saved_file_is_human_readable_json(tmp_path):
    calibration = calibration_from_points(_corner_points())
    path = tmp_path / "calibration.json"
    calibration.save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["points"]) == 4
    assert payload["points"][0]["name"] == "corner_0"


def test_a_point_sits_on_the_ground_by_default():
    point = CalibrationPoint(name="corner", court_xy=(-5.0, -10.0), image_xy=(210.0, 960.0))
    assert point.height == 0.0
    assert point.court_xyz == (-5.0, -10.0, 0.0)


def test_a_point_can_carry_a_height():
    point = CalibrationPoint(
        name="glass_top", court_xy=(-5.0, -10.0), image_xy=(168.0, 531.0), height=3.0
    )
    assert point.court_xyz == (-5.0, -10.0, 3.0)


def test_the_homography_ignores_points_above_the_ground():
    """A homography only knows the ground plane: a point above the ground skews it."""
    ground = _corner_points()
    elevated = CalibrationPoint(
        name="glass_top", court_xy=(-5.0, -10.0), image_xy=(168.0, 531.0), height=3.0
    )
    without = calibration_from_points(ground)
    with_it = calibration_from_points([*ground, elevated])
    np.testing.assert_allclose(
        with_it.projector.court_to_image(np.array([[1.0, 2.0]])),
        without.projector.court_to_image(np.array([[1.0, 2.0]])),
        atol=1e-9,
    )


def test_an_elevated_control_point_does_not_enter_the_reported_error():
    ground = _corner_points()
    elevated = CalibrationPoint(
        name="glass_top",
        court_xy=(-5.0, -10.0),
        image_xy=(168.0, 531.0),
        is_control=True,
        height=3.0,
    )
    assert calibration_from_points([*ground, elevated]).error is None


def test_a_height_survives_a_save_and_a_load(tmp_path):
    points = [
        *_corner_points(),
        CalibrationPoint(
            name="glass_top", court_xy=(-5.0, -10.0), image_xy=(168.0, 531.0), height=3.0
        ),
    ]
    path = tmp_path / "calibration.json"
    calibration_from_points(points).save(path)
    assert [p.height for p in Calibration.load(path).points] == [0.0, 0.0, 0.0, 0.0, 3.0]


def test_an_old_file_without_heights_still_loads(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "points": [
                    {
                        "name": p.name,
                        "court_xy": list(p.court_xy),
                        "image_xy": list(p.image_xy),
                        "is_control": p.is_control,
                    }
                    for p in _corner_points()
                ]
            }
        ),
        encoding="utf-8",
    )
    assert all(p.height == 0.0 for p in Calibration.load(path).points)
