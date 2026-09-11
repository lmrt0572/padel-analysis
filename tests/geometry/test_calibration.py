import json

import numpy as np
import pytest

from padel_analysis.geometry.calibration import (
    Calibration,
    CalibrationPoint,
    calibration_from_points,
)
from padel_analysis.geometry.court import Court

# Memes correspondances que dans test_projector, pour rester coherent.
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
    # Un cinquieme point, marque comme point de controle.
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
