import numpy as np
import pytest

from padel_analysis.geometry.court import Court
from padel_analysis.geometry.projector import CourtProjector

# A plausible view from a high camera behind the negative back wall: the far end
# (positive y) appears higher in the picture, so at a smaller image y, and narrower
# horizontally.
NEAR_LEFT = (240.0, 940.0)
NEAR_RIGHT = (1680.0, 940.0)
FAR_RIGHT = (1180.0, 330.0)
FAR_LEFT = (740.0, 330.0)


@pytest.fixture
def projector() -> CourtProjector:
    court = Court()
    image_corners = np.array([NEAR_LEFT, NEAR_RIGHT, FAR_RIGHT, FAR_LEFT], dtype=np.float64)
    return CourtProjector.from_correspondences(court.corners(), image_corners)


def test_roundtrip_returns_the_original_court_points(projector):
    court_points = np.array([[0.0, 0.0], [3.0, -5.0], [-4.5, 8.0]])
    image_points = projector.court_to_image(court_points)
    back = projector.image_to_court(image_points)
    np.testing.assert_allclose(back, court_points, atol=1e-9)


def test_corners_map_to_the_correspondences_they_were_built_from(projector):
    court = Court()
    image_corners = projector.court_to_image(court.corners())
    expected = np.array([NEAR_LEFT, NEAR_RIGHT, FAR_RIGHT, FAR_LEFT])
    np.testing.assert_allclose(image_corners, expected, atol=1e-6)


def test_far_baseline_is_higher_in_the_image_than_the_near_one(projector):
    """Catches a flipped y axis, which a roundtrip test cannot detect."""
    near = projector.court_to_image(np.array([[0.0, -10.0]]))
    far = projector.court_to_image(np.array([[0.0, 10.0]]))
    assert far[0, 1] < near[0, 1]


def test_positive_x_is_to_the_right_in_the_image(projector):
    """Catches a flipped x axis, which a roundtrip test cannot detect."""
    left = projector.court_to_image(np.array([[-4.0, 0.0]]))
    right = projector.court_to_image(np.array([[4.0, 0.0]]))
    assert right[0, 0] > left[0, 0]


def test_net_centre_projects_inside_the_court_quadrilateral(projector):
    centre = projector.court_to_image(np.array([[0.0, 0.0]]))[0]
    xs = [NEAR_LEFT[0], NEAR_RIGHT[0], FAR_RIGHT[0], FAR_LEFT[0]]
    ys = [NEAR_LEFT[1], NEAR_RIGHT[1], FAR_RIGHT[1], FAR_LEFT[1]]
    assert min(xs) < centre[0] < max(xs)
    assert min(ys) < centre[1] < max(ys)


def test_reprojection_error_on_held_out_points_is_reported(projector):
    control_court = np.array([[0.0, 0.0], [2.0, 4.0]])
    control_image = projector.court_to_image(control_court)
    error = projector.reprojection_error(control_court, control_image)
    assert error.rmse_pixels == pytest.approx(0.0, abs=1e-6)
    assert error.rmse_metres == pytest.approx(0.0, abs=1e-6)
    assert error.median_pixels == pytest.approx(0.0, abs=1e-6)
    assert error.median_metres == pytest.approx(0.0, abs=1e-6)


def test_degenerate_correspondences_are_rejected():
    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    image = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    with pytest.raises(ValueError, match="homography"):
        CourtProjector.from_correspondences(collinear, image)
