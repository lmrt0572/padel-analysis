import cv2
import numpy as np
import pytest

from padel_analysis.geometry.camera import CameraPose, WallPlane, back_wall_planes
from padel_analysis.geometry.court import Court


@pytest.fixture
def synthetic_pose() -> tuple[CameraPose, np.ndarray, np.ndarray]:
    """A camera 7.6 m up, 6 m behind the negative baseline, looking down the court."""
    court = Court()
    image_size = (1920, 1080)
    focal = 1400.0
    intrinsics = np.array(
        [[focal, 0.0, image_size[0] / 2], [0.0, focal, image_size[1] / 2], [0.0, 0.0, 1.0]]
    )
    # Points 3D : les 4 coins au sol, plus les 2 coins hauts du mur de fond eloigne.
    h = court.back_wall_total_height
    corners = court.corners()
    object_points = np.array(
        [
            [corners[0][0], corners[0][1], 0.0],
            [corners[1][0], corners[1][1], 0.0],
            [corners[2][0], corners[2][1], 0.0],
            [corners[3][0], corners[3][1], 0.0],
            [corners[2][0], corners[2][1], h],
            [corners[3][0], corners[3][1], h],
        ]
    )
    # Rotation : camera regardant vers les y positifs, inclinee vers le bas.
    tilt = np.deg2rad(20.0)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -np.sin(tilt), -np.cos(tilt)],
            [0.0, np.cos(tilt), -np.sin(tilt)],
        ]
    )
    camera_centre = np.array([0.0, -court.half_length - 6.0, 7.6])
    translation = -rotation @ camera_centre

    rvec, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(object_points, rvec, translation, intrinsics, None)
    return (
        CameraPose(rotation=rotation, translation=translation, intrinsics=intrinsics),
        object_points,
        projected.reshape(-1, 2),
    )


def test_rotation_fixture_is_a_valid_rotation_matrix(synthetic_pose):
    truth, _, _ = synthetic_pose
    np.testing.assert_allclose(truth.rotation @ truth.rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(truth.rotation) == pytest.approx(1.0)


def test_pose_recovered_from_projections_matches_the_original(synthetic_pose):
    truth, object_points, image_points = synthetic_pose
    recovered = CameraPose.from_correspondences(
        object_points, image_points, truth.intrinsics
    )
    np.testing.assert_allclose(recovered.camera_centre, truth.camera_centre, atol=1e-3)


def test_projected_points_match_the_synthetic_ones(synthetic_pose):
    truth, object_points, image_points = synthetic_pose
    np.testing.assert_allclose(truth.project(object_points), image_points, atol=1e-6)


def test_camera_centre_is_above_the_ground_and_behind_the_baseline(synthetic_pose):
    truth, _, _ = synthetic_pose
    court = Court()
    assert truth.camera_centre[2] > 0
    assert truth.camera_centre[1] < -court.half_length


def test_back_wall_planes_are_vertical_and_at_the_baselines():
    court = Court()
    planes = back_wall_planes(court)
    assert len(planes) == 2
    for plane in planes:
        assert isinstance(plane, WallPlane)
        # Normale horizontale : une paroi verticale a une normale sans composante z.
        assert plane.normal[2] == pytest.approx(0.0)
        assert abs(plane.offset) == pytest.approx(court.half_length)
        assert plane.height == court.back_wall_total_height


def test_wall_top_edge_projects_above_its_ground_edge(synthetic_pose):
    truth, _, _ = synthetic_pose
    court = Court()
    ground = np.array([[0.0, court.half_length, 0.0]])
    top = np.array([[0.0, court.half_length, court.back_wall_total_height]])
    assert truth.project(top)[0, 1] < truth.project(ground)[0, 1]
