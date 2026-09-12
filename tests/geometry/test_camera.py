import numpy as np
import pytest

from padel_analysis.geometry.camera import CameraPose, Surface, court_surfaces
from padel_analysis.geometry.court import Court


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


def test_the_back_walls_are_vertical_and_at_the_baselines():
    court = Court()
    walls = [s for s in court_surfaces(court) if s.name.startswith("back_wall")]
    assert len(walls) == 2
    for wall in walls:
        assert isinstance(wall, Surface)
        # Normale horizontale : une paroi verticale a une normale sans composante z.
        assert wall.normal[2] == pytest.approx(0.0)
        assert abs(wall.offset) == pytest.approx(court.half_length)


def test_wall_top_edge_projects_above_its_ground_edge(synthetic_pose):
    truth, _, _ = synthetic_pose
    court = Court()
    ground = np.array([[0.0, court.half_length, 0.0]])
    top = np.array([[0.0, court.half_length, court.back_wall_total_height]])
    assert truth.project(top)[0, 1] < truth.project(ground)[0, 1]


def test_a_ray_starts_at_the_camera(synthetic_pose):
    pose, _, _ = synthetic_pose
    origin, _ = pose.ray((960.0, 540.0))
    np.testing.assert_allclose(origin, pose.camera_centre, atol=1e-9)


def test_a_ray_through_a_projected_point_returns_to_it(synthetic_pose):
    pose, _, _ = synthetic_pose
    point = np.array([2.0, -3.0, 1.5])
    origin, direction = pose.ray(pose.project(point.reshape(1, 3))[0])
    steps = (point[2] - origin[2]) / direction[2]
    np.testing.assert_allclose(origin + steps * direction, point, atol=1e-6)


def test_a_ray_meets_the_ground_where_the_point_was(synthetic_pose):
    pose, _, _ = synthetic_pose
    floor = court_surfaces(Court())[0]
    assert floor.name == "floor"
    pixel = pose.project(np.array([[1.0, 2.0, 0.0]]))[0]
    np.testing.assert_allclose(floor.intersect(*pose.ray(pixel)), [1.0, 2.0, 0.0], atol=1e-6)


def test_a_plane_the_ray_runs_along_is_never_met():
    floor = court_surfaces(Court())[0]
    assert floor.intersect(np.array([0.0, 0.0, 5.0]), np.array([1.0, 0.0, 0.0])) is None


def test_a_surface_behind_the_camera_is_never_met(synthetic_pose):
    pose, _, _ = synthetic_pose
    floor = court_surfaces(Court())[0]
    upwards = np.array([0.0, 0.0, 1.0])
    assert floor.intersect(pose.camera_centre, upwards) is None


def test_there_are_five_surfaces_and_they_are_named():
    assert [s.name for s in court_surfaces(Court())] == [
        "floor",
        "back_wall_negative_y",
        "back_wall_positive_y",
        "side_wall_negative_x",
        "side_wall_positive_x",
    ]


def test_a_point_outside_the_court_is_not_on_the_floor():
    floor = court_surfaces(Court())[0]
    assert floor.contains(np.array([0.0, 0.0, 0.0]))
    assert not floor.contains(np.array([0.0, 12.0, 0.0]))


def test_a_point_above_a_wall_is_not_on_it():
    wall = court_surfaces(Court())[1]
    assert wall.contains(np.array([0.0, -10.0, 2.0]))
    assert not wall.contains(np.array([0.0, -10.0, 5.0]))


def test_the_back_wall_is_glass_below_three_metres():
    wall = court_surfaces(Court())[1]
    assert wall.material_at(np.array([0.0, -10.0, 2.0])) == "verre"
    assert wall.material_at(np.array([0.0, -10.0, 3.5])) == "grillage"


def test_the_side_wall_is_glass_near_each_end():
    wall = court_surfaces(Court())[3]
    assert wall.material_at(np.array([-5.0, -9.0, 1.0])) == "verre"
    assert wall.material_at(np.array([-5.0, 9.0, 1.0])) == "verre"
    assert wall.material_at(np.array([-5.0, 0.0, 1.0])) == "grillage"


def test_the_floor_has_no_material():
    assert court_surfaces(Court())[0].material_at(np.array([0.0, 0.0, 0.0])) is None
