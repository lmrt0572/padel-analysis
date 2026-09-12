import numpy as np

from padel_analysis.geometry.court import Court


def test_court_is_symmetric_about_the_net():
    court = Court()
    assert court.half_length == court.length / 2
    assert court.half_width == court.width / 2


def test_service_line_lies_between_net_and_baseline():
    court = Court()
    assert 0 < court.service_line_distance < court.half_length


def test_reference_points_are_four_court_corners_in_order():
    court = Court()
    corners = court.corners()
    assert corners.shape == (4, 2)
    # ordre : arriere-gauche, arriere-droit, avant-droit, avant-gauche
    expected = np.array(
        [
            [-court.half_width, -court.half_length],
            [court.half_width, -court.half_length],
            [court.half_width, court.half_length],
            [-court.half_width, court.half_length],
        ]
    )
    np.testing.assert_allclose(corners, expected)


def test_offensive_zone_contains_a_point_near_the_net():
    court = Court()
    near_net = np.array([[0.0, 1.0]])
    assert court.is_in_offensive_zone(near_net, side=+1)[0]


def test_offensive_zone_excludes_a_point_behind_the_service_line():
    court = Court()
    deep = np.array([[0.0, court.service_line_distance + 1.0]])
    assert not court.is_in_offensive_zone(deep, side=+1)[0]


def test_the_side_wall_glass_leaves_mesh_in_the_middle():
    """Variante Crystal : du verre a chaque bout, du grillage entre les deux."""
    court = Court()
    assert court.side_wall_glass_length * 2 < court.length


def test_the_side_walls_are_lower_than_the_back_walls():
    court = Court()
    assert court.side_wall_total_height < court.back_wall_total_height
