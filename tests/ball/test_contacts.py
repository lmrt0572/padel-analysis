import pytest

from padel_analysis.ball.contacts import sharpness_of, turn_of, velocities


def test_velocities_reads_both_sides_of_a_frame():
    path = {0: (0.0, 0.0), 2: (10.0, 0.0), 4: (30.0, 0.0)}
    incoming, outgoing = velocities(path, 2, span=2)
    assert incoming == pytest.approx((10.0, 0.0))
    assert outgoing == pytest.approx((20.0, 0.0))


def test_velocities_needs_all_three_points():
    path = {0: (0.0, 0.0), 2: (10.0, 0.0)}
    assert velocities(path, 2, span=2) is None


def test_velocities_refuses_an_absent_neighbour():
    """Une frame sans position ne peut pas porter une vitesse."""
    path = {0: None, 2: (10.0, 0.0), 4: (30.0, 0.0)}
    assert velocities(path, 2, span=2) is None


def test_a_straight_line_does_not_turn():
    assert turn_of((10.0, 0.0), (10.0, 0.0)) == pytest.approx(0.0)


def test_turn_is_the_length_of_the_velocity_change():
    assert turn_of((10.0, 0.0), (10.0, 6.0)) == pytest.approx(6.0)


def test_a_full_reversal_turns_by_both_speeds():
    assert turn_of((10.0, 0.0), (-10.0, 0.0)) == pytest.approx(20.0)


def test_sharpness_is_scale_free():
    """Le meme demi-tour vaut pareil a vitesse lente et rapide."""
    slow = sharpness_of((5.0, 0.0), (-5.0, 0.0))
    fast = sharpness_of((50.0, 0.0), (-50.0, 0.0))
    assert slow == pytest.approx(fast)
    assert slow == pytest.approx(1.0)


def test_sharpness_of_a_straight_line_is_zero():
    assert sharpness_of((10.0, 0.0), (10.0, 0.0)) == pytest.approx(0.0)


def test_sharpness_refuses_to_divide_by_nothing():
    assert sharpness_of((0.0, 0.0), (0.0, 0.0)) == pytest.approx(0.0)
