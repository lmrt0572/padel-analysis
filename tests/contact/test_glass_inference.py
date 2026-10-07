import numpy as np
import pytest

from padel_analysis.contact.glass_inference import Touch, inferred_walls, wall_ahead

# stroke from the near side, bounce at the far end at 30 m/s, taken by the far player
STRIKE = Touch(0, "raquette", "near", (0.0, -8.0))
BOUNCE = Touch(15, "sol", "far", (0.0, 7.0))


def _peak_at(frame, length=60):
    probability = np.zeros(length)
    probability[frame] = 0.4
    return probability


def test_the_wall_ahead_is_the_back_glass_for_a_ball_going_deep():
    reach, meeting = wall_ahead((0.0, 7.0), (0.0, 1.0))
    assert reach == pytest.approx(3.0)
    assert meeting == pytest.approx((0.0, 10.0))


def test_the_wall_ahead_is_the_side_one_for_a_ball_going_across():
    reach, meeting = wall_ahead((3.0, 7.0), (1.0, 0.0))
    assert reach == pytest.approx(2.0)
    assert meeting == pytest.approx((5.0, 7.0))


def test_a_ball_too_slow_to_have_gone_straight_went_by_the_wall():
    answer = Touch(45, "raquette", "far", (0.0, 8.0))  # one metre in one second
    walls = inferred_walls([STRIKE, BOUNCE, answer], _peak_at(25), 0)
    assert [(w.frame, w.bounce, w.replaces_bounce) for w in walls] == [(25, 15, False)]


def test_a_ball_met_at_its_pace_went_straight():
    answer = Touch(20, "raquette", "far", (1.0, 9.0))
    assert inferred_walls([STRIKE, BOUNCE, answer], _peak_at(17), 0) == []


def test_a_wall_found_on_the_bounce_frame_means_the_bounce_was_the_wall():
    answer = Touch(45, "raquette", "far", (0.0, 8.0))
    walls = inferred_walls([STRIKE, BOUNCE, answer], _peak_at(16), 0)
    assert walls[0].replaces_bounce


def test_nothing_is_added_when_a_wall_was_already_found():
    answer = Touch(45, "raquette", "far", (0.0, 8.0))
    wall = Touch(25, "verre", None, None)
    assert inferred_walls([STRIKE, BOUNCE, wall, answer], _peak_at(25), 0) == []


def test_a_strike_from_the_half_of_the_bounce_is_not_the_one_that_sent_it():
    same_half = Touch(0, "raquette", "far", (0.0, 8.0))
    answer = Touch(45, "raquette", "far", (0.0, 8.0))
    assert inferred_walls([same_half, BOUNCE, answer], _peak_at(25), 0) == []
