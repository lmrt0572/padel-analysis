from padel_analysis.demo import ball_near


def test_the_ball_at_the_contact_frame_is_taken_first():
    assert ball_near(10, {10: (5.0, 5.0), 11: (6.0, 6.0)}, {}, 2) == (5.0, 5.0)


def test_a_ball_dropped_at_the_turn_is_taken_from_the_nearest_frame():
    shown = {7: (1.0, 1.0), 11: (2.0, 2.0)}
    assert ball_near(10, shown, {}, 2) == (2.0, 2.0)


def test_the_raw_path_fills_in_before_a_farther_displayed_ball():
    assert ball_near(10, {12: (2.0, 2.0)}, {10: (3.0, 3.0)}, 2) == (3.0, 3.0)


def test_no_ball_within_reach_leaves_the_contact_undrawn():
    assert ball_near(10, {13: (2.0, 2.0)}, {}, 2) is None
