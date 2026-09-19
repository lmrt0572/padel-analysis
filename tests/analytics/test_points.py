from padel_analysis.analytics.points import point_winner
from padel_analysis.io.scoreboard import ScoreState


def _s(points, games=(3, 1), set_number=1):
    return ScoreState(set_number, games, points, server=1)


def test_a_point_step_names_the_pair_that_won_it():
    assert point_winner(_s(("15", "0")), _s(("30", "0"))) == 1
    assert point_winner(_s(("15", "0")), _s(("15", "15"))) == 2


def test_a_won_game_names_its_winner_and_resets_the_points():
    assert point_winner(_s(("40", "30")), _s(("0", "0"), games=(4, 1))) == 1
    assert point_winner(_s(("40", "40")), _s(("0", "0"), games=(3, 2))) == 2


def test_a_game_won_without_the_points_resetting_is_a_misreading():
    assert point_winner(_s(("40", "30")), _s(("15", "0"), games=(4, 1))) is None


def test_two_points_between_readings_are_left_undecided():
    assert point_winner(_s(("0", "0")), _s(("30", "0"))) is None
    assert point_winner(_s(("15", "0")), _s(("30", "15"))) is None


def test_the_same_score_twice_has_no_winner():
    assert point_winner(_s(("30", "15")), _s(("30", "15"))) is None


def test_a_won_set_goes_to_the_pair_that_led_it():
    assert point_winner(_s(("40", "0"), games=(5, 3)), _s(("0", "0"), games=(0, 0), set_number=2)) == 1
