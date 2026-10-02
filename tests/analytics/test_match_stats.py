import pytest

from padel_analysis.analytics.match_stats import (
    Orientation,
    PairStats,
    contact_stats,
    movement_stats,
    orient,
    point_stats,
    stretch_points,
)
from padel_analysis.analytics.segmentation import RallySpan
from padel_analysis.io.scoreboard import ScoreState


def _state(games, points, server=1, set_number=1):
    return ScoreState(set_number, games, points, server)


def _pairs():
    return {1: PairStats(1), 2: PairStats(2)}


STILL = Orientation((), "near", 1, 0)


def test_the_serves_vote_where_the_top_row_starts():
    readings = [(0, _state((0, 0), ("0", "0"), server=1)),
                (100, _state((0, 0), ("15", "0"), server=1)),
                (200, _state((1, 0), ("0", "0"), server=2))]
    # Le service de la ligne du haut part du fond ; apres le premier jeu, on change de
    # cote, et la ligne du bas sert depuis le fond, donc la ligne du haut est proche.
    serves = {0: "far", 100: "far", 200: "far"}
    found = orient(readings, serves)
    assert found.changes == (200,)
    assert found.first_half == "far"
    assert (found.votes_for, found.votes_against) == (3, 0)
    assert found.half_of(1, 250) == "near" and found.half_of(2, 250) == "far"


def test_the_board_settles_who_won_each_point():
    readings = [(0, _state((0, 0), ("0", "0"))), (100, _state((0, 0), ("15", "0"))),
                (200, _state((0, 0), ("15", "15"))), (300, _state((0, 0), ("15", "15")))]
    assert stretch_points(readings) == {0: 1, 100: 2}
    pairs = _pairs()
    point_stats(stretch_points(readings), pairs)
    assert (pairs[1].points_won, pairs[2].points_won) == (1, 1)


def test_strikes_are_split_into_volleys_and_shots_after_a_bounce_or_the_glass():
    rally = RallySpan(0, 100, ((10, "raquette"), (20, "sol"), (30, "raquette"),
                               (40, "raquette"), (50, "sol"), (55, "verre"), (60, "raquette")))
    strikers = {10: "near_1", 30: "far_1", 40: "near_2", 60: "far_2"}
    pairs = _pairs()
    contact_stats([rally], strikers, STILL, {0: 2}, pairs)
    near, far = pairs[1], pairs[2]
    assert (near.strikes, near.volleys, near.after_bounce) == (2, 1, 0)
    assert (far.strikes, far.after_bounce, far.after_glass) == (2, 2, 1)
    assert far.won_by_length == {"4 à 7 coups": 1}


def test_distance_is_never_joined_across_a_splice_and_time_at_the_net_is_counted():
    positions = {f: {"near_1": (f / 30.0, -3.0), "far_1": (0.0, 8.0)} for f in range(61)}
    pairs = _pairs()
    movement_stats(positions, [30], STILL, pairs, window=1)
    assert pairs[1].distance_m == pytest.approx(2.0 - 1 / 30.0, abs=1e-6)
    assert pairs[1].net_share == pytest.approx(1.0)
    assert pairs[2].distance_m == pytest.approx(0.0) and pairs[2].net_share == 0.0


def test_a_pair_is_folded_onto_the_near_half_whichever_end_it_plays():
    from padel_analysis.analytics.match_stats import pair_positions

    changes = Orientation((100,), "near", 1, 0)
    positions = {50: {"near_1": (1.0, -6.0)}, 150: {"far_1": (1.0, 6.0)}}
    folded = pair_positions(positions, changes)
    assert folded[1].tolist() == [[1.0, -6.0], [-1.0, -6.0]]
    assert folded[2].shape == (0, 2)
