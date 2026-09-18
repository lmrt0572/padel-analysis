import math

import numpy as np
import pytest

from padel_analysis.analytics.rally import (
    RACKET_HEIGHT,
    Rally,
    RallyContact,
    after_each_shot,
    ball_speeds,
    impacts,
    movements,
    shots_by_player,
    summary,
)

FPS = 30.0


def _rally(contacts, positions=None, start=0, stop=299):
    return Rally(start=start, stop=stop, fps=FPS, contacts=tuple(contacts),
                 positions=positions or {})


def _hit(frame, player, point=None):
    return RallyContact(frame, "raquette", point, player)


def _bounce(frame, kind="sol", point=(0.0, 5.0, 0.0)):
    return RallyContact(frame, kind, point, None)


def test_a_rally_counts_the_strikes_of_each_player():
    rally = _rally([_hit(0, "near_1"), _bounce(15), _hit(30, "far_1"), _hit(60, "near_1")])
    assert shots_by_player(rally) == {"near_1": 2, "far_1": 1}


def test_after_a_shot_comes_what_the_ball_touched_next():
    rally = _rally([
        _hit(0, "near_1"), _bounce(15), _hit(30, "far_1"),
        _bounce(40, "verre", (5.0, -8.0, 1.0)), _hit(60, "near_1"),
    ])
    after = after_each_shot(rally)
    assert after["near_1"] == {"sol": 1, "fin": 1}
    assert after["far_1"] == {"verre": 1}


def test_impacts_are_the_contacts_placed_on_a_surface():
    rally = _rally([_hit(0, "near_1"), _bounce(15), _bounce(40, "verre", (5.0, -8.0, 1.0))])
    assert [(c.frame, c.kind) for c in impacts(rally)] == [(15, "sol"), (40, "verre")]


def test_a_contact_without_a_court_point_is_not_an_impact():
    rally = _rally([RallyContact(15, "sol", None, None)])
    assert impacts(rally) == []


def test_speed_between_two_bounces_is_their_distance_over_the_time():
    rally = _rally([_bounce(0, point=(0.0, 0.0, 0.0)), _bounce(30, point=(3.0, 4.0, 0.0))])
    [segment] = ball_speeds(rally)
    assert segment.metres_per_second == pytest.approx(5.0)
    assert segment.kmh == pytest.approx(18.0)
    assert not segment.estimated


def test_a_strike_is_placed_at_its_player_and_the_speed_is_marked_estimated():
    positions = {"near_1": {0: (0.0, -6.0)}}
    rally = _rally([_hit(0, "near_1"), _bounce(15, point=(0.0, 6.0, 0.0))], positions)
    [segment] = ball_speeds(rally)
    expected = math.dist((0.0, -6.0, RACKET_HEIGHT), (0.0, 6.0, 0.0)) / 0.5
    assert segment.metres_per_second == pytest.approx(expected)
    assert segment.estimated


def test_no_speed_is_given_when_an_end_cannot_be_placed():
    rally = _rally([_hit(0, "near_1"), _bounce(15)])  # position du joueur inconnue
    assert ball_speeds(rally) == []


def test_a_still_player_travels_nothing():
    positions = {"near_1": {f: (1.0, -5.0) for f in range(60)}}
    move = movements(_rally([], positions, stop=59))["near_1"]
    assert move.distance == pytest.approx(0.0)


def test_a_player_walking_one_metre_per_second_travels_that_far():
    positions = {"near_1": {f: (f / FPS, -5.0) for f in range(61)}}
    move = movements(_rally([], positions, stop=60))["near_1"]
    # Le lissage sur 9 images rogne environ quatre images de trajet a chaque bout.
    assert move.distance == pytest.approx(2.0, abs=0.15)


def test_a_player_close_to_the_net_spends_the_rally_there():
    positions = {"near_1": {f: (0.0, -3.0) for f in range(30)},
                 "far_1": {f: (0.0, 8.0) for f in range(30)}}
    moves = movements(_rally([], positions, stop=29))
    assert moves["near_1"].net_share == pytest.approx(1.0)
    assert moves["far_1"].net_share == pytest.approx(0.0)


def test_the_summary_tells_how_long_how_many_shots_and_how_it_ended():
    rally = _rally([
        _hit(0, "near_1"), _bounce(15), _hit(30, "far_1"),
        _bounce(40, "verre", (5.0, -8.0, 1.0)), _bounce(45, "grillage", (5.0, 0.0, 3.5)),
        _bounce(52, "sol"),
    ], start=0, stop=89)
    result = summary(rally)
    assert result.duration == pytest.approx(3.0)
    assert result.shots == 2
    assert result.walls == 2
    assert result.last == "sol"


def test_an_empty_rally_has_a_summary_too():
    result = summary(_rally([], stop=29))
    assert result.shots == 0 and result.walls == 0 and result.last is None


def test_contacts_are_kept_in_time_order_whatever_the_input_order():
    rally = _rally([_bounce(40), _hit(0, "near_1")])
    assert [c.frame for c in rally.contacts] == [0, 40]


def test_a_contact_time_is_counted_from_the_rally_start():
    rally = _rally([_bounce(115)], start=100, stop=200)
    assert rally.time_of(rally.contacts[0]) == pytest.approx(0.5)


def test_positions_outside_the_rally_are_ignored():
    positions = {"near_1": {**{f: (0.0, -5.0) for f in range(30)}, 500: (4.0, -9.0)}}
    move = movements(_rally([], positions, stop=29))["near_1"]
    assert move.distance == pytest.approx(0.0)
    np.testing.assert_allclose(move.path[-1], (0.0, -5.0))
