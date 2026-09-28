import math

import pytest

from padel_analysis.analytics.live_stats import LiveTimeline, PointOutcome, volley_flags
from padel_analysis.analytics.rally import Rally, RallyContact


def _rally():
    positions = {
        "near_1": {f: (f / 30.0, -3.0) for f in range(91)},  # au filet, 1 m/s
        "far_1": {f: (0.0, 8.0) for f in range(91)},  # au fond, immobile
    }
    contacts = (
        RallyContact(10, "raquette", None, "near_1"),
        RallyContact(25, "sol", (0.0, 6.0, 0.0), None),
        RallyContact(40, "raquette", None, "far_1"),
        RallyContact(55, "verre", (5.0, -8.0, 1.0), None),
        RallyContact(70, "raquette", None, "near_1"),
    )
    return Rally(0, 90, 30.0, contacts, positions)


def test_nothing_is_counted_before_it_happens():
    stats = LiveTimeline(_rally()).at(9)
    assert stats.shots == 0 and stats.walls == 0
    assert all(line.shots == 0 for line in stats.players)


def test_shots_and_walls_add_up_as_the_rally_goes():
    timeline = LiveTimeline(_rally())
    assert timeline.at(40).shots == 2
    assert timeline.at(60).walls == 1
    end = timeline.at(90)
    assert {line.slot: line.shots for line in end.players}["near_1"] == 2
    assert end.pair_shots == {"proche": 2, "fond": 1}


def test_the_last_contact_is_shown_for_a_moment_only():
    timeline = LiveTimeline(_rally(), recent=0.2)  # 6 images
    assert timeline.at(56).last == "verre"
    assert timeline.at(62).last is None


def test_distance_grows_with_the_player_who_moves_and_not_the_other():
    stats = LiveTimeline(_rally()).at(90)
    lines = {line.slot: line for line in stats.players}
    assert lines["near_1"].distance == pytest.approx(3.0, abs=0.2)
    assert lines["far_1"].distance == pytest.approx(0.0)


def test_net_share_splits_the_player_at_the_net_from_the_one_at_the_back():
    stats = LiveTimeline(_rally()).at(90)
    lines = {line.slot: line for line in stats.players}
    assert lines["near_1"].net_share == pytest.approx(1.0)
    assert lines["far_1"].net_share == pytest.approx(0.0)
    assert stats.pair_net["proche"] == pytest.approx(1.0)


def test_a_player_never_seen_has_no_net_share():
    stats = LiveTimeline(_rally()).at(90)
    near_2 = {line.slot: line for line in stats.players}["near_2"]
    assert math.isnan(near_2.net_share) and near_2.distance == 0.0


def test_elapsed_time_counts_from_the_start_and_stops_at_the_end():
    timeline = LiveTimeline(_rally())
    assert timeline.at(45).elapsed == pytest.approx(1.5)
    assert timeline.at(500).elapsed == pytest.approx(3.0)


def test_a_strike_with_no_bounce_since_the_last_one_is_a_volley():
    rally = Rally(0, 90, 30.0, (
        RallyContact(0, "raquette", None, "near_1"),
        RallyContact(10, "sol", (0.0, 6.0, 0.0), None),
        RallyContact(20, "raquette", None, "far_1"),  # apres rebond
        RallyContact(30, "verre", (5.0, -9.0, 1.0), None),
        RallyContact(40, "raquette", None, "near_1"),  # vitre sans sol : volee
    ))
    assert volley_flags(rally) == {0: None, 20: False, 40: True}
    lines = {line.slot: line for line in LiveTimeline(rally).at(90).players}
    assert (lines["near_1"].volleys, lines["near_1"].after_bounce) == (1, 0)
    assert (lines["far_1"].volleys, lines["far_1"].after_bounce) == (0, 1)


def test_top_running_speed_follows_the_player_and_ignores_a_tracking_jump():
    positions = {"near_1": {f: (f * 5.0 / 30.0, -6.0) for f in range(61)}}  # 5 m/s
    positions["near_1"][30] = (9.0, 9.0)  # un saut du suivi
    rally = Rally(0, 60, 30.0, (), positions)
    top = {line.slot: line for line in LiveTimeline(rally).at(60).players}["near_1"].top_speed
    assert top == pytest.approx(18.0, abs=1.5)


def test_a_shot_speed_is_known_once_the_ball_has_landed():
    positions = {"near_1": {f: (0.0, -6.0) for f in range(61)}}
    rally = Rally(0, 60, 30.0, (
        RallyContact(0, "raquette", None, "near_1"),
        RallyContact(30, "sol", (0.0, 6.0, 0.0), None),
    ), positions)
    timeline = LiveTimeline(rally)
    assert timeline.at(29).last_shot_speed is None
    expected = math.dist((0.0, -6.0, 1.0), (0.0, 6.0, 0.0)) * 3.6
    assert timeline.at(30).last_shot_speed == pytest.approx(expected)


def test_the_minimap_trail_ends_on_the_current_position():
    stats = LiveTimeline(_rally()).at(60)
    assert stats.positions["near_1"][-1] == pytest.approx((2.0, -3.0), abs=0.01)
    assert len(stats.positions["near_1"]) <= 46


def test_a_short_jump_does_not_count_as_a_sprint():
    """Un saut leve les chevilles : le point au sol recule d'un metre en 0,2 s."""
    track = {f: (0.0, 6.0) for f in range(91)}
    for f in range(40, 46):
        track[f] = (0.0, 6.0 + 0.2 * (f - 39))
    rally = Rally(0, 90, 30.0, (), {"far_1": track})
    top = {line.slot: line for line in LiveTimeline(rally).at(90).players}["far_1"].top_speed
    assert top < 5.0


def test_the_panel_follows_rallies_opened_by_splices():
    contacts = (
        RallyContact(5, "raquette", None, "near_1"), RallyContact(15, "sol", (0, 5, 0), None),
        RallyContact(25, "raquette", None, "far_1"), RallyContact(35, "raquette", None, "near_1"),
        RallyContact(60, "raquette", None, "far_1"), RallyContact(70, "raquette", None, "near_1"),
    )
    timeline = LiveTimeline(Rally(0, 90, 30.0, contacts, {}), splices=[50])
    during_first = timeline.at(30)
    assert (during_first.rally_number, during_first.rally_shots) == (1, 2)
    during_second = timeline.at(65)
    assert (during_second.rally_number, during_second.rally_shots) == (2, 1)
    assert during_second.longest_rally == 3


def test_without_splices_the_whole_clip_is_one_rally():
    stats = LiveTimeline(_rally()).at(90)
    assert stats.rally_number == 1 and stats.rally_shots == stats.shots


def test_points_are_credited_once_over_and_not_before():
    outcomes = [PointOutcome(40, "near", "near_1", "gagnant"),
                PointOutcome(80, "near", "far_1", "faute")]
    timeline = LiveTimeline(_rally(), points=outcomes)
    assert timeline.at(39).pair_points == {"proche": 0, "fond": 0}
    end = timeline.at(90)
    assert end.pair_points == {"proche": 2, "fond": 0}
    lines = {line.slot: line for line in end.players}
    assert (lines["near_1"].winners, lines["far_1"].errors) == (1, 1)


def test_without_a_score_read_no_points_are_shown():
    assert LiveTimeline(_rally()).at(90).pair_points is None
