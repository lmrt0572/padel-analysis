import math

import pytest

from padel_analysis.analytics.live_stats import LiveTimeline
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
