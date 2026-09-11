import numpy as np
import pytest

from padel_analysis.analytics.net_control import (
    HYSTERESIS,
    NET_THRESHOLD,
    NetControl,
    at_net_states,
    net_control,
)
from padel_analysis.analytics.trajectories import MatchTrajectories
from padel_analysis.pipeline.cache import FramePositions, PositionCache


def _trajectories(rows: list[dict[str, tuple[float, float]]]) -> MatchTrajectories:
    cache = PositionCache()
    for frame, positions in enumerate(rows):
        cache.add(FramePositions(frame=frame, positions=positions))
    return MatchTrajectories.from_cache(cache, fps=30.0)


def _row(near_depth: float, far_depth: float) -> dict[str, tuple[float, float]]:
    return {
        "near_1": (-1.0, -near_depth), "near_2": (1.0, -near_depth),
        "far_1": (-1.0, far_depth), "far_2": (1.0, far_depth),
    }


def test_threshold_sits_in_the_measured_valley():
    """3,5-4,5 m en attaque, 6,5-9 m en defense : le seuil doit separer les deux."""
    assert 4.5 < NET_THRESHOLD < 6.5


def test_the_default_hysteresis_is_not_zero():
    """Sans cette garde, mettre la constante a zero ne casserait aucun test :
    les autres passent leur propre valeur en argument."""
    assert 0.0 < HYSTERESIS < 1.0


def test_a_player_closer_than_the_threshold_is_at_the_net():
    depths = np.array([3.0, 3.0, 3.0])
    np.testing.assert_array_equal(at_net_states(depths), [True, True, True])


def test_a_player_deeper_than_the_threshold_is_not():
    depths = np.array([8.0, 8.0, 8.0])
    np.testing.assert_array_equal(at_net_states(depths), [False, False, False])


def test_hysteresis_prevents_flicker_around_the_line():
    """Un joueur qui oscille sur le seuil ne doit pas changer d'etat a chaque frame."""
    depths = np.array([3.0, 5.4, 5.6, 5.4, 5.6, 5.4])
    states = at_net_states(depths, hysteresis=0.4)
    assert states.tolist() == [True, True, True, True, True, True]


def test_hysteresis_still_allows_a_genuine_transition():
    depths = np.array([3.0, 3.0, 7.0, 8.0, 9.0])
    states = at_net_states(depths, hysteresis=0.4)
    assert states.tolist() == [True, True, False, False, False]


def test_an_absent_depth_keeps_the_previous_state():
    depths = np.array([3.0, np.nan, 3.0])
    states = at_net_states(depths)
    assert states.tolist() == [True, True, True]


def test_a_team_controls_when_it_is_up_and_the_other_is_back():
    result = net_control(_trajectories([_row(3.0, 8.0)] * 10))
    assert isinstance(result, NetControl)
    assert result.near_percent == pytest.approx(100.0)
    assert result.far_percent == pytest.approx(0.0)


def test_both_teams_up_counts_as_contested():
    result = net_control(_trajectories([_row(3.0, 3.0)] * 10))
    assert result.near_percent == pytest.approx(0.0)
    assert result.far_percent == pytest.approx(0.0)
    assert result.contested_percent == pytest.approx(100.0)


def test_both_teams_back_counts_as_contested():
    result = net_control(_trajectories([_row(8.0, 8.0)] * 10))
    assert result.contested_percent == pytest.approx(100.0)


def test_one_player_up_and_one_back_is_not_control():
    rows = [{
        "near_1": (-1.0, -3.0), "near_2": (1.0, -8.0),
        "far_1": (-1.0, 8.0), "far_2": (1.0, 8.0),
    }] * 10
    result = net_control(_trajectories(rows))
    assert result.near_percent == pytest.approx(0.0)


def test_incomplete_frames_are_excluded_from_the_denominator():
    rows = [_row(3.0, 8.0)] * 8 + [{"near_1": (-1.0, -3.0)}] * 2
    result = net_control(_trajectories(rows))
    assert result.evaluated_frames == 8
    assert result.skipped_frames == 2
    assert result.near_percent == pytest.approx(100.0)


def test_the_three_percentages_sum_to_one_hundred():
    rows = [_row(3.0, 8.0)] * 5 + [_row(8.0, 3.0)] * 3 + [_row(3.0, 3.0)] * 2
    result = net_control(_trajectories(rows))
    total = result.near_percent + result.far_percent + result.contested_percent
    assert total == pytest.approx(100.0)


def test_timeline_has_one_entry_per_evaluated_frame():
    rows = [_row(3.0, 8.0)] * 5 + [{"near_1": (-1.0, -3.0)}] * 2
    result = net_control(_trajectories(rows))
    assert result.timeline.shape == (5,)


def test_a_run_with_no_complete_frame_reports_nothing_rather_than_crashing():
    rows = [{"near_1": (-1.0, -3.0)}] * 5
    result = net_control(_trajectories(rows))
    assert result.evaluated_frames == 0
    assert np.isnan(result.near_percent)
