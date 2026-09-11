import numpy as np
import pytest

from padel_analysis.analytics.trajectories import SLOTS, MatchTrajectories
from padel_analysis.pipeline.cache import FramePositions, PositionCache


def _cache(entries: dict[int, dict[str, tuple[float, float]]]) -> PositionCache:
    cache = PositionCache()
    for frame, positions in entries.items():
        cache.add(FramePositions(frame=frame, positions=positions))
    return cache


def _full(x: float, y: float) -> dict[str, tuple[float, float]]:
    return {
        "near_1": (-x, -y), "near_2": (x, -y),
        "far_1": (-x, y), "far_2": (x, y),
    }


def test_slots_are_the_four_of_the_tracker():
    assert SLOTS == ("near_1", "near_2", "far_1", "far_2")


def test_frames_are_sorted_and_complete():
    trajectories = MatchTrajectories.from_cache(
        _cache({2: _full(1, 5), 0: _full(1, 5)}), fps=30.0
    )
    np.testing.assert_array_equal(trajectories.frames, [0, 2])


def test_every_slot_gets_a_row_per_frame():
    trajectories = MatchTrajectories.from_cache(
        _cache({0: _full(1, 5), 1: _full(1, 5)}), fps=30.0
    )
    for slot in SLOTS:
        assert trajectories.positions[slot].shape == (2, 2)


def test_a_missing_player_becomes_nan_not_an_interpolation():
    cache = _cache({0: _full(1, 5), 1: {"near_1": (-1.0, -5.0)}, 2: _full(1, 5)})
    trajectories = MatchTrajectories.from_cache(cache, fps=30.0)
    far_1 = trajectories.positions["far_1"]
    assert np.isnan(far_1[1]).all()
    assert not np.isnan(far_1[0]).any()
    assert not np.isnan(far_1[2]).any()


def test_present_marks_the_frames_where_a_player_was_located():
    cache = _cache({0: _full(1, 5), 1: {"near_1": (-1.0, -5.0)}})
    trajectories = MatchTrajectories.from_cache(cache, fps=30.0)
    np.testing.assert_array_equal(trajectories.present("near_1"), [True, True])
    np.testing.assert_array_equal(trajectories.present("far_1"), [True, False])


def test_complete_mask_requires_all_four_players():
    cache = _cache({0: _full(1, 5), 1: {"near_1": (-1.0, -5.0)}, 2: _full(1, 5)})
    trajectories = MatchTrajectories.from_cache(cache, fps=30.0)
    np.testing.assert_array_equal(trajectories.complete_mask(), [True, False, True])


def test_depth_is_the_absolute_distance_to_the_net():
    cache = _cache({0: {"near_1": (0.0, -7.0), "near_2": (0.0, -2.0),
                        "far_1": (0.0, 3.0), "far_2": (0.0, 8.0)}})
    trajectories = MatchTrajectories.from_cache(cache, fps=30.0)
    assert trajectories.depth("near_1")[0] == pytest.approx(7.0)
    assert trajectories.depth("far_1")[0] == pytest.approx(3.0)


def test_side_of_a_slot_follows_its_name():
    trajectories = MatchTrajectories.from_cache(_cache({0: _full(1, 5)}), fps=30.0)
    assert trajectories.side("near_1") == -1
    assert trajectories.side("far_2") == +1


def test_duration_uses_the_frame_rate():
    cache = _cache({i: _full(1, 5) for i in range(60)})
    trajectories = MatchTrajectories.from_cache(cache, fps=30.0)
    assert trajectories.duration_seconds() == pytest.approx(2.0)


def test_an_empty_cache_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        MatchTrajectories.from_cache(PositionCache(), fps=30.0)
