import numpy as np
import pytest

from padel_analysis.analytics.basic_stats import (
    distance_travelled,
    mean_position,
    smooth_positions,
    speed_percentile,
    step_distances,
)


def _line(n: int, step: float) -> np.ndarray:
    """A player moving in a straight line by `step` metres per frame."""
    xs = np.arange(n, dtype=np.float64) * step
    return np.stack([xs, np.zeros(n)], axis=1)


def test_step_distances_measure_each_consecutive_move():
    positions = _line(4, 0.5)
    frames = np.arange(4)
    np.testing.assert_allclose(step_distances(frames, positions), [0.5, 0.5, 0.5])


def test_a_gap_in_the_frames_is_never_bridged():
    """The main trap: linking two positions separated by a gap."""
    frames = np.array([0, 1, 50, 51])
    positions = np.array([[0.0, 0.0], [0.5, 0.0], [30.0, 0.0], [30.5, 0.0]])
    np.testing.assert_allclose(step_distances(frames, positions), [0.5, 0.5])


def test_an_absent_position_breaks_the_chain():
    frames = np.arange(4)
    positions = np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 0.0], [2.5, 0.0]])
    np.testing.assert_allclose(step_distances(frames, positions), [0.5])


def test_distance_travelled_sums_the_steps():
    assert distance_travelled(np.arange(5), _line(5, 0.4)) == pytest.approx(1.6)


def test_distance_of_a_motionless_player_is_zero():
    positions = np.zeros((10, 2))
    assert distance_travelled(np.arange(10), positions) == pytest.approx(0.0)


def test_smoothing_reduces_the_distance_of_a_noisy_still_player():
    """Noise inflates the distance: that is the effect we want to be able to quantify."""
    rng = np.random.default_rng(0)
    frames = np.arange(300)
    positions = rng.normal(0.0, 0.05, size=(300, 2))
    raw = distance_travelled(frames, positions)
    smoothed = distance_travelled(frames, smooth_positions(positions, window=9))
    assert raw > 5.0
    assert smoothed < raw / 2


def test_smoothing_preserves_a_genuine_straight_move():
    positions = _line(100, 0.1)
    smoothed = smooth_positions(positions, window=9)
    raw_distance = distance_travelled(np.arange(100), positions)
    smoothed_distance = distance_travelled(np.arange(100), smoothed)
    assert smoothed_distance == pytest.approx(raw_distance, rel=0.08)


def test_smoothing_leaves_gaps_as_gaps():
    positions = np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 0.0]])
    smoothed = smooth_positions(positions, window=3)
    assert np.isnan(smoothed[1]).all()


def test_speed_percentile_converts_metres_per_frame_to_per_second():
    frames = np.arange(50)
    positions = _line(50, 0.1)  # 0.1 m par frame a 30 fps = 3 m/s
    assert speed_percentile(frames, positions, fps=30.0, percentile=50) == pytest.approx(3.0)


def test_speed_percentile_is_robust_to_a_single_spike():
    """The raw maximum would be driven by a single outlier; the p95 is not."""
    frames = np.arange(100)
    positions = _line(100, 0.05)
    positions[50] += 5.0
    p95 = speed_percentile(frames, positions, fps=30.0, percentile=95)
    assert p95 < 10.0


def test_mean_position_ignores_absent_frames():
    positions = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]])
    np.testing.assert_allclose(mean_position(positions), [2.0, 3.0])


def test_mean_position_of_an_entirely_absent_player_is_nan():
    positions = np.full((3, 2), np.nan)
    assert np.isnan(mean_position(positions)).all()


def test_a_track_shorter_than_the_window_is_smoothed_over_its_own_length():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    smoothed = smooth_positions(positions, window=9)
    assert smoothed.shape == positions.shape
    assert smoothed[1] == pytest.approx([1.0, 0.0])
