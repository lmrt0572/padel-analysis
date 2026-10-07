import numpy as np

from padel_analysis.ball.smoothing import despike, smooth_path


def test_a_lone_spike_is_removed():
    path = {f: (float(f) * 10, 0.0) for f in range(10)}
    path[5] = (50.0, 90.0)
    assert despike(path, max_deviation=25.0)[5] is None
    assert despike(path, max_deviation=25.0)[4] == (40.0, 0.0)


def test_a_smooth_curve_keeps_all_its_points():
    path = {f: (float(f) * 10, float(f) ** 2) for f in range(10)}
    assert all(p is not None for p in despike(path, max_deviation=25.0).values())


def test_smoothing_brings_noisy_points_closer_to_the_truth():
    rng = np.random.default_rng(0)
    truth = {f: (10.0 * f, 5.0 * f) for f in range(40)}
    noisy = {f: (x + rng.normal(0, 3), y + rng.normal(0, 3)) for f, (x, y) in truth.items()}
    smooth = smooth_path(noisy, cuts=[], max_gap=3)

    def error(path):
        return np.mean([np.hypot(path[f][0] - truth[f][0], path[f][1] - truth[f][1])
                        for f in range(5, 35)])

    assert error(smooth) < error(noisy)


def test_a_cut_keeps_the_corner_of_a_bounce():
    """A continuous smoothing would round the bounce; cut at the contact, it keeps it."""
    path = {f: (10.0 * f, 0.0) for f in range(21)}
    path.update({f: (200.0 - 10.0 * (f - 20), 0.0) for f in range(21, 41)})
    cut = smooth_path(path, cuts=[20], max_gap=3)
    uncut = smooth_path(path, cuts=[], max_gap=3)
    assert abs(cut[20][0] - 200.0) < abs(uncut[20][0] - 200.0)


def test_short_gaps_are_filled_and_long_ones_left_open():
    path = {f: (10.0 * f, 0.0) for f in range(30)}
    for f in (10, 11):
        path[f] = None
    for f in range(18, 26):
        path[f] = None
    smooth = smooth_path(path, cuts=[], max_gap=3)
    assert smooth[10] is not None and smooth[11] is not None
    assert all(smooth[f] is None for f in range(18, 26))
