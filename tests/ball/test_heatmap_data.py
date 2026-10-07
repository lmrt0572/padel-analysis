import numpy as np
import pytest

from padel_analysis.ball.heatmap_data import gaussian_target, stacked_indices


def test_the_stack_reaches_both_sides():
    assert stacked_indices(300, spacing=3) == [297, 300, 303]


def test_the_stack_follows_its_spacing():
    assert stacked_indices(300, spacing=2) == [298, 300, 302]


def test_the_target_peaks_on_the_ball():
    target = gaussian_target((32, 64), (10.0, 20.0), sigma=2.0)
    assert target.shape == (32, 64)
    assert np.unravel_index(target.argmax(), target.shape) == (20, 10)
    assert target.max() == pytest.approx(1.0)


def test_the_target_fades_with_distance():
    target = gaussian_target((32, 64), (10.0, 20.0), sigma=2.0)
    assert target[20, 10] > target[20, 12] > target[20, 16]


def test_a_ball_outside_the_frame_leaves_an_empty_target():
    assert gaussian_target((32, 64), (200.0, 200.0), sigma=2.0).max() < 1e-6


def test_no_ball_leaves_an_empty_target():
    """A frame without an annotated ball is not a frame without a ball: it teaches nothing."""
    assert gaussian_target((32, 64), None, sigma=2.0).max() == 0.0


def test_the_target_is_not_a_binary_mask():
    """A Gaussian target says where the centre is; a binary mask does not."""
    target = gaussian_target((32, 64), (10.0, 20.0), sigma=2.0)
    assert 0.0 < target[20, 12] < 1.0


def test_a_wider_sigma_spreads_further():
    narrow = gaussian_target((32, 64), (10.0, 20.0), sigma=1.0)
    wide = gaussian_target((32, 64), (10.0, 20.0), sigma=4.0)
    assert wide[20, 14] > narrow[20, 14]
