import numpy as np
import pytest

from padel_analysis.eval.airborne import airborne_mask, ground_projection_bias


def test_a_standing_player_is_never_airborne():
    heights = np.full(100, 80.0)
    assert not airborne_mask(heights).any()


def test_a_jump_is_detected_as_a_rise_above_the_local_baseline():
    heights = np.full(100, 80.0)
    heights[50:55] = 55.0  # ankles clearly higher in the picture
    mask = airborne_mask(heights, window=31, tolerance=10.0)
    assert mask[50:55].all()
    assert not mask[:40].any()


def test_a_slow_drift_is_not_mistaken_for_a_jump():
    """A player stepping back moves away, so looks higher: it is not a jump."""
    heights = np.linspace(80.0, 60.0, 200)
    mask = airborne_mask(heights, window=31, tolerance=10.0)
    assert mask.sum() < 10


def test_absent_heights_are_never_airborne():
    heights = np.full(50, np.nan)
    assert not airborne_mask(heights).any()


def test_the_bias_formula_matches_the_geometry():
    """Un point a hauteur h se projette a d * H / (H - h) de l'aplomb camera."""
    bias = ground_projection_bias(distance_m=15.0, height_m=0.4, camera_height_m=7.6)
    assert bias == pytest.approx(15.0 * 7.6 / 7.2 - 15.0, rel=1e-6)


def test_a_player_on_the_ground_has_no_bias():
    assert ground_projection_bias(15.0, 0.0, 7.6) == pytest.approx(0.0)


def test_the_bias_grows_with_distance_from_the_camera():
    near = ground_projection_bias(8.0, 0.4, 7.6)
    far = ground_projection_bias(20.0, 0.4, 7.6)
    assert far > near
