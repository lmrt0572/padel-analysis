import math

import pytest

from padel_analysis.eval.contact_metrics import (
    bounces_between,
    chance_precision,
    interval_coverage,
)
from padel_analysis.eval.shots import ShotEvent


def _events():
    return [ShotEvent(10, 14, "Forehand"), ShotEvent(50, 54, "Backhand")]


def test_coverage_is_the_share_of_frames_inside_an_interval():
    assert interval_coverage(list(range(100)), _events()) == pytest.approx(0.10)


def test_coverage_ignores_frames_nobody_annotated():
    assert interval_coverage([10, 11, 12], _events()) == pytest.approx(1.0)


def test_coverage_of_nothing_is_not_a_number():
    assert math.isnan(interval_coverage([], _events()))


def test_bounces_counts_what_falls_between_two_shots():
    assert bounces_between([20, 30, 40], _events(), max_gap=120) == {3: 1}


def test_bounces_ignores_a_gap_too_long_to_be_a_rally():
    assert bounces_between([20], _events(), max_gap=10) == {}


def test_bounces_of_a_single_shot_has_no_pair():
    assert bounces_between([20], [ShotEvent(10, 14, "Serve")], max_gap=120) == {}


def test_bounces_does_not_count_a_contact_inside_a_shot():
    """A contact inside the annotated interval is the stroke, not a bounce."""
    assert bounces_between([12, 30], _events(), max_gap=120) == {1: 1}


def test_chance_precision_matches_the_coverage_it_draws_from():
    """Un tirage aleatoire touche les intervalles a hauteur de leur couverture."""
    rate = chance_precision(list(range(100)), _events(), count=20, draws=200, seed=0)
    assert rate == pytest.approx(0.10, abs=0.04)


def test_chance_precision_is_one_when_every_frame_is_covered():
    frames = list(range(10, 15))
    assert chance_precision(frames, _events(), count=3, draws=10, seed=0) == 1.0


def test_chance_precision_refuses_to_draw_more_than_it_has():
    assert math.isnan(chance_precision([1, 2], _events(), count=5, draws=10, seed=0))
