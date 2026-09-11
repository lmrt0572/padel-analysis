import numpy as np
import pytest

pytest.importorskip("motmetrics")

from padel_analysis.eval.tracking_metrics import TrackingAccumulator, TrackingScore


def _box(x, y):
    return np.array([x, y, x + 40, y + 100], dtype=np.float64)


def test_a_perfect_tracker_scores_one():
    accumulator = TrackingAccumulator()
    for _ in range(10):
        accumulator.add(
            truth={"near_1": _box(0, 0), "near_2": _box(200, 0)},
            hypothesis={"a": _box(0, 0), "b": _box(200, 0)},
        )
    score = accumulator.score()
    assert isinstance(score, TrackingScore)
    assert score.mota == pytest.approx(1.0)
    assert score.idf1 == pytest.approx(1.0)
    assert score.id_switches == 0


def test_a_swap_is_counted_as_an_identity_switch():
    accumulator = TrackingAccumulator()
    for frame in range(10):
        swapped = frame >= 5
        accumulator.add(
            truth={"near_1": _box(0, 0), "near_2": _box(200, 0)},
            hypothesis={
                "a": _box(200, 0) if swapped else _box(0, 0),
                "b": _box(0, 0) if swapped else _box(200, 0),
            },
        )
    assert accumulator.score().id_switches >= 1


def test_an_extra_track_lowers_mota():
    """Une piste attribuee a un spectateur est un faux positif."""
    clean = TrackingAccumulator()
    noisy = TrackingAccumulator()
    for _ in range(10):
        truth = {"near_1": _box(0, 0)}
        clean.add(truth=truth, hypothesis={"a": _box(0, 0)})
        noisy.add(truth=truth, hypothesis={"a": _box(0, 0), "b": _box(600, 0)})
    assert noisy.score().mota < clean.score().mota


def test_a_missed_player_lowers_mota():
    accumulator = TrackingAccumulator()
    for frame in range(10):
        hypothesis = {} if frame < 5 else {"a": _box(0, 0)}
        accumulator.add(truth={"near_1": _box(0, 0)}, hypothesis=hypothesis)
    assert accumulator.score().mota < 1.0


def test_the_number_of_frames_is_reported():
    accumulator = TrackingAccumulator()
    for _ in range(7):
        accumulator.add(truth={"near_1": _box(0, 0)}, hypothesis={"a": _box(0, 0)})
    assert accumulator.score().frames == 7


def test_an_empty_accumulator_reports_nan_rather_than_crashing():
    score = TrackingAccumulator().score()
    assert score.frames == 0
    assert np.isnan(score.mota)


def test_boxes_too_far_apart_are_not_matched():
    accumulator = TrackingAccumulator()
    for _ in range(10):
        accumulator.add(truth={"near_1": _box(0, 0)}, hypothesis={"a": _box(900, 0)})
    assert accumulator.score().mota < 0.5
