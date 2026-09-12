import math

import pytest

from padel_analysis.eval.ball_metrics import BallScore, ball_score


def test_a_perfect_prediction_scores_one_everywhere():
    score = ball_score(
        predicted={1: (100.0, 100.0), 2: (200.0, 200.0)},
        annotated={1: (100.0, 100.0), 2: (200.0, 200.0)},
    )
    assert isinstance(score, BallScore)
    for tolerance in (5, 10, 20):
        assert score.recall[tolerance] == pytest.approx(1.0)
        assert score.precision[tolerance] == pytest.approx(1.0)


def test_the_tolerance_decides_whether_a_prediction_counts():
    score = ball_score(
        predicted={1: (108.0, 100.0)},
        annotated={1: (100.0, 100.0)},
    )
    assert score.recall[5] == pytest.approx(0.0)
    assert score.recall[10] == pytest.approx(1.0)
    assert score.recall[20] == pytest.approx(1.0)


def test_a_missing_prediction_lowers_recall_but_not_precision():
    score = ball_score(
        predicted={1: (100.0, 100.0), 2: None},
        annotated={1: (100.0, 100.0), 2: (200.0, 200.0)},
    )
    assert score.recall[10] == pytest.approx(0.5)
    assert score.precision[10] == pytest.approx(1.0)


def test_a_prediction_on_an_unannotated_frame_is_not_counted_as_wrong():
    """Rien ne dit si la balle y est invisible ou seulement non annotee."""
    score = ball_score(
        predicted={1: (100.0, 100.0), 2: (500.0, 500.0)},
        annotated={1: (100.0, 100.0)},
    )
    assert score.precision[10] == pytest.approx(1.0)
    assert score.unscorable == 1


def test_the_counts_are_reported():
    score = ball_score(
        predicted={1: (100.0, 100.0), 2: None, 3: (300.0, 300.0)},
        annotated={1: (100.0, 100.0), 2: (200.0, 200.0)},
    )
    assert score.annotated == 2
    assert score.predicted == 2
    assert score.unscorable == 1


def test_an_empty_evaluation_reports_nan_rather_than_crashing():
    score = ball_score(predicted={}, annotated={})
    assert score.annotated == 0
    assert math.isnan(score.recall[10])
    assert math.isnan(score.precision[10])


def test_an_annotated_frame_never_predicted_is_a_miss():
    score = ball_score(predicted={}, annotated={1: (100.0, 100.0)})
    assert score.recall[10] == pytest.approx(0.0)


def test_the_tolerances_can_be_chosen():
    score = ball_score(
        predicted={1: (103.0, 100.0)},
        annotated={1: (100.0, 100.0)},
        tolerances=(2, 4),
    )
    assert set(score.recall) == {2, 4}
    assert score.recall[2] == pytest.approx(0.0)
    assert score.recall[4] == pytest.approx(1.0)
