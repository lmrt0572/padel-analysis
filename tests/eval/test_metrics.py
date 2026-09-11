import numpy as np
import pytest

from padel_analysis.eval.metrics import (
    DetectionScore,
    LocalisationError,
    detection_score,
    localisation_error,
)


def test_a_perfect_frame_scores_one():
    score = detection_score(matched=4, predicted=4, annotated=4)
    assert isinstance(score, DetectionScore)
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1.0)
    assert score.f1 == pytest.approx(1.0)


def test_extra_predictions_lower_precision_not_recall():
    """Le detecteur trouve des spectateurs : la precision chute, le rappel non."""
    score = detection_score(matched=4, predicted=6, annotated=4)
    assert score.precision == pytest.approx(4 / 6)
    assert score.recall == pytest.approx(1.0)


def test_missed_people_lower_recall_not_precision():
    score = detection_score(matched=3, predicted=3, annotated=4)
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(0.75)


def test_a_frame_with_nothing_scores_zero_rather_than_crashing():
    score = detection_score(matched=0, predicted=0, annotated=0)
    assert score.precision == pytest.approx(0.0)
    assert score.recall == pytest.approx(0.0)
    assert score.f1 == pytest.approx(0.0)


def test_localisation_error_is_zero_for_identical_points():
    points = np.array([[100.0, 200.0], [300.0, 400.0]])
    error = localisation_error(points, points, np.array([-3.0, 5.0]))
    assert isinstance(error, LocalisationError)
    assert error.median_px == pytest.approx(0.0)


def test_localisation_error_measures_pixel_distance():
    predicted = np.array([[103.0, 200.0]])
    annotated = np.array([[100.0, 200.0]])
    error = localisation_error(predicted, annotated, np.array([-3.0]))
    assert error.median_px == pytest.approx(3.0)


def test_errors_are_split_by_half_of_the_court():
    """Un pixel vaut 4,3 fois plus au fond : un chiffre global serait trompeur."""
    predicted = np.array([[102.0, 200.0], [110.0, 200.0]])
    annotated = np.array([[100.0, 200.0], [100.0, 200.0]])
    depths = np.array([-3.0, 5.0])  # un proche, un eloigne
    error = localisation_error(predicted, annotated, depths)
    assert error.median_px_near == pytest.approx(2.0)
    assert error.median_px_far == pytest.approx(10.0)


def test_a_half_with_no_sample_reports_nan():
    predicted = np.array([[102.0, 200.0]])
    annotated = np.array([[100.0, 200.0]])
    error = localisation_error(predicted, annotated, np.array([-3.0]))
    assert np.isnan(error.median_px_far)


def test_the_count_of_compared_points_is_reported():
    predicted = np.array([[102.0, 200.0], [110.0, 200.0]])
    annotated = np.array([[100.0, 200.0], [100.0, 200.0]])
    error = localisation_error(predicted, annotated, np.array([-3.0, 5.0]))
    assert error.samples == 2
