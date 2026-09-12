import math

import pytest

from padel_analysis.eval.ball_metrics import (
    BallScore,
    CandidateScore,
    EventScore,
    ball_score,
    candidate_score,
    event_score,
)
from padel_analysis.eval.shots import ShotEvent


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


def test_a_candidate_on_the_ball_is_found_at_rank_one():
    score = candidate_score(
        candidates={1: [(100.0, 100.0), (500.0, 500.0)]},
        annotated={1: (100.0, 100.0)},
    )
    assert isinstance(score, CandidateScore)
    assert score.recall[10] == pytest.approx(1.0)
    assert score.median_rank == pytest.approx(1.0)


def test_the_rank_counts_from_one():
    score = candidate_score(
        candidates={1: [(500.0, 500.0), (800.0, 800.0), (100.0, 100.0)]},
        annotated={1: (100.0, 100.0)},
    )
    assert score.median_rank == pytest.approx(3.0)


def test_a_ball_absent_from_the_list_lowers_the_ceiling():
    score = candidate_score(
        candidates={1: [(100.0, 100.0)], 2: [(999.0, 999.0)]},
        annotated={1: (100.0, 100.0), 2: (200.0, 200.0)},
    )
    assert score.recall[10] == pytest.approx(0.5)


def test_the_share_inside_the_top_of_the_list_is_reported():
    score = candidate_score(
        candidates={
            1: [(100.0, 100.0)],
            2: [(0.0, 0.0)] * 30 + [(200.0, 200.0)],
        },
        annotated={1: (100.0, 100.0), 2: (200.0, 200.0)},
        top=10,
    )
    assert score.within_top == pytest.approx(0.5)


def test_a_frame_with_no_candidate_is_a_miss():
    score = candidate_score(candidates={1: []}, annotated={1: (100.0, 100.0)})
    assert score.recall[10] == pytest.approx(0.0)


def test_the_median_number_of_candidates_is_reported():
    score = candidate_score(
        candidates={1: [(0.0, 0.0)] * 3, 2: [(0.0, 0.0)] * 9},
        annotated={1: (0.0, 0.0), 2: (0.0, 0.0)},
    )
    assert score.median_candidates == pytest.approx(6.0)


def test_an_empty_candidate_evaluation_reports_nan():
    score = candidate_score(candidates={}, annotated={})
    assert math.isnan(score.recall[10])
    assert math.isnan(score.median_rank)


def test_a_contact_inside_an_event_finds_it():
    score = event_score(
        contacts=[15],
        events=[ShotEvent(start_frame=10, end_frame=20, category="Smash")],
    )
    assert isinstance(score, EventScore)
    assert score.recall == pytest.approx(1.0)
    assert score.precision == pytest.approx(1.0)


def test_an_event_without_any_contact_is_missed():
    score = event_score(
        contacts=[],
        events=[ShotEvent(start_frame=10, end_frame=20, category="Smash")],
    )
    assert score.recall == pytest.approx(0.0)


def test_a_contact_outside_every_event_lowers_precision():
    score = event_score(
        contacts=[15, 500],
        events=[ShotEvent(start_frame=10, end_frame=20, category="Smash")],
    )
    assert score.recall == pytest.approx(1.0)
    assert score.precision == pytest.approx(0.5)


def test_two_contacts_in_one_event_count_that_event_once():
    score = event_score(
        contacts=[12, 18],
        events=[ShotEvent(start_frame=10, end_frame=20, category="Smash")],
    )
    assert score.matched_events == 1
    assert score.recall == pytest.approx(1.0)


def test_the_event_counts_are_reported():
    score = event_score(
        contacts=[15, 500],
        events=[
            ShotEvent(start_frame=10, end_frame=20, category="Smash"),
            ShotEvent(start_frame=60, end_frame=70, category="Serve"),
        ],
    )
    assert score.events == 2
    assert score.contacts == 2
    assert score.matched_events == 1


def test_an_evaluation_without_events_reports_nan_rather_than_crashing():
    score = event_score(contacts=[10], events=[])
    assert math.isnan(score.recall)
