from padel_analysis.ball.candidates import Candidate
from padel_analysis.ball.confidence import confident_path, path_scores


def test_a_path_point_takes_the_score_of_the_candidate_it_came_from():
    path = {1: (10.0, 20.0), 2: None}
    raw = {1: [Candidate(5.0, 5.0, 0.9), Candidate(10.0, 20.0, 0.4)], 2: []}
    assert path_scores(path, raw) == {1: 0.4, 2: 0.0}


def test_a_point_with_no_matching_candidate_scores_nothing():
    assert path_scores({1: (10.0, 20.0)}, {1: [Candidate(1.0, 1.0, 0.9)]}) == {1: 0.0}


def test_weak_points_are_hidden():
    path = {f: (float(f), 0.0) for f in range(10)}
    scores = {f: (0.1 if f == 4 else 0.8) for f in range(10)}
    kept = confident_path(path, scores, threshold=0.5, min_run=1)
    assert kept[4] is None
    assert kept[3] == (3.0, 0.0)


def test_a_confident_run_too_short_to_be_a_trajectory_is_hidden():
    """Trois images sures au milieu du doute ne font pas une trajectoire."""
    path = {f: (float(f), 0.0) for f in range(12)}
    scores = {f: (0.9 if 4 <= f <= 6 else 0.1) for f in range(12)}
    kept = confident_path(path, scores, threshold=0.5, min_run=5)
    assert all(point is None for point in kept.values())


def test_a_long_enough_run_survives_whole():
    path = {f: (float(f), 0.0) for f in range(12)}
    scores = {f: (0.9 if 2 <= f <= 8 else 0.1) for f in range(12)}
    kept = confident_path(path, scores, threshold=0.5, min_run=5)
    assert [f for f, p in kept.items() if p is not None] == list(range(2, 9))
