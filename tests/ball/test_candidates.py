import numpy as np
import pytest

from padel_analysis.ball.candidates import (
    Candidate,
    MotionCandidates,
    demote_inside_boxes,
)


def _blank(height=200, width=300, value=40):
    return np.full((height, width, 3), value, dtype=np.uint8)


def _with_disc(frame, x, y, radius=5, value=230):
    out = frame.copy()
    ys, xs = np.ogrid[: frame.shape[0], : frame.shape[1]]
    mask = (xs - x) ** 2 + (ys - y) ** 2 <= radius**2
    out[mask] = value
    return out


def test_the_frames_needed_are_the_neighbours_at_the_chosen_spacing():
    finder = MotionCandidates(spacing=2)
    assert finder.frames_needed(100) == [98, 100, 102]


def test_a_disc_present_only_in_the_middle_frame_is_found():
    finder = MotionCandidates(spacing=1)
    frames = {9: _blank(), 10: _with_disc(_blank(), 150, 100), 11: _blank()}
    found = finder(frames, 10)
    assert found
    assert isinstance(found[0], Candidate)
    assert found[0].x == pytest.approx(150, abs=2)
    assert found[0].y == pytest.approx(100, abs=2)


def test_a_static_object_is_not_a_candidate():
    """Une ligne peinte ou un logo est identique sur les trois frames."""
    finder = MotionCandidates(spacing=1)
    painted = _with_disc(_blank(), 150, 100)
    assert finder({9: painted, 10: painted, 11: painted}, 10) == []


def test_a_blob_larger_than_a_ball_is_refused():
    """Un joueur bouge aussi : sa surface le disqualifie."""
    finder = MotionCandidates(spacing=1, max_area=400)
    frames = {
        9: _blank(),
        10: _with_disc(_blank(), 150, 100, radius=40),
        11: _blank(),
    }
    assert finder(frames, 10) == []


def test_a_blob_smaller_than_a_ball_is_refused():
    finder = MotionCandidates(spacing=1, min_area=20)
    frames = {9: _blank(), 10: _with_disc(_blank(), 150, 100, radius=1), 11: _blank()}
    assert finder(frames, 10) == []


def test_candidates_come_back_strongest_first():
    finder = MotionCandidates(spacing=1)
    middle = _with_disc(_blank(), 100, 100, value=120)
    middle = _with_disc(middle, 200, 100, value=250)
    found = finder({9: _blank(), 10: middle, 11: _blank()}, 10)
    assert len(found) >= 2
    assert found[0].score >= found[1].score
    assert found[0].x == pytest.approx(200, abs=2)


def test_the_list_is_capped():
    finder = MotionCandidates(spacing=1, max_candidates=3)
    middle = _blank()
    for i in range(10):
        middle = _with_disc(middle, 20 + 25 * i, 100, value=200 + i)
    found = finder({9: _blank(), 10: middle, 11: _blank()}, 10)
    assert len(found) == 3


def test_a_faint_change_is_ignored():
    finder = MotionCandidates(spacing=1, min_peak=50)
    frames = {
        9: _blank(value=40),
        10: _with_disc(_blank(value=40), 150, 100, value=60),
        11: _blank(value=40),
    }
    assert finder(frames, 10) == []


def test_an_object_that_only_darkens_is_ignored():
    """Le critere retenu est 'plus clair que les deux voisines', et il est assume."""
    finder = MotionCandidates(spacing=1)
    frames = {
        9: _blank(value=200),
        10: _with_disc(_blank(value=200), 150, 100, value=30),
        11: _blank(value=200),
    }
    assert finder(frames, 10) == []


def test_a_missing_neighbour_frame_yields_nothing():
    finder = MotionCandidates(spacing=1)
    assert finder({10: _blank()}, 10) == []


def test_an_object_that_arrives_and_stays_is_not_a_candidate():
    """Un joueur qui entre dans le champ, ou un bandeau LED qui change de visuel.

    Il est nouveau par rapport a la frame precedente, mais toujours la ensuite : ce
    n'est pas une balle qui traverse. Seule la comparaison aux DEUX voisines le
    refuse.
    """
    finder = MotionCandidates(spacing=1)
    arrived = _with_disc(_blank(), 150, 100)
    assert finder({9: _blank(), 10: arrived, 11: arrived}, 10) == []


def test_the_default_spacing_is_not_one():
    """Un ecart d'une frame perdait un tiers des balles : la valeur par defaut compte."""
    assert MotionCandidates().frames_needed(100) != [99, 100, 101]


def test_a_candidate_inside_a_box_is_demoted_but_kept():
    """Une balle passant devant un joueur doit rester atteignable."""
    inside = Candidate(x=100.0, y=100.0, score=200.0)
    outside = Candidate(x=500.0, y=500.0, score=100.0)
    ranked = demote_inside_boxes(
        [inside, outside], boxes=[np.array([50.0, 50.0, 150.0, 150.0])], factor=0.25
    )
    assert [c.x for c in ranked] == [500.0, 100.0]
    assert ranked[1].score == pytest.approx(50.0)


def test_a_candidate_outside_every_box_is_untouched():
    candidate = Candidate(x=500.0, y=500.0, score=100.0)
    ranked = demote_inside_boxes(
        [candidate], boxes=[np.array([0.0, 0.0, 10.0, 10.0])], factor=0.25
    )
    assert ranked[0] == candidate


def test_without_any_box_the_order_is_unchanged():
    given = [Candidate(1.0, 1.0, 10.0), Candidate(2.0, 2.0, 30.0)]
    ranked = demote_inside_boxes(given, boxes=[], factor=0.25)
    assert [c.score for c in ranked] == [30.0, 10.0]


def test_the_border_of_a_box_counts_as_inside():
    candidate = Candidate(x=150.0, y=150.0, score=100.0)
    ranked = demote_inside_boxes(
        [candidate], boxes=[np.array([50.0, 50.0, 150.0, 150.0])], factor=0.5
    )
    assert ranked[0].score == pytest.approx(50.0)


def test_a_factor_of_one_changes_nothing():
    given = [Candidate(100.0, 100.0, 10.0), Candidate(500.0, 500.0, 30.0)]
    ranked = demote_inside_boxes(
        given, boxes=[np.array([50.0, 50.0, 150.0, 150.0])], factor=1.0
    )
    assert [c.score for c in ranked] == [30.0, 10.0]
