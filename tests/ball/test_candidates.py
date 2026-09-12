import numpy as np
import pytest

from padel_analysis.ball.candidates import Candidate, MotionCandidates


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
