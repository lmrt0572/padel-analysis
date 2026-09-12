import pytest

from padel_analysis.ball.candidates import Candidate
from padel_analysis.ball.trajectory import (
    Segment,
    build_segments,
    grow,
    positions_of,
)


def _straight(start_frame, count, x0=100.0, y0=100.0, dx=20.0, dy=0.0, noise=0.0):
    """Une balle en ligne droite, un candidat par frame."""
    out = {}
    for i in range(count):
        x, y = x0 + dx * i + noise * (i % 2), y0 + dy * i
        out[start_frame + i] = [Candidate(x, y, 100.0)]
    return out


def test_a_segment_knows_its_extent():
    segment = Segment(positions={5: (0.0, 0.0), 6: (1.0, 1.0), 7: (2.0, 2.0)})
    assert segment.start == 5
    assert segment.stop == 7
    assert segment.length == 3


def test_growth_follows_a_straight_line():
    candidates = _straight(10, 8)
    segment = grow(candidates, first=10, second=11)
    assert segment.start == 10
    assert segment.stop == 17
    assert segment.positions[17] == pytest.approx((240.0, 100.0))


def test_growth_also_goes_backwards():
    candidates = _straight(10, 8)
    segment = grow(candidates, first=14, second=15)
    assert segment.start == 10
    assert segment.stop == 17


def test_a_candidate_outside_the_gate_is_refused():
    """Une balle qui saute ailleurs n'est pas la meme balle."""
    candidates = _straight(10, 5)
    candidates[15] = [Candidate(2000.0, 900.0, 100.0)]
    candidates[16] = [Candidate(2020.0, 900.0, 100.0)]
    segment = grow(candidates, first=10, second=11, gate=30.0, max_misses=0)
    assert segment.stop == 14


def test_a_short_hole_is_crossed_by_prediction():
    candidates = _straight(10, 8)
    del candidates[13]
    segment = grow(candidates, first=10, second=11, gate=30.0, max_misses=2)
    assert segment.stop == 17
    assert segment.positions[13] == pytest.approx((160.0, 100.0))


def test_a_long_hole_ends_the_segment():
    candidates = _straight(10, 10)
    for frame in (13, 14, 15):
        del candidates[frame]
    segment = grow(candidates, first=10, second=11, gate=30.0, max_misses=2)
    assert segment.stop == 12


def test_the_trailing_predictions_are_trimmed():
    """Un segment ne doit pas se terminer sur des positions que rien n'a confirmees."""
    candidates = _straight(10, 6)
    segment = grow(candidates, first=10, second=11, gate=30.0, max_misses=2)
    assert segment.stop == 15
    assert segment.confirmed == set(range(10, 16))


def test_the_nearest_candidate_wins_not_the_strongest():
    """La trajectoire decide, pas le score de mouvement."""
    candidates = _straight(10, 5)
    candidates[12] = [Candidate(300.0, 100.0, 900.0), Candidate(140.0, 100.0, 10.0)]
    segment = grow(candidates, first=10, second=11, gate=30.0)
    assert segment.positions[12] == pytest.approx((140.0, 100.0))


def test_a_seed_without_candidates_yields_nothing():
    assert grow({}, first=10, second=11) is None


def test_a_seed_whose_two_frames_are_too_far_apart_yields_nothing():
    candidates = {10: [Candidate(0.0, 0.0, 100.0)], 11: [Candidate(900.0, 0.0, 100.0)]}
    assert grow(candidates, first=10, second=11, max_step=60.0) is None


def test_the_default_gate_absorbs_the_measured_wobble():
    """L'ecart a la prediction vaut 28,5 px au p90 : la porte par defaut doit le passer.

    Le va-et-vient de dix pixels construit ici produit vingt pixels d'ecart a chaque
    pas. Une porte resserree a quinze le rejetterait, et ce test tomberait.
    """
    candidates = _straight(10, 8, noise=10.0)
    segment = grow(candidates, first=10, second=11)
    assert segment is not None
    assert segment.length >= 6


def test_a_single_straight_run_gives_one_segment():
    segments = build_segments(_straight(10, 12))
    assert len(segments) == 1
    assert segments[0].start == 10
    assert segments[0].stop == 21


def test_two_separated_runs_give_two_segments():
    candidates = _straight(10, 8)
    candidates.update(_straight(60, 8, x0=900.0))
    segments = build_segments(candidates)
    assert len(segments) == 2
    assert [s.start for s in segments] == [10, 60]


def test_a_run_shorter_than_the_minimum_is_dropped():
    candidates = _straight(10, 3)
    assert build_segments(candidates, min_length=5) == []


def test_the_longer_segment_wins_an_overlap():
    """Un segment long a survecu a plus de contraintes qu'un court."""
    candidates = _straight(10, 12)
    for frame in range(14, 18):
        candidates[frame] = list(candidates[frame]) + [
            Candidate(400.0 + 5 * frame, 600.0, 500.0)
        ]
    segments = build_segments(candidates, min_length=3)
    covered = [f for s in segments for f in s.positions]
    assert len(covered) == len(set(covered))


def test_segments_come_back_in_order():
    candidates = _straight(60, 8, x0=900.0)
    candidates.update(_straight(10, 8))
    segments = build_segments(candidates)
    assert [s.start for s in segments] == [10, 60]


def test_no_candidate_at_all_gives_no_segment():
    assert build_segments({}) == []


def test_every_covered_frame_carries_a_position():
    segments = build_segments(_straight(10, 8))
    positions = positions_of(segments, start=10, stop=17)
    assert set(positions) == set(range(10, 18))
    assert all(p is not None for p in positions.values())


def test_a_frame_outside_every_segment_carries_none():
    segments = build_segments(_straight(10, 8))
    positions = positions_of(segments, start=8, stop=19)
    assert positions[8] is None
    assert positions[19] is None
    assert positions[12] is not None


def test_the_requested_range_is_respected():
    segments = build_segments(_straight(10, 20))
    positions = positions_of(segments, start=12, stop=15)
    assert sorted(positions) == [12, 13, 14, 15]


def test_no_segment_gives_none_everywhere():
    positions = positions_of([], start=0, stop=3)
    assert positions == {0: None, 1: None, 2: None, 3: None}
