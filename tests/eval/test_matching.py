import numpy as np
import pytest

from padel_analysis.eval.matching import iou, match_by_iou


def _box(x1, y1, x2, y2):
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def test_identical_boxes_have_an_iou_of_one():
    assert iou(_box(0, 0, 10, 10), _box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_disjoint_boxes_have_an_iou_of_zero():
    assert iou(_box(0, 0, 10, 10), _box(20, 20, 30, 30)) == pytest.approx(0.0)


def test_touching_boxes_have_an_iou_of_zero():
    assert iou(_box(0, 0, 10, 10), _box(10, 0, 20, 10)) == pytest.approx(0.0)


def test_half_overlap_gives_a_third():
    """Two boxes of the same size half of which overlap: 0.5 / 1.5."""
    assert iou(_box(0, 0, 10, 10), _box(5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_each_prediction_matches_at_most_one_annotation():
    predicted = [_box(0, 0, 10, 10), _box(1, 1, 11, 11)]
    annotated = [_box(0, 0, 10, 10)]
    pairs = match_by_iou(predicted, annotated, threshold=0.5)
    assert len(pairs) == 1


def test_a_pair_below_the_threshold_is_not_matched():
    predicted = [_box(0, 0, 10, 10)]
    annotated = [_box(9, 9, 19, 19)]
    assert match_by_iou(predicted, annotated, threshold=0.5) == []


def test_matching_prefers_the_better_overlap():
    predicted = [_box(0, 0, 10, 10)]
    annotated = [_box(5, 0, 15, 10), _box(0, 0, 10, 10)]
    pairs = match_by_iou(predicted, annotated, threshold=0.3)
    assert pairs == [(0, 1)]


def test_empty_inputs_give_no_pairs():
    assert match_by_iou([], [_box(0, 0, 1, 1)], threshold=0.5) == []
    assert match_by_iou([_box(0, 0, 1, 1)], [], threshold=0.5) == []
