from padel_analysis.eval.splits import SPLITS, FrameRange


def test_the_two_evaluation_ranges_are_declared():
    assert "finalf_eval" in SPLITS
    assert "finalm_heldout" in SPLITS


def test_the_womens_evaluation_slice_sits_inside_the_shot_range():
    """Outside 0-20099 there is no stroke, so no contact metric."""
    slice_ = SPLITS["finalf_eval"]
    assert slice_.start >= 0
    assert slice_.stop <= 20099


def test_the_womens_training_ranges_avoid_the_evaluation_slice():
    evaluation = SPLITS["finalf_eval"]
    for training in SPLITS["finalf_train"]:
        assert training.stop < evaluation.start or training.start > evaluation.stop


def test_the_held_out_range_stops_where_the_annotation_stops():
    """Beyond 21472 the absence of an annotation does not mean the absence of a ball."""
    assert SPLITS["finalm_heldout"].stop <= 21472


def test_a_range_reports_whether_it_contains_a_frame():
    span = FrameRange(start=10, stop=20)
    assert span.contains(10) and span.contains(20)
    assert not span.contains(9)
    assert not span.contains(21)


def test_a_range_can_be_turned_into_a_length():
    assert FrameRange(start=10, stop=19).length == 10
