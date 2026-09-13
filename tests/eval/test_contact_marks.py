import pytest

from padel_analysis.eval.contact_marks import ContactMarks, match_contacts


def _marks():
    marks = ContactMarks(video="FinalF", frame_range=(100, 400))
    marks.mark(150, "sol")
    marks.mark(200, "verre")
    marks.mark(300, "raquette")
    return marks


def test_a_mark_outside_the_answers_is_refused():
    with pytest.raises(ValueError):
        _marks().mark(250, "vitre")


def test_marking_the_same_instant_twice_keeps_the_latest():
    marks = _marks()
    marks.mark(151, "verre", merge=2)
    assert marks.marks == {151: "verre", 200: "verre", 300: "raquette"}


def test_undo_removes_the_last_mark():
    marks = _marks()
    assert marks.undo() == (300, "raquette")
    assert 300 not in marks.marks


def test_marks_survive_a_save_and_a_load(tmp_path):
    path = tmp_path / "marks.json"
    marks = _marks()
    marks.position = 222
    marks.save(path)
    back = ContactMarks.load(path)
    assert back.marks == marks.marks
    assert back.position == 222


def test_matching_counts_found_missed_and_invented_contacts():
    detected = {152: "sol", 203: "sol", 250: "raquette"}
    result = match_contacts(detected, _marks().marks, tolerance=3)
    assert result.found == 2
    assert result.missed == 1
    assert result.invented == 1
    assert result.right_surface == 1


def test_one_mark_cannot_be_claimed_by_two_detections():
    result = match_contacts({149: "sol", 151: "sol"}, {150: "sol"}, tolerance=3)
    assert result.found == 1
    assert result.invented == 1
