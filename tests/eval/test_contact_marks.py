import pytest

from padel_analysis.eval.contact_marks import (
    INVENTED,
    MISSED,
    RIGHT,
    WRONG_SURFACE,
    ContactMarks,
    match_contacts,
    pair_contacts,
)


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


def test_two_contacts_a_frame_apart_are_both_kept():
    marks = _marks()
    marks.mark(201, "verre")  # une balle dans un coin touche deux vitres
    assert marks.marks[200] == "verre" and marks.marks[201] == "verre"


def test_marking_the_same_frame_again_changes_its_kind():
    marks = _marks()
    marks.mark(200, "grillage")
    assert marks.marks[200] == "grillage" and len(marks.marks) == 3


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


def test_each_detection_and_each_mark_gets_one_line():
    detected = {10: "sol", 30: "raquette", 80: "verre"}
    marks = {11: "sol", 31: "sol", 50: "raquette"}
    lines = pair_contacts(detected, marks, tolerance=3)
    assert [(line.detected_frame, line.marked_frame, line.status) for line in lines] == [
        (10, 11, RIGHT), (30, 31, WRONG_SURFACE), (80, None, INVENTED), (None, 50, MISSED),
    ]


def test_the_pairing_and_the_counts_always_agree():
    detected = {10: "sol", 12: "sol", 40: "verre", 70: "raquette"}
    marks = {11: "sol", 41: "sol", 90: "raquette"}
    lines = pair_contacts(detected, marks)
    counts = match_contacts(detected, marks)
    assert counts.invented == sum(line.status == INVENTED for line in lines) == 2
    assert counts.missed == sum(line.status == MISSED for line in lines) == 1
    assert counts.right_surface == sum(line.status == RIGHT for line in lines) == 1
