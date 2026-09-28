import pytest

from padel_analysis.analytics.segmentation import RallySpan, rallies

CONTACTS = [
    (5, "raquette"), (20, "sol"), (35, "raquette"),     # premier echange
    (110, "raquette"), (130, "verre"), (150, "raquette"), (170, "raquette"),
    (260, "sol"),                                      # un plan sans frappe
]


def test_each_splice_opens_a_rally():
    spans = rallies([100, 250], CONTACTS, start=0, stop=299)
    assert [(s.start, s.stop) for s in spans] == [(0, 99), (100, 249)]


def test_a_stretch_without_a_strike_is_not_a_rally():
    spans = rallies([100, 250], CONTACTS, start=0, stop=299)
    assert all(s.strikes > 0 for s in spans)
    assert 250 not in [s.start for s in spans]


def test_a_rally_counts_its_strikes_and_lasts_from_first_to_last_contact():
    second = rallies([100, 250], CONTACTS, start=0, stop=299)[1]
    assert second.strikes == 3
    assert second.duration(30.0) == pytest.approx(2.0)


def test_splices_outside_the_clip_are_ignored():
    spans = rallies([-50, 100, 900], CONTACTS, start=0, stop=299)
    assert [s.start for s in spans] == [0, 100]


def test_a_rally_with_one_contact_has_no_duration():
    assert RallySpan(0, 10, ((3, "raquette"),)).duration(30.0) == 0.0
