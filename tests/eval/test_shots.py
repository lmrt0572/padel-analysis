from padel_analysis.eval.shots import ShotEvent, ShotEvents


def _write(path, rows):
    """rows : [(frame, has_shot, category), ...]"""
    lines = ["file_name;has_shot;category"]
    for frame, has, category in rows:
        lines.append(f"frame_{frame:06d}.PNG;{has};{category}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_consecutive_marked_frames_form_one_event(tmp_path):
    path = _write(
        tmp_path / "shots.csv",
        [(0, 0, "0"), (1, 1, "Smash"), (2, 1, "Smash"), (3, 1, "Smash"), (4, 0, "0")],
    )
    assert ShotEvents.load(path).events() == [
        ShotEvent(start_frame=1, end_frame=3, category="Smash")
    ]


def test_a_gap_separates_two_events(tmp_path):
    path = _write(
        tmp_path / "shots.csv",
        [(1, 1, "Serve"), (2, 0, "0"), (3, 1, "Forehand")],
    )
    events = ShotEvents.load(path).events()
    assert [e.category for e in events] == ["Serve", "Forehand"]
    assert [(e.start_frame, e.end_frame) for e in events] == [(1, 1), (3, 3)]


def test_a_file_without_any_shot_holds_no_event(tmp_path):
    path = _write(tmp_path / "shots.csv", [(0, 0, "0"), (1, 0, "0")])
    assert ShotEvents.load(path).events() == []


def test_the_covered_range_is_reported(tmp_path):
    path = _write(tmp_path / "shots.csv", [(10, 0, "0"), (11, 1, "Smash")])
    shots = ShotEvents.load(path)
    assert shots.covered_range() == (10, 11)


def test_events_can_be_restricted_to_a_range(tmp_path):
    path = _write(
        tmp_path / "shots.csv",
        [(1, 1, "Serve"), (2, 0, "0"), (50, 1, "Smash"), (51, 1, "Smash")],
    )
    events = ShotEvents.load(path).events(start=40, stop=60)
    assert [(e.start_frame, e.end_frame) for e in events] == [(50, 51)]


def test_an_event_reports_whether_it_contains_a_frame():
    event = ShotEvent(start_frame=10, end_frame=20, category="Smash")
    assert event.contains(10) and event.contains(20) and event.contains(15)
    assert not event.contains(9)
    assert not event.contains(21)


def test_a_category_change_without_a_gap_starts_a_new_event(tmp_path):
    """Two strokes that follow each other without an empty frame between them."""
    path = _write(
        tmp_path / "shots.csv",
        [(1, 1, "Serve"), (2, 1, "Serve"), (3, 1, "Forehand"), (4, 1, "Forehand")],
    )
    events = ShotEvents.load(path).events()
    assert [(e.start_frame, e.end_frame, e.category) for e in events] == [
        (1, 2, "Serve"),
        (3, 4, "Forehand"),
    ]
