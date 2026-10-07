
import pytest

from padel_analysis.eval.surface_truth import (
    ANSWERS,
    NO_CONTACT,
    NOT_A_SURFACE,
    UNREADABLE,
    SurfaceGroundTruth,
    SurfaceTask,
    class_of,
)


def _truth():
    return SurfaceGroundTruth(
        video="FinalF",
        frame_range=(16000, 20099),
        parameters={"wrist_distance": 80.0, "margin": 0.30},
        tasks=[
            SurfaceTask(frame=100, stratum="ambigu"),
            SurfaceTask(frame=200, stratum="tranche"),
        ],
    )


def test_every_answer_maps_to_a_class():
    assert class_of("sol") == "sol"
    assert class_of("verre") == "mur"
    assert class_of("grillage") == "mur"
    assert class_of("raquette") == "raquette"


def test_an_unreadable_answer_has_no_class():
    assert class_of(UNREADABLE) is None


def test_an_absent_contact_has_no_class_either():
    assert class_of(NO_CONTACT) is None


def test_the_two_non_surfaces_stay_distinct():
    """One is a non-measurement, the other is a measured false positive: never confused."""
    assert NO_CONTACT != UNREADABLE
    assert set(NOT_A_SURFACE) == {NO_CONTACT, UNREADABLE}


def test_the_answer_vocabulary_is_closed():
    assert set(ANSWERS) == {
        "sol",
        "verre",
        "grillage",
        "filet",
        "raquette",
        NO_CONTACT,
        UNREADABLE,
    }


def test_the_net_is_its_own_class():
    """Neither floor nor wall: the net dampens differently, and the ball leaves it differently."""
    assert class_of("filet") == "filet"


def test_an_absent_contact_can_be_recorded():
    truth = _truth()
    truth.answer(100, NO_CONTACT)
    assert truth.answers[100] == NO_CONTACT


def test_a_fresh_truth_has_everything_pending():
    assert [t.frame for t in _truth().pending()] == [100, 200]


def test_answering_removes_a_task_from_pending():
    truth = _truth()
    truth.answer(100, "sol")
    assert [t.frame for t in truth.pending()] == [200]


def test_an_answer_outside_the_vocabulary_is_refused():
    with pytest.raises(ValueError):
        _truth().answer(100, "vitre")


def test_an_answer_for_an_unknown_frame_is_refused():
    with pytest.raises(KeyError):
        _truth().answer(999, "sol")


def test_undoing_puts_the_task_back():
    truth = _truth()
    truth.answer(100, "sol")
    assert truth.undo(100) == "sol"
    assert [t.frame for t in truth.pending()] == [100, 200]


def test_undoing_an_unanswered_task_returns_nothing():
    assert _truth().undo(100) is None


def test_it_survives_a_save_and_a_load(tmp_path):
    truth = _truth()
    truth.answer(100, "grillage")
    path = tmp_path / "surfaces.json"
    truth.save(path)

    back = SurfaceGroundTruth.load(path)
    assert back.answers == {100: "grillage"}
    assert back.frame_range == (16000, 20099)
    assert [t.stratum for t in back.tasks] == ["ambigu", "tranche"]
    assert back.parameters["margin"] == 0.30


def test_the_file_never_names_a_predicted_surface(tmp_path):
    """The prediction must not pass through the file the tool reads."""
    path = tmp_path / "surfaces.json"
    _truth().save(path)
    assert "predicted" not in path.read_text(encoding="utf-8")


def test_a_save_leaves_no_temporary_behind(tmp_path):
    path = tmp_path / "surfaces.json"
    _truth().save(path)
    assert [p.name for p in tmp_path.iterdir()] == ["surfaces.json"]


def test_a_failed_save_does_not_destroy_the_previous_file(tmp_path, monkeypatch):
    """A cut during the write must cost the answer in progress, not the campaign."""
    path = tmp_path / "surfaces.json"
    _truth().save(path)
    before = path.read_text(encoding="utf-8")

    def explode(*args, **kwargs):
        raise OSError("disque plein")

    monkeypatch.setattr("padel_analysis.io.atomic.os.replace", explode)
    with pytest.raises(OSError):
        _truth().save(path)
    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["surfaces.json"]
