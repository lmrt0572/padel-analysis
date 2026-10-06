import json

import pytest

from padel_analysis.io.atomic import write_json_atomically


def test_the_payload_is_written_and_nothing_else_is_left(tmp_path):
    path = tmp_path / "deep" / "truth.json"
    write_json_atomically(path, {"a": [1, 2]}, indent=1)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": [1, 2]}
    assert [p.name for p in path.parent.iterdir()] == ["truth.json"]


def test_a_write_cut_short_keeps_the_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "truth.json"
    write_json_atomically(path, {"kept": True})

    def interrupted(*args, **kwargs):
        raise OSError("coupure de courant")

    monkeypatch.setattr("padel_analysis.io.atomic.os.replace", interrupted)
    with pytest.raises(OSError):
        write_json_atomically(path, {"kept": False})
    assert json.loads(path.read_text(encoding="utf-8")) == {"kept": True}
    assert [p.name for p in tmp_path.iterdir()] == ["truth.json"]


def test_a_payload_that_cannot_be_written_leaves_the_previous_file(tmp_path):
    path = tmp_path / "truth.json"
    write_json_atomically(path, {"kept": True})
    with pytest.raises(TypeError):
        write_json_atomically(path, {"bad": object()})
    assert json.loads(path.read_text(encoding="utf-8")) == {"kept": True}
    assert [p.name for p in tmp_path.iterdir()] == ["truth.json"]
