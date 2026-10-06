import json

import numpy as np

from padel_analysis import analyse
from padel_analysis.pipeline.cache import FramePositions, PositionCache


def _cache(path, frames=300):
    """Four players pacing a few metres each, the near pair staying close to the net."""
    cache = PositionCache()
    t = np.linspace(0.0, 2 * np.pi, frames)
    for i in range(frames):
        cache.add(FramePositions(i, {
            "near_1": (-2.5 + np.sin(t[i]), -2.0),
            "near_2": (2.5, -3.0 + 0.5 * np.cos(t[i])),
            "far_1": (-2.5, 8.0 + 0.5 * np.sin(t[i])),
            "far_2": (2.5 + np.cos(t[i]), 9.0),
        }))
    cache.save(path, video="v.mp4", calibration="c.json")


def test_a_cache_becomes_a_report_and_its_figures(tmp_path, monkeypatch, capsys):
    _cache(tmp_path / "cache.json")
    report_path, figures = tmp_path / "report.json", tmp_path / "figures"
    monkeypatch.setattr("sys.argv", ["analyse", "--cache", str(tmp_path / "cache.json"),
                                     "--out", str(report_path), "--figures", str(figures)])
    analyse.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["frames"] == 300 and report["complete_frames"] == 300
    assert set(report["players"]) == {"near_1", "near_2", "far_1", "far_2"}
    assert report["net_control"]["near_percent"] > report["net_control"]["far_percent"]
    for name in ("heatmaps.png", "net_control.png"):
        assert (figures / name).read_bytes()[:4] == b"\x89PNG"
    printed = capsys.readouterr().out
    assert "controle du filet" in printed and "near_1" in printed


def test_no_figure_is_drawn_unless_asked(tmp_path, monkeypatch):
    _cache(tmp_path / "cache.json", frames=120)
    monkeypatch.setattr("sys.argv", ["analyse", "--cache", str(tmp_path / "cache.json"),
                                     "--out", str(tmp_path / "report.json")])
    analyse.main()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cache.json", "report.json"]
