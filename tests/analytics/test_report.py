import json

import numpy as np
import pytest

from padel_analysis.analytics.report import build_report, save_report
from padel_analysis.analytics.trajectories import MatchTrajectories
from padel_analysis.pipeline.cache import FramePositions, PositionCache


def _trajectories(n: int = 60) -> MatchTrajectories:
    cache = PositionCache()
    for frame in range(n):
        drift = frame * 0.01
        cache.add(FramePositions(frame=frame, positions={
            "near_1": (-1.0 + drift, -3.0), "near_2": (1.0, -3.0),
            "far_1": (-1.0, 8.0), "far_2": (1.0, 8.0),
        }))
    return MatchTrajectories.from_cache(cache, fps=30.0)


def test_report_covers_the_four_players():
    report = build_report(_trajectories())
    assert set(report["players"]) == {"near_1", "near_2", "far_1", "far_2"}


def test_each_player_reports_distance_speed_and_mean_position():
    player = build_report(_trajectories())["players"]["near_1"]
    for key in ("distance_m", "distance_smoothed_m", "speed_p95_ms",
                "mean_position", "frames_located", "half"):
        assert key in player


def test_distance_is_reported_both_raw_and_smoothed():
    """L'ecart entre les deux chiffre la part de bruit, au lieu de la masquer."""
    player = build_report(_trajectories())["players"]["near_2"]
    assert player["distance_m"] >= player["distance_smoothed_m"]


def test_each_player_is_labelled_with_its_half_of_the_court():
    players = build_report(_trajectories())["players"]
    assert players["near_1"]["half"] == "near"
    assert players["far_2"]["half"] == "far"


def test_net_control_is_included():
    control = build_report(_trajectories())["net_control"]
    assert control["near_percent"] == pytest.approx(100.0)
    assert control["threshold_m"] == pytest.approx(5.5)


def test_the_report_records_its_coverage():
    report = build_report(_trajectories(60))
    assert report["frames"] == 60
    assert report["duration_s"] == pytest.approx(2.0)
    assert report["complete_frames"] == 60


def test_report_roundtrips_through_json(tmp_path):
    report = build_report(_trajectories())
    path = tmp_path / "report.json"
    save_report(report, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["players"]["near_1"]["half"] == "near"


def test_saved_json_contains_no_numpy_scalars(tmp_path):
    """json.dump echoue sur les types numpy : le rapport doit etre en types Python."""
    path = tmp_path / "report.json"
    save_report(build_report(_trajectories()), path)
    assert path.stat().st_size > 0


def test_a_player_never_located_reports_nan_rather_than_crashing():
    cache = PositionCache()
    for frame in range(10):
        cache.add(FramePositions(frame=frame, positions={"near_1": (0.0, -3.0)}))
    report = build_report(MatchTrajectories.from_cache(cache, fps=30.0))
    assert report["players"]["far_1"]["frames_located"] == 0
    assert np.isnan(report["players"]["far_1"]["mean_position"][0])
