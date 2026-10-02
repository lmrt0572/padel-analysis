import numpy as np
import pytest

from padel_analysis.geometry.court import Court
from padel_analysis.render import figures

PNG = b"\x89PNG"


def _written(path):
    return path.exists() and path.read_bytes()[:4] == PNG


def _counts(right, shown, real):
    return {"right": right, "shown": shown, "real": real}


VERDICTS = {"judges": [
    {"name": "Juge 1", "rules": _counts(60, 90, 100), "model": _counts(80, 85, 100)},
    {"name": "Juge 2", "rules": _counts(55, 80, 100), "model": _counts(78, 84, 100)},
]}
RALLY = {
    "title": "Un echange", "duration": 3.0,
    "contacts": [
        {"t": 0.0, "kind": "raquette", "player": "near_1", "point": None},
        {"t": 0.5, "kind": "sol", "player": None, "point": [0.0, 6.0, 0.0]},
        {"t": 1.5, "kind": "verre", "player": None, "point": [5.0, -8.0, 1.0]},
    ],
    "impacts": [
        {"t": 0.5, "kind": "sol", "point": [0.0, 6.0, 0.0]},
        {"t": 1.5, "kind": "verre", "point": [5.0, -8.0, 1.0]},
    ],
    "players": {
        "near_1": {"name": "Proche 1", "path": [[0.0, 0.0, -6.0], [0.1, 0.2, -5.8]],
                   "shots": 1, "after": {"sol": 1}},
        "far_1": {"name": "Fond 1", "path": [[0.0, 1.0, 8.0]], "shots": 0, "after": {}},
    },
}


def test_the_judges_chart_is_written_with_or_without_cross_validation(tmp_path):
    figures.judges_chart(VERDICTS, None, tmp_path / "a.png")
    cv = {"regles": _counts(500, 700, 860), "modele": _counts(700, 800, 860)}
    figures.judges_chart(VERDICTS, cv, tmp_path / "b.png")
    assert _written(tmp_path / "a.png") and _written(tmp_path / "b.png")


def test_the_learning_curve_is_written(tmp_path):
    curve = [{"minutes": n, "right": 60 + n, "shown": 90, "real": 100} for n in (2, 4, 6)]
    figures.learning_curve_chart(curve, tmp_path / "c.png")
    assert _written(tmp_path / "c.png")


def test_the_confusion_chart_takes_missed_and_invented_contacts(tmp_path):
    confusion = [
        {"marked": "sol", "detected": "sol", "count": 40},
        {"marked": "sol", "detected": "raquette", "count": 5},
        {"marked": "verre", "detected": "aucun", "count": 7},
        {"marked": "aucun", "detected": "raquette", "count": 3},
    ]
    figures.confusion_chart(confusion, tmp_path / "d.png")
    assert _written(tmp_path / "d.png")


@pytest.mark.parametrize("chart", ["rally_court_chart", "rally_timeline_chart",
                                   "rally_shots_chart"])
def test_each_rally_chart_is_written(tmp_path, chart):
    getattr(figures, chart)(RALLY, tmp_path / f"{chart}.png")
    assert _written(tmp_path / f"{chart}.png")


def test_the_player_charts_are_written(tmp_path):
    rng = np.random.default_rng(0)
    positions = {"near_1": np.column_stack([rng.uniform(-4, 4, 200), rng.uniform(-9, -1, 200)])}
    figures.heatmaps_chart(positions, Court(), tmp_path / "e.png")
    figures.net_control_chart(np.abs(positions["near_1"][:, 1]), 5.85,
                              {"near_percent": 40.0, "far_percent": 30.0,
                               "contested_percent": 30.0}, tmp_path / "f.png")
    assert _written(tmp_path / "e.png") and _written(tmp_path / "f.png")


def _match_report(match):
    pair = {"points_won": 5, "strikes": 40, "volleys": 18, "after_bounce": 15,
            "after_glass": 9, "distance_m": 800.0, "net_share": 0.5,
            "won_by_length": {"1 à 3 coups": 1, "8 coups et plus": 4}}
    grid = [[0.0] * 20 for _ in range(40)]
    grid[10][5] = 1.0
    occupancy = {"grid": grid, "extent": [-5.0, 5.0, -10.0, 10.0]}
    return {"match": match, "minutes": 5.0, "pairs": {"A / B": pair, "C / D": dict(pair)},
            "occupancy": {"A / B": occupancy, "C / D": occupancy}}


def test_the_match_figures_are_drawn(tmp_path):
    from padel_analysis.render.figures import (
        pair_duel_chart,
        pair_occupancy_chart,
        points_by_length_chart,
    )

    reports = [_match_report("FinalF"), _match_report("FinalM")]
    for draw, name in ((pair_duel_chart, "duel.png"), (points_by_length_chart, "longueur.png"),
                       (pair_occupancy_chart, "occupation.png")):
        draw(reports, tmp_path / name)
        assert (tmp_path / name).stat().st_size > 0
