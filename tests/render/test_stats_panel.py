import math

from padel_analysis.analytics.live_stats import LiveStats, PlayerLine
from padel_analysis.render.stats_panel import StatsPanel


def _stats(**changes):
    players = tuple(PlayerLine(slot, 2, 1, 1, 12.0, 15.0, 0.5)
                    for slot in ("near_1", "near_2", "far_1", "far_2"))
    base = {"elapsed": 12.0, "shots": 8, "walls": 3, "last": "verre", "players": players,
                "pair_shots": {"proche": 4, "fond": 4}, "pair_net": {"proche": 0.5, "fond": 0.5},
                "last_shot_speed": 42.0, "top_shot_speed": 61.0,
                "positions": {"near_1": [(0.0, -6.0), (0.5, -5.5)], "far_1": []}}
    base.update(changes)
    return LiveStats(**base)


def test_the_panel_has_the_requested_size_and_colour_depth():
    image = StatsPanel(480, 1080, {}).draw(_stats())
    assert image.shape == (1080, 480, 3)


def test_the_panel_draws_the_empty_start_of_a_clip():
    empty = _stats(shots=0, walls=0, last=None, last_shot_speed=None, top_shot_speed=None,
                   players=tuple(PlayerLine(s, 0, 0, 0, 0.0, 0.0, math.nan)
                                 for s in ("near_1", "near_2", "far_1", "far_2")))
    assert StatsPanel(480, 720, {}).draw(empty).shape == (720, 480, 3)
