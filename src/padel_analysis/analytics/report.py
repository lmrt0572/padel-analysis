"""Assembles the analytics into one JSON report.

Distance is given twice, raw and smoothed. The gap between them is the share the
position noise contributed, and it is larger for the far half, where a pixel is
worth 4.3 times what it is at the near end. Reporting both puts that in the open.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from .basic_stats import (
    distance_travelled,
    mean_position,
    smooth_positions,
    speed_percentile,
)
from .net_control import HYSTERESIS, NET_THRESHOLD, net_control
from .trajectories import SLOTS, MatchTrajectories

SMOOTHING_WINDOW = 9


def build_report(trajectories: MatchTrajectories) -> dict[str, Any]:
    """Every figure this milestone produces, as plain Python types."""
    players: dict[str, Any] = {}
    for slot in SLOTS:
        positions = trajectories.positions[slot]
        smoothed = smooth_positions(positions, window=SMOOTHING_WINDOW)
        centre = mean_position(positions)
        located = trajectories.present(slot)
        players[slot] = {
            "half": "near" if trajectories.side(slot) < 0 else "far",
            "frames_located": int(located.sum()),
            "distance_m": distance_travelled(trajectories.frames, positions),
            "distance_smoothed_m": distance_travelled(trajectories.frames, smoothed),
            "speed_p95_ms": speed_percentile(
                trajectories.frames, smoothed, trajectories.fps
            ),
            "mean_position": [float(centre[0]), float(centre[1])],
            "mean_depth_m": float(np.nanmean(trajectories.depth(slot)))
            if located.any()
            else float("nan"),
        }

    control = net_control(trajectories)
    return {
        "frames": int(trajectories.frames.size),
        "duration_s": trajectories.duration_seconds(),
        "complete_frames": int(trajectories.complete_mask().sum()),
        "fps": trajectories.fps,
        "players": players,
        "net_control": {
            "threshold_m": NET_THRESHOLD,
            "hysteresis_m": HYSTERESIS,
            "near_percent": control.near_percent,
            "far_percent": control.far_percent,
            "contested_percent": control.contested_percent,
            "evaluated_frames": control.evaluated_frames,
            "skipped_frames": control.skipped_frames,
        },
        "smoothing_window_frames": SMOOTHING_WINDOW,
    }


def save_report(report: dict[str, Any], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
