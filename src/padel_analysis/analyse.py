"""Turn a position cache into statistics and figures.

Usage:
    python -m padel_analysis.analyse --cache cache/<name>.json \
        --out outputs/<name>_report.json --figures outputs/
"""

import argparse
from pathlib import Path

import numpy as np

from .analytics.net_control import NET_THRESHOLD
from .analytics.report import build_report, save_report
from .analytics.trajectories import SLOTS, MatchTrajectories
from .geometry.court import Court
from .pipeline.cache import PositionCache
from .render.figures import heatmaps_chart, net_control_chart


def draw_heatmaps(trajectories: MatchTrajectories, court: Court, path: Path) -> None:
    """Draw one occupancy map per player, on a single figure."""
    heatmaps_chart({slot: trajectories.positions[slot] for slot in SLOTS}, court, path)


def draw_net_control(trajectories: MatchTrajectories, control: dict, path: Path) -> None:
    """Draw the depth distribution the net threshold comes from, and who held the net."""
    depths = np.concatenate([trajectories.depth(slot) for slot in SLOTS])
    net_control_chart(depths, NET_THRESHOLD, control, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    trajectories = MatchTrajectories.from_cache(PositionCache.load(args.cache), args.fps)
    report = build_report(trajectories)
    save_report(report, args.out)

    control = report["net_control"]
    print(f"frames            : {report['frames']} "
          f"({report['duration_s'] / 60:.1f} min)")
    print(f"complete frames   : {report['complete_frames']}")
    print()
    print(f"net control (threshold {control['threshold_m']} m)")
    print(f"  near side    : {control['near_percent']:5.1f} %")
    print(f"  far side     : {control['far_percent']:5.1f} %")
    print(f"  contested    : {control['contested_percent']:5.1f} %")
    print()
    print(f"{'player':>8} {'frames':>8} {'raw dist':>11} {'smooth dist':>12} "
          f"{'noise':>7} {'v p95':>7} {'depth':>9}")
    for slot, player in report["players"].items():
        raw, smooth = player["distance_m"], player["distance_smoothed_m"]
        noise = 100.0 * (raw - smooth) / raw if raw > 0 else float("nan")
        print(f"{slot:>8} {player['frames_located']:>8} {raw:>10.1f}m "
              f"{smooth:>11.1f}m {noise:>6.1f}% {player['speed_p95_ms']:>6.2f} "
              f"{player['mean_depth_m']:>8.2f}m")

    if args.figures is not None:
        court = Court()
        draw_heatmaps(trajectories, court, args.figures / "heatmaps.png")
        draw_net_control(trajectories, report["net_control"], args.figures / "net_control.png")
        print(f"\nfigures written to {args.figures}")

    print(f"\nreport: {args.out}")


if __name__ == "__main__":
    main()
