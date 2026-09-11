"""Turns a position cache into statistics and figures.

Usage:
    python -m padel_analysis.analyse --cache cache/<name>.json \
        --out outputs/<name>_report.json --figures outputs/
"""

import argparse
from pathlib import Path

import numpy as np

from .analytics.heatmap import occupancy_grid
from .analytics.net_control import NET_THRESHOLD
from .analytics.report import build_report, save_report
from .analytics.trajectories import SLOTS, MatchTrajectories
from .geometry.court import Court
from .pipeline.cache import PositionCache


def draw_heatmaps(trajectories: MatchTrajectories, court: Court, path: Path) -> None:
    """One occupancy map per player, on a single figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 4, figsize=(16, 7))
    for axis, slot in zip(axes, SLOTS):
        grid, extent = occupancy_grid(trajectories.positions[slot], court)
        axis.imshow(grid, origin="lower", extent=extent, aspect="equal", cmap="hot")
        axis.axhline(0.0, color="cyan", linewidth=1)
        for depth in (-court.service_line_distance, court.service_line_distance):
            axis.axhline(depth, color="white", linewidth=0.5)
        axis.set_title(slot)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def draw_depth_histogram(trajectories: MatchTrajectories, path: Path) -> None:
    """The distribution the net threshold was chosen from."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = np.concatenate([trajectories.depth(slot) for slot in SLOTS])
    depths = depths[~np.isnan(depths)]

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(depths, bins=40, range=(0, 10), color="#3b6ea5")
    axis.axvline(NET_THRESHOLD, color="crimson", linewidth=2,
                 label=f"seuil {NET_THRESHOLD} m")
    axis.set_xlabel("distance au filet (m)")
    axis.set_ylabel("frames")
    axis.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


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
    print(f"frames completes  : {report['complete_frames']}")
    print()
    print(f"controle du filet (seuil {control['threshold_m']} m)")
    print(f"  cote proche  : {control['near_percent']:5.1f} %")
    print(f"  cote eloigne : {control['far_percent']:5.1f} %")
    print(f"  dispute      : {control['contested_percent']:5.1f} %")
    print()
    print(f"{'joueur':>8} {'frames':>8} {'dist brute':>11} {'dist lissee':>12} "
          f"{'bruit':>7} {'v p95':>7} {'prof moy':>9}")
    for slot, player in report["players"].items():
        raw, smooth = player["distance_m"], player["distance_smoothed_m"]
        noise = 100.0 * (raw - smooth) / raw if raw > 0 else float("nan")
        print(f"{slot:>8} {player['frames_located']:>8} {raw:>10.1f}m "
              f"{smooth:>11.1f}m {noise:>6.1f}% {player['speed_p95_ms']:>6.2f} "
              f"{player['mean_depth_m']:>8.2f}m")

    if args.figures is not None:
        court = Court()
        draw_heatmaps(trajectories, court, args.figures / "heatmaps.png")
        draw_depth_histogram(trajectories, args.figures / "depth_histogram.png")
        print(f"\nfigures ecrites dans {args.figures}")

    print(f"\nrapport : {args.out}")


if __name__ == "__main__":
    main()
