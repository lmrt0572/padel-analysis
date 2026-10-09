"""Produce the figures of the README and of the report, in docs/figures/.

The numbers come from files written by the measurement scripts:
  - docs/figures/verdicts.json: the judges, recorded once given;
  - outputs/measures/cv.json: cross-validation, confusion and learning curve
    (scripts/train_contact_model.py --cv --curve --results ...);
  - a position cache of a whole match, for the player figures;
  - outputs/match_stats/<match>.json: the report of each final, by pair
    (scripts/match_stats.py).
A figure whose source is missing is skipped, and the script says so.

Usage:
    python scripts/make_figures.py --cache cache/FinalF_full.json
"""

import argparse
import json
from pathlib import Path

from padel_analysis.analytics.report import build_report
from padel_analysis.analytics.trajectories import SLOTS, MatchTrajectories
from padel_analysis.geometry.court import Court
from padel_analysis.pipeline.cache import PositionCache
from padel_analysis.render import figures


def read(path: Path) -> object | None:
    if not path.exists():
        print(f"  skipped: {path} not found")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", type=Path, default=Path("docs/figures/verdicts.json"))
    parser.add_argument("--measures", type=Path, default=Path("outputs/measures/cv.json"))
    parser.add_argument("--cache", type=Path, help="position cache of a whole match")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--reports", type=Path, default=Path("outputs/match_stats"),
                        help="folder of the match reports, by pair")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    verdicts = read(args.verdicts)
    measures = read(args.measures) or {}
    if verdicts is not None:
        figures.judges_chart(verdicts, measures.get("cv"), args.out / "judges.png")
    if "curve" in measures:
        figures.learning_curve_chart(measures["curve"], args.out / "learning_curve.png")
    if "confusion" in measures:
        figures.confusion_chart(measures["confusion"], args.out / "confusion.png")

    if args.cache is not None:
        trajectories = MatchTrajectories.from_cache(PositionCache.load(args.cache), args.fps)
        court = Court()
        figures.heatmaps_chart({slot: trajectories.positions[slot] for slot in SLOTS}, court,
                               args.out / "occupancy.png")
        import numpy as np

        depths = np.concatenate([trajectories.depth(slot) for slot in SLOTS])
        report = build_report(trajectories)
        figures.net_control_chart(depths, report["net_control"]["threshold_m"],
                                  report["net_control"], args.out / "net_control.png")

    reports = [read(args.reports / f"{match}.json") for match in ("FinalF", "FinalM")]
    reports = [report for report in reports if report]
    if reports:
        figures.pair_duel_chart(reports, args.out / "pair_report.png")
        figures.points_by_length_chart(reports, args.out / "points_by_length.png")
        figures.pair_occupancy_chart(reports, args.out / "pair_occupancy.png")

    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
