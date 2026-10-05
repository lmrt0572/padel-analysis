"""Produit les figures du README et du rapport, dans docs/figures/.

Les chiffres viennent de fichiers ecrits par les scripts de mesure, jamais recopies a la
main :
  - docs/figures/verdicts.json : les trois juges, consignes une fois rendus ;
  - outputs/measures/cv.json : validation croisee, confusion et courbe d'apprentissage
    (scripts/train_contact_model.py --cv --curve --results ...) ;
  - un cache de positions d'un match entier, pour les figures des joueurs ;
  - outputs/match_stats/<match>.json : le bilan de chaque finale, par paire
    (scripts/match_stats.py).
Une figure dont la source manque est sautee, et le script le dit.

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
        print(f"  saute : {path} introuvable")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", type=Path, default=Path("docs/figures/verdicts.json"))
    parser.add_argument("--measures", type=Path, default=Path("outputs/measures/cv.json"))
    parser.add_argument("--cache", type=Path, help="cache de positions d'un match entier")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--bilans", type=Path, default=Path("outputs/match_stats"),
                        help="dossier des bilans de match, par paire")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    verdicts = read(args.verdicts)
    measures = read(args.measures) or {}
    if verdicts is not None:
        figures.judges_chart(verdicts, measures.get("cv"), args.out / "juges.png")
    if "curve" in measures:
        figures.learning_curve_chart(measures["curve"], args.out / "courbe.png")
    if "confusion" in measures:
        figures.confusion_chart(measures["confusion"], args.out / "confusion.png")

    if args.cache is not None:
        trajectories = MatchTrajectories.from_cache(PositionCache.load(args.cache), args.fps)
        court = Court()
        figures.heatmaps_chart({slot: trajectories.positions[slot] for slot in SLOTS}, court,
                               args.out / "occupation.png")
        import numpy as np

        depths = np.concatenate([trajectories.depth(slot) for slot in SLOTS])
        report = build_report(trajectories)
        figures.net_control_chart(depths, report["net_control"]["threshold_m"],
                                  report["net_control"], args.out / "filet.png")

    reports = [read(args.bilans / f"{match}.json") for match in ("FinalF", "FinalM")]
    reports = [report for report in reports if report]
    if reports:
        figures.pair_duel_chart(reports, args.out / "bilan_paires.png")
        figures.points_by_length_chart(reports, args.out / "points_longueur.png")
        figures.pair_occupancy_chart(reports, args.out / "occupation_paires.png")

    print(f"figures ecrites dans {args.out}")


if __name__ == "__main__":
    main()
