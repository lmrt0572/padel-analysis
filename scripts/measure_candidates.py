"""Mesure le plafond de rappel des candidats de balle, pour plusieurs ecarts.

L'ecart temporel n'est pas choisi : il est balaye et le tableau tranche.

Usage:
    python scripts/measure_candidates.py --video <video.mp4> \
        --annotations <ball.json> --start 16000 --stop 20099 \
        --spacings 1 2 3 4 --out outputs/<nom>_candidates.json
"""

import argparse
import json
from pathlib import Path

from padel_analysis.ball.candidates import MotionCandidates
from padel_analysis.eval.ball_dataset import BallAnnotations
from padel_analysis.eval.ball_metrics import candidate_score
from padel_analysis.io.video_source import VideoSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--spacings", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    balls = BallAnnotations.load(args.annotations)
    annotated = {
        f: c for f, c in balls.centres().items() if args.start <= f <= args.stop
    }
    print(f"balles annotees dans la plage : {len(annotated)}", flush=True)

    widest = max(args.spacings)
    finders = {s: MotionCandidates(spacing=s) for s in args.spacings}
    candidates: dict[int, dict[int, list[tuple[float, float]]]] = {
        s: {} for s in args.spacings
    }
    window: dict[int, object] = {}

    with VideoSource(args.video) as source:
        for index, frame in source.iter_frames(
            start=max(0, args.start - widest), stop=args.stop + widest + 1
        ):
            window[index] = frame
            for old in [f for f in window if f < index - 2 * widest]:
                del window[old]

            middle = index - widest
            if middle in annotated:
                for spacing, finder in finders.items():
                    found = finder(window, middle)
                    candidates[spacing][middle] = [(c.x, c.y) for c in found]

    results = {}
    for spacing in args.spacings:
        score = candidate_score(candidates[spacing], annotated)
        results[spacing] = {
            "recall": score.recall,
            "median_rank": score.median_rank,
            "within_top_10": score.within_top,
            "median_candidates": score.median_candidates,
            "annotated": score.annotated,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'ecart':>6} {'rappel 5px':>11} {'10px':>7} {'20px':>7} "
          f"{'rang med.':>10} {'top 10':>8} {'candidats':>10}")
    for spacing in args.spacings:
        r = results[spacing]
        print(f"{spacing:>6} {r['recall'][5]:>11.3f} {r['recall'][10]:>7.3f} "
              f"{r['recall'][20]:>7.3f} {r['median_rank']:>10.1f} "
              f"{r['within_top_10']:>8.3f} {r['median_candidates']:>10.0f}")
    print(f"\nresultats : {args.out}")


if __name__ == "__main__":
    main()
