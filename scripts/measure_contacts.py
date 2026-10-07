"""Measures the contacts on a range, reconstructed path and annotated ball.

Both sources go through the same detector: the gap between them is therefore
attributable to the trajectory, and to nothing else.

Precision is printed with what a random draw would get. The two being equal on this
dataset, it is the distribution of bounces per exchange that carries the measurement:
padel expects zero, one or two between two strokes.

Usage:
    python scripts/measure_contacts.py --video <video.mp4> \
        --annotations <ball.json> --shots <shots.csv> \
        --identity ground_truth/identity/<name>.json \
        --start 0 --stop 20099 --out outputs/<name>_contacts.json
"""

import argparse
import json
from pathlib import Path

from measure_trajectory import collect

from padel_analysis.ball.contacts import find_contacts
from padel_analysis.ball.path import best_path
from padel_analysis.eval.ball_dataset import BallAnnotations
from padel_analysis.eval.ball_metrics import event_score
from padel_analysis.eval.contact_metrics import (
    bounces_between,
    chance_precision,
    interval_coverage,
)
from padel_analysis.eval.identity import IdentityGroundTruth
from padel_analysis.eval.shots import ShotEvents


def report(name, positions, events, cuts):
    contacts = find_contacts(positions, cuts=cuts)
    frames = [c.frame for c in contacts]
    usable = sorted(f for f, p in positions.items() if p is not None)
    score = event_score(frames, events)
    chance = chance_precision(usable, events, len(frames))
    tally = bounces_between(frames, events)
    pairs = sum(tally.values())
    mean = sum(k * v for k, v in tally.items()) / pairs if pairs else float("nan")

    print(f"\n{name}")
    print(f"  contacts            : {len(frames)}")
    print(f"  recall of strokes   : {score.recall:.3f}")
    print(f"  precision           : {score.precision:.3f}  (chance {chance:.3f})")
    print(f"  bounces per exchange: {mean:.2f}  {dict(sorted(tally.items()))}")
    return {
        "contacts": len(frames),
        "recall": score.recall,
        "precision": score.precision,
        "chance_precision": chance,
        "bounces_mean": mean,
        "bounces": {str(k): v for k, v in sorted(tally.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    events = ShotEvents.load(args.shots).events(args.start, args.stop)
    cuts = [c.frame for c in IdentityGroundTruth.load(args.identity).cuts]
    annotated = {
        f: p
        for f, p in BallAnnotations.load(args.annotations).centres().items()
        if args.start <= f <= args.stop
    }

    coverage = interval_coverage(sorted(annotated), events)
    print(
        f"{len(events)} strokes, {len(cuts)} cuts, "
        f"intervals over {coverage:.1%} of the annotated frames"
    )

    candidates = collect(args.video, args.start, args.stop, 2, 0.25)
    path = best_path(candidates, args.start, args.stop)

    payload = {
        "start": args.start,
        "stop": args.stop,
        "events": len(events),
        "cuts": len(cuts),
        "interval_coverage": coverage,
        "annotated": report("ANNOTATED BALL (ceiling)", annotated, events, cuts),
        "path": report("CHEMIN RECONSTRUIT", path, events, cuts),
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nresults: {args.out}")


if __name__ == "__main__":
    main()
