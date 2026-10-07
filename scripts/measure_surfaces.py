"""Measures the surface rule against the human judgements.

Reported by class, by stratum, and with the count next to each rate: a rate on two
examples is not a rate.

The two answers that name no surface are counted separately and never added together.
"x" is a non-measurement; "aucun" is a false positive of the contact stage, and its
complement is the precision of that stage, the only one this project can produce, the
stroke annotation covering half the frames.

Usage:
    python scripts/measure_surfaces.py --truth ground_truth/surfaces/<name>.json \
        --annotations <ball.json> --poses <pose.json> \
        --calibration ground_truth/calibrations/<name>.json \
        --out outputs/<name>_surfaces.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from build_surface_tasks import wrists_on

from padel_analysis.contact.surfaces import RACKET, classify
from padel_analysis.eval.ball_dataset import BallAnnotations
from padel_analysis.eval.dataset import PoseAnnotations
from padel_analysis.eval.surface_metrics import per_class, share_of
from padel_analysis.eval.surface_truth import (
    NO_CONTACT,
    UNREADABLE,
    SurfaceGroundTruth,
    class_of,
)
from padel_analysis.geometry.calibration import Calibration
from padel_analysis.geometry.camera import (
    CameraPose,
    court_surfaces,
    estimate_intrinsics,
)
from padel_analysis.geometry.court import Court

CLASS_OF_SURFACE = {RACKET: "raquette", "floor": "sol", "net": "filet"}


def predicted_class(name: str | None) -> str:
    """The class the rule announces, 'rien' if it found no surface."""
    if name is None:
        return "rien"
    return CLASS_OF_SURFACE.get(name, "mur")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fit = [p for p in Calibration.load(args.calibration).points if not p.is_control]
    objects = np.array([p.court_xyz for p in fit], dtype=np.float64)
    pixels = np.array([p.image_xy for p in fit], dtype=np.float64)
    pose = CameraPose.from_correspondences(
        objects, pixels, estimate_intrinsics(objects, pixels, (args.width, args.height))
    )
    surfaces = court_surfaces(Court())

    truth = SurfaceGroundTruth.load(args.truth)
    ball = BallAnnotations.load(args.annotations).centres()
    poses = PoseAnnotations.load(args.poses)
    strata = {t.frame: t.stratum for t in truth.tasks}

    judged, predicted, actual = [], [], []
    isolated: Counter[str] = Counter()
    for task in truth.tasks:
        answer = truth.answers.get(task.frame)
        if answer is None:
            continue
        verdict = classify(
            ball[task.frame], wrists_on(poses, task.frame), pose, surfaces
        )
        judged.append(task.frame)
        predicted.append(predicted_class(verdict.surface))
        actual.append(answer)
        if verdict.candidates == 1:
            isolated[answer] += 1

    truths = [class_of(a) or a for a in actual]
    scores = per_class(predicted, truths)
    # The filter is on the ANSWER, not on the class: class_of translates an answer
    # and so does not know "mur", which is already one. Filtering on the class would
    # silently discard the seventeen walls.
    real = [
        (p, t)
        for p, t, a in zip(predicted, truths, actual)
        if class_of(a) is not None
    ]

    print(f"{len(judged)} contacts judged\n")
    print(f"{'class':10} {'count':>9} {'precision':>10} {'recall':>8} {'F1':>7}")
    for label, score in scores.items():
        print(
            f"{label:10} {score.support:9} {score.precision:10.3f} "
            f"{score.recall:8.3f} {score.f1:7.3f}"
        )
    accuracy = sum(1 for p, t in real if p == t) / len(real)
    print(f"\noverall accuracy: {accuracy:.3f} over {len(real)} real contacts")

    absent = share_of(actual, NO_CONTACT)
    unreadable = share_of(actual, UNREADABLE)
    print(f"\ndetection false positives        : {absent:.3f}")
    print(f"precision of the contact stage    : {1 - absent:.3f}")
    print(f"unreadable                        : {unreadable:.3f}")

    total_isolated = sum(isolated.values())
    if total_isolated:
        spurious = isolated.get(NO_CONTACT, 0)
        print(
            f"\nisolated ray: {total_isolated} cases, {spurious} of them without a contact "
            f"({spurious / total_isolated:.0%})"
        )

    print("\nby stratum:")
    for stratum in ("raquette", "isole", "ambigu"):
        rows = [
            (p, t, a)
            for p, t, a, frame in zip(predicted, truths, actual, judged)
            if strata[frame] == stratum
        ]
        kept = [(p, t) for p, t, a in rows if class_of(a) is not None]
        if not kept:
            print(f"  {stratum:9} n={len(rows):3}  no real contact")
            continue
        good = sum(1 for p, t in kept if p == t)
        print(
            f"  {stratum:9} n={len(rows):3}  real {len(kept):3}  "
            f"accuracy {good / len(kept):.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "judged": len(judged),
                "accuracy": accuracy,
                "real_contacts": len(real),
                "no_contact_rate": absent,
                "unreadable_rate": unreadable,
                "isolated_spurious": isolated.get(NO_CONTACT, 0),
                "isolated_total": total_isolated,
                "per_class": {
                    label: {
                        "support": score.support,
                        "precision": score.precision,
                        "recall": score.recall,
                        "f1": score.f1,
                    }
                    for label, score in scores.items()
                },
                "answers": dict(Counter(actual)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nresults: {args.out}")


if __name__ == "__main__":
    main()
