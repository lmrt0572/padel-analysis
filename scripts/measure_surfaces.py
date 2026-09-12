"""Mesure la regle de surface contre les jugements humains.

Rapporte par classe, par strate, et avec l'effectif a cote de chaque taux : un taux sur
deux exemples n'est pas un taux.

Les deux reponses qui ne nomment aucune surface sont comptees a part et jamais
additionnees. "x" est une non-mesure ; "aucun" est un faux positif de l'etage des
contacts, et son complement est la precision de cet etage - la seule que ce projet
puisse produire, l'annotation de frappes couvrant la moitie des frames.

Usage:
    python scripts/measure_surfaces.py --truth ground_truth/surfaces/<nom>.json \
        --annotations <ball.json> --poses <pose.json> \
        --calibration ground_truth/calibrations/<nom>.json \
        --out outputs/<nom>_surfaces.json
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
    """La classe que la regle annonce, 'rien' si elle n'a trouve aucune surface."""
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
    # Le filtre porte sur la REPONSE, pas sur la classe : class_of traduit une
    # reponse et ne connait donc pas "mur", qui en est deja une. Filtrer sur la
    # classe ecarterait silencieusement les dix-sept murs.
    real = [
        (p, t)
        for p, t, a in zip(predicted, truths, actual)
        if class_of(a) is not None
    ]

    print(f"{len(judged)} contacts juges\n")
    print(f"{'classe':10} {'effectif':>9} {'precision':>10} {'rappel':>8} {'F1':>7}")
    for label, score in scores.items():
        print(
            f"{label:10} {score.support:9} {score.precision:10.3f} "
            f"{score.recall:8.3f} {score.f1:7.3f}"
        )
    accuracy = sum(1 for p, t in real if p == t) / len(real)
    print(f"\nexactitude globale : {accuracy:.3f} sur {len(real)} contacts reels")

    absent = share_of(actual, NO_CONTACT)
    unreadable = share_of(actual, UNREADABLE)
    print(f"\nfaux positifs de detection        : {absent:.3f}")
    print(f"precision de l'etage des contacts : {1 - absent:.3f}")
    print(f"illisibles                        : {unreadable:.3f}")

    total_isolated = sum(isolated.values())
    if total_isolated:
        spurious = isolated.get(NO_CONTACT, 0)
        print(
            f"\nrayon isole : {total_isolated} cas, dont {spurious} sans contact "
            f"({spurious / total_isolated:.0%})"
        )

    print("\npar strate :")
    for stratum in ("raquette", "isole", "ambigu"):
        rows = [
            (p, t, a)
            for p, t, a, frame in zip(predicted, truths, actual, judged)
            if strata[frame] == stratum
        ]
        kept = [(p, t) for p, t, a in rows if class_of(a) is not None]
        if not kept:
            print(f"  {stratum:9} n={len(rows):3}  aucun contact reel")
            continue
        good = sum(1 for p, t in kept if p == t)
        print(
            f"  {stratum:9} n={len(rows):3}  reels {len(kept):3}  "
            f"exactitude {good / len(kept):.3f}"
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
    print(f"\nresultats : {args.out}")


if __name__ == "__main__":
    main()
