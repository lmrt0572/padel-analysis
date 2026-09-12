"""Dresse la liste des contacts a soumettre au jugement humain.

Les contacts sont detectes sur la balle ANNOTEE, pas sur la trajectoire reconstruite.
La question posee est "ce contact reel, quelle surface" : y meler des faux positifs de
detection ferait juger des evenements inexistants, et melangerait deux erreurs
distinctes - celle du jalon B.3, deja mesuree, et celle de C.

La strate enregistree dit si la regle a hesite, jamais ce qu'elle a conclu. Elle sert a
decouper les resultats par difficulte, et l'outil d'arbitrage ne la montre pas.

Aucune video n'est decodee : balle, poses et calibration sont des fichiers JSON.

Usage:
    python scripts/build_surface_tasks.py --annotations <ball.json> \
        --poses <pose.json> --calibration ground_truth/calibrations/<nom>.json \
        --start 16000 --stop 20099 --video FinalF \
        --out ground_truth/surfaces/FinalF.json
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from padel_analysis.ball.contacts import find_contacts
from padel_analysis.contact.surfaces import RACKET, classify
from padel_analysis.eval.ball_dataset import BallAnnotations
from padel_analysis.eval.dataset import PoseAnnotations
from padel_analysis.eval.surface_truth import SurfaceGroundTruth, SurfaceTask
from padel_analysis.geometry.calibration import Calibration
from padel_analysis.geometry.camera import (
    CameraPose,
    court_surfaces,
    estimate_intrinsics,
)
from padel_analysis.geometry.court import Court

LEFT_WRIST, RIGHT_WRIST = 9, 10


def wrists_on(poses: PoseAnnotations, frame: int) -> list[tuple[float, float]]:
    """Every visible wrist on that frame, all players together."""
    return [
        (float(k[0]), float(k[1]))
        for person in poses.for_frame(frame)
        for k in (person.keypoints[LEFT_WRIST], person.keypoints[RIGHT_WRIST])
        if k[2] > 0
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--wrist-distance", type=float, default=80.0)
    parser.add_argument("--margin", type=float, default=0.30)
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

    centres = BallAnnotations.load(args.annotations).centres()
    ball = {f: p for f, p in centres.items() if args.start <= f <= args.stop}
    poses = PoseAnnotations.load(args.poses)

    tasks: list[SurfaceTask] = []
    strata: Counter[str] = Counter()
    for contact in find_contacts(ball):
        verdict = classify(
            ball[contact.frame],
            wrists_on(poses, contact.frame),
            pose,
            surfaces,
            wrist_distance=args.wrist_distance,
            margin=args.margin,
        )
        if verdict.surface == RACKET:
            stratum = "raquette"
        elif verdict.candidates == 1:
            stratum = "tranche"
        else:
            stratum = "ambigu"
        tasks.append(SurfaceTask(frame=contact.frame, stratum=stratum))
        strata[stratum] += 1

    if not tasks:
        raise SystemExit("aucun contact detecte sur cette plage")

    SurfaceGroundTruth(
        video=args.video,
        frame_range=(args.start, args.stop),
        parameters={
            "wrist_distance": args.wrist_distance,
            "margin": args.margin,
            "focal": float(pose.intrinsics[0, 0]),
        },
        tasks=tasks,
    ).save(args.out)

    print(f"{len(tasks)} contacts a juger, ecrits dans {args.out}")
    for stratum, count in strata.most_common():
        print(f"  {stratum:10} {count:4}  ({count / len(tasks):.0%})")


if __name__ == "__main__":
    main()
