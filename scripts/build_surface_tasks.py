"""Draws up the list of contacts to submit to human judgement.

The contacts are detected on the ANNOTATED ball, not on the reconstructed trajectory.
The question asked is "this real contact, which surface": mixing in detection false
positives would have non-existent events judged, and would blend two distinct errors,
that of milestone B.3, already measured, and that of C.

The stratum recorded says whether the rule hesitated, never what it concluded. It is
used to split the results by difficulty, and the arbitration tool does not show it.

Three strata. "raquette": a wrist is close. "isole": the ray meets a single admissible
surface, which looked like a sign of confidence and turned out to be the opposite, the
24 cases of the tuning match being 24 non-events. A real contact happens in the volume
of play, where the near back wall is always admissible too since the camera is behind
it; a ray that meets only one surface therefore points outside the play. "ambigu":
several surfaces remain possible.

--sample draws a subset at random while keeping the proportion of each stratum. The
men's match has 886 contacts, that is two hours of arbitration; a sample is enough to
separate 0.83 from 0.75, and its seed is recorded.

--refresh recomputes the strata while keeping the answers already given. The answers
are about instants, which the rule does not change; only their stratum moves.

No video is decoded: ball, poses and calibration are JSON files.

Usage:
    python scripts/build_surface_tasks.py --annotations <ball.json> \
        --poses <pose.json> --calibration ground_truth/calibrations/<name>.json \
        --start 16000 --stop 20099 --video FinalF \
        --out ground_truth/surfaces/FinalF.json
"""

import argparse
import random
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


def _stratified(tasks: list[SurfaceTask], size: int, seed: int) -> list[SurfaceTask]:
    """Draws `size` tasks while keeping the proportion of each stratum.

    The size and the seed are recorded in the file: without them the draw would not be
    reproducible, and a result that cannot be redone is not a measurement.
    """
    by_stratum: dict[str, list[SurfaceTask]] = {}
    for task in tasks:
        by_stratum.setdefault(task.stratum, []).append(task)

    generator = random.Random(seed)
    drawn: list[SurfaceTask] = []
    for stratum, group in sorted(by_stratum.items()):
        share = max(1, round(size * len(group) / len(tasks)))
        drawn += generator.sample(group, min(share, len(group)))
    return sorted(drawn, key=lambda t: t.frame)


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
    parser.add_argument(
        "--sample",
        type=int,
        help="keep only N tasks, drawn at random in proportion to each stratum",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="recompute the strata while keeping the answers already given",
    )
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
            stratum = "isole"
        else:
            stratum = "ambigu"
        tasks.append(SurfaceTask(frame=contact.frame, stratum=stratum))
        strata[stratum] += 1

    if not tasks:
        raise SystemExit("no contact detected on this range")

    if args.sample and args.sample < len(tasks):
        tasks = _stratified(tasks, args.sample, args.seed)
        strata = Counter(t.stratum for t in tasks)

    kept: dict[int, str] = {}
    if args.refresh and args.out.exists():
        previous = SurfaceGroundTruth.load(args.out)
        frames = {t.frame for t in tasks}
        kept = {f: a for f, a in previous.answers.items() if f in frames}
        lost = len(previous.answers) - len(kept)
        print(f"resuming: {len(kept)} answers kept, {lost} lost")
    elif args.out.exists():
        raise SystemExit(
            f"{args.out} already exists: use --refresh to keep its answers"
        )

    SurfaceGroundTruth(
        video=args.video,
        frame_range=(args.start, args.stop),
        parameters={
            "wrist_distance": args.wrist_distance,
            "margin": args.margin,
            "focal": float(pose.intrinsics[0, 0]),
            **({"sample": args.sample, "seed": args.seed} if args.sample else {}),
        },
        tasks=tasks,
        answers=kept,
    ).save(args.out)

    print(f"{len(tasks)} contacts to judge, written to {args.out}")
    for stratum, count in strata.most_common():
        print(f"  {stratum:10} {count:4}  ({count / len(tasks):.0%})")


if __name__ == "__main__":
    main()
