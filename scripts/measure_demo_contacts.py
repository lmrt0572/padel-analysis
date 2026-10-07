"""Note les contacts de la chaine de demonstration contre le pointage complet d'une plage.

Precision, rappel et justesse de surface, pour plusieurs variantes de la chaine, sur les
memes marques. Pour les contacts rates, dit aussi a quel etage ils se sont perdus : le
chemin n'etait pas sur la balle, le chemin y etait mais sans virage detecte, ou le
contact a ete trouve puis masque par un filtre d'affichage.

Usage:
    python scripts/measure_demo_contacts.py --analysis outputs/demo_FinalF_analysis.pkl \
        --marks ground_truth/contact_marks/FinalF_16000.json \
        --annotations <ball.json> --calibration ground_truth/calibrations/FinalF.json
"""

import argparse
import math
import pickle
from collections import Counter
from pathlib import Path

from padel_analysis.ball.confidence import confident_path, path_scores
from padel_analysis.ball.contacts import find_contacts
from padel_analysis.ball.path import best_path
from padel_analysis.ball.smoothing import despike, smooth_path
from padel_analysis.contact.surfaces import classify
from padel_analysis.demo import build_events
from padel_analysis.eval.ball_dataset import BallAnnotations
from padel_analysis.eval.contact_marks import ContactMarks, match_contacts
from padel_analysis.geometry.calibration import Calibration
from padel_analysis.geometry.camera import court_surfaces, pose_from_calibration
from padel_analysis.geometry.court import Court
from padel_analysis.render.ball_overlay import contact_label

ANSWER_OF_LABEL = {"SOL": "sol", "VITRE": "verre", "GRILLAGE": "grillage", "FILET": "filet",
                   "RAQUETTE": "raquette"}


def labelled(contacts, positions, frames, pose, surfaces, isolated_filter):
    """Contact frame -> answer shown, for the contacts a variant would display."""
    shown: dict[int, str] = {}
    for contact in contacts:
        ball = positions.get(contact.frame)
        if ball is None:
            continue
        verdict = classify(ball, frames[contact.frame]["wrists"], pose, surfaces)
        label = contact_label(verdict)
        if label is None:
            continue
        if isolated_filter and label != "RAQUETTE" and verdict.candidates == 1:
            continue
        shown[contact.frame] = ANSWER_OF_LABEL[label]
    return shown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--marks", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--tolerance", type=int, default=3)
    args = parser.parse_args()

    analysis = pickle.loads(args.analysis.read_bytes())
    frames, start, stop = analysis["frames"], analysis["start"], analysis["stop"]
    marks = ContactMarks.load(args.marks).marks
    truth_ball = BallAnnotations.load(args.annotations).centres()
    pose = pose_from_calibration(Calibration.load(args.calibration).points, analysis["size"])
    surfaces = court_surfaces(Court())
    raw = {f: v["raw"] for f, v in frames.items()}

    relative = analysis["path"]
    absolute = best_path(raw, start, stop, weight=960.0, absent_cost=150.0, absolute=True)
    shown = despike(confident_path(absolute, path_scores(absolute, raw), 0.7, 8))
    variants = {
        "chemin relatif, sans filtre": labelled(
            find_contacts(relative), relative, frames, pose, surfaces, False),
        "chemin absolu, sans filtre": labelled(
            find_contacts(absolute), absolute, frames, pose, surfaces, False),
        "demo without the gesture": labelled(
            find_contacts(smooth_path(shown, cuts=[], process_noise=100.0)), shown, frames,
            pose, surfaces, True),
        "demo actuelle": {
            e.frame: ANSWER_OF_LABEL[e.label]
            for e in build_events(analysis, Calibration.load(args.calibration).points)[2]
        },
    }

    print(f"{len(marks)} marked contacts: {dict(Counter(marks.values()))}\n")
    print(f"{'variant':30} {'shown':>8} {'precision':>9} {'recall':>7} {'right surface':>14}")
    for name, detected in variants.items():
        m = match_contacts(detected, marks, args.tolerance)
        print(f"{name:30} {len(detected):8} {m.precision:9.0%} {m.recall:7.0%} "
              f"{m.right_surface:>5} of {m.found:<4}")

    demo = variants["demo actuelle"]
    raw_contacts = {c.frame for c in find_contacts(absolute)}
    causes: Counter[tuple[str, str]] = Counter()
    for frame, answer in marks.items():
        if any(abs(frame - f) <= args.tolerance for f in demo):
            continue
        near = [f for f in range(frame - args.tolerance, frame + args.tolerance + 1)]
        on_ball = any(
            absolute.get(f) is not None and f in truth_ball
            and math.dist(absolute[f], truth_ball[f]) <= 15 for f in near
        )
        if not on_ball:
            cause = "path not on the ball"
        elif not any(abs(frame - f) <= args.tolerance for f in raw_contacts):
            cause = "path on the ball, turn not detected"
        else:
            cause = "found then hidden or misplaced"
        causes[(answer, cause)] += 1

    print("\nmarked contacts missing from the demo, by surface and by cause:")
    for (answer, cause), count in sorted(causes.items(), key=lambda kv: -kv[1]):
        print(f"  {answer:9} {cause:42} {count:3}")


if __name__ == "__main__":
    main()
