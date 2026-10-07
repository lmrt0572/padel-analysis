"""Mesure l'erreur de localisation des chevilles selon la resolution d'inference.

Downscaler 1920 vers 640 signifie qu'un pixel du modele vaut trois pixels d'image.
Au fond du court, ou un pixel vaut 6,47 cm, l'effet est brutal. On le chiffre.

Usage:
    python scripts/experiment_imgsz.py --video <video.mp4> --annotations <pose.json> \
        --calibration ground_truth/calibrations/<nom>.json --start 5000 --frames 200
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from padel_analysis.eval.dataset import PoseAnnotations
from padel_analysis.geometry.calibration import Calibration
from padel_analysis.io.video_source import VideoSource
from padel_analysis.perception.pose_detector import PoseDetector

SIZES = (640, 960, 1280, 1600)
MAX_MATCH_DISTANCE = 100.0


def match_by_position(predicted: list, annotated: list) -> list[tuple[int, int]]:
    """Matches predictions and annotations by the smallest distance between ankles."""
    if not predicted or not annotated:
        return []
    cost = np.zeros((len(predicted), len(annotated)))
    for i, p in enumerate(predicted):
        for j, a in enumerate(annotated):
            cost[i, j] = np.linalg.norm(p.ankle_midpoint() - a.ankle_midpoint())
    rows, cols = linear_sum_assignment(cost)
    return [
        (int(r), int(c))
        for r, c in zip(rows, cols)
        if cost[r, c] < MAX_MATCH_DISTANCE
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()

    print("loading the annotations...", flush=True)
    annotations = PoseAnnotations.load(args.annotations)
    projector = Calibration.load(args.calibration).projector

    print()
    print(f"{'imgsz':>7} {'ms/frame':>9} {'err px':>8} {'cm near':>11} "
          f"{'cm far':>9} {'matched':>10}")
    print("-" * 60)

    for size in SIZES:
        detector = PoseDetector(imgsz=size)
        pixel_errors: list[float] = []
        near_errors: list[float] = []
        far_errors: list[float] = []
        matched = 0
        started = time.perf_counter()
        processed = 0

        with VideoSource(args.video) as source:
            for index, frame in source.iter_frames(
                start=args.start, stop=args.start + args.frames
            ):
                truth = annotations.for_frame(index)
                processed += 1
                if not truth:
                    continue
                predicted = detector.detect(frame)
                for pi, ai in match_by_position(predicted, truth):
                    p_img = predicted[pi].ankle_midpoint()
                    a_img = truth[ai].ankle_midpoint()
                    pixel_errors.append(float(np.linalg.norm(p_img - a_img)))

                    p_court = projector.image_to_court(p_img.reshape(1, 2))[0]
                    a_court = projector.image_to_court(a_img.reshape(1, 2))[0]
                    centimetres = float(np.linalg.norm(p_court - a_court)) * 100
                    (near_errors if a_court[1] < 0 else far_errors).append(centimetres)
                    matched += 1

        elapsed = (time.perf_counter() - started) / max(processed, 1) * 1000
        median = np.median(pixel_errors) if pixel_errors else float("nan")
        near = np.median(near_errors) if near_errors else float("nan")
        far = np.median(far_errors) if far_errors else float("nan")
        print(f"{size:>7} {elapsed:>9.0f} {median:>8.2f} {near:>11.1f} "
              f"{far:>9.1f} {matched:>10}", flush=True)


if __name__ == "__main__":
    main()
