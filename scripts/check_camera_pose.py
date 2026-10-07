"""Fits the camera pose on the calibration, and judges it on its control points.

The points marked is_control are never used for the fit. It is the only way to get a
figure that means something: a pose judged on what produced it measures nothing.

What the gap is worth in metres depends on depth: a pixel is 1.51 cm near the camera
and 6.47 cm at the far end. The control points cover both, and that is deliberate.

Usage:
    python scripts/check_camera_pose.py --calibration ground_truth/calibrations/FinalF.json
"""

import argparse
from pathlib import Path

import numpy as np

from padel_analysis.geometry.calibration import Calibration
from padel_analysis.geometry.camera import CameraPose, estimate_intrinsics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    points = Calibration.load(args.calibration).points
    fit = [p for p in points if not p.is_control]
    control = [p for p in points if p.is_control]

    objects = np.array([p.court_xyz for p in fit], dtype=np.float64)
    pixels = np.array([p.image_xy for p in fit], dtype=np.float64)
    if not (objects[:, 2] > 0).any():
        raise SystemExit(
            "no point above the ground among the fitting points: the vertical "
            "direction would be free and the pose would mean nothing"
        )

    intrinsics = estimate_intrinsics(objects, pixels, (args.width, args.height))
    pose = CameraPose.from_correspondences(objects, pixels, intrinsics)
    centre = pose.camera_centre
    fitted = float(np.sqrt(((pose.project(objects) - pixels) ** 2).sum(axis=1).mean()))

    print(f"focal length      : {intrinsics[0, 0]:.0f} px")
    print(
        f"camera, on court : x={centre[0]:+.2f}  y={centre[1]:+.2f}  "
        f"z={centre[2]:+.2f} m"
    )
    print(f"fitting error     : {fitted:.2f} px over {len(fit)} points\n")

    print(f"{'control point':28} {'height':>9} {'gap':>9}")
    errors = []
    for point in control:
        projected = pose.project(np.array([point.court_xyz], dtype=np.float64))[0]
        gap = float(np.hypot(*(projected - np.array(point.image_xy))))
        errors.append(gap)
        print(f"{point.name:28} {point.height:7.2f} m {gap:6.1f} px")

    elevated = [g for p, g in zip(control, errors) if p.height > 0]
    print(f"\nmedian, all controls    : {np.median(errors):.1f} px")
    if elevated:
        print(f"median, above the ground: {np.median(elevated):.1f} px")
    print(f"maximum                  : {max(errors):.1f} px")


if __name__ == "__main__":
    main()
