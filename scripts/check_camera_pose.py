"""Ajuste la pose de camera sur la calibration, et la juge sur ses points de controle.

Les points marques is_control ne servent jamais a l'ajustement. C'est la seule facon
d'obtenir un chiffre qui veuille dire quelque chose : une pose jugee sur ce qui l'a
produite ne mesure rien.

Ce que l'ecart vaut en metres depend de la profondeur - un pixel fait 1,51 cm pres de
la camera et 6,47 cm au fond eloigne. Les points de controle couvrent les deux, et
c'est voulu.

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
            "aucun point en hauteur parmi les points d'ajustement : la direction "
            "verticale serait libre et la pose n'aurait aucun sens"
        )

    intrinsics = estimate_intrinsics(objects, pixels, (args.width, args.height))
    pose = CameraPose.from_correspondences(objects, pixels, intrinsics)
    centre = pose.camera_centre
    fitted = float(np.sqrt(((pose.project(objects) - pixels) ** 2).sum(axis=1).mean()))

    print(f"focale            : {intrinsics[0, 0]:.0f} px")
    print(
        f"camera en court   : x={centre[0]:+.2f}  y={centre[1]:+.2f}  "
        f"z={centre[2]:+.2f} m"
    )
    print(f"erreur ajustement : {fitted:.2f} px sur {len(fit)} points\n")

    print(f"{'point de controle':28} {'hauteur':>9} {'ecart':>9}")
    errors = []
    for point in control:
        projected = pose.project(np.array([point.court_xyz], dtype=np.float64))[0]
        gap = float(np.hypot(*(projected - np.array(point.image_xy))))
        errors.append(gap)
        print(f"{point.name:28} {point.height:7.2f} m {gap:6.1f} px")

    elevated = [g for p, g in zip(control, errors) if p.height > 0]
    print(f"\nmediane, tous controles  : {np.median(errors):.1f} px")
    if elevated:
        print(f"mediane, en hauteur      : {np.median(elevated):.1f} px")
    print(f"maximum                  : {max(errors):.1f} px")


if __name__ == "__main__":
    main()
