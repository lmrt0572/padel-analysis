"""Ajoute les coupures de plan a une verite terrain d'identite existante.

L'association au plus proche voisin suppose une image continue. Un changement de
plan la brise : si deux partenaires ont echange leur poste pendant la coupure,
l'association suit le mauvais, sans que rien ne paraisse ambigu.

Le fichier est enrichi sur place : les assignations et les arbitrages deja rendus
sont conserves.

Usage:
    python scripts/detect_cuts.py --annotations <pose.json> \
        --calibration ground_truth/calibrations/<nom>.json --identity ground_truth/identity/<nom>.json
"""

import argparse
from pathlib import Path

import numpy as np

from padel_analysis.eval.cuts import find_camera_cuts
from padel_analysis.eval.dataset import PoseAnnotations
from padel_analysis.eval.identity import SLOTS, IdentityGroundTruth
from padel_analysis.geometry.calibration import Calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args()

    print("chargement des annotations...", flush=True)
    annotations = PoseAnnotations.load(args.annotations)
    truth = IdentityGroundTruth.load(args.identity)
    projector = Calibration.load(args.calibration).projector

    positions: dict[int, dict[str, np.ndarray]] = {}
    for frame, row in truth.assignments.items():
        people = annotations.for_frame(frame)
        image = {s: people[i].ankle_midpoint() for s, i in row.items() if i < len(people)}
        if len(image) != len(SLOTS):
            continue
        stacked = np.array([image[s] for s in SLOTS])
        positions[frame] = dict(zip(SLOTS, projector.image_to_court(stacked)))

    cuts = find_camera_cuts(positions, threshold=args.threshold)
    detected = {c.frame for c in cuts}

    # Un arbitrage deja rendu sur une coupure toujours detectee est conserve ; s'il
    # porte sur une coupure que le nouveau seuil ne retient plus, il tombe.
    kept = [f for f in truth.resolved_cuts if f in detected]
    dropped = len(truth.resolved_cuts) - len(kept)

    truth.cuts = cuts
    truth.resolved_cuts = kept
    truth.save(args.identity)

    both = sum(1 for c in cuts if len(c.sides) == 2)
    print(f"frames exploitables : {len(positions)}")
    print(f"coupures detectees ({args.threshold} m) : {len(cuts)}")
    print(f"  dont les deux paires en risque : {both}")
    print(f"  dont une seule paire : {len(cuts) - both}")
    print(f"episodes de rapprochement conserves : {len(truth.resolved)}/"
          f"{len(truth.episodes)} arbitres")
    print(f"coupures deja arbitrees conservees : {len(kept)}")
    if dropped:
        print(f"arbitrages abandonnes (coupure non retenue) : {dropped}")
    print(f"\necrit {args.identity}")


if __name__ == "__main__":
    main()
