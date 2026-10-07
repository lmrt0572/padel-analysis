"""Add the broadcast cuts to an existing identity ground truth.

The file is enriched in place: assignments and arbitrations already given are kept.

Usage:
    python scripts/detect_cuts.py --annotations <pose.json> \
        --calibration ground_truth/calibrations/<name>.json \
        --identity ground_truth/identity/<name>.json
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

    print("loading the annotations...", flush=True)
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

    # an arbitration on a cut that is still detected is kept; one on a cut the new
    # threshold drops is discarded
    kept = [f for f in truth.resolved_cuts if f in detected]
    dropped = len(truth.resolved_cuts) - len(kept)

    truth.cuts = cuts
    truth.resolved_cuts = kept
    truth.save(args.identity)

    both = sum(1 for c in cuts if len(c.sides) == 2)
    print(f"usable frames: {len(positions)}")
    print(f"cuts detected ({args.threshold} m): {len(cuts)}")
    print(f"  of which both pairs at risk: {both}")
    print(f"  of which a single pair: {len(cuts) - both}")
    print(f"close-approach episodes kept: {len(truth.resolved)}/"
          f"{len(truth.episodes)} arbitrated")
    print(f"cuts already arbitrated and kept: {len(kept)}")
    if dropped:
        print(f"arbitrations dropped (cut no longer kept): {dropped}")
    print(f"\nwrote {args.identity}")


if __name__ == "__main__":
    main()
