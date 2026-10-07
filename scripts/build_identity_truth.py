"""Build the identity ground truth from the dataset annotations.

The automatic association covers the whole match; ambiguous episodes are listed for
human review.

Usage:
    python scripts/build_identity_truth.py --annotations <pose.json> \
        --calibration ground_truth/calibrations/<name>.json \
        --out ground_truth/identity/<name>.json
"""

import argparse
from pathlib import Path

import numpy as np

from padel_analysis.eval.dataset import PoseAnnotations
from padel_analysis.eval.identity import (
    IdentityGroundTruth,
    assign_by_proximity,
    find_ambiguous_episodes,
)
from padel_analysis.geometry.calibration import Calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1.5)
    args = parser.parse_args()

    print("loading the annotations...", flush=True)
    annotations = PoseAnnotations.load(args.annotations)
    projector = Calibration.load(args.calibration).projector

    positions_by_frame: dict[int, np.ndarray] = {}
    for frame in annotations.frame_indices():
        people = annotations.for_frame(frame)
        if len(people) != 4:
            continue
        image_points = np.array([p.ankle_midpoint() for p in people])
        positions_by_frame[frame] = projector.image_to_court(image_points)

    print(f"frames with four people: {len(positions_by_frame)}", flush=True)

    assignments = assign_by_proximity(positions_by_frame)
    episodes = find_ambiguous_episodes(positions_by_frame, threshold=args.threshold)

    truth = IdentityGroundTruth(assignments=assignments, episodes=episodes, resolved=[])
    truth.save(args.out)

    print(f"frames assigned: {len(assignments)}")
    print(f"episodes to arbitrate ({args.threshold} m): {len(episodes)}")
    for i, episode in enumerate(episodes):
        length = episode.end_frame - episode.start_frame + 1
        print(f"  {i:>3}  frames {episode.start_frame}-{episode.end_frame} "
              f"({length} fr)  {episode.slots[0]}/{episode.slots[1]}  "
              f"min {episode.min_separation_m:.2f} m")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
