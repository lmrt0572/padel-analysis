"""Lance la campagne d'evaluation et ecrit ses resultats.

Usage:
    python scripts/run_evaluation.py --video <video.mp4> --annotations <pose.json> \
        --identity ground_truth/identity/<nom>.json \
        --calibration ground_truth/calibrations/<nom>.json --out outputs/<nom>_eval.json \
        --frames 3000
"""

import argparse
import json
from pathlib import Path

from padel_analysis.eval.campaign import run_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    results = run_campaign(
        video=args.video,
        annotations_path=args.annotations,
        calibration_path=args.calibration,
        identity_path=args.identity,
        frames=args.frames,
        start=args.start,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    detection = results["detection"]
    ground = results["ablation_ground_point"]
    tracker = results["ablation_tracker"]

    print(f"frames evaluated: {results['frames']}\n")
    print("DETECTION")
    print(f"  precision {detection['precision']:.3f}   "
          f"recall {detection['recall']:.3f}   F1 {detection['f1']:.3f}")
    print(f"  {detection['predicted']} predicted, {detection['annotated']} annotated, "
          f"{detection['matched']} matched\n")

    print(f"ABLATION 1: ground point   ({ground['samples']} samples)")
    print(f"{'':>12} {'overall':>9} {'near':>9} {'far':>9}")
    print(f"{'ankles':>12} {ground['ankles_median_px']:>8.2f}p "
          f"{ground['ankles_median_px_near']:>8.2f}p "
          f"{ground['ankles_median_px_far']:>8.2f}p")
    print(f"{'bbox bottom':>12} {ground['bbox_median_px']:>8.2f}p "
          f"{ground['bbox_median_px_near']:>8.2f}p "
          f"{ground['bbox_median_px_far']:>8.2f}p\n")

    print(f"ABLATION 2: tracker   ({tracker['scored_frames']} frames with identity)")
    print(f"{'':>14} {'mean tracks':>12} {'frames > 4':>11} "
          f"{'MOTA':>7} {'IDF1':>7} {'switches':>9}")
    print(f"{'constrained':>14} {tracker['constrained_mean_tracks']:>12.2f} "
          f"{tracker['constrained_frames_over_four']:>11} "
          f"{tracker['constrained_mota']:>7.3f} {tracker['constrained_idf1']:>7.3f} "
          f"{tracker['constrained_id_switches']:>9}")
    print(f"{'ByteTrack only':>14} {tracker['bytetrack_mean_tracks']:>12.2f} "
          f"{tracker['bytetrack_frames_over_four']:>11} "
          f"{tracker['bytetrack_mota']:>7.3f} {tracker['bytetrack_idf1']:>7.3f} "
          f"{tracker['bytetrack_id_switches']:>9}")
    print(f"\nresults: {args.out}")


if __name__ == "__main__":
    main()
