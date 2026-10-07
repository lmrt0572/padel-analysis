"""Compare greedy growth and global optimisation, on the same candidates.

--weights swaps the motion detector for the trained network; nothing else changes. A
single pass over the video feeds both, and the player boxes come from the detector.

Usage:
    python scripts/measure_trajectory.py --video <video.mp4> \
        --annotations <ball.json> --start 16000 --stop 20099 \
        --out outputs/<name>_trajectory.json
"""

import argparse
import json
from pathlib import Path

from padel_analysis.ball.candidates import MotionCandidates, demote_inside_boxes
from padel_analysis.ball.path import best_path
from padel_analysis.ball.trajectory import build_segments, positions_of
from padel_analysis.eval.ball_dataset import BallAnnotations
from padel_analysis.eval.ball_metrics import ball_score
from padel_analysis.io.video_source import VideoSource
from padel_analysis.perception.pose_detector import PoseDetector


def collect(video, start, stop, spacing, factor, weights=None):
    """Return the candidates of each frame, penalised by the boxes of the detector.

    `weights` swaps the motion detector for the trained network.
    """
    if weights is None:
        finder = MotionCandidates(spacing=spacing)
    else:
        if spacing != 3:
            raise SystemExit(
                "the network was trained with a spacing of 3: pass "
                "--spacing 3. Without it the window of frames would be too short "
                "and the detector would return an empty list, without saying anything."
            )
        from padel_analysis.ball.heatmap_net import NetCandidates

        finder = NetCandidates(weights, spacing=spacing)
    detector = PoseDetector()
    candidates: dict[int, list] = {}
    window: dict[int, object] = {}

    with VideoSource(video) as source:
        for index, frame in source.iter_frames(
            start=max(0, start - spacing), stop=stop + spacing + 1
        ):
            window[index] = frame
            for old in [f for f in window if f < index - 2 * spacing]:
                del window[old]
            middle = index - spacing
            if middle < start or middle > stop:
                continue
            found = finder(window, middle)
            if not found:
                continue
            boxes = [d.bbox for d in detector.detect(window[middle])]
            candidates[middle] = demote_inside_boxes(found, boxes, factor)
            if middle % 2000 == 0:
                print(f"  frame {middle}", flush=True)
    return candidates


def report(name, predicted, annotated):
    score = ball_score(predicted, annotated)
    covered = sum(1 for p in predicted.values() if p is not None)
    print(f"\n{name}  ({covered} frames covered)")
    print(f"{'tolerance':>10} {'recall':>8} {'precision':>10}")
    for tolerance in (5, 10, 20):
        print(f"{tolerance:>9}px {score.recall[tolerance]:>8.3f} "
              f"{score.precision[tolerance]:>10.3f}")
    return {
        "covered_frames": covered,
        "recall": score.recall,
        "precision": score.precision,
        "unscorable": score.unscorable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--spacing", type=int, default=2)
    parser.add_argument("--factor", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--gate", type=float, default=320.0)
    parser.add_argument("--weight", type=float, default=240.0)
    parser.add_argument("--absent-cost", type=float, default=1200.0)
    parser.add_argument(
        "--weights",
        type=Path,
        help="weights of the network; without this option, detection by motion",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    balls = BallAnnotations.load(args.annotations)
    annotated = {
        f: c for f, c in balls.centres().items() if args.start <= f <= args.stop
    }
    print(f"annotated balls in the range: {len(annotated)}", flush=True)

    candidates = collect(
        args.video, args.start, args.stop, args.spacing, args.factor, args.weights
    )
    print(f"frames with candidates: {len(candidates)}", flush=True)

    segments = build_segments(candidates)
    greedy = positions_of(segments, args.start, args.stop)
    global_ = best_path(
        candidates,
        args.start,
        args.stop,
        width=args.width,
        gate=args.gate,
        weight=args.weight,
        absent_cost=args.absent_cost,
    )

    results = {
        "frames": args.stop - args.start + 1,
        "annotated": len(annotated),
        "segments": len(segments),
        "greedy": report("LIGNE DE BASE - croissance gloutonne", greedy, annotated),
        "global": report("OPTIMISATION GLOBALE", global_, annotated),
        "settings": {
            "width": args.width,
            "gate": args.gate,
            "weight": args.weight,
            "absent_cost": args.absent_cost,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults: {args.out}")


if __name__ == "__main__":
    main()
