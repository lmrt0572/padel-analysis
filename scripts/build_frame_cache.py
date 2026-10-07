"""Extracts the training frames as reduced JPEGs, with the ball scaled to match.

One frame in three. At 30 frames per second two consecutive frames carry almost the
same information: subsampling costs little and divides the epoch time by three. The
stacked frames being spaced 3 apart too, the neighbours of a centre fall on the same
grid and a single cache is enough for the three channels.

The evaluation slice is excluded. It must stay intact: a network that had seen it would
no longer measure anything.

Usage:
    python scripts/build_frame_cache.py --video <video.mp4> --annotations <ball.json> \
        --exclude 16000 20099 --step 3 --out cache/FinalF
"""

import argparse
import json
from pathlib import Path

import cv2

from padel_analysis.eval.ball_dataset import BallAnnotations
from padel_analysis.io.video_source import VideoSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--exclude", type=int, nargs=2, required=True)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.width % 8 or args.height % 8:
        raise SystemExit(
            "width and height must be multiples of 8: the network halves the "
            "resolution three times, and 540 does not allow it"
        )
    low, high = args.exclude
    centres = BallAnnotations.load(args.annotations).centres()
    args.out.mkdir(parents=True, exist_ok=True)

    scale_x, scale_y = args.width / 1920, args.height / 1080
    balls: dict[str, list[float]] = {}
    written = 0
    with VideoSource(args.video) as source:
        for index, frame in source.iter_frames():
            if index % args.step or low <= index <= high:
                continue
            cv2.imwrite(
                str(args.out / f"{index:06d}.jpg"),
                cv2.resize(frame, (args.width, args.height)),
                [cv2.IMWRITE_JPEG_QUALITY, args.quality],
            )
            written += 1
            point = centres.get(index)
            if point is not None:
                balls[str(index)] = [point[0] * scale_x, point[1] * scale_y]
            if written % 2000 == 0:
                print(f"  {written} frames", flush=True)

    (args.out / "balls.json").write_text(
        json.dumps(
            {"size": [args.width, args.height], "step": args.step, "balls": balls}
        ),
        encoding="utf-8",
    )
    print(f"{written} frames written, {len(balls)} of them with an annotated ball")


if __name__ == "__main__":
    main()
