"""End-to-end pipeline: video in, annotated video and cached court positions out.

Usage:
    python -m padel_analysis.cli --video <video.mp4> \
        --calibration data/calibrations/<name>.json \
        --out outputs/annotated.mp4 --cache cache/<name>.json --frames 900
"""

import argparse
import contextlib
import time
from pathlib import Path

from .geometry.calibration import Calibration
from .geometry.court import Court
from .io.video_source import VideoSource
from .perception.appearance import torso_histogram
from .perception.ground_point import AnkleMidpoint, BboxBottom
from .perception.pose_detector import PoseDetector
from .pipeline.cache import FramePositions, PositionCache
from .render.minimap import Minimap
from .render.overlay import draw_people, paste_minimap
from .render.video_writer import VideoWriter
from .tracking.court_constraint import CourtObservation, CourtSlotTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true",
                        help="only produce the cache, without writing a video")
    parser.add_argument("--frames", type=int, default=None,
                        help="number of frames to process; all of them by default")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=1600)
    parser.add_argument("--ground-point", choices=("ankles", "bbox"), default="ankles")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.no_video and args.out is None:
        raise SystemExit("--out is required unless --no-video is given")

    court = Court()
    projector = Calibration.load(args.calibration).projector
    detector = PoseDetector(imgsz=args.imgsz)
    strategy = AnkleMidpoint() if args.ground_point == "ankles" else BboxBottom()
    tracker = CourtSlotTracker()
    minimap = Minimap(court)
    cache = PositionCache()

    with VideoSource(args.video) as source:
        meta = source.metadata
        remaining = meta.frame_count - args.start
        count = remaining if args.frames is None else min(args.frames, remaining)
        stop = args.start + count
        started = time.perf_counter()

        with contextlib.ExitStack() as stack:
            writer = None
            if not args.no_video:
                writer = stack.enter_context(
                    VideoWriter(args.out, meta.fps, (meta.width, meta.height))
                )
            for index, frame in source.iter_frames(start=args.start, stop=stop):
                detections = detector.detect(frame)

                observations: list[CourtObservation] = []
                for detection in detections:
                    point, confidence = strategy(detection)
                    court_xy = projector.image_to_court(point.reshape(1, 2))[0]
                    observations.append(
                        CourtObservation(
                            court_xy=court_xy,
                            confidence=confidence,
                            appearance=torso_histogram(frame, detection.bbox),
                        )
                    )

                assignment = tracker.update(observations)
                positions = {
                    name: (
                        float(observations[i].court_xy[0]),
                        float(observations[i].court_xy[1]),
                    )
                    for name, i in assignment.items()
                }
                cache.add(FramePositions(frame=index, positions=positions))

                if writer is not None:
                    annotated = draw_people(frame, detections, assignment)
                    annotated = paste_minimap(annotated, minimap.draw(positions))
                    writer.write(annotated)

                done = index - args.start + 1
                if done % 300 == 0:
                    rate = done / (time.perf_counter() - started)
                    print(f"  {done}/{count} frames  {rate:.1f} fps", flush=True)

    cache.save(args.cache, video=str(args.video), calibration=str(args.calibration))
    elapsed = time.perf_counter() - started
    complete = sum(1 for f in cache.frames() if len(cache.at(f).positions) == 4)
    if not args.no_video:
        print(f"wrote {args.out}")
    print(f"cache {args.cache}")
    print(f"{count} frames in {elapsed:.0f} s ({count / max(elapsed, 1e-9):.1f} fps)")
    print(f"frames with 4 identified players: {complete}/{count} "
          f"({100 * complete / max(count, 1):.2f} %)")


if __name__ == "__main__":
    main()
