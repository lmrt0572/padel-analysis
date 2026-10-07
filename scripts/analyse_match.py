"""Analyse a whole match, minute by minute, for the match statistics.

The expensive pass, cut into slices of 1,800 frames. Each slice is saved as soon as it
is finished, so running the command again resumes at the next one.

Usage:
    python scripts/analyse_match.py --weights weights/ball_net.pt
    python scripts/analyse_match.py --weights weights/ball_net.pt --match FinalF
"""

import argparse
import pickle
import time
from pathlib import Path

import minutes

from padel_analysis.demo import analyse
from padel_analysis.io.video_source import VideoSource

MATCHES = ("FinalF", "FinalM")
OUT = Path("outputs/match")


def chunk_path(match: str, start: int) -> Path:
    return OUT / f"{match}_{start:05d}.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--match", choices=MATCHES, help="a single match; both by default")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    plan = []
    for match in (args.match,) if args.match else MATCHES:
        with VideoSource(Path(minutes.video(match))) as source:
            count = source.metadata.frame_count
        plan += [(match, start, min(minutes.FRAMES, count - start))
                 for start in range(0, count, minutes.FRAMES)]
    todo = [(m, s, n) for m, s, n in plan if not chunk_path(m, s).exists()]
    print(f"{len(plan)} slices, {len(plan) - len(todo)} already done, {len(todo)} to do",
          flush=True)
    began = time.time()
    for number, (match, start, frames) in enumerate(todo, 1):
        print(f"[{number}/{len(todo)}] {match} frames {start} to {start + frames - 1}", flush=True)
        result = analyse(argparse.Namespace(
            video=Path(minutes.video(match)), calibration=Path(minutes.calibration(match)),
            weights=args.weights, start=start, frames=frames,
        ))
        partial = chunk_path(match, start).with_suffix(".tmp")
        partial.write_bytes(pickle.dumps(result))
        partial.replace(chunk_path(match, start))
        left = (time.time() - began) / number * (len(todo) - number)
        print(f"  saved; about {left / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
