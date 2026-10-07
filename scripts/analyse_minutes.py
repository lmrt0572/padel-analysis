"""Analyses every marked minute with one set of weights, without producing a video.

It is the expensive pass (pose detector and ball network on every frame) and the only
one that needs the graphics card. Its result is saved: every measurement afterwards is
redone without it.

Usage:
    python scripts/analyse_minutes.py --weights weights/ball_net.pt --tag 360
    python scripts/analyse_minutes.py --weights weights/ball_net.pt --tag 360         --match FinalF --minute 8000
"""

import argparse
import pickle
from pathlib import Path

import minutes

from padel_analysis.demo import analyse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="short name of the set of weights, e.g. 360 or 720")
    parser.add_argument("--match", help="analyse only one minute: its match")
    parser.add_argument("--minute", type=int, help="analyse only one minute: its first frame")
    args = parser.parse_args()

    todo = (minutes.TUNING + minutes.USED + minutes.EXTRA + minutes.JUDGE + minutes.JUDGE_2
            + minutes.JUDGE_3 + minutes.JUDGE_4 + minutes.JUDGE_5 + minutes.IDENTITY_JUDGE)
    if args.match is not None and args.minute is not None:
        todo = [(args.match, args.minute)]
    for number, (match, start) in enumerate(todo, 1):
        out = Path(minutes.analysis(match, start, args.tag))
        if out.exists():
            print(f"[{number}/{len(todo)}] {out.name} already exists")
            continue
        print(f"[{number}/{len(todo)}] {match} {start}", flush=True)
        result = analyse(argparse.Namespace(
            video=Path(minutes.video(match)), calibration=Path(minutes.calibration(match)),
            weights=args.weights, start=start, frames=minutes.FRAMES,
        ))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(pickle.dumps(result))


if __name__ == "__main__":
    main()
