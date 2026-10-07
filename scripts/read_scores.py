"""Read the score off the scoreboard at each rally of a match, and deduce the points.

The scoreboard is read on a few frames at the start of each rally and the most frequent
reading is kept. Between two successive rallies, the grammar of the score says who won
the point, or that a reading is doubtful. With --mosaique, scoreboards drawn at random
are written with their reading, to check it by eye.

Usage:
    python scripts/read_scores.py --match FinalF --out outputs/scores/FinalF.json
"""

import argparse
import json
import random
from collections import Counter
from itertools import pairwise
from pathlib import Path

import cv2
import minutes
import numpy as np

from padel_analysis.analytics.points import point_winner
from padel_analysis.io.scoreboard import Scoreboard
from padel_analysis.io.splices import SPLICE
from padel_analysis.io.video_source import VideoSource

EXAMPLES = Path("ground_truth/scoreboard/templates.json")
READS = (15, 30, 45, 60, 75)  # frames after the cut on which to read the scoreboard


def frame_of(match: str, index: int) -> np.ndarray:
    with VideoSource(minutes.video(match)) as source:
        for _, image in source.iter_frames(start=index, stop=index + 1):
            return image
    raise ValueError(f"{match} has no frame {index}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mosaique", type=Path, help="check image to write")
    args = parser.parse_args()

    board = Scoreboard.from_examples(EXAMPLES, frame_of)
    stretches: list[dict] = []
    previous = None
    crops = {}
    with VideoSource(minutes.video(args.match)) as source:
        for index, image in source.iter_frames(start=0):
            small = cv2.cvtColor(cv2.resize(image, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            change = 0.0 if previous is None else float(np.abs(small - previous).mean())
            previous = small
            if index == 0 or change > SPLICE:
                stretches.append({"start": index, "reads": []})
            offset = index - stretches[-1]["start"]
            if offset in READS:
                state = board.read(image)
                if state is not None:
                    stretches[-1]["reads"].append(state)
                    if offset == READS[0]:
                        crops[stretches[-1]["start"]] = (image[70:190, 40:460].copy(), state)

    points, undecided, same = [], 0, 0
    readings = []
    for stretch in stretches:
        if not stretch["reads"]:
            continue
        state, _ = Counter(stretch["reads"]).most_common(1)[0]
        readings.append((stretch["start"], state))
    for (start, before), (after_start, after) in pairwise(readings):
        if before == after:
            same += 1
            continue
        winner = point_winner(before, after)
        if winner is None:
            undecided += 1
        points.append({"start": start, "next": after_start, "winner": winner,
                       "before": [before.set_number, *before.games, *before.points],
                       "after": [after.set_number, *after.games, *after.points]})
    decided = sum(1 for p in points if p["winner"] is not None)
    print(f"{args.match}: {len(stretches)} sequences, {len(readings)} read; {len(points)} "
          f"score changes, {decided} points attributed, {undecided} doubtful, "
          f"{same} sequences without a change (same point)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "points": points,
        # one reading per sequence: its start, the score and the row of the serving pair
        "readings": [{"start": start, "set": state.set_number, "games": list(state.games),
                      "points": list(state.points), "server": state.server}
                     for start, state in readings],
        "stretches": [stretch["start"] for stretch in stretches],
    }
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    if args.mosaique:
        chosen = random.Random(0).sample(sorted(crops), min(24, len(crops)))
        tiles = []
        for start in chosen:
            crop, state = crops[start]
            tile = cv2.copyMakeBorder(cv2.resize(crop, (315, 90)), 0, 26, 0, 4,
                                      cv2.BORDER_CONSTANT)
            text = (f"set {state.set_number}  jeux {state.games[0]}-{state.games[1]}  "
                    f"pts {state.points[0]}-{state.points[1]}  serv {state.server}")
            cv2.putText(tile, text, (4, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            tiles.append(tile)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        cv2.imwrite(str(args.mosaique), np.vstack(rows))


if __name__ == "__main__":
    main()
