"""Measure the cutting into rallies against the serves annotated in the dataset.

An announced rally start is right if it falls within two seconds of a serve. The
minutes the contact model has never seen are only scored with --juge, once.

Usage:
    python scripts/rally_bench.py --contact-model weights/contact_net.pt
    python scripts/rally_bench.py --contact-model weights/contact_net.pt --juge
"""

import argparse
import csv
import pickle
from pathlib import Path

import minutes

from padel_analysis.analytics.segmentation import rallies
from padel_analysis.contact.learned import ContactModel
from padel_analysis.demo import learned_events
from padel_analysis.geometry.calibration import Calibration
from padel_analysis.io.splices import frame_changes, splices
from padel_analysis.io.video_source import VideoSource
from padel_analysis.rallies import ANSWER_OF_LABEL

TUNE = [("FinalF", 16000), ("FinalF", 17800), ("FinalM", 5000), ("FinalM", 8000),
        ("FinalM", 18000)]
JUDGE = [("FinalF", 3000), ("FinalF", 8000), ("FinalF", 12000), ("FinalM", 1000),
         ("FinalM", 12000), ("FinalM", 14000)]
CACHE = Path("outputs/segmentation")
TOLERANCE = 60  # frames
EDGE = 30  # a serve stuck to the edge of the minute cannot be judged


def serves(match: str) -> list[int]:
    """Return the first frame of each annotated serve; a serve split in two counts once."""
    path = Path(f"data/padeltracker100/extracted/labels/2022_BCN_{match}_1_shots.csv")
    starts, previous, last = [], None, -1000
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            frame, category = int(row["file_name"][6:12]), row["category"]
            if category == "Serve" and previous != "Serve" and frame - last > 90:
                starts.append(frame)
            if category == "Serve":
                last = frame
            previous = category
    return starts


def minute(match: str, start: int, model: ContactModel) -> dict:
    """Return the contacts and the frame-to-frame changes of an analysed minute; cached."""
    path = CACHE / f"{match}_{start}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    analysis = pickle.loads(Path(minutes.analysis(match, start, "360")).read_bytes())
    points = Calibration.load(minutes.calibration(match)).points
    _, _, events, _ = learned_events(analysis, points, model)
    with VideoSource(minutes.video(match)) as source:
        changes = frame_changes(source.iter_frames(start=analysis["start"],
                                                   stop=analysis["stop"] + 1))
    data = {"start": analysis["start"], "stop": analysis["stop"], "changes": changes,
            "contacts": [(e.frame, ANSWER_OF_LABEL[e.label]) for e in events]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(data))
    return data


def score(keys, model: ContactModel) -> tuple[int, int, int]:
    """Return serves found, rally starts announced, serves annotated."""
    found = announced = real = 0
    for match, start in keys:
        data = minute(match, start, model)
        low, high = data["start"] + EDGE, data["stop"] - EDGE
        truth = [s for s in serves(match) if low <= s <= high]
        spans = rallies(splices(data["changes"]), data["contacts"], data["start"], data["stop"])
        guesses = [s.start for s in spans if low <= s.start <= high]
        free = list(truth)
        for guess in guesses:
            near = [t for t in free if abs(t - guess) <= TOLERANCE]
            if near:
                free.remove(min(near, key=lambda t: abs(t - guess)))
                found += 1
        announced += len(guesses)
        real += len(truth)
    return found, announced, real


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-model", type=Path, required=True)
    parser.add_argument("--juge", action="store_true")
    args = parser.parse_args()
    keys = JUDGE if args.juge else TUNE
    found, announced, real = score(keys, ContactModel.load(args.contact_model))
    title = "JUGE" if args.juge else "reglage"
    print(f"{title}: {found}/{real} serves found, {announced} starts announced, "
          f"precision {found / max(announced, 1):.0%}, recall {found / max(real, 1):.0%}")


if __name__ == "__main__":
    main()
