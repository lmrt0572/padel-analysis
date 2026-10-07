"""Test bench of the identity tracking, on the minutes already analysed.

The analysed minutes already carry the detections; with the torso colours cached, any
tracking is replayed in a few seconds. The judge minutes are only scored with --juge,
once, when the tracking is frozen.

Usage:
    python scripts/identity_bench.py            # builds the cache, scores the tuning
    python scripts/identity_bench.py --juge     # the verdict, once
"""

import argparse
import pickle
from pathlib import Path

import minutes

from padel_analysis.demo import observe
from padel_analysis.eval.dataset import PoseAnnotations
from padel_analysis.eval.identity import IdentityGroundTruth
from padel_analysis.eval.tracking_metrics import TrackingAccumulator
from padel_analysis.geometry.calibration import Calibration
from padel_analysis.io.video_source import VideoSource
from padel_analysis.perception.ground_point import AnkleMidpoint
from padel_analysis.tracking.court_constraint import CourtSlotTracker

CACHE = Path("outputs/identity")
ALL = (minutes.TUNING + minutes.USED + minutes.EXTRA + minutes.JUDGE + minutes.JUDGE_2
       + minutes.JUDGE_3)


def observations(match: str, start: int) -> dict:
    """Return every frame's detections placed on the court, with torso colours; cached."""
    path = CACHE / f"{match}_{start}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    analysis = pickle.loads(Path(minutes.analysis(match, start, "360")).read_bytes())
    calibration = Calibration.load(minutes.calibration(match))
    strategy = AnkleMidpoint()
    frames = {}
    with VideoSource(minutes.video(match)) as source:
        for index, image in source.iter_frames(start=analysis["start"],
                                                stop=analysis["stop"] + 1):
            people = analysis["frames"][index]["people"]
            frames[index] = {
                "people": people,
                "observations": observe(people, image, calibration, strategy),
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(frames))
    return frames


def score(chosen, make_tracker=CourtSlotTracker) -> dict:
    """Return IDF1 and identity switches of a tracker over the chosen minutes."""
    accumulator = TrackingAccumulator()
    truths, poses = {}, {}
    for match, start in chosen:
        if match not in truths:
            truths[match] = IdentityGroundTruth.load(Path(f"ground_truth/identity/{match}.json"))
            poses[match] = PoseAnnotations.load(
                Path(f"data/padeltracker100/extracted/labels/2022_BCN_{match}_1_pose.json"))
        truth, pose = truths[match], poses[match]
        tracker = make_tracker()
        for index, frame in sorted(observations(match, start).items()):
            assignment = tracker.update(frame["observations"])
            row = truth.assignments.get(index)
            if not row:
                continue
            annotated = pose.for_frame(index)
            name = f"{match}{start}#{truth.segment_of(index)}"
            accumulator.add(
                truth={f"{slot}@{name}": annotated[i].bbox for slot, i in row.items()
                       if i < len(annotated)},
                hypothesis={f"{slot}@{name}": frame["people"][i].bbox
                            for slot, i in assignment.items()},
            )
    result = accumulator.score()
    return {"idf1": result.idf1, "switches": result.id_switches, "frames": result.frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--juge", action="store_true")
    args = parser.parse_args()
    chosen = minutes.IDENTITY_JUDGE if args.juge else ALL
    result = score(chosen)
    title = "JUGE, huit minutes neuves" if args.juge else "reglage, vingt minutes"
    print(f"{title}: {len(chosen)} minutes, {result['frames']} frames scored")
    print(f"  IDF1 {result['idf1']:.3f}   identity changes {result['switches']}")


if __name__ == "__main__":
    main()
