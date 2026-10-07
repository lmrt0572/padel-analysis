"""Report a whole match, by pair: points, strokes, movement.

Needs the analysis of the match (scripts/analyse_match.py) and the reading of its
scoreboard (scripts/read_scores.py). Player observations, contacts and broadcast cuts
are each cached.

Usage:
    python scripts/match_stats.py --match FinalF --contact-model weights/contact_net.pt
"""

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import minutes

from padel_analysis.analytics.heatmap import occupancy_grid
from padel_analysis.analytics.match_stats import (
    PairStats,
    contact_stats,
    movement_stats,
    orient,
    pair_positions,
    point_stats,
    stretch_points,
)
from padel_analysis.analytics.points import side_of
from padel_analysis.analytics.segmentation import RACKET, rallies
from padel_analysis.contact.learned import ContactModel
from padel_analysis.demo import learned_events, observe
from padel_analysis.geometry.calibration import Calibration
from padel_analysis.geometry.court import Court
from padel_analysis.io.scoreboard import ScoreState
from padel_analysis.io.splices import frame_changes, splices
from padel_analysis.io.video_source import VideoSource
from padel_analysis.perception.ground_point import AnkleMidpoint
from padel_analysis.rallies import ANSWER_OF_LABEL, strikers
from padel_analysis.tracking.court_constraint import CourtSlotTracker

CHUNKS = Path("outputs/match")
CACHE = Path("outputs/match_obs")
OUT = Path("outputs/match_stats")
# the pairs as the scoreboard writes them, top row then bottom row
PAIRS = {"FinalF": {1: "JOS / SAN", 2: "SAL / TRI"}, "FinalM": {1: "LEB / GAL", 2: "DIN / CHI"}}
FPS = 30.0


def chunk_paths(match: str) -> list[Path]:
    return sorted(CHUNKS.glob(f"{match}_*.pkl"))


def observations(match: str, path: Path) -> dict:
    """Return every detection of a chunk on the court, with its torso colour; cached."""
    cached = CACHE / path.name
    if cached.exists():
        return pickle.loads(cached.read_bytes())
    analysis = pickle.loads(path.read_bytes())
    calibration = Calibration.load(minutes.calibration(match))
    strategy = AnkleMidpoint()
    frames = {}
    with VideoSource(minutes.video(match)) as source:
        for index, image in source.iter_frames(start=analysis["start"],
                                                stop=analysis["stop"] + 1):
            if index not in analysis["frames"]:
                continue
            people = analysis["frames"][index]["people"]
            frames[index] = {"people": people,
                             "observations": observe(people, image, calibration, strategy),
                             "chunk_assignment": analysis["frames"][index]["assignment"]}
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(pickle.dumps(frames))
    return frames


def tracked(match: str) -> tuple[dict, dict]:
    """Return the players followed by one tracker over the match: frames and assignments."""
    frames = {}
    for path in chunk_paths(match):
        frames.update(observations(match, path))
    cached = CACHE / f"{match}_suivi.pkl"
    if cached.exists():
        return frames, pickle.loads(cached.read_bytes())
    tracker = CourtSlotTracker()
    whole = {index: tracker.update(frames[index]["observations"]) for index in sorted(frames)}
    cached.write_bytes(pickle.dumps(whole))
    return frames, whole


def broadcast_cuts(match: str) -> list[int]:
    cached = CACHE / f"{match}_raccords.json"
    if not cached.exists():
        with VideoSource(Path(minutes.video(match))) as source:
            cached.write_text(json.dumps(splices(frame_changes(source.iter_frames()))))
    return json.loads(cached.read_text())


def contacts(match: str, frames: dict, whole: dict, model_path: Path) -> list:
    """Return the model's contacts over the match, chunk by chunk, tracked match-long."""
    cached = OUT / f"{match}_contacts.pkl"
    if cached.exists():
        return pickle.loads(cached.read_bytes())
    model = ContactModel.load(model_path)
    points = Calibration.load(minutes.calibration(match)).points
    events = []
    for path in chunk_paths(match):
        analysis = pickle.loads(path.read_bytes())
        for index, data in analysis["frames"].items():
            assignment = whole.get(index, {})
            obs = frames.get(index, {}).get("observations", [])
            data["assignment"] = assignment
            data["positions"] = {slot: tuple(float(v) for v in obs[i].court_xy)
                                 for slot, i in assignment.items() if i < len(obs)}
        analysis["stop"] = max(analysis["frames"])
        events += learned_events(analysis, points, model)[2]
        print(f"  contacts {path.name}: {len(events)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(pickle.dumps(events))
    return events


def scoreboard(match: str) -> list[tuple[int, ScoreState]]:
    payload = json.loads(Path(f"outputs/scores/{match}.json").read_text(encoding="utf-8"))
    return [(r["start"], ScoreState(r["set"], tuple(r["games"]), tuple(r["points"]), r["server"]))
            for r in payload["readings"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True, choices=sorted(PAIRS))
    parser.add_argument("--contact-model", type=Path, required=True)
    args = parser.parse_args()

    frames, whole = tracked(args.match)
    events = contacts(args.match, frames, whole, args.contact_model)
    cuts = broadcast_cuts(args.match)
    readings = scoreboard(args.match)
    first = min(f for f, _ in readings)
    last = max(frames)

    people = {index: {"people": frames[index]["people"], "assignment": whole.get(index, {})}
              for index in frames}
    who = strikers(people, [(e.frame, e.box, e.pixel) for e in events if e.label == "RAQUETTE"])
    found = [(e.frame, ANSWER_OF_LABEL[e.label]) for e in events]
    spans = rallies(cuts, found, 0, last)
    serves = {}
    for span in spans:
        strikes = [f for f, kind in span.contacts if kind == RACKET and who.get(f)]
        if strikes:
            serves[span.start] = side_of(who[min(strikes)])
    orientation = orient(readings, serves)
    if orientation is None:
        raise SystemExit("no serve read: the pairs cannot be placed")

    pairs = {row: PairStats(row) for row in (1, 2)}
    winners = stretch_points(readings)
    point_stats(winners, pairs)
    played = [span for span in spans if span.start >= first]
    contact_stats(played, who, orientation, winners, pairs)
    positions = {index: {slot: tuple(float(v) for v in frames[index]["observations"][i].court_xy)
                         for slot, i in whole.get(index, {}).items()}
                 for index in frames if index >= first}
    movement_stats(positions, cuts, orientation, pairs)
    occupancy = {}
    for row, points in pair_positions(positions, orientation).items():
        grid, extent = occupancy_grid(points, Court())
        occupancy[PAIRS[args.match][row]] = {"grid": grid.round(6).tolist(), "extent": extent}

    report = {
        "match": args.match,
        "from_frame": first,
        "to_frame": last,
        "minutes": (last - first) / FPS / 60,
        "end_changes": list(orientation.changes),
        "serves_agreeing": orientation.votes_for,
        "serves_disagreeing": orientation.votes_against,
        "rallies": len(played),
        "points_settled": len([s for s in winners if s >= first]),
        "pairs": {PAIRS[args.match][row]: {**asdict(stats), "net_share": stats.net_share}
                  for row, stats in pairs.items()},
        "occupancy": occupancy,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.match}.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "occupancy"}, indent=1,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
