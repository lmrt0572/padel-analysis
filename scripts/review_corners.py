"""Review the glass contacts marked near a corner, in the minutes marked before 29/09.

The marking tool then kept a single mark where a ball touched two glass panels from one
frame to the next. This opens the tool on each minute, `n` and `p` jumping between the
glass contacts marked within 1.5 m of a corner. Add the second contact where there were
two. A minute closed with `q` is recorded as reviewed.

Usage:
    python scripts/review_corners.py            # review, minute after minute
    python scripts/review_corners.py --liste    # only count the moments
"""

import argparse
import json
import pickle
from pathlib import Path

import minutes
from mark_contacts import run

from padel_analysis.demo import ball_near, build_events
from padel_analysis.eval.contact_marks import ContactMarks
from padel_analysis.geometry.calibration import Calibration
from padel_analysis.geometry.camera import court_surfaces
from padel_analysis.geometry.court import Court

BEFORE_FIX = (minutes.TUNING + minutes.USED + minutes.EXTRA + minutes.JUDGE + minutes.JUDGE_2
              + minutes.JUDGE_3)
CORNER = 1.5
WALLS = ("verre", "grillage")
PROGRESS = Path("outputs/coins/relues.json")


def corner_moments(match: str, start: int) -> list[int]:
    """Return the wall marks of one minute near a corner, with no other wall beside."""
    court = Court()
    surfaces = court_surfaces(court)
    analysis = pickle.loads(Path(minutes.analysis(match, start, "360")).read_bytes())
    path, shown, _, pose = build_events(analysis, Calibration.load(minutes.calibration(match)).points)
    marks = ContactMarks.load(minutes.marks(match, start)).marks
    moments = []
    for frame, answer in sorted(marks.items()):
        if answer not in WALLS:
            continue
        if any(f != frame and abs(f - frame) <= 5 and a in WALLS for f, a in marks.items()):
            continue
        ball = ball_near(frame, shown, path, 3)
        if ball is None:
            continue
        origin, direction = pose.ray(ball)
        for surface in surfaces[1:-1]:
            meeting = surface.intersect(origin, direction)
            if meeting is None or not surface.contains(meeting, 0.30):
                continue
            if surface.name.startswith("back"):
                near = abs(abs(meeting[0]) - court.half_width) < CORNER
            else:
                near = abs(abs(meeting[1]) - court.half_length) < CORNER
            if near:
                moments.append(frame)
                break
    return moments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liste", action="store_true", help="count without opening the tool")
    args = parser.parse_args()
    done = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else []
    plan = [(match, start, corner_moments(match, start)) for match, start in BEFORE_FIX]
    total = sum(len(m) for *_, m in plan)
    left = [(match, start, m) for match, start, m in plan if m and f"{match}_{start}" not in done]
    print(f"{total} glass contacts near a corner in {len(BEFORE_FIX)} minutes; "
          f"{sum(len(m) for *_, m in left)} to review in {len(left)} minutes")
    if args.liste:
        for match, start, moments in plan:
            print(f"  {match} {start:5} : {len(moments)}")
        return
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    for number, (match, start, moments) in enumerate(left, 1):
        print(f"[{number}/{len(left)}] {match} {start} : {len(moments)} moments", flush=True)
        run(Path(minutes.video(match)), match, start, minutes.FRAMES,
            Path(minutes.marks(match, start)), moments)
        done.append(f"{match}_{start}")
        PROGRESS.write_text(json.dumps(done), encoding="utf-8")


if __name__ == "__main__":
    main()
