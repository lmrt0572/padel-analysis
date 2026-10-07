"""Demonstration video with a statistics panel to the right of the picture.

The statistics (shots, glass contacts, strokes, distance and time at the net per player
and per pair) are accumulated from the start of the clip and move with the video. The
broadcast picture is not covered: the panel is added beside it.

The points come from the scoreboard, read at the start of each rally and of the one
that follows: the winning pair, then the last striker of the rally, credited with a
winner if they are of that pair, with an error otherwise.

Needs the saved analysis of the minute (scripts/analyse_minutes.py) and ffmpeg.

Usage:
    python scripts/stats_video.py --match FinalF --minute 12000 --start 12888 \
        --stop 13799 --contact-model weights/contact_net.pt --out outputs/stats.mp4
"""

import argparse
import pickle
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import minutes
import numpy as np
from match_stats import PAIRS
from read_scores import EXAMPLES, READS, frame_of

from padel_analysis.analytics.live_stats import LiveTimeline, PointOutcome
from padel_analysis.analytics.points import credit, point_winner, serving_side
from padel_analysis.analytics.segmentation import rallies
from padel_analysis.ball.smoothing import smooth_path
from padel_analysis.contact.learned import ContactModel
from padel_analysis.demo import learned_events, render, retrack
from padel_analysis.geometry.calibration import Calibration
from padel_analysis.io.scoreboard import Scoreboard
from padel_analysis.io.splices import SPLICE
from padel_analysis.io.video_source import VideoSource
from padel_analysis.rallies import DEFAULT_NAMES, RallySpec, build_rally
from padel_analysis.render.court_drawing import court_backdrop
from padel_analysis.render.figure_style import PLAYER
from padel_analysis.render.stats_panel import StatsPanel

FPS = 30.0
PANEL_WIDTH = 480
# The colours of the players of the panel, in BGR for OpenCV: a player has the same
# colour in the picture and in the figures.
PLAYER_BGR = {slot: tuple(int(c[i:i + 2], 16) for i in (5, 3, 1)) for slot, c in PLAYER.items()}
AFTER = 900  # frames read after the clip, for the score that follows its last rally


def to_h264(source: Path, target: Path) -> None:
    """Reencode for browsers: OpenCV writes MPEG-4 part 2, which they do not play."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg was not found in the PATH")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(target)],
        check=True,
    )


def splices_and_scores(match: str, start: int, stop: int, board: Scoreboard):
    """The splices inside the clip, and the score read at the start of each stretch."""
    cuts, readings = [], []
    stretch, previous, reads = start, None, []
    with VideoSource(Path(minutes.video(match))) as source:
        for index, image in source.iter_frames(start=start, stop=stop + AFTER):
            small = cv2.cvtColor(cv2.resize(image, (160, 90)), cv2.COLOR_BGR2GRAY)
            small = small.astype(np.float32)
            change = 0.0 if previous is None else float(np.abs(small - previous).mean())
            previous = small
            if change > SPLICE:
                if reads:
                    readings.append((stretch, Counter(reads).most_common(1)[0][0]))
                if index > stop and readings and readings[-1][0] > stop:
                    break
                stretch, reads = index, []
                if index <= stop:
                    cuts.append(index)
            if index - stretch in READS:
                state = board.read(image)
                if state is not None:
                    reads.append(state)
    if reads:
        readings.append((stretch, Counter(reads).most_common(1)[0][0]))
    return cuts, readings


def point_outcomes(rally, cuts, readings, start, stop) -> list[PointOutcome]:
    """Each rally of the clip whose score change is clear, credited to a player."""
    contacts = [(c.frame, c.kind) for c in rally.contacts]
    outcomes = []
    for span in rallies(cuts, contacts, start, stop):
        before = next((state for first, state in readings if first == span.start), None)
        after = next((state for first, state in readings if first > span.start), None)
        if before is None or after is None:
            continue
        row = point_winner(before, after)
        strikes = [(c.frame, c.player) for c in rally.contacts
                   if c.kind == "raquette" and c.player and span.start <= c.frame <= span.stop]
        sides = serving_side(before.server, strikes)
        if row is None or row not in sides:
            continue
        credited = credit(sides[row], strikes)
        # The point is counted at the end of the rally: one second after the last contact.
        end = min(span.stop, span.contacts[-1][0] + 30)
        outcomes.append(PointOutcome(end, sides[row], *(credited or (None, None))))
    return outcomes


def pair_names(match, rally, cuts, readings, start, stop) -> dict[str, str] | None:
    """The scoreboard's name of the pair in each half, voted by the serves of the clip."""
    rows = PAIRS.get(match)
    if rows is None:
        return None
    contacts = [(c.frame, c.kind) for c in rally.contacts]
    votes = Counter()
    for span in rallies(cuts, contacts, start, stop):
        before = next((state for first, state in readings if first == span.start), None)
        strikes = [(c.frame, c.player) for c in rally.contacts
                   if c.kind == "raquette" and c.player and span.start <= c.frame <= span.stop]
        if before is None:
            continue
        for row, side in serving_side(before.server, strikes).items():
            if side == "near":
                votes[row] += 1
    if not votes:
        return None
    near = votes.most_common(1)[0][0]
    return {"proche": rows[near], "fond": rows[3 - near]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True)
    parser.add_argument("--minute", type=int, required=True)
    parser.add_argument("--start", type=int, help="by default, the start of the minute")
    parser.add_argument("--stop", type=int, help="by default, the end of the minute")
    parser.add_argument("--contact-model", type=Path, required=True)
    parser.add_argument("--tag", default="360")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay", action="store_true",
                        help="draw on the rebuilt court, without the broadcast picture")
    args = parser.parse_args()

    analysis = pickle.loads(
        Path(minutes.analysis(args.match, args.minute, args.tag)).read_bytes()
    )
    start = analysis["start"] if args.start is None else args.start
    stop = analysis["stop"] if args.stop is None else args.stop
    if not analysis["start"] <= start <= stop <= analysis["stop"]:
        raise SystemExit("the clip goes beyond the analysed minute")

    calibration = Calibration.load(minutes.calibration(args.match))
    _, shown, events, pose = learned_events(analysis, calibration.points,
                                            ContactModel.load(args.contact_model))
    # The contacts come from the analysis as it was measured; the identities of the
    # players, from the tracking as it is today.
    print("player tracking replayed", flush=True)
    analysis = retrack(analysis, Path(minutes.video(args.match)), calibration)
    spec = RallySpec("extrait", args.match, args.minute, start, stop, "")
    print("scoreboard read", flush=True)
    board = Scoreboard.from_examples(EXAMPLES, frame_of)
    cuts, readings = splices_and_scores(args.match, start, stop, board)
    rally = build_rally(analysis, events, spec, FPS)
    outcomes = point_outcomes(rally, cuts, readings, start, stop)
    print(f"{len(outcomes)} points attributed", flush=True)
    timeline = LiveTimeline(rally, splices=cuts, points=outcomes)
    panel = StatsPanel(PANEL_WIDTH, analysis["size"][1], dict(DEFAULT_NAMES),
                       pair_names(args.match, rally, cuts, readings, start, stop))
    drawn = smooth_path(shown, cuts=[e.frame for e in events])

    with tempfile.TemporaryDirectory() as scratch:
        raw = Path(scratch) / "brut.mp4"
        render(Path(minutes.video(args.match)), analysis, events, pose, drawn, raw, start,
               stop, side=lambda frame: panel.draw(timeline.at(frame)), minimap=False,
               labels=dict(DEFAULT_NAMES), colours=PLAYER_BGR, tracked_only=True,
               backdrop=court_backdrop(pose, analysis["size"]) if args.replay else None)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        to_h264(raw, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
