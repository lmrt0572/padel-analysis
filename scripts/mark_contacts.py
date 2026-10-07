"""Mark by hand every contact of a stretch of match.

The whole stretch is watched without seeing what the system detected, so any variant
can then be scored in precision and in recall.

    space   play / pause
    j  l    step back / forward one frame
    h  ;    step back / forward one second
    +  -    faster / slower playback

    s  the FLOOR      v  a GLASS panel      g  the MESH      t  the NET      f  a STROKE
       mark a contact on the frame shown; typing again on that frame changes its kind

    r  cancel the last mark
    n  p    with --revoir, go to the next / previous moment to review
    q  quit (everything is already saved)

Usage:
    python scripts/mark_contacts.py --video <video.mp4> --start 16000 --frames 1800 \
        --video-name FinalF --out ground_truth/contact_marks/FinalF_16000.json
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import cv2

from padel_analysis.eval.contact_marks import ContactMarks
from padel_analysis.io.video_source import VideoSource

WINDOW = "pointage des contacts"
KEYS = {"s": "sol", "v": "verre", "g": "grillage", "t": "filet", "f": "raquette"}
HELP = "espace lecture   j l image   h ; seconde   + - vitesse   s v g t f marquer   r annuler   q quitter"
REVIEW_HELP = "n p moment a revoir suivant / precedent"
DISPLAY = (1280, 720)
BUFFER = 150
CLOSE = 2
LEAD = 20  # frames shown before a moment to review


class Frames:
    """Fast sequential reader, with a buffer to go back without seeking."""

    def __init__(self, source: VideoSource) -> None:
        self.source = source
        self.buffer: dict[int, object] = {}
        self.stream = None
        self.next = None

    def get(self, index: int):
        if index in self.buffer:
            return self.buffer[index]
        if self.stream is None or index != self.next:
            self.stream = self.source.iter_frames(start=index)
            self.next = index
        while True:
            got, frame = next(self.stream)
            self.next = got + 1
            self.buffer[got] = cv2.resize(frame, DISPLAY)
            for old in [f for f in self.buffer if abs(f - got) > BUFFER]:
                del self.buffer[old]
            if got >= index:
                return self.buffer[index]


def draw(frame, marks: ContactMarks, index: int, playing: bool, delay: int, flash: str,
         review: Sequence[int] = ()):
    canvas = frame.copy()
    start, stop = marks.frame_range
    seconds = (index - start) / 30.0
    status = "LECTURE" if playing else "PAUSE"
    header = (
        f"image {index}   {seconds:5.1f} s   {status}   {1000 // max(delay, 1)} img/s   "
        f"{len(marks.marks)} contacts marques"
    )
    lines = [
        (header, 36, 0.8),
        (HELP + ("   " + REVIEW_HELP if review else ""), 70, 0.5),
    ]
    if flash:
        lines.append((flash, 118, 1.1))
    for text, y, scale in lines:
        cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
        cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)

    # timeline of the range: its own marks only, never the detections of the system
    width, height = DISPLAY
    top = height - 26
    cv2.rectangle(canvas, (16, top), (width - 16, top + 12), (40, 40, 40), -1)
    span = max(stop - start, 1)
    for frame_marked in marks.marks:
        x = 16 + round((frame_marked - start) / span * (width - 32))
        cv2.line(canvas, (x, top - 4), (x, top + 16), (0, 220, 255), 2)
    for moment in review:
        x = 16 + round((moment - start) / span * (width - 32))
        cv2.circle(canvas, (x, top - 10), 4, (255, 120, 0), -1)
    x = 16 + round((index - start) / span * (width - 32))
    cv2.line(canvas, (x, top - 8), (x, top + 20), (255, 255, 255), 3)
    return canvas


def mark_message(marks: ContactMarks, index: int, answer: str) -> str:
    """Return what the screen says after a mark, including any contact a frame or two away."""
    close = sorted(f for f in marks.marks if f != index and abs(f - index) <= CLOSE)
    message = f"MARQUE : {answer} a l'image {index}"
    if close:
        gaps = ", ".join(f"{marks.marks[f]} a {f - index:+d}" for f in close)
        message += f"   (aussi : {gaps})"
    return message


def run(video: Path, video_name: str, start: int, frames: int, out: Path,
        review: Sequence[int] = ()) -> None:
    """Open the marking window on one stretch; `review` lists moments to jump between."""
    stop = start + frames - 1
    if out.exists():
        marks = ContactMarks.load(out)
        if marks.frame_range != (start, stop):
            raise SystemExit(f"{out} covers the range {marks.frame_range}, not {(start, stop)}")
    else:
        marks = ContactMarks(video=video_name, frame_range=(start, stop), position=start)
        marks.save(out)

    review = sorted(review)
    index = min(max(marks.position, start), stop)
    playing, delay, flash, flash_left, current = False, 100, "", 0, -1
    if review:
        current, index = 0, max(review[0] - LEAD, start)
        flash, flash_left = f"A REVOIR 1/{len(review)} : contact pointe a l'image {review[0]}", 60
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    with VideoSource(video) as source:
        shown = Frames(source)
        while True:
            cv2.imshow(WINDOW, draw(shown.get(index), marks, index, playing, delay, flash, review))
            key = cv2.waitKey(delay if playing else 0) & 0xFF
            flash_left -= 1
            if flash_left <= 0:
                flash = ""

            if key == 255:
                if playing:
                    index = min(index + 1, stop)
                    playing = index < stop
                continue
            char = chr(key)
            if char == "q":
                break
            if char == " ":
                playing = not playing
            elif char in "jlh;":
                playing = False
                step = {"j": -1, "l": 1, "h": -30, ";": 30}[char]
                index = min(max(index + step, start), stop)
            elif char in "np" and review:
                playing = False
                current = min(max(current + (1 if char == "n" else -1), 0), len(review) - 1)
                index = max(review[current] - LEAD, start)
                flash = (f"A REVOIR {current + 1}/{len(review)} : contact pointe a l'image "
                         f"{review[current]}")
                flash_left = 60
            elif char == "+":
                delay = max(15, delay // 2)
            elif char == "-":
                delay = min(400, delay * 2)
            elif char in KEYS:
                marks.mark(index, KEYS[char])
                flash, flash_left = mark_message(marks, index, KEYS[char]), 40
            elif char == "r":
                undone = marks.undo()
                if undone:
                    flash, flash_left = f"ANNULE : {undone[1]} a l'image {undone[0]}", 25
            marks.position = index
            marks.save(out)
    marks.position = index
    marks.save(out)
    cv2.destroyWindow(WINDOW)
    print(f"{len(marks.marks)} contacts marked, position saved at frame {index}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revoir", type=Path,
                        help="JSON file of frames to review, browsed with n and p")
    args = parser.parse_args()
    review = json.loads(args.revoir.read_text(encoding="utf-8")) if args.revoir else ()
    run(args.video, args.video_name, args.start, args.frames, args.out, review)


if __name__ == "__main__":
    main()
