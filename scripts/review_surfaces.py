"""Human arbitration: what did the ball bounce off?

Each contact is replayed in a loop, the ball circled, with its trail before and after.
The contact to judge is the one under the magenta cross.

    s  the FLOOR
    v  a GLASS panel
    g  the MESH
    t  the NET
    f  a STROKE, so a racket
    n  NO contact: the trajectory goes straight through
    x  unreadable, I cannot decide

    r  go back to the previous clip and cancel its answer
    q  quit, keeping the answers given

`x` is left out of the computation; `n` counts as a false positive of the contact stage.
The tool never shows what the rule predicts.

Usage:
    python scripts/review_surfaces.py --video <video.mp4> \
        --annotations <ball.json> --truth ground_truth/surfaces/<name>.json
"""

import argparse
import json
from pathlib import Path

import cv2

from padel_analysis.eval.ball_dataset import BallAnnotations
from padel_analysis.eval.surface_truth import SurfaceGroundTruth
from padel_analysis.io.video_source import VideoSource

WINDOW = "contre quoi la balle a-t-elle rebondi ?"
KEYS = (
    "s sol   v vitre   g grillage   t filet   f frappe   n aucun   x illisible"
    "   r retour   q quitter"
)
ANSWER_KEYS = {
    "s": "sol",
    "v": "verre",
    "g": "grillage",
    "t": "filet",
    "f": "raquette",
    "n": "aucun",
    "x": "x",
}
SPAN = 12
TRAIL = (60, 200, 255)
BALL = (0, 220, 255)
MARK = (255, 80, 255)


def draw(frame, centres, index, contact, caption):
    """Return the frame with the ball's trail and the fixed marker of the instant to judge."""
    canvas = frame.copy()
    for offset in range(-SPAN, SPAN + 1):
        point = centres.get(contact + offset)
        if point is None:
            continue
        cv2.circle(canvas, (int(point[0]), int(point[1])), 3, TRAIL, 2)

    impact = centres.get(contact)
    if impact is not None:
        x, y = int(impact[0]), int(impact[1])
        cv2.circle(canvas, (x, y), 30, MARK, 2)
        cv2.line(canvas, (x - 48, y), (x - 16, y), MARK, 2)
        cv2.line(canvas, (x + 16, y), (x + 48, y), MARK, 2)
        cv2.line(canvas, (x, y - 48), (x, y - 16), MARK, 2)
        cv2.line(canvas, (x, y + 16), (x, y + 48), MARK, 2)

    current = centres.get(index)
    if current is not None:
        cv2.circle(canvas, (int(current[0]), int(current[1])), 10, BALL, 2)

    if index == contact:
        cv2.putText(canvas, "CONTACT", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 0), 6)
        cv2.putText(canvas, "CONTACT", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    MARK, 2)

    for text, y, scale in ((caption, 44, 0.9), (KEYS, 82, 0.65)):
        cv2.putText(
            canvas, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5
        )
        cv2.putText(
            canvas, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2
        )
    return canvas


def ask(source, centres, contact, caption):
    """Replay the sequence in a loop until a valid key is pressed."""
    accepted = set(ANSWER_KEYS) | {"r", "q"}
    while True:
        for index, frame in source.iter_frames(
            start=max(0, contact - SPAN), stop=contact + SPAN + 1
        ):
            cv2.imshow(WINDOW, draw(frame, centres, index, contact, caption))
            # the playback lingers on the instant to judge
            key = cv2.waitKey(320 if index == contact else 55) & 0xFF
            if key != 255 and chr(key) in accepted:
                return chr(key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument(
        "--positions",
        type=Path,
        help="computed positions to show in place of the annotated ball, to judge "
        "what the system claims and not what the annotation shows",
    )
    args = parser.parse_args()

    truth = SurfaceGroundTruth.load(args.truth)
    centres = BallAnnotations.load(args.annotations).centres()
    if args.positions is not None:
        claimed = json.loads(args.positions.read_text(encoding="utf-8"))
        centres = {int(f): tuple(xy) for f, xy in claimed.items()}
    total = len(truth.tasks)
    answered = [t for t in truth.tasks if t.frame in truth.answers]

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    with VideoSource(args.video) as source:
        while True:
            pending = truth.pending()
            if not pending:
                print(f"\ncampaign finished: {total} contacts arbitrated")
                break
            task = pending[0]
            caption = f"{total - len(pending) + 1} / {total}   frame {task.frame}"
            key = ask(source, centres, task.frame, caption)

            if key == "q":
                print(f"\nstopped: {len(truth.answers)} / {total} arbitrated")
                break
            if key == "r":
                if answered:
                    truth.undo(answered.pop().frame)
                    truth.save(args.truth)
                continue
            truth.answer(task.frame, ANSWER_KEYS[key])
            truth.save(args.truth)
            answered.append(task)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
