"""Draw the calibrated court model over a real frame, to check the calibration by eye.

Usage:
    python scripts/overlay_court.py --video data/clips/match.mp4 --frame 500 \
        --calibration ground_truth/calibrations/match.json --out outputs/overlay.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from padel_analysis.geometry.calibration import Calibration
from padel_analysis.geometry.court import Court

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {args.frame}")

    court = Court()
    calibration = Calibration.load(args.calibration)
    projector = calibration.projector

    outline = projector.court_to_image(court.corners()).astype(np.int32)
    cv2.polylines(frame, [outline], isClosed=True, color=GREEN, thickness=2)

    s = court.service_line_distance
    for sign in (-1, 1):
        line = projector.court_to_image(
            np.array([[-court.half_width, sign * s], [court.half_width, sign * s]])
        ).astype(np.int32)
        cv2.polylines(frame, [line], isClosed=False, color=GREEN, thickness=2)

    net = projector.court_to_image(
        np.array([[-court.half_width, 0.0], [court.half_width, 0.0]])
    ).astype(np.int32)
    cv2.polylines(frame, [net], isClosed=False, color=YELLOW, thickness=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), frame)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
