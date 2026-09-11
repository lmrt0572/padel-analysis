"""Interactive court calibration: click known court locations on a video frame.

Usage:
    python scripts/calibrate.py --video data/clips/match.mp4 --frame 500 \
        --out data/calibrations/match.json

Click each requested point in the order prompted on screen. Press 'u' to undo the
last click, 'q' to abort. The last two points are treated as control points: they
are excluded from the fit so that the reported error is honest.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from padel_analysis.geometry.calibration import CalibrationPoint, calibration_from_points
from padel_analysis.geometry.court import Court

WINDOW = "calibration - click the prompted point"
N_CONTROL = 2


def requested_points(court: Court) -> list[tuple[str, tuple[float, float]]]:
    """Court locations to click, in order. The last two become control points."""
    corners = court.corners()
    s = court.service_line_distance
    return [
        ("corner_near_left", tuple(corners[0])),
        ("corner_near_right", tuple(corners[1])),
        ("corner_far_right", tuple(corners[2])),
        ("corner_far_left", tuple(corners[3])),
        ("service_far_left", (-court.half_width, s)),
        ("service_far_right", (court.half_width, s)),
        # Points de controle : exclus de l'ajustement.
        ("service_near_centre", (0.0, -s)),
        ("net_centre", (0.0, 0.0)),
    ]


def grab_frame(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {index} from {video}")
    return frame


def collect_clicks(frame: np.ndarray, court: Court) -> list[CalibrationPoint]:
    wanted = requested_points(court)
    clicks: list[tuple[float, float]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(wanted):
            clicks.append((float(x), float(y)))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while len(clicks) < len(wanted):
        canvas = frame.copy()
        for cx, cy in clicks:
            cv2.circle(canvas, (int(cx), int(cy)), 6, (0, 255, 0), 2)
        name, court_xy = wanted[len(clicks)]
        kind = "CONTROL" if len(clicks) >= len(wanted) - N_CONTROL else "fit"
        label = f"[{len(clicks) + 1}/{len(wanted)}] {kind} - click: {name}  court={court_xy}"
        cv2.putText(
            canvas, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
        cv2.putText(
            canvas, "u = undo    q = abort", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (200, 200, 200), 1
        )
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("aborted")
        if key == ord("u") and clicks:
            clicks.pop()

    cv2.destroyAllWindows()

    return [
        CalibrationPoint(
            name=name,
            court_xy=court_xy,
            image_xy=click,
            is_control=(i >= len(wanted) - N_CONTROL),
        )
        for i, ((name, court_xy), click) in enumerate(zip(wanted, clicks))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    court = Court()
    frame = grab_frame(args.video, args.frame)
    points = collect_clicks(frame, court)
    calibration = calibration_from_points(points)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    calibration.save(args.out)

    print(f"saved {args.out}")
    if calibration.error is None:
        print("no control points: accuracy unknown")
    else:
        e = calibration.error
        print(f"control points: {calibration.n_control_points}")
        print(f"  reprojection RMSE  : {e.rmse_pixels:.2f} px | {e.rmse_metres * 100:.1f} cm")
        print(f"  reprojection median: {e.median_pixels:.2f} px | {e.median_metres * 100:.1f} cm")


if __name__ == "__main__":
    main()
