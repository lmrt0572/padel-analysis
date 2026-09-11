"""Interactive court calibration: click known court locations on a video frame.

Usage:
    python scripts/calibrate.py --video data/clips/match.mp4 --frame 500 \
        --out data/calibrations/match.json

A schematic of the court is drawn in the corner of the window with the point being
asked for highlighted, so there is no ambiguity about which intersection to click.
Press 'u' to undo the last click, 'q' to abort.

The last four points are control points: they are excluded from the homography fit
and used only to measure its accuracy, so the reported error is never optimistic.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from padel_analysis.geometry.calibration import CalibrationPoint, calibration_from_points
from padel_analysis.geometry.court import Court

WINDOW = "calibration"
N_CONTROL = 4

FIT_COLOR = (60, 220, 60)
CTRL_COLOR = (60, 160, 255)
DONE_COLOR = (140, 140, 140)
HILITE = (0, 255, 255)


def requested_points(court: Court) -> list[tuple[str, tuple[float, float]]]:
    """Court locations to click, in order. The last N_CONTROL become control points."""
    w, ln, s = court.half_width, court.half_length, court.service_line_distance
    return [
        # --- ajustement ---
        ("corner_near_left", (-w, -ln)),
        ("corner_near_right", (w, -ln)),
        ("corner_far_right", (w, ln)),
        ("corner_far_left", (-w, ln)),
        ("service_near_left", (-w, -s)),
        ("service_near_right", (w, -s)),
        ("service_far_left", (-w, s)),
        ("service_far_right", (w, s)),
        ("net_left", (-w, 0.0)),
        # --- controle ---
        ("net_right", (w, 0.0)),
        ("service_near_centre", (0.0, -s)),
        ("service_far_centre", (0.0, s)),
        ("net_centre", (0.0, 0.0)),
    ]


def court_schema(court: Court, wanted, current: int, width: int = 300) -> np.ndarray:
    """Top-down schematic of the court with the requested point highlighted."""
    w, ln, s = court.half_width, court.half_length, court.service_line_distance
    margin = 34
    scale = (width - 2 * margin) / (2 * w)
    height = int(2 * ln * scale) + 2 * margin
    img = np.full((height, width, 3), 35, dtype=np.uint8)

    def px(x: float, y: float) -> tuple[int, int]:
        # y court positif = fond eloigne = haut de l'image
        return (int(margin + (x + w) * scale), int(margin + (ln - y) * scale))

    cv2.rectangle(img, px(-w, ln), px(w, -ln), (110, 55, 28), -1)
    cv2.rectangle(img, px(-w, ln), px(w, -ln), (235, 235, 235), 2)
    for yy in (-s, s):
        cv2.line(img, px(-w, yy), px(w, yy), (235, 235, 235), 1)
    cv2.line(img, px(0, -s), px(0, s), (235, 235, 235), 1)
    cv2.line(img, px(-w, 0), px(w, 0), (80, 220, 255), 2)

    cv2.putText(img, "FOND ELOIGNE", (margin - 26, margin - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)
    cv2.putText(img, "FOND PROCHE", (margin - 24, height - margin + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)

    n_fit = len(wanted) - N_CONTROL
    for i, (_, (cx, cy)) in enumerate(wanted):
        p = px(cx, cy)
        if i < current:
            color, radius = DONE_COLOR, 7
        elif i == current:
            color, radius = HILITE, 13
        else:
            color, radius = (FIT_COLOR if i < n_fit else CTRL_COLOR), 9
        cv2.circle(img, p, radius, color, -1)
        cv2.circle(img, p, radius, (15, 15, 15), 1)
        label = str(i + 1)
        off = -4 if i + 1 < 10 else -8
        cv2.putText(img, label, (p[0] + off, p[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (10, 10, 10), 1)
    return img


def magnifier(frame: np.ndarray, cursor: tuple[int, int], half: int = 44,
              zoom: int = 5) -> np.ndarray:
    """A zoomed view around the cursor, with a crosshair on the exact pixel.

    The far end of the court is heavily foreshortened - the far baseline and the far
    service line sit about thirty pixels apart - so clicking there unaided would
    dominate the calibration error.
    """
    h, w = frame.shape[:2]
    cx = int(np.clip(cursor[0], half, w - half - 1))
    cy = int(np.clip(cursor[1], half, h - half - 1))
    patch = frame[cy - half:cy + half, cx - half:cx + half]
    big = cv2.resize(patch, (2 * half * zoom, 2 * half * zoom),
                     interpolation=cv2.INTER_NEAREST)

    centre = half * zoom
    cv2.line(big, (centre, 0), (centre, big.shape[0]), HILITE, 1)
    cv2.line(big, (0, centre), (big.shape[1], centre), HILITE, 1)
    cv2.circle(big, (centre, centre), zoom, HILITE, 1)
    cv2.rectangle(big, (0, 0), (big.shape[1] - 1, big.shape[0] - 1), (235, 235, 235), 2)
    cv2.putText(big, f"x{zoom}  ({cx},{cy})", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 4)
    cv2.putText(big, f"x{zoom}  ({cx},{cy})", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (235, 235, 235), 1)
    return big


def paste_inset(canvas: np.ndarray, inset: np.ndarray, top: bool, left: bool) -> None:
    """Paste an inset into a corner of the canvas, in place."""
    ch, cw = canvas.shape[:2]
    ih, iw = inset.shape[:2]
    if ih + 20 >= ch or iw + 20 >= cw:
        return
    y0 = 10 if top else ch - ih - 10
    x0 = 10 if left else cw - iw - 10
    canvas[y0:y0 + ih, x0:x0 + iw] = inset


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
    n_fit = len(wanted) - N_CONTROL
    clicks: list[tuple[float, float]] = []
    cursor = [frame.shape[1] // 2, frame.shape[0] // 2]

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        cursor[0], cursor[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(wanted):
            clicks.append((float(x), float(y)))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while len(clicks) < len(wanted):
        canvas = frame.copy()
        for cx, cy in clicks:
            cv2.drawMarker(canvas, (int(cx), int(cy)), FIT_COLOR,
                           cv2.MARKER_CROSS, 22, 2)
        i = len(clicks)
        name, court_xy = wanted[i]
        kind = "CONTROLE" if i >= n_fit else "ajustement"
        text = f"[{i + 1}/{len(wanted)}] {kind} - {name}  court={court_xy}"
        for color, thick in (((0, 0, 0), 5), (HILITE, 2)):
            cv2.putText(canvas, text, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        color, thick)
            cv2.putText(canvas, "u = annuler le dernier clic    q = abandonner",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,
                        max(1, thick - 2))

        # Les encarts basculent du cote oppose au curseur, pour ne jamais masquer
        # le point que l'on cherche a cliquer.
        on_left = cursor[0] > canvas.shape[1] // 2
        paste_inset(canvas, court_schema(court, wanted, i), top=True, left=on_left)
        paste_inset(canvas, magnifier(frame, (cursor[0], cursor[1])),
                    top=False, left=on_left)

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
            is_control=(i >= n_fit),
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
