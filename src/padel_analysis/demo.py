"""Demonstration video: players, minimap, ball, and the court zone each contact lit.

Two passes, and they cannot be one. The ball is chosen over the whole sequence at
once, so nothing can be drawn while frames are still being read. The first pass
analyses every frame and saves what the drawing needs; the second reads the video
again and draws. `--reuse` skips the first pass, so the display can be tuned without
paying for the analysis again.

What is shown is filtered, and the filter is for display only. The path answers on
every frame, so when the ball leaves the picture or rests in a server's hand it still
invents a trajectory. Points the detector was not confident about are hidden, lone
aberrant points are dropped, and each piece between two contacts is smoothed so the
trail reads fluidly without rounding the bounces. The figures in the evaluation report are computed without
this filter and are not affected by it.

Usage:
    python -m padel_analysis.demo --video <video.mp4> \
        --calibration ground_truth/calibrations/<nom>.json \
        --weights weights/ball_net.pt --start 16000 --frames 1800 --out outputs/demo.mp4
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from .ball.candidates import demote_inside_boxes
from .ball.confidence import confident_path, path_scores
from .ball.contacts import find_contacts
from .ball.path import best_path
from .ball.smoothing import despike, smooth_path
from .contact.surfaces import classify
from .geometry.calibration import Calibration
from .geometry.camera import court_surfaces, pose_from_calibration
from .geometry.court import Court
from .io.video_source import VideoSource
from .perception.appearance import torso_histogram
from .perception.ground_point import AnkleMidpoint
from .perception.pose_detector import PoseDetector
from .render.ball_overlay import (
    ContactEvent,
    contact_label,
    draw_ball,
    draw_hitter,
    hitter_box,
    trail,
    visible_events,
)
from .render.court_zones import draw_zone, zone_of
from .render.minimap import Minimap
from .render.overlay import draw_people, paste_minimap
from .render.video_writer import VideoWriter
from .tracking.court_constraint import CourtObservation, CourtSlotTracker

SPACING = 3
LEFT_WRIST, RIGHT_WRIST = 9, 10


def analyse(args: argparse.Namespace) -> dict:
    """First pass: everything the drawing needs, frame by frame, then the ball path."""
    from .ball.heatmap_net import NetCandidates

    calibration = Calibration.load(args.calibration)
    detector = PoseDetector()
    finder = NetCandidates(args.weights, spacing=SPACING)
    tracker = CourtSlotTracker()
    strategy = AnkleMidpoint()
    start, stop = args.start, args.start + args.frames - 1
    frames: dict[int, dict] = {}

    with VideoSource(args.video) as source:
        size = (source.metadata.width, source.metadata.height)
        window: dict[int, np.ndarray] = {}
        for index, frame in source.iter_frames(
            start=max(0, start - SPACING), stop=stop + SPACING + 1
        ):
            window[index] = frame
            for old in [f for f in window if f < index - 2 * SPACING]:
                del window[old]
            middle = index - SPACING
            if not start <= middle <= stop:
                continue

            detections = detector.detect(window[middle])
            observations = []
            for detection in detections:
                point, confidence = strategy(detection)
                court_xy = calibration.projector.image_to_court(point.reshape(1, 2))[0]
                observations.append(
                    CourtObservation(
                        court_xy=court_xy,
                        confidence=confidence,
                        appearance=torso_histogram(window[middle], detection.bbox),
                    )
                )
            assignment = tracker.update(observations)
            raw = finder(window, middle)
            frames[middle] = {
                "people": detections,
                "assignment": assignment,
                "positions": {
                    name: (
                        float(observations[i].court_xy[0]),
                        float(observations[i].court_xy[1]),
                    )
                    for name, i in assignment.items()
                },
                "wrists": [
                    (float(k[0]), float(k[1]))
                    for d in detections
                    for k in (d.keypoints[LEFT_WRIST], d.keypoints[RIGHT_WRIST])
                    if k[2] > 0.3
                ],
                "raw": raw,
                "candidates": demote_inside_boxes(raw, [d.bbox for d in detections]),
            }
            if (middle - start + 1) % 300 == 0:
                print(f"  {middle - start + 1}/{args.frames}", flush=True)

    path = best_path({f: v["candidates"] for f, v in frames.items()}, start, stop)
    return {"start": start, "stop": stop, "size": size, "frames": frames, "path": path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--start", type=int, default=16000)
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="score du reseau sous lequel la balle n'est pas affichee. Mesure sur une "
        "minute de match annotee : a 0,7 et 8 images, les trajectoires affichees la ou "
        "aucune balle n'est annotee passent de 222 a 36, pour 97,5 %% des positions "
        "justes conservees",
    )
    parser.add_argument(
        "--min-run",
        type=int,
        default=8,
        help="images consecutives sures pour afficher une trajectoire",
    )
    parser.add_argument("--glow", type=int, default=30, help="frames d'eclairage d'une zone")
    parser.add_argument("--reuse", action="store_true", help="relire l'analyse sauvegardee")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    saved = args.out.with_name(args.out.stem + "_analysis.pkl")
    if args.reuse:
        analysis = pickle.loads(saved.read_bytes())
    else:
        print("passe 1 : analyse", flush=True)
        analysis = analyse(args)
        saved.write_bytes(pickle.dumps(analysis))

    court = Court()
    frames, path = analysis["frames"], analysis["path"]
    scores = path_scores(path, {f: v["raw"] for f, v in frames.items()})
    # Pour l'affichage seulement : confiance, puis retrait des points isoles aberrants.
    shown = despike(confident_path(path, scores, args.threshold, args.min_run))

    pose = pose_from_calibration(Calibration.load(args.calibration).points, analysis["size"])
    surfaces = court_surfaces(court)
    events: list[ContactEvent] = []
    for contact in find_contacts(shown):
        ball = shown[contact.frame]
        verdict = classify(ball, frames[contact.frame]["wrists"], pose, surfaces)
        label = contact_label(verdict)
        if label is None:
            continue
        # Un rayon qui ne rencontre qu'une surface pointe hors du jeu. Mesure sur les
        # deux matchs annotes : ces contacts portent 38 des 42 faux murs, pour 2 vrais
        # murs sur 33. Filtre d'affichage, les chiffres mesures ne le connaissent pas.
        if label != "RAQUETTE" and verdict.candidates == 1:
            continue
        box = hitter_box(ball, frames[contact.frame]["people"]) if label == "RAQUETTE" else None
        events.append(
            ContactEvent(contact.frame, label, ball, zone_of(verdict, court), box)
        )
    drawn = smooth_path(shown, cuts=[e.frame for e in events])
    hidden = sum(1 for f in path if path[f] is not None and shown[f] is None)
    print(
        f"balle masquee sur {hidden} images par manque de confiance ; "
        f"{len(events)} contacts affiches",
        flush=True,
    )

    minimap = Minimap(court)
    print("passe 2 : rendu", flush=True)
    with VideoSource(args.video) as source, VideoWriter(
        args.out, source.metadata.fps, analysis["size"]
    ) as writer:
        for index, frame in source.iter_frames(
            start=analysis["start"], stop=analysis["stop"] + 1
        ):
            data = frames.get(index, {})
            canvas = draw_people(frame, data.get("people", []), data.get("assignment", {}))
            for event in visible_events(events, index, args.glow):
                strength = 1.0 - (index - event.frame) / args.glow
                if event.zone is not None:
                    canvas = draw_zone(canvas, event.zone, pose, strength)
                elif event.box is not None:
                    canvas = draw_hitter(canvas, event.box, strength)
            canvas = draw_ball(canvas, trail(drawn, index), [])
            canvas = paste_minimap(canvas, minimap.draw(data.get("positions", {})))
            writer.write(canvas)
    print(f"ecrit {args.out}")


if __name__ == "__main__":
    main()
