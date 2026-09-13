"""Demonstration video: players, minimap, ball, and what each contact hit.

Two passes, and they cannot be one. The ball is chosen over the whole sequence at
once, so nothing can be drawn while the frames are still being read. The first pass
analyses every frame and keeps what the drawing needs; the second reads the video
again and draws.

Contacts here come from the reconstructed path, not from annotated positions as in
the measurements, so expect more of them than really happened - the contact stage
flags one that did not happen in four on annotated positions, and more on this path.

Usage:
    python -m padel_analysis.demo --video <video.mp4> \
        --calibration ground_truth/calibrations/<nom>.json \
        --weights weights/ball_net.pt --start 16000 --frames 1800 --out outputs/demo.mp4
"""

import argparse
from pathlib import Path

import numpy as np

from .ball.candidates import demote_inside_boxes
from .ball.contacts import find_contacts
from .ball.heatmap_net import NetCandidates
from .ball.path import best_path
from .contact.surfaces import classify
from .geometry.calibration import Calibration
from .geometry.camera import court_surfaces, pose_from_calibration
from .geometry.court import Court
from .io.video_source import VideoSource
from .perception.appearance import torso_histogram
from .perception.ground_point import AnkleMidpoint
from .perception.pose_detector import PoseDetector
from .render.ball_overlay import (
    LABEL_COLOURS,
    ContactEvent,
    contact_label,
    draw_ball,
    trail,
    visible_events,
)
from .render.minimap import Minimap
from .render.overlay import draw_people, paste_minimap
from .render.video_writer import VideoWriter
from .tracking.court_constraint import CourtObservation, CourtSlotTracker

SPACING = 3
LEFT_WRIST, RIGHT_WRIST = 9, 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--start", type=int, default=16000)
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--hold", type=int, default=20, help="frames d'affichage d'un contact")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    court = Court()
    calibration = Calibration.load(args.calibration)
    detector = PoseDetector()
    finder = NetCandidates(args.weights, spacing=SPACING)
    tracker = CourtSlotTracker()
    strategy = AnkleMidpoint()

    start, stop = args.start, args.start + args.frames - 1
    people, assignments, positions, wrists, candidates = {}, {}, {}, {}, {}

    print("passe 1 : analyse", flush=True)
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

            people[middle] = detections
            assignments[middle] = assignment
            positions[middle] = {
                name: (float(observations[i].court_xy[0]), float(observations[i].court_xy[1]))
                for name, i in assignment.items()
            }
            wrists[middle] = [
                (float(k[0]), float(k[1]))
                for d in detections
                for k in (d.keypoints[LEFT_WRIST], d.keypoints[RIGHT_WRIST])
                if k[2] > 0.3
            ]
            candidates[middle] = demote_inside_boxes(
                finder(window, middle), [d.bbox for d in detections]
            )
            if (middle - start + 1) % 300 == 0:
                print(f"  {middle - start + 1}/{args.frames}", flush=True)

    path = best_path(candidates, start, stop)
    pose = pose_from_calibration(calibration.points, size)
    surfaces = court_surfaces(court)

    events: list[ContactEvent] = []
    for contact in find_contacts(path):
        ball = path[contact.frame]
        verdict = classify(ball, wrists.get(contact.frame, []), pose, surfaces)
        label = contact_label(verdict)
        if label is None:
            continue
        court_xy = None if verdict.point is None else (verdict.point[0], verdict.point[1])
        events.append(ContactEvent(contact.frame, label, ball, court_xy))
    print(f"{len(events)} contacts etiquetes", flush=True)


    minimap = Minimap(court)
    print("passe 2 : rendu", flush=True)
    with VideoSource(args.video) as source, VideoWriter(
        args.out, source.metadata.fps, size
    ) as writer:
        for index, frame in source.iter_frames(start=start, stop=stop + 1):
            shown = visible_events(events, index, args.hold)
            canvas = draw_people(frame, people.get(index, []), assignments.get(index, {}))
            canvas = draw_ball(canvas, trail(path, index), shown)
            impacts = [
                (e.court_xy, LABEL_COLOURS[e.label])
                for e in visible_events(events, index, hold=90)
                if e.court_xy is not None
            ]
            canvas = paste_minimap(
                canvas, minimap.draw(positions.get(index, {}), impacts=impacts)
            )
            writer.write(canvas)
    print(f"ecrit {args.out}")


if __name__ == "__main__":
    main()
