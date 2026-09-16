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
from .contact.gesture import gesture_near, strikes
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
GESTURE_SPEED = 8.0
STRIKE_SPEED = 20.0
CONTACT_SPAN = 3
CONTACT_SHARPNESS = 0.40
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


def build_events(
    analysis: dict, calibration_points: list, threshold: float = 0.7, min_run: int = 8
) -> tuple[dict, dict, list[ContactEvent], object]:
    """The displayed ball path and the contacts shown on it, from a saved analysis.

    Kept apart from the drawing so the measurement scripts score exactly what the
    video shows, and not a copy of it that could drift.
    """
    court = Court()
    frames = analysis["frames"]
    raw = {f: v["raw"] for f, v in frames.items()}
    # Le chemin d'affichage lit les scores du reseau tels quels, et peut donc renoncer
    # la ou la balle n'est pas. Mesure sur une minute annotee, avec le filtre de
    # confiance : 1 374 positions justes, 87 fausses, 2 fantomes, contre 1 174, 199 et
    # 27 avec le chemin relatif regle pour le rappel.
    path = best_path(
        raw, analysis["start"], analysis["stop"], weight=960.0, absent_cost=150.0,
        absolute=True,
    )
    scores = path_scores(path, raw)
    # Pour l'affichage seulement : confiance, puis retrait des points isoles aberrants.
    shown = despike(confident_path(path, scores, threshold, min_run))

    pose = pose_from_calibration(calibration_points, analysis["size"])
    surfaces = court_surfaces(court)
    events: list[ContactEvent] = []
    # Les contacts se cherchent sur une trajectoire lissee : chaque zigzag du chemin brut
    # passerait pour un virage. Mesure sur la minute annotee du match de reglage : 2 faux
    # contacts et 3 non juges en moins, pour 1 vrai perdu. Un lissage plus fort en perd 7.
    # Sur la trajectoire affichee, la vitesse se mesure sur 3 images et non 2, avec un
    # seuil de nettete plus bas : balaye sur quatre minutes pointees (316 contacts),
    # 177 -> 194 contacts justes ; sur une minute de l'autre match, 42 -> 47.
    players = {f: (v["people"], v["assignment"]) for f, v in frames.items()}
    turns = find_contacts(
        smooth_path(shown, cuts=[], process_noise=100.0), span=CONTACT_SPAN,
        sharpness=CONTACT_SHARPNESS,
    )
    for contact in turns:
        ball = shown.get(contact.frame)
        if ball is None:
            continue
        # Un poignet proche mais immobile ne fait pas une frappe : sans geste, la
        # raquette est ecartee et la surface se decide par la geometrie seule.
        gesture = max(
            gesture_near(players, f, shown.get(f)) for f in range(contact.frame - 3, contact.frame + 4)
        )
        wrists = frames[contact.frame]["wrists"] if gesture >= GESTURE_SPEED else []
        verdict = classify(ball, wrists, pose, surfaces)
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
    # Les frappes que le virage ne voit pas - un coup dans l'axe de la camera plie a
    # peine la trajectoire a l'image - se lisent au geste du frappeur. Mesure sur un
    # pointage complet : +11 contacts justes sur le match de reglage, +9 sur le match
    # tenu a l'ecart, reglage fige. Une frappe sans virage demande un geste plus franc
    # que celle qui confirme un virage : 20 px/image contre 8, balaye sur 316 contacts,
    # 19 contacts inventes en moins pour le meme nombre de justes.
    found = [e.frame for e in events]
    for frame in strikes(players, shown, analysis["start"], analysis["stop"], STRIKE_SPEED,
                         taken=found):
        ball = shown[frame]
        events.append(ContactEvent(frame, "RAQUETTE", ball, None,
                                   hitter_box(ball, frames[frame]["people"])))
    events.sort(key=lambda e: e.frame)
    return path, shown, events, pose


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

    path, shown, events, pose = build_events(
        analysis, Calibration.load(args.calibration).points, args.threshold, args.min_run
    )
    court = Court()
    frames = analysis["frames"]
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
