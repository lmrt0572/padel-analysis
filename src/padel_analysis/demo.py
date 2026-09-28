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
from .contact.surfaces import Verdict, classify
from .geometry.calibration import Calibration
from .geometry.camera import court_surfaces, pose_from_calibration
from .geometry.court import Court
from .io.video_source import VideoSource
from .perception.appearance import torso_histogram
from .perception.ground_point import AnkleMidpoint
from .perception.pose_detector import PoseDetector
from .rallies import strikers
from .render.ball_overlay import (
    ContactEvent,
    contact_label,
    draw_ball,
    draw_hitter,
    following_box,
    hitter_box,
    trail,
    visible_events,
)
from .render.court_zones import draw_zone, impact_patch, zone_of
from .render.minimap import Minimap
from .render.overlay import draw_people, paste_minimap
from .render.video_writer import VideoWriter
from .tracking.court_constraint import CourtObservation, CourtSlotTracker

SPACING = 3
GESTURE_SPEED = 8.0
STRIKE_SPEED = 20.0
CONTACT_SPAN = 3
CONTACT_SHARPNESS = 0.40
CONTACT_REACH = 2
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
            observations = observe(detections, window[middle], calibration, strategy)
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


def observe(detections, image, calibration, strategy) -> list[CourtObservation]:
    """Each detected person placed on the court, with the colours of their torso."""
    observations = []
    for detection in detections:
        point, confidence = strategy(detection)
        court_xy = calibration.projector.image_to_court(point.reshape(1, 2))[0]
        observations.append(CourtObservation(
            court_xy=court_xy,
            confidence=confidence,
            appearance=torso_histogram(image, detection.bbox),
        ))
    return observations


def retrack(analysis: dict, video: Path, calibration: Calibration) -> dict:
    """The same analysis with the players' identities tracked again.

    Detections are kept; only the assignment to the four slots is redone, with the
    tracker as it stands now. Reading the video again is needed for the torso colours,
    running the pose detector again is not. The saved analysis is left untouched: the
    measured figures were made on it.
    """
    tracker = CourtSlotTracker()
    strategy = AnkleMidpoint()
    frames = {}
    with VideoSource(video) as source:
        for index, image in source.iter_frames(start=analysis["start"],
                                                stop=analysis["stop"] + 1):
            data = analysis["frames"][index]
            observations = observe(data["people"], image, calibration, strategy)
            assignment = tracker.update(observations)
            frames[index] = {
                **data,
                "assignment": assignment,
                "positions": {
                    name: (float(observations[i].court_xy[0]), float(observations[i].court_xy[1]))
                    for name, i in assignment.items()
                },
            }
    return {**analysis, "frames": frames}


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


LABEL_OF_ANSWER = {"raquette": "RAQUETTE", "sol": "SOL", "verre": "VITRE",
                   "grillage": "GRILLAGE", "filet": "FILET"}


def learned_events(
    analysis: dict, calibration_points: list, model, threshold: float = 0.7,
    whole_zone: bool = True, infer_walls: bool = True,
) -> tuple[dict, dict, list[ContactEvent], object]:
    """The same as `build_events`, with the contacts decided by a trained model.

    The rule chain still runs: its decisions are one of the cues the model reads.
    Validated minute by minute over nine hand-marked minutes, each predicted by a
    model that never saw it: 556 contacts of 703 given the right surface, against 411
    for the rules.
    """
    from .contact.learned import decode, frame_features

    path, shown, rule_events, pose = build_events(analysis, calibration_points)
    court = Court()
    surfaces = court_surfaces(court)
    features, material = frame_features(analysis, path, shown, rule_events, pose, surfaces)
    probabilities = model.probabilities(features)
    answers = decode(probabilities, material, analysis["start"], threshold)
    if infer_walls:
        answers = with_inferred_walls(answers, probabilities, material, analysis, shown, path,
                                      pose, surfaces)

    frames = analysis["frames"]
    events: list[ContactEvent] = []
    for frame, answer in answers.items():
        ball = ball_near(frame, shown, path, CONTACT_REACH)
        if ball is None:
            continue
        label = LABEL_OF_ANSWER[answer]
        if label == "RAQUETTE":
            box = hitter_box(ball, frames[frame]["people"])
            events.append(ContactEvent(frame, label, ball, None, box))
            continue
        verdict = _verdict_on(label, ball, pose, surfaces)
        light = zone_of if whole_zone else impact_patch
        zone = light(verdict, court) if verdict is not None else None
        point = verdict.point if verdict is not None else None
        events.append(ContactEvent(frame, label, ball, zone, point=point))
    return path, shown, events, pose


def with_inferred_walls(answers: dict[int, str], probabilities: np.ndarray, material: np.ndarray,
                        analysis: dict, shown: dict, path: dict, pose, surfaces) -> dict[int, str]:
    """The model's contacts, with the walls that the pace of the ball implies.

    Each bounce is placed on the floor by the ray through the ball, each strike at the
    striker's feet; `inferred_walls` does the rest. Measured over the eleven training
    minutes, each predicted by a model that never saw it: 64 to 70 glass contacts of
    128 found at the right frame, the other contacts unchanged.
    """
    from .contact.glass_inference import Touch, inferred_walls
    from .contact.learned import CLASS_OF_ANSWER, MESH_WALL

    start, frames = analysis["start"], analysis["frames"]
    balls = {frame: ball_near(frame, shown, path, CONTACT_REACH) for frame in answers}
    players = strikers(frames, [(frame, hitter_box(balls[frame], frames[frame]["people"]),
                                 balls[frame])
                                for frame, answer in answers.items()
                                if answer == "raquette" and balls[frame] is not None])
    touches = []
    for frame, answer in sorted(answers.items()):
        ball = balls[frame]
        side = place = None
        if ball is not None and answer == "sol":
            meeting = surfaces[0].intersect(*pose.ray(ball))
            if meeting is not None:
                place = (float(meeting[0]), float(meeting[1]))
                side = "far" if place[1] > 0 else "near"
        elif players.get(frame) is not None:
            slot = players[frame]
            side = slot.split("_")[0]
            place = frames[frame]["positions"].get(slot)
        touches.append(Touch(frame, answer, side, place))

    result = dict(answers)
    for wall in inferred_walls(touches, probabilities[:, CLASS_OF_ANSWER["verre"]], start):
        frame = wall.bounce if wall.replaces_bounce else wall.frame
        if not wall.replaces_bounce and any(abs(frame - other) <= 3 for other in result):
            frame = max(frame, wall.bounce + 4)
        kind = "grillage" if material[frame - start] == MESH_WALL else "verre"
        if wall.replaces_bounce:
            result[frame] = kind
        else:
            result.setdefault(frame, kind)
    return dict(sorted(result.items()))


def ball_near(frame: int, shown: dict, path: dict, reach: int) -> tuple | None:
    """The ball at a contact, or at the nearest frame that has it, up to `reach` away.

    The ball is often missing at the very frame of a contact: the display filter drops
    the vertex of a sharp turn as a spike, and a strike hides the ball behind the
    racket. The contact itself is decided on every frame, ball or not.
    """
    for gap in range(reach + 1):
        for candidate in (frame - gap, frame + gap):
            ball = shown.get(candidate) or path.get(candidate)
            if ball is not None:
                return ball
    return None


def _verdict_on(label: str, ball, pose, surfaces) -> Verdict | None:
    """Where the ray through the ball meets the surface the model chose, for drawing.

    Several walls can be admissible at once, and the list order used to decide, which
    lit a back wall for a contact on a side one. The ball is on the first surface the
    ray reaches: anything behind it would be hidden by it.
    """
    origin, direction = pose.ray(ball)
    reached = []
    for surface in surfaces:
        is_floor, is_net = surface.name == "floor", surface.name == "net"
        if label == "SOL":
            wanted = is_floor
        elif label == "FILET":
            wanted = is_net
        else:
            wanted = not (is_floor or is_net)
        if not wanted:
            continue
        meeting = surface.intersect(origin, direction)
        if meeting is None:
            continue
        if not (is_floor or is_net) and not surface.contains(meeting, 0.30):
            continue
        reached.append((float(np.linalg.norm(meeting - origin)), surface, meeting))
    if not reached:
        return None
    _, surface, meeting = min(reached, key=lambda item: item[0])
    return Verdict(surface.name, meeting, surface.material_at(meeting), candidates=len(reached))


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
    parser.add_argument(
        "--hitter-glow",
        type=int,
        default=12,
        help="frames d'eclairage du frappeur. Plus court qu'une zone : le geste dure moins "
        "longtemps qu'un rebond ne se lit",
    )
    parser.add_argument("--reuse", action="store_true", help="relire l'analyse sauvegardee")
    parser.add_argument(
        "--contact-model",
        type=Path,
        help="modele appris de contacts (scripts/train_contact_model.py) ; sans lui, les "
        "contacts sont decides par les regles",
    )
    parser.add_argument(
        "--impact-patch",
        action="store_true",
        help="eclairer seulement le point d'impact, sur environ 1,5 m, au lieu de toute "
        "la zone touchee",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    saved = args.out.with_name(args.out.stem + "_analysis.pkl")
    if args.reuse:
        analysis = pickle.loads(saved.read_bytes())
    else:
        print("passe 1 : analyse", flush=True)
        analysis = analyse(args)
        saved.write_bytes(pickle.dumps(analysis))

    points = Calibration.load(args.calibration).points
    if args.contact_model is not None:
        from .contact.learned import ContactModel

        path, shown, events, pose = learned_events(
            analysis, points, ContactModel.load(args.contact_model),
            whole_zone=not args.impact_patch,
        )
    else:
        path, shown, events, pose = build_events(analysis, points, args.threshold, args.min_run)
    drawn = smooth_path(shown, cuts=[e.frame for e in events])
    hidden = sum(1 for f in path if path[f] is not None and shown[f] is None)
    print(
        f"balle masquee sur {hidden} images par manque de confiance ; "
        f"{len(events)} contacts affiches",
        flush=True,
    )

    print("passe 2 : rendu", flush=True)
    render(args.video, analysis, events, pose, drawn, args.out,
           analysis["start"], analysis["stop"], args.glow, args.hitter_glow)
    print(f"ecrit {args.out}")


def render(
    video: Path, analysis: dict, events: list[ContactEvent], pose, drawn: dict, out: Path,
    start: int, stop: int, glow: int = 30, hitter_glow: int = 12, side=None,
    minimap: bool = True, labels: dict | None = None, colours: dict | None = None,
) -> None:
    """Second pass: draw players, minimap, ball trail and lit contacts on each frame.

    Args:
        side: optional panel drawn to the right of the picture, as `side(frame)` returning
            an image of the video's height. The picture itself is left whole.
        minimap: paste the minimap in the picture's top-right corner.
        labels, colours: names and BGR colours of the players' boxes, by slot.
    """
    frames = analysis["frames"]
    court_map = Minimap(Court()) if minimap else None
    width, height = analysis["size"]
    if side is not None:
        width += side(start).shape[1]
    with VideoSource(video) as source, VideoWriter(
        out, source.metadata.fps, (width, height)
    ) as writer:
        for index, frame in source.iter_frames(start=start, stop=stop + 1):
            data = frames.get(index, {})
            canvas = draw_people(frame, data.get("people", []), data.get("assignment", {}),
                                 labels=labels, colours=colours)
            for event in visible_events(events, index, glow):
                if event.zone is not None:
                    strength = 1.0 - (index - event.frame) / glow
                    canvas = draw_zone(canvas, event.zone, pose, strength)
            for event in visible_events(events, index, hitter_glow):
                if event.box is not None:
                    strength = 1.0 - (index - event.frame) / hitter_glow
                    box = following_box(event.box, data.get("people", []))
                    canvas = draw_hitter(canvas, box, strength)
            canvas = draw_ball(canvas, trail(drawn, index), [])
            if court_map is not None:
                canvas = paste_minimap(canvas, court_map.draw(data.get("positions", {})))
            if side is not None:
                canvas = np.hstack([canvas, side(index)])
            writer.write(canvas)


if __name__ == "__main__":
    main()
