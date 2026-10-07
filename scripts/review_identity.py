"""Human arbitration of the moments where identity may have slipped.

Two kinds of doubt, arbitrated one after the other.

CLOSE APPROACHES: two partners pass close enough for the association to hesitate. The
video is replayed in a loop with the two players concerned circled in the colour of
their slot.

    n  they have NOT switched, the automatic association was right
    s  they HAVE switched: the two slots are swapped from this episode on

BROADCAST CUTS: the camera changes, the players reappear elsewhere, and the association
ties them to the nearest to their position before. If two partners swapped positions
during the cut, it follows the wrong one. The two pairs can switch independently, hence
four answers.

    n  no switch
    p  the NEAR pair has switched
    e  the FAR pair has switched
    b  BOTH pairs have switched
    c  the TEAMS HAVE CHANGED ENDS

The change of ends is apart. The slots stand for a half of the court, not a person:
after a change of ends, `near_1` is someone else, and no swap of labels can express it.
The identity of the player stops there and starts again from zero: it is a boundary,
not a correction.

Do not judge the bodies but the colours: does the same player wear the same colour
before and after? A stable detail (cap, shoes, sleeves) is worth more than a general
impression.

    r  go back to the previous clip and cancel its answer
    q  give up (the arbitrations already given are kept)

Usage:
    python scripts/review_identity.py --video <video.mp4> \
        --annotations <pose.json> --identity ground_truth/identity/<name>.json
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from padel_analysis.eval.dataset import PoseAnnotations
from padel_analysis.eval.identity import (
    FAR_SLOTS,
    NEAR_SLOTS,
    SLOTS,
    IdentityGroundTruth,
)
from padel_analysis.io.video_source import VideoSource
from padel_analysis.render.minimap import TEAM_COLOURS

WINDOW = "revue d'identite"
MARGIN = 45


def draw_frame(
    frame: np.ndarray,
    people: list,
    row: dict[str, int],
    slots: tuple[str, ...],
    caption: str,
    keys: str,
) -> np.ndarray:
    canvas = frame.copy()
    for slot in slots:
        index = row.get(slot)
        if index is None or index >= len(people):
            continue
        x1, y1, x2, y2 = (round(float(v)) for v in people[index].bbox)
        colour = TEAM_COLOURS.get(slot, (200, 200, 200))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(canvas, slot, (x1, max(16, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(canvas, slot, (x1, max(16, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    for colour, thickness in (((0, 0, 0), 5), ((0, 255, 255), 2)):
        cv2.putText(canvas, caption, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    colour, thickness)
        cv2.putText(canvas, keys, (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    colour, max(1, thickness - 2))
    return canvas


def ask(
    source: VideoSource,
    annotations: PoseAnnotations,
    truth: IdentityGroundTruth,
    start: int,
    stop: int,
    slots: tuple[str, ...],
    caption: str,
    keys: str,
    allowed: str,
) -> str:
    """Replays the sequence in a loop until a valid key is pressed."""
    accepted = set(allowed) | {"q"}
    while True:
        for index, frame in source.iter_frames(start=max(0, start), stop=stop):
            people = annotations.for_frame(index)
            row = truth.assignments.get(index, {})
            cv2.imshow(
                WINDOW, draw_frame(frame, people, row, slots, caption, keys)
            )
            key = cv2.waitKey(40) & 0xFF
            if key != 255 and chr(key) in accepted:
                return chr(key)


def undo_episode(truth: IdentityGroundTruth, episode) -> str | None:
    """Undoes the answer given on a close approach. The swap is its own inverse."""
    key = truth.decisions.pop(f"episode:{episode.start_frame}", None)
    if key == "s":
        truth.apply_swap(episode.start_frame, episode.slots)
    if episode.start_frame in truth.resolved:
        truth.resolved.remove(episode.start_frame)
    return key


def undo_cut(truth: IdentityGroundTruth, cut) -> str | None:
    key = truth.decisions.pop(f"cut:{cut.frame}", None)
    if key is None and cut.frame in truth.boundaries:
        key = "c"  # answered before the answers were traced
    if key == "c" and cut.frame in truth.boundaries:
        truth.boundaries.remove(cut.frame)
    if key in ("p", "b"):
        truth.apply_swap(cut.frame, NEAR_SLOTS)
    if key in ("e", "b"):
        truth.apply_swap(cut.frame, FAR_SLOTS)
    if cut.frame in truth.resolved_cuts:
        truth.resolved_cuts.remove(cut.frame)
    return key


def review_episodes(source, annotations, truth, path) -> bool:
    """Retourne False si l'utilisateur a demande a quitter."""
    pending = [e for e in truth.episodes if e.start_frame not in truth.resolved]
    print(f"close approaches to arbitrate: {len(pending)}/{len(truth.episodes)}")

    position = 0
    while position < len(pending):
        episode = pending[position]
        caption = (f"[rapprochement {position + 1}/{len(pending)}] frames "
                   f"{episode.start_frame}-{episode.end_frame}  "
                   f"{episode.slots[0]} / {episode.slots[1]}  "
                   f"min {episode.min_separation_m:.2f} m")
        decision = ask(
            source, annotations, truth,
            episode.start_frame - MARGIN, episode.end_frame + MARGIN,
            episode.slots, caption,
            "n = pas de permutation    s = permutation    "
            "r = retour    q = quitter", "nsr",
        )
        if decision == "q":
            return False
        if decision == "r":
            if position > 0:
                position -= 1
                previous = pending[position]
            elif truth.resolved:
                done = truth.resolved[-1]
                previous = next(
                    (e for e in truth.episodes if e.start_frame == done), None
                )
                if previous is None:
                    print("  nothing to cancel")
                    continue
                pending.insert(0, previous)
            else:
                print("  nothing to cancel: no close approach answered")
                continue
            undone = undo_episode(truth, previous)
            truth.save(path)
            print(f"  back to {previous.start_frame} "
                  f"(reponse annulee : {undone or 'inconnue'})")
            continue

        if decision == "s":
            truth.apply_swap(episode.start_frame, episode.slots)
            print(f"  close approach {episode.start_frame}: SWITCH applied")
        else:
            print(f"  close approach {episode.start_frame}: no switch")
        truth.resolved.append(episode.start_frame)
        truth.decisions[f"episode:{episode.start_frame}"] = decision
        truth.save(path)
        position += 1
    return True


def review_cuts(source, annotations, truth, path) -> bool:
    pending = [c for c in truth.cuts if c.frame not in truth.resolved_cuts]
    print(f"cuts to arbitrate: {len(pending)}/{len(truth.cuts)}")

    position = 0
    while position < len(pending):
        cut = pending[position]
        # All five answers are always offered. The detector is used to find the clips to
        # watch; what is seen in them does not belong to it. Two close partners who swap
        # places travel less than the threshold, and the pair is therefore not announced,
        # without having any less switched.
        keys = ("n = aucune    p = proche    e = eloignee    b = les deux    "
                "c = CHANGEMENT DE COTE    r = retour    q = quitter")
        caption = (f"[coupure {position + 1}/{len(pending)}] frame {cut.frame}  "
                   f"paires signalees : {' + '.join(cut.sides)}  "
                   f"saut {cut.displacement_m:.1f} m")
        decision = ask(
            source, annotations, truth,
            cut.frame - MARGIN, cut.frame + MARGIN,
            SLOTS, caption, keys, "npebcr",
        )
        if decision == "q":
            return False

        if decision == "r":
            # At the first clip of the queue, the previous one was answered in an earlier
            # session: it is no longer pending, it has to be taken back from the record.
            if position > 0:
                position -= 1
                previous = pending[position]
            elif truth.resolved_cuts:
                done = truth.resolved_cuts[-1]
                previous = next((c for c in truth.cuts if c.frame == done), None)
                if previous is None:
                    print("  nothing to cancel")
                    continue
                pending.insert(0, previous)
            else:
                print("  nothing to cancel: no cut answered")
                continue
            undone = undo_cut(truth, previous)
            truth.save(path)
            if undone is None:
                print(f"  back to {previous.frame}: previous answer NOT "
                      f"traced, only the mark is removed. If it was a "
                      f"switch, apply it again then cancel once more")
            else:
                print(f"  back to {previous.frame} (answer cancelled: {undone})")
            continue

        if decision == "c":
            truth.boundaries.append(cut.frame)
            truth.boundaries.sort()
            print(f"  cut {cut.frame}: CHANGE OF ENDS, identity restarted")
        else:
            swapped = []
            if decision in ("p", "b"):
                truth.apply_swap(cut.frame, NEAR_SLOTS)
                swapped.append("proche")
            if decision in ("e", "b"):
                truth.apply_swap(cut.frame, FAR_SLOTS)
                swapped.append("eloignee")
            print(f"  cut {cut.frame}: "
                  f"{'SWITCH ' + ' and '.join(swapped) if swapped else 'none'}")

        truth.resolved_cuts.append(cut.frame)
        truth.decisions[f"cut:{cut.frame}"] = decision
        truth.save(path)
        position += 1
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    args = parser.parse_args()

    print("loading...", flush=True)
    annotations = PoseAnnotations.load(args.annotations)
    truth = IdentityGroundTruth.load(args.identity)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    with VideoSource(args.video) as source:
        if review_episodes(source, annotations, truth, args.identity):
            review_cuts(source, annotations, truth, args.identity)
    cv2.destroyAllWindows()

    print(f"\nclose approaches: {len(truth.resolved)}/{len(truth.episodes)}")
    print(f"cuts            : {len(truth.resolved_cuts)}/{len(truth.cuts)}")
    print(f"saved to {args.identity}")


if __name__ == "__main__":
    main()
