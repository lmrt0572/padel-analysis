"""Arbitrage humain des episodes ou deux partenaires se croisent.

Pour chaque episode, la video est rejouee en boucle autour du moment le plus serre,
avec les deux joueurs concernes entoures de leur couleur de slot. La question est
binaire : ont-ils permute, ou non ?

Touches :
    n  ils n'ont PAS permute, l'association automatique etait bonne
    s  ils ONT permute : les deux slots sont echanges a partir de cet episode
    q  abandonner (les arbitrages deja rendus sont conserves)

Usage:
    python scripts/review_identity.py --video <video.mp4> \
        --annotations <pose.json> --identity data/identity/<nom>.json
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from padel_analysis.eval.dataset import PoseAnnotations
from padel_analysis.eval.identity import IdentityGroundTruth
from padel_analysis.io.video_source import VideoSource
from padel_analysis.render.minimap import TEAM_COLOURS

WINDOW = "revue d'identite"
MARGIN = 45


def draw_episode_frame(
    frame: np.ndarray,
    people: list,
    row: dict[str, int],
    slots: tuple[str, str],
    caption: str,
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
        cv2.putText(canvas, "n = pas de permutation    s = permutation    q = quitter",
                    (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour,
                    max(1, thickness - 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    args = parser.parse_args()

    print("chargement...", flush=True)
    annotations = PoseAnnotations.load(args.annotations)
    truth = IdentityGroundTruth.load(args.identity)

    pending = [e for e in truth.episodes if e.start_frame not in truth.resolved]
    print(f"{len(pending)} episodes a arbitrer sur {len(truth.episodes)}")
    if not pending:
        print("rien a faire")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    with VideoSource(args.video) as source:
        for position, episode in enumerate(pending, start=1):
            start = max(0, episode.start_frame - MARGIN)
            stop = episode.end_frame + MARGIN
            caption = (f"[{position}/{len(pending)}] frames "
                       f"{episode.start_frame}-{episode.end_frame}  "
                       f"{episode.slots[0]} / {episode.slots[1]}  "
                       f"min {episode.min_separation_m:.2f} m")

            decision = None
            while decision is None:
                for index, frame in source.iter_frames(start=start, stop=stop):
                    people = annotations.for_frame(index)
                    row = truth.assignments.get(index, {})
                    cv2.imshow(
                        WINDOW,
                        draw_episode_frame(frame, people, row, episode.slots, caption),
                    )
                    key = cv2.waitKey(40) & 0xFF
                    if key in (ord("n"), ord("s"), ord("q")):
                        decision = chr(key)
                        break

            if decision == "q":
                break
            if decision == "s":
                truth.apply_swap(episode.start_frame, episode.slots)
                print(f"  episode {episode.start_frame}: PERMUTATION appliquee")
            else:
                print(f"  episode {episode.start_frame}: pas de permutation")
            truth.resolved.append(episode.start_frame)
            truth.save(args.identity)

    cv2.destroyAllWindows()
    print(f"\n{len(truth.resolved)}/{len(truth.episodes)} episodes arbitres")
    print(f"enregistre dans {args.identity}")


if __name__ == "__main__":
    main()
