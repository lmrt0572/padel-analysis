"""Arbitrage humain des moments ou l'identite peut avoir decroche.

Deux natures de doute, arbitrees l'une apres l'autre.

RAPPROCHEMENTS - deux partenaires passent assez pres pour que l'association
hesite. La video est rejouee en boucle avec les deux joueurs concernes entoures
de leur couleur de slot.

    n  ils n'ont PAS permute, l'association automatique etait bonne
    s  ils ONT permute : les deux slots sont echanges a partir de cet episode

COUPURES DE PLAN - la camera change, les joueurs reapparaissent ailleurs, et
l'association les rattache au plus proche de leur position d'avant. Si deux
partenaires ont echange leur poste pendant la coupure, elle suit le mauvais. Les
deux paires peuvent permuter independamment, d'ou quatre reponses.

    n  aucune permutation
    p  la paire PROCHE a permute
    e  la paire ELOIGNEE a permute
    b  les DEUX paires ont permute

Ne juge pas les corps mais les couleurs : le meme joueur porte-t-il la meme
couleur avant et apres ? Un detail stable - casquette, chaussures, manches - vaut
mieux qu'une impression generale.

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
    """Rejoue la sequence en boucle jusqu'a ce qu'une touche valide soit frappee."""
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


def review_episodes(source, annotations, truth, path) -> bool:
    """Retourne False si l'utilisateur a demande a quitter."""
    pending = [e for e in truth.episodes if e.start_frame not in truth.resolved]
    print(f"rapprochements a arbitrer : {len(pending)}/{len(truth.episodes)}")

    for position, episode in enumerate(pending, start=1):
        caption = (f"[rapprochement {position}/{len(pending)}] frames "
                   f"{episode.start_frame}-{episode.end_frame}  "
                   f"{episode.slots[0]} / {episode.slots[1]}  "
                   f"min {episode.min_separation_m:.2f} m")
        decision = ask(
            source, annotations, truth,
            episode.start_frame - MARGIN, episode.end_frame + MARGIN,
            episode.slots, caption,
            "n = pas de permutation    s = permutation    q = quitter", "ns",
        )
        if decision == "q":
            return False
        if decision == "s":
            truth.apply_swap(episode.start_frame, episode.slots)
            print(f"  rapprochement {episode.start_frame}: PERMUTATION appliquee")
        else:
            print(f"  rapprochement {episode.start_frame}: pas de permutation")
        truth.resolved.append(episode.start_frame)
        truth.save(path)
    return True


def review_cuts(source, annotations, truth, path) -> bool:
    pending = [c for c in truth.cuts if c.frame not in truth.resolved_cuts]
    print(f"coupures a arbitrer : {len(pending)}/{len(truth.cuts)}")

    for position, cut in enumerate(pending, start=1):
        allowed = "n"
        wording = ["n = aucune"]
        if "near" in cut.sides:
            allowed += "p"
            wording.append("p = proche")
        if "far" in cut.sides:
            allowed += "e"
            wording.append("e = eloignee")
        if len(cut.sides) == 2:
            allowed += "b"
            wording.append("b = les deux")
        wording.append("q = quitter")

        caption = (f"[coupure {position}/{len(pending)}] frame {cut.frame}  "
                   f"paires en risque : {' + '.join(cut.sides)}  "
                   f"saut {cut.displacement_m:.1f} m")
        decision = ask(
            source, annotations, truth,
            cut.frame - MARGIN, cut.frame + MARGIN,
            SLOTS, caption, "    ".join(wording), allowed,
        )
        if decision == "q":
            return False

        swapped = []
        if decision in ("p", "b"):
            truth.apply_swap(cut.frame, NEAR_SLOTS)
            swapped.append("proche")
        if decision in ("e", "b"):
            truth.apply_swap(cut.frame, FAR_SLOTS)
            swapped.append("eloignee")
        print(f"  coupure {cut.frame}: "
              f"{'PERMUTATION ' + ' et '.join(swapped) if swapped else 'aucune'}")
        truth.resolved_cuts.append(cut.frame)
        truth.save(path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    args = parser.parse_args()

    print("chargement...", flush=True)
    annotations = PoseAnnotations.load(args.annotations)
    truth = IdentityGroundTruth.load(args.identity)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    with VideoSource(args.video) as source:
        if review_episodes(source, annotations, truth, args.identity):
            review_cuts(source, annotations, truth, args.identity)
    cv2.destroyAllWindows()

    print(f"\nrapprochements : {len(truth.resolved)}/{len(truth.episodes)}")
    print(f"coupures       : {len(truth.resolved_cuts)}/{len(truth.cuts)}")
    print(f"enregistre dans {args.identity}")


if __name__ == "__main__":
    main()
