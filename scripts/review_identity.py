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
    c  les EQUIPES ONT CHANGE DE COTE

Le changement de cote est a part. Les slots designent une moitie de court, pas une
personne : apres un changement de cote, `near_1` est quelqu'un d'autre, et aucun
echange d'etiquettes ne peut l'exprimer. L'identite du joueur s'arrete la et repart
a zero - c'est une frontiere, pas une correction.

Ne juge pas les corps mais les couleurs : le meme joueur porte-t-il la meme
couleur avant et apres ? Un detail stable - casquette, chaussures, manches - vaut
mieux qu'une impression generale.

    r  revenir au clip precedent et annuler sa reponse
    q  abandonner (les arbitrages deja rendus sont conserves)

Usage:
    python scripts/review_identity.py --video <video.mp4> \
        --annotations <pose.json> --identity ground_truth/identity/<nom>.json
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


def undo_episode(truth: IdentityGroundTruth, episode) -> str | None:
    """Defait la reponse rendue sur un rapprochement. L'echange est son inverse."""
    key = truth.decisions.pop(f"episode:{episode.start_frame}", None)
    if key == "s":
        truth.apply_swap(episode.start_frame, episode.slots)
    if episode.start_frame in truth.resolved:
        truth.resolved.remove(episode.start_frame)
    return key


def undo_cut(truth: IdentityGroundTruth, cut) -> str | None:
    key = truth.decisions.pop(f"cut:{cut.frame}", None)
    if key is None and cut.frame in truth.boundaries:
        key = "c"  # repondu avant que les reponses soient tracees
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
    print(f"rapprochements a arbitrer : {len(pending)}/{len(truth.episodes)}")

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
                    print("  rien a annuler")
                    continue
                pending.insert(0, previous)
            else:
                print("  rien a annuler : aucun rapprochement rendu")
                continue
            undone = undo_episode(truth, previous)
            truth.save(path)
            print(f"  retour sur {previous.start_frame} "
                  f"(reponse annulee : {undone or 'inconnue'})")
            continue

        if decision == "s":
            truth.apply_swap(episode.start_frame, episode.slots)
            print(f"  rapprochement {episode.start_frame}: PERMUTATION appliquee")
        else:
            print(f"  rapprochement {episode.start_frame}: pas de permutation")
        truth.resolved.append(episode.start_frame)
        truth.decisions[f"episode:{episode.start_frame}"] = decision
        truth.save(path)
        position += 1
    return True


def review_cuts(source, annotations, truth, path) -> bool:
    pending = [c for c in truth.cuts if c.frame not in truth.resolved_cuts]
    print(f"coupures a arbitrer : {len(pending)}/{len(truth.cuts)}")

    position = 0
    while position < len(pending):
        cut = pending[position]
        # Les cinq reponses sont toujours offertes. Le detecteur sert a trouver les
        # clips a regarder ; ce qu'on y voit ne lui appartient pas. Deux partenaires
        # proches qui echangent leurs places parcourent moins que le seuil, et la
        # paire n'est donc pas annoncee - sans cesser d'avoir permute.
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
            # Au premier clip de la file, le precedent a ete rendu dans une session
            # anterieure : il n'est plus en attente, il faut le reprendre au dossier.
            if position > 0:
                position -= 1
                previous = pending[position]
            elif truth.resolved_cuts:
                done = truth.resolved_cuts[-1]
                previous = next((c for c in truth.cuts if c.frame == done), None)
                if previous is None:
                    print("  rien a annuler")
                    continue
                pending.insert(0, previous)
            else:
                print("  rien a annuler : aucune coupure rendue")
                continue
            undone = undo_cut(truth, previous)
            truth.save(path)
            if undone is None:
                print(f"  retour sur {previous.frame} : reponse precedente NON "
                      f"tracee, seule la marque est retiree - si c'etait une "
                      f"permutation, la reappliquer puis annuler a nouveau")
            else:
                print(f"  retour sur {previous.frame} (reponse annulee : {undone})")
            continue

        if decision == "c":
            truth.boundaries.append(cut.frame)
            truth.boundaries.sort()
            print(f"  coupure {cut.frame}: CHANGEMENT DE COTE, identite relancee")
        else:
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
