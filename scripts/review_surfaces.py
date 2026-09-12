"""Arbitrage humain : contre quoi la balle a-t-elle rebondi ?

Chaque contact est rejoue en boucle, la balle entouree, avec sa trace avant et apres
pour rendre le changement de direction lisible.

    s  le SOL
    v  une VITRE
    g  le GRILLAGE
    f  une FRAPPE, donc une raquette
    x  illisible, je ne peux pas trancher

    r  revenir au clip precedent et annuler sa reponse
    q  quitter en conservant les reponses rendues

Verre et grillage sont demandes separement parce que l'oeil les distingue. C'est ce
qui rend la deduction geometrique verifiable au lieu d'etre supposee juste.

Cet outil n'affiche jamais ce que la regle predit, ni meme si elle hesite. Une verite
terrain construite sur l'hypothese qu'elle doit juger ne mesure que deux erreurs qui
s'accordent - au sous-projet A, corriger ce defaut avait fait passer l'IDF1 de 0,956
a 0,819, et le chiffre flatteur etait l'artefact.

Le fichier est reecrit de facon atomique apres chaque reponse : une coupure de courant
coute le clip en cours, pas la campagne.

Usage:
    python scripts/review_surfaces.py --video <video.mp4> \
        --annotations <ball.json> --truth ground_truth/surfaces/<nom>.json
"""

import argparse
from pathlib import Path

import cv2

from padel_analysis.eval.ball_dataset import BallAnnotations
from padel_analysis.eval.surface_truth import SurfaceGroundTruth
from padel_analysis.io.video_source import VideoSource

WINDOW = "contre quoi la balle a-t-elle rebondi ?"
KEYS = "s sol   v vitre   g grillage   f frappe   x illisible   r retour   q quitter"
ANSWER_KEYS = {"s": "sol", "v": "verre", "g": "grillage", "f": "raquette", "x": "x"}
SPAN = 20
TRAIL = (60, 200, 255)
BALL = (0, 220, 255)


def draw(frame, centres, index, contact, caption):
    """La frame, la trace de la balle autour du contact, et les touches."""
    canvas = frame.copy()
    for offset in range(-SPAN, SPAN + 1):
        point = centres.get(contact + offset)
        if point is None:
            continue
        here = contact + offset == index
        cv2.circle(
            canvas,
            (int(point[0]), int(point[1])),
            9 if here else 3,
            BALL if here else TRAIL,
            2,
        )
    current = centres.get(index)
    if current is not None:
        cv2.circle(canvas, (int(current[0]), int(current[1])), 26, BALL, 2)
    for text, y, scale in ((caption, 44, 0.9), (KEYS, 82, 0.65)):
        cv2.putText(
            canvas, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5
        )
        cv2.putText(
            canvas, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2
        )
    return canvas


def ask(source, centres, contact, caption):
    """Rejoue la sequence en boucle jusqu'a ce qu'une touche valide soit frappee."""
    accepted = set(ANSWER_KEYS) | {"r", "q"}
    while True:
        for index, frame in source.iter_frames(
            start=max(0, contact - SPAN), stop=contact + SPAN + 1
        ):
            cv2.imshow(WINDOW, draw(frame, centres, index, contact, caption))
            key = cv2.waitKey(45) & 0xFF
            if key != 255 and chr(key) in accepted:
                return chr(key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    args = parser.parse_args()

    truth = SurfaceGroundTruth.load(args.truth)
    centres = BallAnnotations.load(args.annotations).centres()
    total = len(truth.tasks)
    answered = [t for t in truth.tasks if t.frame in truth.answers]

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    with VideoSource(args.video) as source:
        while True:
            pending = truth.pending()
            if not pending:
                print(f"\ncampagne terminee : {total} contacts arbitres")
                break
            task = pending[0]
            caption = f"{total - len(pending) + 1} / {total}   frame {task.frame}"
            key = ask(source, centres, task.frame, caption)

            if key == "q":
                print(f"\narret : {len(truth.answers)} / {total} arbitres")
                break
            if key == "r":
                if answered:
                    truth.undo(answered.pop().frame)
                    truth.save(args.truth)
                continue
            truth.answer(task.frame, ANSWER_KEYS[key])
            truth.save(args.truth)
            answered.append(task)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
