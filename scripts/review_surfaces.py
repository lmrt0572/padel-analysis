"""Arbitrage humain : contre quoi la balle a-t-elle rebondi ?

Chaque contact est rejoue en boucle, la balle entouree, avec sa trace avant et apres
pour rendre le changement de direction lisible.

    s  le SOL
    v  une VITRE
    g  le GRILLAGE
    t  le FILET
    f  une FRAPPE, donc une raquette
    n  AUCUN contact : la trajectoire passe tout droit
    x  illisible, je ne peux pas trancher

Le clip peut contenir plusieurs evenements - un rebond puis une frappe. Celui qui est
soumis au jugement est le seul marque par la croix magenta, et la lecture s'y attarde
en affichant CONTACT. Les autres sont du contexte.

`n` et `x` ne disent pas la meme chose et ne doivent pas etre confondus. `x` veut dire
"je ne peux pas trancher" : c'est une non-mesure, ecartee du calcul. `n` veut dire "il
ne s'est rien passe ici" : c'est un faux positif de l'etage des contacts, et c'est la
seule facon de mesurer sa precision, l'annotation de frappes etant trop grossiere.

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
KEYS = (
    "s sol   v vitre   g grillage   t filet   f frappe   n aucun   x illisible"
    "   r retour   q quitter"
)
ANSWER_KEYS = {
    "s": "sol",
    "v": "verre",
    "g": "grillage",
    "t": "filet",
    "f": "raquette",
    "n": "aucun",
    "x": "x",
}
SPAN = 12
TRAIL = (60, 200, 255)
BALL = (0, 220, 255)
MARK = (255, 80, 255)


def draw(frame, centres, index, contact, caption):
    """La frame, la trace de la balle, et le marqueur fixe de l'instant a juger.

    Le marqueur ne bouge pas : il reste sur la position de la balle a la frame du
    contact. Sans lui, un clip contenant a la fois un rebond et une frappe ne dit
    pas lequel des deux est soumis au jugement.
    """
    canvas = frame.copy()
    for offset in range(-SPAN, SPAN + 1):
        point = centres.get(contact + offset)
        if point is None:
            continue
        cv2.circle(canvas, (int(point[0]), int(point[1])), 3, TRAIL, 2)

    impact = centres.get(contact)
    if impact is not None:
        x, y = int(impact[0]), int(impact[1])
        cv2.circle(canvas, (x, y), 30, MARK, 2)
        cv2.line(canvas, (x - 48, y), (x - 16, y), MARK, 2)
        cv2.line(canvas, (x + 16, y), (x + 48, y), MARK, 2)
        cv2.line(canvas, (x, y - 48), (x, y - 16), MARK, 2)
        cv2.line(canvas, (x, y + 16), (x, y + 48), MARK, 2)

    current = centres.get(index)
    if current is not None:
        cv2.circle(canvas, (int(current[0]), int(current[1])), 10, BALL, 2)

    if index == contact:
        cv2.putText(canvas, "CONTACT", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 0), 6)
        cv2.putText(canvas, "CONTACT", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    MARK, 2)

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
            # On s'attarde sur l'instant a juger : c'est celui-la que l'oeil doit voir.
            key = cv2.waitKey(320 if index == contact else 55) & 0xFF
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
