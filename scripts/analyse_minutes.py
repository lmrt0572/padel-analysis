"""Analyse toutes les minutes pointees avec un jeu de poids, sans produire de video.

C'est la passe couteuse - detecteur de poses et reseau de balle sur chaque image - et
la seule qui demande la carte graphique. Son resultat est sauvegarde : toutes les
mesures ensuite se refont sans elle.

Usage:
    python scripts/analyse_minutes.py --weights weights/ball_net.pt --tag 360
"""

import argparse
import pickle
from pathlib import Path

import minutes

from padel_analysis.demo import analyse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="nom court du jeu de poids, ex. 360 ou 720")
    args = parser.parse_args()

    todo = minutes.TUNING + minutes.USED + minutes.JUDGE
    for number, (match, start) in enumerate(todo, 1):
        out = Path(minutes.analysis(match, start, args.tag))
        if out.exists():
            print(f"[{number}/{len(todo)}] {out.name} existe deja")
            continue
        print(f"[{number}/{len(todo)}] {match} {start}", flush=True)
        result = analyse(argparse.Namespace(
            video=Path(minutes.video(match)), calibration=Path(minutes.calibration(match)),
            weights=args.weights, start=start, frames=minutes.FRAMES,
        ))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(pickle.dumps(result))


if __name__ == "__main__":
    main()
