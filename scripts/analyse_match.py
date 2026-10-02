"""Analyse un match entier, minute par minute, pour les statistiques du match.

C'est la passe couteuse - detecteur de poses et reseau de balle sur chaque image -
decoupee en tranches de 1 800 images contigues, de la premiere image a la derniere.
Chaque tranche est sauvegardee des qu'elle est finie : une interruption ne perd que la
tranche en cours, et relancer la commande reprend a la suivante. Le suivi des joueurs
de chaque tranche repart de zero ; il est rejoue ensuite d'un seul tenant sur le match.

Usage:
    python scripts/analyse_match.py --weights weights/ball_net.pt
    python scripts/analyse_match.py --weights weights/ball_net.pt --match FinalF
"""

import argparse
import pickle
import time
from pathlib import Path

import minutes

from padel_analysis.demo import analyse
from padel_analysis.io.video_source import VideoSource

MATCHES = ("FinalF", "FinalM")
OUT = Path("outputs/match")


def chunk_path(match: str, start: int) -> Path:
    return OUT / f"{match}_{start:05d}.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--match", choices=MATCHES, help="un seul match ; les deux par defaut")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    plan = []
    for match in (args.match,) if args.match else MATCHES:
        with VideoSource(Path(minutes.video(match))) as source:
            count = source.metadata.frame_count
        plan += [(match, start, min(minutes.FRAMES, count - start))
                 for start in range(0, count, minutes.FRAMES)]
    todo = [(m, s, n) for m, s, n in plan if not chunk_path(m, s).exists()]
    print(f"{len(plan)} tranches, {len(plan) - len(todo)} deja faites, {len(todo)} a faire",
          flush=True)
    began = time.time()
    for number, (match, start, frames) in enumerate(todo, 1):
        print(f"[{number}/{len(todo)}] {match} images {start} a {start + frames - 1}", flush=True)
        result = analyse(argparse.Namespace(
            video=Path(minutes.video(match)), calibration=Path(minutes.calibration(match)),
            weights=args.weights, start=start, frames=frames,
        ))
        partial = chunk_path(match, start).with_suffix(".tmp")
        partial.write_bytes(pickle.dumps(result))
        partial.replace(chunk_path(match, start))
        left = (time.time() - began) / number * (len(todo) - number)
        print(f"  sauvegardee ; reste environ {left / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
