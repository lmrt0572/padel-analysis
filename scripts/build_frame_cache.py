"""Extrait les frames d'entrainement en JPEG reduit, avec la balle mise a l'echelle.

Une frame sur trois. A 30 images par seconde deux frames consecutives portent presque
la meme information : le sous-echantillonnage coute peu et divise le temps d'epoque par
trois. Les frames empilees etant espacees de 3 elles aussi, les voisins d'un centre
tombent sur la meme grille et un seul cache suffit aux trois canaux.

La tranche d'evaluation est exclue. Elle doit rester intacte : un reseau qui l'aurait
vue ne mesurerait plus rien.

Usage:
    python scripts/build_frame_cache.py --video <video.mp4> --annotations <ball.json> \
        --exclude 16000 20099 --step 3 --out cache/FinalF
"""

import argparse
import json
from pathlib import Path

import cv2

from padel_analysis.eval.ball_dataset import BallAnnotations
from padel_analysis.io.video_source import VideoSource

WIDTH, HEIGHT = 640, 360


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--exclude", type=int, nargs=2, required=True)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    low, high = args.exclude
    centres = BallAnnotations.load(args.annotations).centres()
    args.out.mkdir(parents=True, exist_ok=True)

    scale_x, scale_y = WIDTH / 1920, HEIGHT / 1080
    balls: dict[str, list[float]] = {}
    written = 0
    with VideoSource(args.video) as source:
        for index, frame in source.iter_frames():
            if index % args.step or low <= index <= high:
                continue
            cv2.imwrite(
                str(args.out / f"{index:06d}.jpg"),
                cv2.resize(frame, (WIDTH, HEIGHT)),
                [cv2.IMWRITE_JPEG_QUALITY, args.quality],
            )
            written += 1
            point = centres.get(index)
            if point is not None:
                balls[str(index)] = [point[0] * scale_x, point[1] * scale_y]
            if written % 2000 == 0:
                print(f"  {written} frames", flush=True)

    (args.out / "balls.json").write_text(
        json.dumps({"size": [WIDTH, HEIGHT], "step": args.step, "balls": balls}),
        encoding="utf-8",
    )
    print(f"{written} frames ecrites, dont {len(balls)} avec une balle annotee")


if __name__ == "__main__":
    main()
