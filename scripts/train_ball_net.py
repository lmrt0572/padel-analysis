"""Entraine le reseau de carte de chaleur sur le cache de frames.

FP32 uniquement. La precision mixte a ete mesuree trois fois plus lente sur cette
carte : la TU117 n'a pas de coeurs tensoriels, donc le demi-format ne gagne rien et
les conversions coutent tout. Ne pas y revenir.

Lot de 4, largeur 16, 360x640 : 1,82 Go mesures sur les 3,45 libres.

Seules les frames portant une balle annotee servent. Une frame sans annotation n'est
pas une frame sans balle - 17,5 % des frames annotees n'en portent pas, et rien ne dit
si la balle etait absente ou seulement non etiquetee.

La cible etant nulle presque partout, une entropie croisee nue apprendrait a repondre
zero. Les pixels positifs sont donc ponderes.

Les poids sont ecrits apres chaque epoque, de facon atomique : deux heures de calcul ne
doivent pas dependre de la fin du script.

Usage:
    python scripts/train_ball_net.py --cache cache/FinalF --epochs 10 --out weights/ball_net
"""

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from padel_analysis.ball.heatmap_data import gaussian_target, stacked_indices
from padel_analysis.ball.heatmap_net import BallHeatmapNet, load_stack


class CachedBalls(Dataset):
    """Les frames du cache qui portent une balle et dont les deux voisines existent."""

    def __init__(self, cache: Path, spacing: int, sigma: float, indices: list[int]):
        self.cache = cache
        self.spacing = spacing
        self.sigma = sigma
        self.indices = indices
        meta = json.loads((cache / "balls.json").read_text(encoding="utf-8"))
        self.width, self.height = meta["size"]
        self.balls = {int(k): tuple(v) for k, v in meta["balls"].items()}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        images = [
            cv2.imread(str(self.cache / f"{f:06d}.jpg"))
            for f in stacked_indices(index, self.spacing)
        ]
        target = gaussian_target(
            (self.height, self.width), self.balls[index], self.sigma
        )
        return load_stack(images), torch.from_numpy(target).unsqueeze(0)


def usable(cache: Path, spacing: int) -> list[int]:
    """Les frames annotees dont les deux voisines sont bien dans le cache."""
    meta = json.loads((cache / "balls.json").read_text(encoding="utf-8"))
    present = {int(p.stem) for p in cache.glob("*.jpg")}
    return sorted(
        int(k)
        for k in meta["balls"]
        if all(f in present for f in stacked_indices(int(k), spacing))
    )


def save_atomically(state: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f"{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--spacing", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--rate", type=float, default=1e-3)
    parser.add_argument("--positive-weight", type=float, default=200.0)
    parser.add_argument("--validation", type=float, default=0.05)
    parser.add_argument("--limit", type=int, help="sonde : ne garder que N frames")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    frames = usable(args.cache, args.spacing)
    if args.limit:
        frames = frames[: args.limit]
    random.Random(args.seed).shuffle(frames)

    cut = max(1, int(len(frames) * args.validation))
    held, trained = frames[:cut], frames[cut:]
    print(f"{len(trained)} frames d'entrainement, {len(held)} de validation")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = BallHeatmapNet(width=args.width).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=args.rate)
    weight = torch.tensor([args.positive_weight], device=device)

    loaders = {
        name: DataLoader(
            CachedBalls(args.cache, args.spacing, args.sigma, group),
            batch_size=args.batch,
            shuffle=name == "train",
            num_workers=2,
            pin_memory=True,
        )
        for name, group in (("train", trained), ("held", held))
    }

    for epoch in range(1, args.epochs + 1):
        net.train()
        started, total, seen = time.time(), 0.0, 0
        for stack, target in loaders["train"]:
            stack, target = stack.to(device), target.to(device)
            optimiser.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(stack), target, pos_weight=weight
            )
            loss.backward()
            optimiser.step()
            total += float(loss) * len(stack)
            seen += len(stack)

        net.eval()
        checked, validation = 0, 0.0
        with torch.no_grad():
            for stack, target in loaders["held"]:
                stack, target = stack.to(device), target.to(device)
                validation += float(
                    nn.functional.binary_cross_entropy_with_logits(
                        net(stack), target, pos_weight=weight
                    )
                ) * len(stack)
                checked += len(stack)

        elapsed = time.time() - started
        print(
            f"epoque {epoch:2}  perte {total / seen:.4f}  "
            f"validation {validation / max(checked, 1):.4f}  "
            f"{elapsed / 60:.1f} min  {seen / elapsed:.1f} frames/s",
            flush=True,
        )
        save_atomically(net.state_dict(), args.out.with_suffix(".pt"))

    print(f"poids ecrits : {args.out.with_suffix('.pt')}")


if __name__ == "__main__":
    main()
