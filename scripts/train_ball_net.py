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
doivent pas dependre de la fin du script, et on peut arreter des que la validation
cesse de descendre sans rien perdre. Deux fichiers : le dernier etat, et celui de la
meilleure validation - un pic passager, comme celui de l'epoque 8 du premier
entrainement, ne doit pas pouvoir faire perdre le meilleur modele.

Le debit mesure est de 10,7 frames/s, contre 15,1 en synthetique : le decodage JPEG est
devenu le goulot, chaque frame etant lue trois fois - une par position dans la pile.
Augmenter le nombre de processus de chargement n'a pas aide.

Une session interrompue se reprend : --resume repart des poids ecrits, --first-epoch
numerote la suite, et la meilleure validation initiale est recalculee sur les poids
repris plutot que supposee. L'etat de l'optimiseur est ecrit a cote des poids depuis
cette option ; un entrainement plus ancien repart donc avec des moments d'Adam nuls, et
le script le dit.

Usage:
    python scripts/train_ball_net.py --cache cache/FinalF --epochs 10 --out weights/ball_net
    python scripts/train_ball_net.py --cache cache/FinalF_step1 --epochs 10         --resume weights/ball_net_full.pt --first-epoch 7 --out weights/ball_net_full
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
from padel_analysis.ball.heatmap_net import BallHeatmapNet, load_stack, meta_path


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


def optimiser_path(weights: Path) -> Path:
    """Ou vit l'etat de l'optimiseur qui accompagne ces poids."""
    return weights.with_name(weights.stem + "_optimiser.pt")


def validate(net: nn.Module, loader: DataLoader, weight: torch.Tensor, device: str) -> float:
    """Perte moyenne sur les frames de validation, ponderee comme a l'entrainement."""
    net.eval()
    checked, total = 0, 0.0
    with torch.no_grad():
        for stack, target in loader:
            stack, target = stack.to(device), target.to(device)
            total += float(
                nn.functional.binary_cross_entropy_with_logits(
                    net(stack), target, pos_weight=weight
                )
            ) * len(stack)
            checked += len(stack)
    return total / max(checked, 1)


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
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        type=Path,
        help="repartir de ces poids ; memes --cache, --seed et --limit, sinon la "
        "validation ne porterait plus sur les memes frames",
    )
    parser.add_argument(
        "--first-epoch",
        type=int,
        default=1,
        help="numero de la premiere epoque a jouer, pour reprendre ou l'on s'est arrete",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), help="par defaut : cuda si present")
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

    size = json.loads((args.cache / "balls.json").read_text(encoding="utf-8"))["size"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta_path(args.out.with_suffix(".pt")).write_text(
        json.dumps({"size": size, "width": args.width, "spacing": args.spacing,
                    "sigma": args.sigma}),
        encoding="utf-8",
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = BallHeatmapNet(width=args.width).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=args.rate)
    weight = torch.tensor([args.positive_weight], device=device)

    if args.resume is not None:
        net.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        moments = optimiser_path(args.resume)
        if moments.exists():
            optimiser.load_state_dict(
                torch.load(moments, map_location=device, weights_only=True)
            )
            print(f"reprise : poids et etat de l'optimiseur depuis {args.resume}")
        else:
            print(
                f"reprise : poids depuis {args.resume}, mais aucun etat d'optimiseur - "
                "Adam repart de moments nuls, ce qui secoue les premieres iterations"
            )

    loaders = {
        name: DataLoader(
            CachedBalls(args.cache, args.spacing, args.sigma, group),
            batch_size=args.batch,
            shuffle=name == "train",
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        for name, group in (("train", trained), ("held", held))
    }

    if args.first_epoch > args.epochs:
        raise SystemExit("--first-epoch depasse --epochs : rien a jouer")
    best = float("inf")
    if args.resume is not None:
        best = validate(net, loaders["held"], weight, device)
        print(f"validation des poids repris : {best:.4f}")

    for epoch in range(args.first_epoch, args.epochs + 1):
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

        held_loss = validate(net, loaders["held"], weight, device)
        elapsed = time.time() - started
        print(
            f"epoque {epoch:2}  perte {total / seen:.4f}  "
            f"validation {held_loss:.4f}  "
            f"{elapsed / 60:.1f} min  {seen / elapsed:.1f} frames/s",
            flush=True,
        )
        save_atomically(net.state_dict(), args.out.with_suffix(".pt"))
        save_atomically(optimiser.state_dict(), optimiser_path(args.out.with_suffix(".pt")))
        if held_loss < best:
            best = held_loss
            save_atomically(
                net.state_dict(), args.out.with_name(args.out.name + "_best.pt")
            )
            print(f"          meilleure validation jusqu'ici, epoque {epoch} conservee",
                  flush=True)

    print(f"poids ecrits : {args.out.with_suffix('.pt')} (dernier), "
          f"{args.out.name}_best.pt (meilleure validation {best:.4f})")


if __name__ == "__main__":
    main()
