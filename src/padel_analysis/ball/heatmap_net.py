"""A heatmap network that answers the same protocol as the motion detector.

The point of this module is an ablation, not a model. `MotionCandidates` and
`NetCandidates` both satisfy `BallCandidates`, so stage two never learns which
produced its input and the measured gap belongs to the detector alone.

The shape is a narrow U-Net over three stacked colour frames, nine channels in and one
heatmap out at the input resolution. Its size is not a taste: a GTX 1650 leaves 3.45 GB
free, and the configuration measured on 2026-09-12 puts width 16 at a batch of four in
1.82 GB and 15.1 frames per second. Width 32 reaches 3.61 GB and a third of the speed.

Mixed precision is deliberately absent. Measured on this card it runs three times
slower - the TU117 has no tensor cores, so half precision buys nothing and the casts
cost everything.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .candidates import Candidate
from .heatmap_data import stacked_indices


def _block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1),
        nn.BatchNorm2d(outputs),
        nn.ReLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1),
        nn.BatchNorm2d(outputs),
        nn.ReLU(inplace=True),
    )


class BallHeatmapNet(nn.Module):
    """Stacked frames in, one heatmap out, at the resolution it was given."""

    def __init__(self, width: int = 16, stacked: int = 3) -> None:
        super().__init__()
        w = width
        self.down1 = _block(3 * stacked, w)
        self.down2 = _block(w, 2 * w)
        self.down3 = _block(2 * w, 4 * w)
        self.down4 = _block(4 * w, 8 * w)
        self.up3 = _block(12 * w, 4 * w)
        self.up2 = _block(6 * w, 2 * w)
        self.up1 = _block(3 * w, w)
        self.head = nn.Conv2d(w, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.down1(x)
        second = self.down2(self.pool(first))
        third = self.down3(self.pool(second))
        fourth = self.down4(self.pool(third))
        x = self.up3(torch.cat([self.upsample(fourth), third], dim=1))
        x = self.up2(torch.cat([self.upsample(x), second], dim=1))
        x = self.up1(torch.cat([self.upsample(x), first], dim=1))
        return self.head(x)


def peaks_of(
    heatmap: np.ndarray,
    threshold: float = 0.1,
    suppression: int = 6,
    scale: tuple[float, float] = (1.0, 1.0),
    limit: int = 20,
) -> list[Candidate]:
    """The heatmap's local maxima, strongest first, in the original frame's pixels.

    Suppression exists because one ball makes one blob, not thirty neighbouring
    candidates - the same reason the contact stage keeps one peak per burst.

    Args:
        heatmap: the network's answer, already through a sigmoid.
        threshold: least value worth reporting.
        suppression: least distance in map pixels between two kept peaks.
        scale: (x, y) factors back to the original frame.
        limit: most candidates to return, strongest first.
    """
    ys, xs = np.nonzero(heatmap >= threshold)
    if not len(ys):
        return []
    order = np.argsort(heatmap[ys, xs])[::-1]

    kept: list[Candidate] = []
    for position in order:
        y, x = float(ys[position]), float(xs[position])
        if any(abs(x - c.x / scale[0]) <= suppression
               and abs(y - c.y / scale[1]) <= suppression for c in kept):
            continue
        kept.append(
            Candidate(x * scale[0], y * scale[1], float(heatmap[int(y), int(x)]))
        )
        if len(kept) >= limit:
            break
    return kept


def meta_path(weights: Path) -> Path:
    """Where the training settings of these weights are written.

    The best-validation weights share the file of their training run, so `_best` is
    dropped from the name before looking.
    """
    stem = Path(weights).stem.removesuffix("_best")
    return Path(weights).with_name(stem + "_meta.json")


class NetCandidates:
    """The candidate protocol of `candidates.py`, backed by a trained network.

    Interchangeable with `MotionCandidates` by construction, which is what makes the
    ablation exact rather than merely suggestive.
    """

    def __init__(
        self,
        weights: Path,
        spacing: int = 3,
        width: int | None = None,
        size: tuple[int, int] | None = None,
        threshold: float = 0.1,
        suppression: int | None = None,
        limit: int = 20,
        device: str = "cuda",
    ) -> None:
        meta_file = meta_path(weights)
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        if meta and meta["spacing"] != spacing:
            raise ValueError(
                f"these weights were trained with spacing {meta['spacing']}, not {spacing}"
            )
        # Without a companion file: the earlier weights, trained at 640x360.
        size = size or tuple(meta.get("size", (640, 360)))
        width = width or meta.get("width", 16)
        if size[0] % 8 or size[1] % 8:
            raise ValueError("the network needs a size whose sides are multiples of 8")
        # Suppression is in map pixels: at 1280 wide, a ball is twice as big there as
        # at 640, and the same real distance is worth twice as much.
        if suppression is None:
            suppression = round(6 * size[0] / 640)
        self.spacing = spacing
        self.size = size
        self.threshold = threshold
        self.suppression = suppression
        self.limit = limit
        self.device = device if torch.cuda.is_available() else "cpu"
        self.net = BallHeatmapNet(width=width).to(self.device).eval()
        self.net.load_state_dict(
            torch.load(Path(weights), map_location=self.device, weights_only=True)
        )

    def frames_needed(self, index: int) -> list[int]:
        return stacked_indices(index, self.spacing)

    def __call__(
        self, frames: dict[int, np.ndarray], index: int
    ) -> list[Candidate]:
        import cv2

        wanted = self.frames_needed(index)
        if any(frames.get(f) is None for f in wanted):
            return []

        width, height = self.size
        original = frames[index].shape
        stack = np.concatenate(
            [cv2.resize(frames[f], (width, height)) for f in wanted], axis=2
        )
        tensor = (
            torch.from_numpy(stack).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        )
        with torch.no_grad():
            heatmap = torch.sigmoid(self.net(tensor.to(self.device)))
        answer = heatmap[0, 0].cpu().numpy()

        return peaks_of(
            answer,
            threshold=self.threshold,
            suppression=self.suppression,
            scale=(original[1] / width, original[0] / height),
            limit=self.limit,
        )


def load_stack(images: Sequence[np.ndarray]) -> torch.Tensor:
    """Stack cached frames into the tensor the network reads, values in [0, 1]."""
    stacked = np.concatenate(images, axis=2)
    return torch.from_numpy(stacked).permute(2, 0, 1).float().div_(255.0)
