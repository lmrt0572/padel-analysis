"""Contacts and their surface, read by a small temporal network instead of thresholds.

The rule chain decides each cue on its own: a turn sharp enough, a wrist fast enough,
a ray that meets one surface. Each threshold was swept, and the chain still plateaued,
because the cues are only conclusive together - a soft turn next to a swinging wrist
is a shot, the same turn with the ball at a player's feet is a bounce. A network that
sees every cue over a few dozen frames learns those combinations from the hand-marked
minutes.

Every frame gets a vector of cues: the path and its turns, what else the detector
proposed there, the nearest player's limbs and apparent size, and the surfaces the ray
can meet. A dilated 1D convolution labels each frame as no
contact, racket, floor, wall or net. Contacts are the local peaks of the contact
probability. Glass or mesh is then read from the geometry, as the rule chain does:
the three mesh contacts in the marked minutes are too few to learn from.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..ball.confidence import path_scores
from ..ball.contacts import sharpness_of, turn_of, velocities
from ..ball.smoothing import smooth_path
from ..geometry.camera import MESH, CameraPose, Surface
from .gesture import gesture_near

CLASSES = ("aucun", "raquette", "sol", "mur", "filet")
CLASS_OF_ANSWER = {"raquette": 1, "sol": 2, "verre": 3, "grillage": 3, "filet": 4}
RULE_LABELS = ("RAQUETTE", "SOL", "VITRE", "GRILLAGE", "FILET")
NO_WALL, GLASS_WALL, MESH_WALL = 0, 1, 2
IGNORED = -100
SEEDS = tuple(range(18))
"""One network per seed, their probabilities averaged. Over the eleven training
minutes, each predicted by an ensemble that never saw it, 3 networks scored 0.844 on
average over six sets of seeds, 9 networks 0.847 and 0.856, 18 networks 0.855, with
fewer invented contacts: 67 on average, then 57."""

Point = tuple[float, float]

try:
    from numpy._core.multiarray import _reconstruct as _rebuild_array
except ImportError:  # NumPy 1.x
    from numpy.core.multiarray import _reconstruct as _rebuild_array
_NUMPY_ARRAYS = [_rebuild_array, np.ndarray, np.dtype, type(np.dtype(np.float32))]
"""What a model file may hold besides tensors: an array of 32-bit floats."""


def frame_features(
    analysis: dict,
    path: Mapping[int, Point | None],
    shown: Mapping[int, Point | None],
    rule_events: Sequence,
    pose: CameraPose,
    surfaces: Sequence[Surface],
) -> tuple[np.ndarray, np.ndarray]:
    """One row of cues per frame of the analysed range, and the wall material there.

    Args:
        analysis: the saved first pass of the demonstration.
        path: the absolute ball path, whose network scores are a cue.
        shown: the displayed path, after the confidence filter.
        rule_events: what the rule chain decided, given as one more cue.
        pose: the camera pose of this video.
        surfaces: the court surfaces, floor first and net last.

    Returns:
        (features, material): features is (frames, cues) float32; material is, per
        frame, whether the first admissible wall under the ball is glass or mesh.
    """
    frames = analysis["frames"]
    width, height = analysis["size"]
    scores = path_scores(path, {f: v["raw"] for f, v in frames.items()})
    smooth = smooth_path(dict(shown), cuts=[], process_noise=100.0)
    players = {f: (v["people"], v["assignment"]) for f, v in frames.items()}
    rule = {e.frame: e.label for e in rule_events}

    rows, material = [], []
    for frame in range(analysis["start"], analysis["stop"] + 1):
        ball = smooth.get(frame)
        row: list[float] = [ball is not None]
        row += [ball[0] / width, ball[1] / height] if ball else [0.0, 0.0]
        for span in (1, 2, 3):
            pair = velocities(smooth, frame, span)
            row += [pair is not None]
            row += [c / 20 for c in (*pair[0], *pair[1])] if pair else [0.0] * 4
            if span > 1:
                row += [sharpness_of(*pair), turn_of(*pair) / 50] if pair else [0.0, 0.0]
        score = scores.get(frame)
        row += [score if score is not None else 0.0]
        row += [gesture_near(players, frame, ball) / 20 if ball else 0.0]
        wrists = frames[frame]["wrists"]
        nearest = min((math.dist(ball, w) for w in wrists), default=300.0) if ball else 300.0
        row += [min(nearest / 100, 3.0)]
        row += _height_by_player(ball, frames[frame]["people"])
        wall, cues = _geometry(ball, pose, surfaces)
        row += cues
        row += _candidates(ball, frames[frame]["raw"], smooth.get(frame - 1))
        row += _skeleton(ball, frames[frame]["people"], height)
        row += [rule.get(frame) == label for label in RULE_LABELS]
        rows.append(row)
        material.append(wall)
    return np.asarray(rows, dtype=np.float32), np.asarray(material, dtype=np.int64)


ELBOWS, WRISTS, HIPS, ANKLES = (7, 8), (9, 10), (11, 12), (15, 16)


def _candidates(ball: Point | None, raw: Sequence, previous: Point | None) -> list[float]:
    """What the detector proposed at this frame, beyond the point the path kept.

    A frame where the path sits far from the detector's own best peak is a frame where
    the ball was hidden or confused, which is where contacts get lost.
    """
    best = sorted(raw, key=lambda c: -c.score)
    row = [len(best) / 10]
    row += [best[index].score if index < len(best) else 0.0 for index in range(3)]
    away = math.dist(ball, (best[0].x, best[0].y)) / 100 if ball and best else 3.0
    row += [min(away, 3.0)]
    row += [min(math.dist(ball, previous) / 20, 3.0) if ball and previous else 0.0]
    return row


def _skeleton(ball: Point | None, people: Sequence, height: int) -> list[float]:
    """The nearest player's limbs around the ball, and how far the next player stands.

    A shot is taken with the arm, a bounce happens near the feet. The apparent size of
    the player stands in for depth, which one camera cannot measure.
    """
    if ball is None or not people:
        return [0.0, *[3.0] * 4, 0.0, 3.0]
    ordered = sorted(people, key=lambda p: math.dist(ball, _centre(p)))
    near = ordered[0]
    top, bottom = float(near.bbox[1]), float(near.bbox[3])
    row = [(bottom - top) / height]
    for joints in (ELBOWS, WRISTS, HIPS, ANKLES):
        gaps = [math.dist(ball, near.keypoints[j][:2]) / 100 for j in joints
                if near.keypoints[j][2] > 0.3]
        row += [min(min(gaps, default=3.0), 3.0)]
    ankles = [near.keypoints[j][1] for j in ANKLES if near.keypoints[j][2] > 0.3]
    foot = max(ankles, default=bottom)
    # Height of the ball above the feet, in player heights: unlike pixels, it does
    # not depend on depth.
    row += [max(min((foot - ball[1]) / max(bottom - top, 1.0), 3.0), -3.0)]
    other = min((math.dist(ball, _centre(p)) for p in ordered[1:]), default=1500.0)
    row += [min(other / 500, 3.0)]
    return row


def _centre(person) -> Point:
    x1, y1, x2, y2 = person.bbox[:4]
    return (float(x1 + x2) / 2, float(y1 + y2) / 2)


def _height_by_player(ball: Point | None, people: Sequence) -> list[float]:
    """How far the ball is sideways from the nearest player, and how high along them.

    Height is 0 at the top of the box and 1 at the feet: a bounce sits near 1, a shot
    higher up.
    """
    if ball is None:
        return [3.0, 0.0]
    side, along = 3.0, 0.0
    for person in people:
        x1, y1, x2, y2 = person.bbox[:4]
        gap = max(x1 - ball[0], 0.0, ball[0] - x2) / 100
        if gap < side:
            side, along = gap, (ball[1] - y1) / max(y2 - y1, 1.0)
    return [min(side, 3.0), max(min(along, 6.0), -3.0)]


def _geometry(
    ball: Point | None, pose: CameraPose, surfaces: Sequence[Surface]
) -> tuple[int, list[float]]:
    """Which surfaces the ray through the ball may meet, and where."""
    if ball is None:
        return NO_WALL, [0.0] * (2 * len(surfaces) + 1)
    origin, direction = pose.ray(ball)
    cues: list[float] = []
    wall = NO_WALL
    admissible = 0
    for surface in surfaces:
        meeting = surface.intersect(origin, direction)
        inside = meeting is not None and surface.contains(meeting, 0.30)
        admissible += inside
        where = 0.0
        if inside:
            # Depth for the floor, height for a wall or the net.
            where = meeting[1] / 10 if surface.name == "floor" else meeting[2] / 4
            if wall == NO_WALL and surface.name not in ("floor", "net"):
                wall = MESH_WALL if surface.material_at(meeting) == MESH else GLASS_WALL
        cues += [float(inside), float(where)]
    return wall, [*cues, admissible / 3]


def frame_labels(marks: Mapping[int, str], start: int, count: int) -> np.ndarray:
    """The training target: each marked contact labels its frame and both neighbours.

    Marks are placed by hand and can be a frame off, so the frames two away from a
    contact are ignored rather than taught as "no contact".
    """
    labels = np.zeros(count, dtype=np.int64)
    for frame in marks:
        for offset in (-2, 2):
            index = frame + offset - start
            if 0 <= index < count and labels[index] == 0:
                labels[index] = IGNORED
    for frame, answer in marks.items():
        for offset in (-1, 0, 1):
            index = frame + offset - start
            if 0 <= index < count:
                labels[index] = CLASS_OF_ANSWER[answer]
    return labels


class ContactNet(nn.Module):
    """Dilated 1D convolutions over the cue sequence: about 60 frames of context."""

    def __init__(self, cues: int, width: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = cues
        for dilation in (1, 2, 4, 8):
            layers += [
                nn.Conv1d(channels, width, 5, padding=2 * dilation, dilation=dilation),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            channels = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(width, len(CLASSES), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, cues, frames) -> (batch, classes, frames) logits."""
        return self.head(self.body(x))


def decode(
    probabilities: np.ndarray,
    material: np.ndarray,
    start: int,
    threshold: float = 0.7,
    radius: int = 4,
    other_radius: int = 3,
    other_threshold: float = 0.85,
) -> dict[int, str]:
    """Contact frame -> answer, from per-frame class probabilities.

    A contact is a frame whose contact probability reaches `threshold` and where its
    own class is the likeliest within `radius` frames. A contact of another kind may
    stand closer, down to `other_radius` frames, when surer than `other_threshold`: a
    bounce is often followed by the back glass five or six frames later, and one peak
    used to hide the other. 0.7 was chosen by cross-validation over the marked minutes,
    each predicted by a model that never saw it; letting another kind stand closer
    brought 708 to 713 contacts of 863 right instead of 697 to 703, over three sets of
    seeds, and 72 to 75 glass contacts of 128 instead of 70 to 73.
    """
    contact = 1.0 - probabilities[:, 0]
    kinds = probabilities[:, 1:].argmax(axis=1) + 1
    found: dict[int, int] = {}
    for index in np.argsort(-contact, kind="stable"):
        if contact[index] < min(threshold, other_threshold):
            break
        kind = int(kinds[index])
        near = [other for other in found if abs(index - other) <= max(radius, other_radius)]
        if any(found[other] == kind and abs(index - other) <= radius for other in near):
            continue
        others = [other for other in near if found[other] != kind]
        if any(abs(index - other) <= other_radius for other in others):
            continue
        if contact[index] < (other_threshold if others else threshold):
            continue
        low, high = max(0, index - radius), index + radius + 1
        if probabilities[index, kind] < probabilities[low:high, kind].max():
            continue
        found[int(index)] = kind
    answers = {}
    for index, kind in sorted(found.items()):
        if kind == CLASS_OF_ANSWER["verre"]:
            answers[start + index] = "grillage" if material[index] == MESH_WALL else "verre"
        else:
            answers[start + index] = CLASSES[kind]
    return answers


class ContactModel:
    """An ensemble of trained networks, with the normalisation they were trained with."""

    def __init__(self, nets: Sequence[ContactNet], mean: np.ndarray, std: np.ndarray) -> None:
        self.nets = list(nets)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """(frames, cues) -> (frames, classes), averaged over the ensemble."""
        x = torch.from_numpy((features - self.mean) / self.std).T[None]
        total = np.zeros((features.shape[0], len(CLASSES)), dtype=np.float64)
        with torch.no_grad():
            for net in self.nets:
                net.eval()
                total += torch.softmax(net(x), dim=1)[0].T.numpy()
        return total / len(self.nets)

    def save(self, path: str | Path) -> None:
        """Write the ensemble as plain tensors, which load without running any code."""
        torch.save(
            {
                "cues": int(self.mean.shape[0]),
                "mean": torch.from_numpy(self.mean),
                "std": torch.from_numpy(self.std),
                "nets": [net.state_dict() for net in self.nets],
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ContactModel":
        """Read a saved ensemble without letting the file run code.

        Files written before this version hold the normalisation as NumPy arrays:
        those, and nothing else, are allowed besides tensors.
        """
        with torch.serialization.safe_globals(_NUMPY_ARRAYS):
            saved = torch.load(path, map_location="cpu", weights_only=True)
        nets = []
        for state in saved["nets"]:
            net = ContactNet(saved["cues"])
            net.load_state_dict(state)
            nets.append(net)
        return cls(nets, np.asarray(saved["mean"]), np.asarray(saved["std"]))


def train(
    sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    seeds: Sequence[int] = SEEDS,
    steps: int = 1500,
    crop: int = 256,
    batch: int = 16,
    contact_weight: float = 6.0,
    device: str = "cpu",
) -> ContactModel:
    """Fit one network per seed on random crops of the marked minutes.

    Args:
        sequences: (features, labels) per marked minute.
        contact_weight: loss weight of every contact class against "no contact",
            which covers about 95 % of frames.
    """
    stacked = np.concatenate([features for features, _ in sequences])
    mean, std = stacked.mean(0), stacked.std(0) + 1e-6
    normalised = [((f - mean) / std, y) for f, y in sequences]
    weights = torch.tensor([1.0] + [contact_weight] * (len(CLASSES) - 1), device=device)
    loss_of = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORED)

    nets = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = ContactNet(stacked.shape[1]).to(device)
        optimiser = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
        net.train()
        for _ in range(steps):
            xs, ys = [], []
            for _ in range(batch):
                features, labels = normalised[rng.integers(len(normalised))]
                offset = rng.integers(0, len(labels) - crop + 1)
                xs.append(features[offset:offset + crop])
                ys.append(labels[offset:offset + crop])
            x = torch.tensor(np.stack(xs), device=device).transpose(1, 2)
            y = torch.tensor(np.stack(ys), device=device)
            loss = loss_of(net(x), y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        nets.append(net.cpu().eval())
    return ContactModel(nets, mean, std)
