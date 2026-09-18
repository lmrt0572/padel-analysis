"""Game and player statistics as they stand at each frame, for the video's side panel.

The video shows the match as it unfolds, so the numbers beside it must not know the
future: every figure here is counted from the start of the clip up to the frame being
drawn. The work is done once, frame by frame, so drawing a frame only reads a row.
"""

import math
from dataclasses import dataclass

import numpy as np

from .basic_stats import smooth_positions
from .net_control import at_net_states
from .rally import RACKET, WALLS, Rally

SLOTS = ("near_1", "near_2", "far_1", "far_2")
PAIRS = {"proche": ("near_1", "near_2"), "fond": ("far_1", "far_2")}


@dataclass(frozen=True)
class PlayerLine:
    slot: str
    shots: int
    distance: float  # metres, lisses
    net_share: float  # part du temps passe au filet, sur les images ou il est vu


@dataclass(frozen=True)
class LiveStats:
    """What the panel shows at one frame."""

    elapsed: float
    shots: int
    walls: int
    last: str | None  # le dernier contact, s'il date de moins d'une seconde
    players: tuple[PlayerLine, ...]
    pair_shots: dict[str, int]
    pair_net: dict[str, float]


class LiveTimeline:
    """Cumulative statistics for every frame of a rally, computed once."""

    def __init__(self, rally: Rally, recent: float = 1.0) -> None:
        self.rally = rally
        self.recent = round(recent * rally.fps)
        frames = np.arange(rally.start, rally.stop + 1)
        self._frames = frames
        self._distance: dict[str, np.ndarray] = {}
        self._at_net: dict[str, np.ndarray] = {}
        self._seen: dict[str, np.ndarray] = {}
        for slot in SLOTS:
            track = rally.positions.get(slot, {})
            raw = np.array([track.get(int(f), (np.nan, np.nan)) for f in frames], dtype=float)
            smooth = smooth_positions(raw)
            present = ~np.isnan(smooth[:, 0])
            steps = np.zeros(len(frames))
            both = present[1:] & present[:-1]
            deltas = np.linalg.norm(np.diff(np.nan_to_num(smooth), axis=0), axis=1)
            steps[1:] = np.where(both, deltas, 0.0)
            self._distance[slot] = np.cumsum(steps)
            at_net = at_net_states(np.where(present, np.abs(smooth[:, 1]), np.nan))
            self._at_net[slot] = np.cumsum(at_net & present)
            self._seen[slot] = np.cumsum(present)

    def at(self, frame: int) -> LiveStats:
        """The statistics counted from the start of the rally up to `frame` included."""
        rally = self.rally
        frame = min(max(frame, rally.start), rally.stop)
        index = frame - rally.start
        past = [c for c in rally.contacts if c.frame <= frame]
        shots = {slot: sum(1 for c in past if c.kind == RACKET and c.player == slot)
                 for slot in SLOTS}
        players = []
        for slot in SLOTS:
            seen = int(self._seen[slot][index])
            players.append(PlayerLine(
                slot=slot,
                shots=shots[slot],
                distance=float(self._distance[slot][index]),
                net_share=float(self._at_net[slot][index]) / seen if seen else math.nan,
            ))
        by_slot = {line.slot: line for line in players}
        last = past[-1] if past and frame - past[-1].frame <= self.recent else None
        return LiveStats(
            elapsed=index / rally.fps,
            shots=sum(1 for c in past if c.kind == RACKET),
            walls=sum(1 for c in past if c.kind in WALLS),
            last=last.kind if last else None,
            players=tuple(players),
            pair_shots={pair: sum(shots[s] for s in slots) for pair, slots in PAIRS.items()},
            pair_net={
                pair: float(np.nanmean([by_slot[s].net_share for s in slots]))
                if any(not math.isnan(by_slot[s].net_share) for s in slots) else math.nan
                for pair, slots in PAIRS.items()
            },
        )
