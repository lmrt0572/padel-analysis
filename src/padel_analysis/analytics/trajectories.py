"""Player trajectories in court metres, loaded from the pipeline cache.

Frames where a player was not located carry NaN rather than an interpolated
position. Bridging those gaps would turn a thirty-frame dropout into a metre of
travelled distance, and the error would be invisible in the output.
"""

from dataclasses import dataclass

import numpy as np

from ..pipeline.cache import PositionCache

SLOTS: tuple[str, ...] = ("near_1", "near_2", "far_1", "far_2")


@dataclass(frozen=True)
class MatchTrajectories:
    """Dense per-frame positions for the four slots, with NaN for absences."""

    frames: np.ndarray  # (N,) frame indices, sorted
    positions: dict[str, np.ndarray]  # slot -> (N, 2) in metres, NaN where absent
    fps: float

    @classmethod
    def from_cache(cls, cache: PositionCache, fps: float) -> "MatchTrajectories":
        frames = np.asarray(cache.frames(), dtype=np.int64)
        if frames.size == 0:
            raise ValueError("cannot build trajectories from an empty cache")

        positions = {slot: np.full((frames.size, 2), np.nan) for slot in SLOTS}
        for row, frame in enumerate(frames):
            for slot, (x, y) in cache.at(int(frame)).positions.items():
                if slot in positions:
                    positions[slot][row] = (x, y)
        return cls(frames=frames, positions=positions, fps=float(fps))

    def present(self, slot: str) -> np.ndarray:
        """Boolean mask of the frames where `slot` was located."""
        return ~np.isnan(self.positions[slot][:, 0])

    def complete_mask(self) -> np.ndarray:
        """Boolean mask of the frames where all four players were located."""
        return np.logical_and.reduce([self.present(slot) for slot in SLOTS])

    def depth(self, slot: str) -> np.ndarray:
        """Distance to the net, in metres. NaN where the player was absent."""
        return np.abs(self.positions[slot][:, 1])

    @staticmethod
    def side(slot: str) -> int:
        """-1 for the negative-y half, +1 for the positive-y half."""
        return -1 if slot.startswith("near") else +1

    def duration_seconds(self) -> float:
        return float(self.frames.size) / self.fps
