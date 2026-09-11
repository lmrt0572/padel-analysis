"""Distance, speed and mean position, computed from court trajectories.

Two traps, both measurable. A dropout must never be crossed as if it were a move,
and position noise inflates travelled distance - the more so at the far end of the
court, where one pixel is worth 4.3 times what it is at the near end. Smoothing is
therefore offered explicitly, so the difference it makes can be reported rather
than hidden.
"""

import numpy as np


def step_distances(frames: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Distances between positions on genuinely consecutive frames.

    Pairs separated by a frame gap, or where either position is absent, are left
    out entirely rather than joined.
    """
    frames = np.asarray(frames)
    positions = np.asarray(positions, dtype=np.float64)
    if frames.size < 2:
        return np.zeros(0)

    adjacent = np.diff(frames) == 1
    present = ~np.isnan(positions[:, 0])
    both_present = present[:-1] & present[1:]

    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return deltas[adjacent & both_present]


def distance_travelled(frames: np.ndarray, positions: np.ndarray) -> float:
    """Total distance in metres, gaps excluded."""
    return float(step_distances(frames, positions).sum())


def smooth_positions(positions: np.ndarray, window: int = 9) -> np.ndarray:
    """Moving average over `window` frames, leaving absences absent.

    A player standing still still moves on paper, because the estimated position
    jitters. Smoothing removes most of that; comparing the smoothed distance with
    the raw one says how much of the total was noise.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if window < 2:
        return positions.copy()

    present = ~np.isnan(positions[:, 0])
    filled = np.where(present[:, None], positions, 0.0)
    kernel = np.ones(window)

    smoothed = np.empty_like(positions)
    counts = np.convolve(present.astype(np.float64), kernel, mode="same")
    for axis in (0, 1):
        totals = np.convolve(filled[:, axis], kernel, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed[:, axis] = totals / counts
    smoothed[~present] = np.nan
    return smoothed


def speed_percentile(
    frames: np.ndarray, positions: np.ndarray, fps: float, percentile: float = 95.0
) -> float:
    """Speed in metres per second at the given percentile of per-frame steps.

    A percentile rather than the maximum: a single bad position would set the
    maximum on its own, and would say nothing about how fast the player ran.
    """
    steps = step_distances(frames, positions)
    if steps.size == 0:
        return float("nan")
    return float(np.percentile(steps, percentile) * fps)


def mean_position(positions: np.ndarray) -> np.ndarray:
    """Average court position over the frames where the player was located."""
    positions = np.asarray(positions, dtype=np.float64)
    present = ~np.isnan(positions[:, 0])
    if not present.any():
        return np.array([np.nan, np.nan])
    return positions[present].mean(axis=0)
