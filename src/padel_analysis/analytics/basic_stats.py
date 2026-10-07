"""Distance, speed and mean position, computed from court trajectories.

A dropout is never crossed as if it were a move, and smoothing is explicit so the
share of position noise in a distance can be reported.
"""

import numpy as np


def step_distances(frames: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Return the distances between positions on consecutive frames, gaps left out."""
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
    """Return the total distance in metres, gaps excluded."""
    return float(step_distances(frames, positions).sum())


def smooth_positions(positions: np.ndarray, window: int = 9) -> np.ndarray:
    """Return the moving average over `window` frames, leaving absences absent."""
    positions = np.asarray(positions, dtype=np.float64)
    # a stretch shorter than the window is smoothed over its own length
    window = min(window, len(positions))
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
    """Return the speed in m/s at the given percentile of per-frame steps.

    A percentile rather than the maximum, which a single bad position would set.
    """
    steps = step_distances(frames, positions)
    if steps.size == 0:
        return float("nan")
    return float(np.percentile(steps, percentile) * fps)


def mean_position(positions: np.ndarray) -> np.ndarray:
    """Return the average court position over the frames where the player was located."""
    positions = np.asarray(positions, dtype=np.float64)
    present = ~np.isnan(positions[:, 0])
    if not present.any():
        return np.array([np.nan, np.nan])
    return positions[present].mean(axis=0)
