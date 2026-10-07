"""Airborne frames, and what they cost a ground-plane projection.

A point at height `h` projected through a ground homography lands at
`d * H / (H - h)` from the camera nadir rather than at `d`. A player is airborne
when the ankles rise clearly above their own recent baseline.
"""

import numpy as np


def airborne_mask(
    ankle_image_y: np.ndarray, window: int = 31, tolerance: float = 10.0
) -> np.ndarray:
    """Return the frames where the ankles sit well above their own local baseline.

    Args:
        ankle_image_y: (N,) ankle height in image pixels; smaller is higher up.
        window: number of frames the baseline is taken over.
        tolerance: how many pixels above the baseline counts as a jump.
    """
    heights = np.asarray(ankle_image_y, dtype=np.float64)
    present = ~np.isnan(heights)
    if not present.any():
        return np.zeros(heights.size, dtype=bool)

    baseline = np.full(heights.size, np.nan)
    half = window // 2
    for i in range(heights.size):
        low, high = max(0, i - half), min(heights.size, i + half + 1)
        chunk = heights[low:high]
        chunk = chunk[~np.isnan(chunk)]
        if chunk.size:
            baseline[i] = np.median(chunk)

    with np.errstate(invalid="ignore"):
        mask = (baseline - heights) > tolerance
    mask[~present] = False
    return mask


def ground_projection_bias(
    distance_m: float, height_m: float, camera_height_m: float
) -> float:
    """Return the extra distance, in metres, that an airborne point appears to gain.

    Args:
        distance_m: horizontal distance from the camera nadir.
        height_m: how far off the ground the point is.
        camera_height_m: the camera's height above the court.
    """
    if height_m <= 0.0:
        return 0.0
    return distance_m * camera_height_m / (camera_height_m - height_m) - distance_m
