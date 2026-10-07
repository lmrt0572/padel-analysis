"""Turning a cached frame and its annotated ball into a training sample.

The target is a Gaussian rather than a binary mask, so the loss points at the centre
of the ball. Frames without an annotated ball are left out of training: nothing says
whether the ball was absent or only unlabelled.
"""

import numpy as np


def stacked_indices(index: int, spacing: int = 3) -> list[int]:
    """Return the frames one sample stacks, earliest first."""
    return [index - spacing, index, index + spacing]


def gaussian_target(
    shape: tuple[int, int], ball: tuple[float, float] | None, sigma: float = 2.0
) -> np.ndarray:
    """Return a heatmap peaking at the ball, or all zeros when there is no ball.

    Args:
        shape: (height, width) of the map.
        ball: ball centre in (x, y) pixels of that map, or None when unannotated.
        sigma: spread in pixels.
    """
    height, width = shape
    target = np.zeros((height, width), dtype=np.float32)
    if ball is None:
        return target

    ys, xs = np.mgrid[0:height, 0:width]
    squared = (xs - ball[0]) ** 2 + (ys - ball[1]) ** 2
    # the gaussian is cut beyond five sigma to keep the background at zero
    reach = (5.0 * sigma) ** 2
    near = squared < reach
    target[near] = np.exp(-squared[near] / (2.0 * sigma**2))
    return target
