"""Turning a cached frame and its annotated ball into a training sample.

The target is a Gaussian rather than a binary mask. A ball covers about three pixels
at this resolution, so a mask would be almost empty and would say nothing about where
inside it the centre lies. A Gaussian gives the loss a gradient pointing at the centre,
and its peak is what the detector reads back.

A frame with no annotated ball gets an empty target - and such frames are excluded from
training rather than taught as background. 17.5 percent of the annotated frames carry
no ball, and nothing says whether the ball was absent or merely unannotated; teaching
them as empty would assert what the data does not support.
"""

import numpy as np


def stacked_indices(index: int, spacing: int = 3) -> list[int]:
    """The frames one sample stacks, earliest first.

    Motion is what separates a ball from a painted line or a reflection, and motion
    needs more than one frame. Spacing three matches the cache grid, and stage B.1
    measured spacings two, three and four equivalent on candidate recall - 0,912,
    0,913 and 0,907 at ten pixels.
    """
    return [index - spacing, index, index + spacing]


def gaussian_target(
    shape: tuple[int, int], ball: tuple[float, float] | None, sigma: float = 2.0
) -> np.ndarray:
    """A heatmap peaking at the ball, or all zeros when there is no ball.

    Args:
        shape: (height, width) of the map.
        ball: ball centre in (x, y) pixels of that map, or None when unannotated.
        sigma: spread in pixels. Two is about the ball's radius at this resolution.
    """
    height, width = shape
    target = np.zeros((height, width), dtype=np.float32)
    if ball is None:
        return target

    ys, xs = np.mgrid[0:height, 0:width]
    squared = (xs - ball[0]) ** 2 + (ys - ball[1]) ** 2
    # Au-dela de cinq sigma la gaussienne vaut moins d'un millionieme : la couper la
    # evite de peindre un fond bruite sur toute la carte.
    reach = (5.0 * sigma) ** 2
    near = squared < reach
    target[near] = np.exp(-squared[near] / (2.0 * sigma**2))
    return target
