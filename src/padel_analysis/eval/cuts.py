"""Camera cuts, which proximity cannot see.

Identity is rebuilt by following each player to the nearest previous position. That
holds as long as the picture is continuous. A broadcast cut breaks it: the players
are somewhere else when the new shot opens, and if two partners exchanged posts
during the cut the association follows the wrong one - silently, because nothing
looks ambiguous. No detected episode, no doubt raised, a wrong ground truth.

A cut is recognised by what makes it dangerous: both players of a side moving at
once. One metre between two frames is thirty metres per second; the ninety-ninth
percentile of real movement is 0.43 m. A single player jumping alone is not a cut -
it is a tracking or annotation error, and it cannot exchange two identities.
"""

from itertools import pairwise

import numpy as np

from .identity import FAR_SLOTS, NEAR_SLOTS, CameraCut

SIDES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("near", NEAR_SLOTS),
    ("far", FAR_SLOTS),
)


def find_camera_cuts(
    positions: dict[int, dict[str, np.ndarray]],
    threshold: float = 1.0,
    max_gap: int = 2,
) -> list[CameraCut]:
    """Frames where a pair could have been exchanged by a change of shot.

    Args:
        positions: court coordinates per frame, keyed by slot name.
        threshold: metres a player must cover between two frames to count.
        max_gap: beyond this many missing frames the players had time to move,
            so the displacement says nothing.
    """
    frames = sorted(positions)
    cuts: list[CameraCut] = []

    for before, after in pairwise(frames):
        if after - before > max_gap:
            continue
        first, second = positions[before], positions[after]

        at_risk: list[str] = []
        smallest: list[float] = []
        for name, slots in SIDES:
            if any(s not in first or s not in second for s in slots):
                continue
            moved = [float(np.linalg.norm(second[s] - first[s])) for s in slots]
            if all(m > threshold for m in moved):
                at_risk.append(name)
                smallest.append(min(moved))

        if at_risk:
            cuts.append(
                CameraCut(
                    frame=after,
                    sides=tuple(at_risk),
                    displacement_m=min(smallest),
                )
            )
    return cuts
