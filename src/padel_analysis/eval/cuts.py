"""Camera cuts, which proximity cannot see.

Identity is rebuilt by following each player to the nearest previous position. That
holds as long as the picture is continuous. A broadcast cut breaks it: the players
are somewhere else when the new shot opens, and if two partners exchanged posts
during the cut the association follows the wrong one - silently, because nothing
looks ambiguous. No detected episode, no doubt raised, a wrong ground truth.

Any player covering more than a metre between two frames marks one: that is thirty
metres per second, where the ninety-ninth percentile of real movement is 0.43 m.

Requiring both players of a pair to move would be the obvious rule, and it is the
wrong one. The assignment being audited follows each player to the nearest previous
position, so when two partners exchange places across a splice the labels follow the
wrong person and the measured displacement collapses to almost nothing. The rule
would be blind to exactly the case it exists to catch. One player above the threshold
is therefore enough, at the price of a few clips raised by a lone annotation glitch -
nine in the women's match, twenty-nine in the men's.

The annotations hold short holes, and identity is carried across them, so they
cannot simply be skipped. What a hole changes is how much movement is credible: the
allowance grows with the time elapsed, from the noise floor between adjacent frames
to several metres across a second. Eight metres over three missing frames is a
splice; ten metres over seventy-seven is a player running.

For the same reason `sides` is a hint and nothing more. Which pair actually swapped
cannot be read from position: a clean exchange and a pair standing still are the same
measurement. Only a human watching the clip can tell them apart.
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
    max_speed_ms: float = 6.0,
    fps: float = 30.0,
) -> list[CameraCut]:
    """Frames where identity could have been exchanged by a splice.

    Args:
        positions: court coordinates per frame, keyed by slot name.
        threshold: metres a player must cover between two adjacent frames to count.
            The floor is set by annotation noise, whose ninety-ninth percentile is
            0.43 m, not by what a body can do in a thirtieth of a second.
        max_speed_ms: fastest a player is credited with moving. Measured peaks sit
            near 3.9 m/s, so six leaves room without excusing a teleport.
        fps: frames per second, to turn a hole into an elapsed time.
    """
    frames = sorted(positions)
    cuts: list[CameraCut] = []

    for before, after in pairwise(frames):
        elapsed = (after - before) / fps
        allowance = max(threshold, max_speed_ms * elapsed)
        first, second = positions[before], positions[after]

        moving: list[str] = []
        largest: list[float] = []
        for name, slots in SIDES:
            present = [s for s in slots if s in first and s in second]
            if not present:
                continue
            moved = [float(np.linalg.norm(second[s] - first[s])) for s in present]
            if any(m > allowance for m in moved):
                moving.append(name)
                largest.append(max(moved))

        if moving:
            cuts.append(
                CameraCut(
                    frame=after,
                    sides=tuple(moving),
                    displacement_m=max(largest),
                )
            )
    return cuts
