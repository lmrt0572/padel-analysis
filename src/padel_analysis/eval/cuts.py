"""Camera cuts, which proximity cannot see.

Identity is rebuilt by following each player to the nearest previous position. A
broadcast cut breaks that silently when two partners exchange posts. A single player
covering too much ground between two frames marks a cut: requiring both would be
blind to a pair that swaps places. Bursts of cuts are merged, and `sides` is only a
hint for the human who arbitrates.
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
    min_separation: int = 15,
) -> list[CameraCut]:
    """Return the frames where identity could have been exchanged by a splice.

    Args:
        positions: court coordinates per frame, keyed by slot name.
        threshold: metres a player must cover between two adjacent frames to count.
        max_speed_ms: fastest a player is credited with moving.
        fps: frames per second, to turn a hole into an elapsed time.
        min_separation: splices closer than this many frames are reported once.
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
    return _merge_bursts(cuts, min_separation)


def _merge_bursts(cuts: list[CameraCut], min_separation: int) -> list[CameraCut]:
    """Keep one entry per burst, the one carrying the largest displacement."""
    merged: list[CameraCut] = []
    burst: list[CameraCut] = []

    for cut in cuts:
        if burst and cut.frame - burst[-1].frame <= min_separation:
            burst.append(cut)
            continue
        if burst:
            merged.append(_representative(burst))
        burst = [cut]
    if burst:
        merged.append(_representative(burst))
    return merged


def _representative(burst: list[CameraCut]) -> CameraCut:
    strongest = max(burst, key=lambda c: c.displacement_m)
    sides = tuple(
        name for name, _ in SIDES if any(name in c.sides for c in burst)
    )
    return CameraCut(
        frame=strongest.frame,
        sides=sides,
        displacement_m=strongest.displacement_m,
    )
