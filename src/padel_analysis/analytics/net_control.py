"""Which pair holds the net, the decisive positional question in padel.

Threshold
---------
Players occupy two distinct depths. Measured over a full match - 182 713 positions -
the attacking mode peaks at 3.95 metres from the net and the defending one at 7.85,
against the back glass. The gap between them bottoms out at 5.85 metres, and the
threshold is placed there.

The service line, at 6.95 metres, is deliberately not used: it is a rule about
serving, not a marker of tactical position, and it falls on the wrong side of the
gap - it would count the whole defending band as attacking.

The gap is broad and shallow, so the exact value carries a degree of convention, and
the resulting percentages move with it: raising the threshold from 5.0 to 6.5 metres
takes the near pair's share from 28 % to 44 %. What does hold steady is the ratio
between the two pairs, which stays near 1.85 across that range. The README publishes
the sensitivity alongside the figures.
"""

from dataclasses import dataclass

import numpy as np

from .trajectories import SLOTS, MatchTrajectories

NET_THRESHOLD = 5.85
HYSTERESIS = 0.4

NEAR_SLOTS = ("near_1", "near_2")
FAR_SLOTS = ("far_1", "far_2")


@dataclass(frozen=True)
class NetControl:
    """How the net was shared, over the frames where all four players were located."""

    near_percent: float
    far_percent: float
    contested_percent: float
    evaluated_frames: int
    skipped_frames: int
    timeline: np.ndarray  # -1 near holds, +1 far holds, 0 contested


def at_net_states(
    depths: np.ndarray,
    threshold: float = NET_THRESHOLD,
    hysteresis: float = HYSTERESIS,
) -> np.ndarray:
    """Whether the player is in the attacking band, frame by frame.

    A player hovering on the line would otherwise switch state every frame, so a
    move towards the net must cross `threshold - hysteresis` and a retreat must
    cross `threshold + hysteresis`. An absent depth holds the previous state.
    """
    depths = np.asarray(depths, dtype=np.float64)
    states = np.zeros(depths.size, dtype=bool)
    current = False
    for i, depth in enumerate(depths):
        if not np.isnan(depth):
            if current and depth > threshold + hysteresis:
                current = False
            elif not current and depth < threshold - hysteresis:
                current = True
        states[i] = current
    return states


def net_control(
    trajectories: MatchTrajectories,
    threshold: float = NET_THRESHOLD,
    hysteresis: float = HYSTERESIS,
) -> NetControl:
    """A pair holds the net when both its players are up and both opponents are back."""
    states = {
        slot: at_net_states(trajectories.depth(slot), threshold, hysteresis)
        for slot in SLOTS
    }
    complete = trajectories.complete_mask()
    evaluated = int(complete.sum())
    skipped = int((~complete).sum())

    if evaluated == 0:
        return NetControl(
            near_percent=float("nan"),
            far_percent=float("nan"),
            contested_percent=float("nan"),
            evaluated_frames=0,
            skipped_frames=skipped,
            timeline=np.zeros(0, dtype=np.int8),
        )

    near_up = np.logical_and.reduce([states[s][complete] for s in NEAR_SLOTS])
    far_up = np.logical_and.reduce([states[s][complete] for s in FAR_SLOTS])

    timeline = np.zeros(evaluated, dtype=np.int8)
    timeline[near_up & ~far_up] = -1
    timeline[far_up & ~near_up] = +1

    return NetControl(
        near_percent=100.0 * float((timeline == -1).sum()) / evaluated,
        far_percent=100.0 * float((timeline == +1).sum()) / evaluated,
        contested_percent=100.0 * float((timeline == 0).sum()) / evaluated,
        evaluated_frames=evaluated,
        skipped_frames=skipped,
        timeline=timeline,
    )
