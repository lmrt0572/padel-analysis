"""Choosing the ball over the whole sequence at once, not frame by frame.

Growing segments greedily was tried first and measured: it covers 52.5 percent of
the annotated balls at best, and 16.8 percent once overlaps are resolved. The reason
is structural rather than a matter of tuning - every decision is local and final, so
a bad start is never revisited and a good segment lost to an overlap is lost for
good.

Here the whole sequence is decided together. Each frame offers its best candidates
plus the option of holding no ball at all, and the cheapest path through all of them
is found by dynamic programming. The state carries two frames rather than one,
because velocity is what makes a trajectory predictable and velocity needs a pair.

The acceleration cost is capped, and that cap is the whole trick. A padel ball
bounces off the floor, the glass, the mesh and the rackets, so a cost that grew
without bound would make the cheapest path one that never bounces - which is to say,
never a ball. Capped, a contact costs a known amount and the path takes it when the
evidence is worth the price.
"""

import math
from collections.abc import Sequence

from .candidates import Candidate

Point = tuple[float, float]


def acceleration_cost(
    previous: Point, current: Point, following: Point, ceiling: float = 80.0
) -> float:
    """How far `following` sits from where constant velocity would put it.

    Args:
        previous, current, following: three consecutive positions.
        ceiling: the most a single step may cost. A contact is a large deviation
            that must stay affordable - free play deviates by 5.4 px in median and
            64 px at the ninety-ninth percentile, so a ceiling above that lets a
            bounce through at a bounded price.
    """
    expected_x = 2 * current[0] - previous[0]
    expected_y = 2 * current[1] - previous[1]
    deviation = math.hypot(following[0] - expected_x, following[1] - expected_y)
    return min(deviation, ceiling)


def emission_cost(
    candidate: Candidate, frame: Sequence[Candidate], weight: float = 30.0
) -> float:
    """What it costs to pick this candidate rather than the best of its frame.

    Scores are normalised inside the frame because the motion peak varies with the
    background: comparing a candidate to its neighbours is meaningful, comparing it
    to a candidate of another frame is not.

    Args:
        candidate: the one being considered.
        frame: every candidate of that frame.
        weight: what a completely worthless candidate costs, in pixel-equivalents,
            so that this cost is commensurable with the acceleration one.
    """
    if not frame or weight == 0.0:
        return 0.0
    best = max(c.score for c in frame)
    if best <= 0.0:
        return 0.0
    return weight * (1.0 - candidate.score / best)
