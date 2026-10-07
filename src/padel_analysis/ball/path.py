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

The acceleration cost is capped, so that a contact costs a known amount rather than
an unaffordable one - a padel ball bounces off the floor, the glass, the mesh and the
rackets, and a path that could never afford to bounce would never be a ball.

That was the reasoning. The measurement does not support it: removing the cap
entirely costs one thousandth of the recall. The cap almost never binds, because the
emission weight dominates and the deviations at a real contact stay under it. It is
kept as a safeguard against a pathological frame, not as the mechanism it was
believed to be.
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
    candidate: Candidate,
    frame: Sequence[Candidate],
    weight: float = 30.0,
    absolute: bool = False,
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
    if weight == 0.0:
        return 0.0
    if absolute:
        # A score already bounded in [0, 1] (the network's) is read as it is: a weak
        # candidate is costly even when it is the best of its frame, and that is what
        # lets the path give up when the ball is not there.
        return weight * (1.0 - min(1.0, max(0.0, candidate.score)))
    if not frame:
        return 0.0
    best = max(c.score for c in frame)
    if best <= 0.0:
        return 0.0
    return weight * (1.0 - candidate.score / best)


ABSENT = -1


def best_path(
    candidates: dict[int, Sequence[Candidate]],
    start: int,
    stop: int,
    width: int = 8,
    gate: float = 320.0,
    weight: float = 240.0,
    absent_cost: float = 1200.0,
    absolute: bool = False,
) -> dict[int, Point | None]:
    """The cheapest explanation of the whole sequence, frame by frame.

    One relation between the two costs matters more than either value: **absence
    must cost more than a bounce**. An absent frame constrains velocity neither on
    the way in nor on the way out, so it launders an arbitrary jump for the price of
    one frame - blinking every third frame makes every transition free. If absence
    were the cheaper of the two, the cheapest path would make the ball vanish at
    every contact - one every fifteen frames - and the measurement would look poor
    without showing why.

    Args:
        candidates: ranked candidates per frame, strongest first.
        start, stop: inclusive bounds of the answer.
        width: how many candidates of each frame are considered. The ball sits in
            the top ten 96 percent of the time, and the cost grows with the cube of
            this number.
        gate: ceiling on the acceleration cost. Swept over 40 to infinity: the
            recall moves by a thousandth, so this bounds a pathological frame
            rather than shaping the answer.
        weight: what the weakest candidate of a frame costs, in pixel-equivalents.
            This one does shape the answer, and its optimum is interior: the recall
            runs 0.739, 0.764, 0.748, 0.712 at 120, 240, 480 and 960.
        absent_cost: what holding no ball costs for one frame. Above about a
            thousand the recall saturates, which is to say the path stops giving up
            at all - and giving up is what the greedy baseline did too much of.
        absolute: read candidate scores as they are rather than relative to the best
            of their frame. Relative scores make the best candidate of every frame
            free however weak it is, so the path can never prefer absence. Only
            meaningful for scores bounded in [0, 1], which the network gives and
            motion detection does not.
    """
    frames = list(range(start, stop + 1))
    kept: dict[int, list[Candidate]] = {
        f: list(candidates.get(f, []))[:width] for f in frames
    }

    def position(frame: int, index: int) -> Point | None:
        if index == ABSENT:
            return None
        return (kept[frame][index].x, kept[frame][index].y)

    def options(frame: int) -> list[int]:
        return [*range(len(kept[frame])), ABSENT]

    first = frames[0]
    if len(frames) == 1:
        only = kept[first]
        if not only:
            return {first: None}
        best = min(only, key=lambda c: emission_cost(c, only, weight, absolute))
        return {first: (best.x, best.y)}

    second = frames[1]
    costs: dict[tuple[int, int], float] = {}
    for i in options(first):
        for j in options(second):
            costs[(i, j)] = (
                absent_cost
                if i == ABSENT
                else emission_cost(kept[first][i], kept[first], weight, absolute)
            ) + (
                absent_cost
                if j == ABSENT
                else emission_cost(kept[second][j], kept[second], weight, absolute)
            )

    back: list[dict[tuple[int, int], tuple[int, int]]] = []
    for step in range(2, len(frames)):
        frame = frames[step]
        previous_frame, current_frame = frames[step - 2], frames[step - 1]
        updated: dict[tuple[int, int], float] = {}
        chosen: dict[tuple[int, int], tuple[int, int]] = {}

        for k in options(frame):
            emission = (
                absent_cost
                if k == ABSENT
                else emission_cost(kept[frame][k], kept[frame], weight, absolute)
            )
            for (i, j), so_far in costs.items():
                here = position(current_frame, j)
                there = position(frame, k)
                behind = position(previous_frame, i)
                if here is None or there is None or behind is None:
                    move = 0.0  # an absent state claims no motion
                else:
                    move = acceleration_cost(behind, here, there, gate)
                total = so_far + move + emission
                if (j, k) not in updated or total < updated[(j, k)]:
                    updated[(j, k)] = total
                    chosen[(j, k)] = (i, j)
        costs = updated
        back.append(chosen)

    final = min(costs, key=lambda state: costs[state])
    indices = [final[1], final[0]]
    state = final
    for chosen in reversed(back):
        state = chosen[state]
        indices.append(state[0])
    indices = list(reversed(indices))[: len(frames)]

    return {frame: position(frame, index) for frame, index in zip(frames, indices)}
