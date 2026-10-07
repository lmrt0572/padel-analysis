"""Choosing the ball over the whole sequence at once, not frame by frame.

Each frame offers its best candidates plus the option of holding no ball, and the
cheapest path through all of them is found by dynamic programming. The state
carries two frames, because velocity needs a pair.
"""

import math
from collections.abc import Sequence

from .candidates import Candidate

Point = tuple[float, float]


def acceleration_cost(
    previous: Point, current: Point, following: Point, ceiling: float = 80.0
) -> float:
    """Return how far `following` sits from where constant velocity would put it.

    Args:
        previous, current, following: three consecutive positions.
        ceiling: the most a single step may cost, so a bounce stays affordable.
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
    """Return what it costs to pick this candidate rather than the best of its frame.

    Args:
        candidate: the one being considered.
        frame: every candidate of that frame.
        weight: what a worthless candidate costs, in pixel-equivalents.
        absolute: read the score as it is rather than relative to the frame.
    """
    if weight == 0.0:
        return 0.0
    if absolute:
        # a score already in [0, 1] is read as it is, so a weak best candidate still
        # costs and the path can give up
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
    """Return the cheapest explanation of the whole sequence, frame by frame.

    Absence must cost more than a bounce, or the ball would vanish at every contact.

    Args:
        candidates: ranked candidates per frame, strongest first.
        start, stop: inclusive bounds of the answer.
        width: how many candidates of each frame are considered.
        gate: ceiling on the acceleration cost.
        weight: what the weakest candidate of a frame costs, in pixel-equivalents.
        absent_cost: what holding no ball costs for one frame.
        absolute: read candidate scores as they are; only meaningful for scores
            bounded in [0, 1].
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
