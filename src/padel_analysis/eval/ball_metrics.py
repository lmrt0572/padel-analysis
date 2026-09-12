"""Scores for the ball, reported at several tolerances.

A single tolerance would hide whether the errors are two pixels or fifteen, so
recall and precision are given at each of them - the same reasoning that put four
inference resolutions in the sub-project A table rather than one.

Frames carrying no annotation are the delicate part. Seventeen percent of the
women's final has none, and nothing says whether the ball is invisible there or
merely unlabelled. A detection on such a frame therefore cannot be called wrong:
those frames are counted apart, as `unscorable`, and left out of precision.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

TOLERANCES: tuple[int, ...] = (5, 10, 20)

Point = tuple[float, float]


@dataclass(frozen=True)
class BallScore:
    recall: dict[int, float]
    precision: dict[int, float]
    annotated: int
    predicted: int
    unscorable: int


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def ball_score(
    predicted: dict[int, Point | None],
    annotated: dict[int, Point],
    tolerances: Sequence[int] = TOLERANCES,
) -> BallScore:
    """Compare one predicted position per frame with the annotated ones.

    Args:
        predicted: position per evaluated frame; None where nothing was found.
        annotated: the annotated position, for annotated frames only.
        tolerances: pixel distances under which a prediction counts as correct.
    """
    found = {f: p for f, p in predicted.items() if p is not None}
    scorable = {f: p for f, p in found.items() if f in annotated}
    unscorable = len(found) - len(scorable)

    recall: dict[int, float] = {}
    precision: dict[int, float] = {}
    for tolerance in tolerances:
        hits = sum(
            1 for f, p in scorable.items() if _distance(p, annotated[f]) <= tolerance
        )
        recall[tolerance] = hits / len(annotated) if annotated else float("nan")
        precision[tolerance] = hits / len(scorable) if scorable else float("nan")

    return BallScore(
        recall=recall,
        precision=precision,
        annotated=len(annotated),
        predicted=len(found),
        unscorable=unscorable,
    )
