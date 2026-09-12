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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .shots import ShotEvent

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


@dataclass(frozen=True)
class CandidateScore:
    """How often the ball is in the list at all, and how far down it sits.

    This is a ceiling, not a performance: it says the ball is available to be
    picked, never that anything picked it.
    """

    recall: dict[int, float]
    median_rank: float
    within_top: float
    median_candidates: float
    annotated: int


def _median(values: list[int]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def candidate_score(
    candidates: dict[int, Sequence[Point]],
    annotated: dict[int, Point],
    tolerances: Sequence[int] = TOLERANCES,
    top: int = 10,
) -> CandidateScore:
    """Measure whether the annotated ball appears among the candidates.

    Args:
        candidates: ranked positions per frame, best first.
        annotated: the annotated position, for annotated frames only.
        tolerances: pixel distances under which a candidate counts as the ball.
        top: rank under which a candidate is considered easy to pick.
    """
    widest = max(tolerances)
    ranks: list[int] = []
    counts: list[int] = []
    hits: dict[int, int] = dict.fromkeys(tolerances, 0)

    for frame, truth in annotated.items():
        listed = list(candidates.get(frame, []))
        counts.append(len(listed))

        for tolerance in tolerances:
            for position, candidate in enumerate(listed, start=1):
                if _distance(candidate, truth) <= tolerance:
                    hits[tolerance] += 1
                    if tolerance == widest:
                        ranks.append(position)
                    break

    total = len(annotated)
    return CandidateScore(
        recall={t: (hits[t] / total if total else float("nan")) for t in tolerances},
        median_rank=_median(ranks),
        within_top=(
            sum(1 for r in ranks if r <= top) / total if total else float("nan")
        ),
        median_candidates=_median(counts),
        annotated=total,
    )


@dataclass(frozen=True)
class EventScore:
    """Detected contacts against the annotated shot intervals.

    The annotation gives an interval, so a contact is credited when it falls
    inside one. How close it lands to the true impact cannot be measured, and is
    therefore not claimed.
    """

    recall: float
    precision: float
    events: int
    contacts: int
    matched_events: int


def event_score(contacts: Sequence[int], events: Sequence["ShotEvent"]) -> EventScore:
    """Args:
        contacts: frames at which a contact was detected.
        events: the annotated shot intervals to find.
    """
    matched = sum(1 for e in events if any(e.contains(c) for c in contacts))
    inside = sum(1 for c in contacts if any(e.contains(c) for e in events))

    return EventScore(
        recall=matched / len(events) if events else float("nan"),
        precision=inside / len(contacts) if contacts else float("nan"),
        events=len(events),
        contacts=len(contacts),
        matched_events=matched,
    )
