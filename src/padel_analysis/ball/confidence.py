"""Deciding where the chosen ball path is trustworthy enough to show.

The path answers on every frame: its costs were swept to maximise recall, and recall
rewards always answering. So when the ball leaves the frame, or sits in a server's
hand, the path still hangs on to the least bad candidate and invents a trajectory.
That is right for the measurement and wrong for a viewer.

This module does not change the path. It reads back how sure the detector was about
each point, and hides what it was not sure of - for display only.
"""

from collections.abc import Sequence

from .candidates import Candidate

Point = tuple[float, float]


def path_scores(
    path: dict[int, Point | None], raw: dict[int, Sequence[Candidate]]
) -> dict[int, float]:
    """The detector's own score for each point of the path, zero where there is none.

    Scores are read from the candidates as the detector produced them, before any
    demotion inside player boxes. A ball about to be struck sits inside a box, and a
    confidence lowered there would hide the very racket contacts worth showing.
    """
    scores: dict[int, float] = {}
    for frame, point in path.items():
        scores[frame] = 0.0
        if point is None:
            continue
        for candidate in raw.get(frame, ()):
            if (candidate.x, candidate.y) == point:
                scores[frame] = candidate.score
                break
    return scores


def confident_path(
    path: dict[int, Point | None],
    scores: dict[int, float],
    threshold: float,
    min_run: int,
) -> dict[int, Point | None]:
    """The path with weak points hidden, and with confident runs too short dropped.

    Args:
        threshold: least detector score for a point to be shown.
        min_run: least number of consecutive confident frames to count as a
            trajectory. A few confident frames amid doubt are more often a flicker
            on a line or a shoe than the ball.
    """
    kept: dict[int, Point | None] = {
        frame: point if point is not None and scores.get(frame, 0.0) >= threshold else None
        for frame, point in path.items()
    }
    frames = sorted(kept)
    run: list[int] = []
    for frame in [*frames, None]:
        if frame is not None and kept[frame] is not None and (not run or frame == run[-1] + 1):
            run.append(frame)
            continue
        if len(run) < min_run:
            for f in run:
                kept[f] = None
        run = [frame] if frame is not None and kept[frame] is not None else []
    return kept
