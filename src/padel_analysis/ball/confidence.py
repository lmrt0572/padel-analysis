"""Deciding where the chosen ball path is trustworthy enough to show.

The path answers on every frame, even when the ball is out of the picture. This
module hides the points the detector was not sure of, for display only.
"""

from collections.abc import Sequence

from .candidates import Candidate

Point = tuple[float, float]


def path_scores(
    path: dict[int, Point | None], raw: dict[int, Sequence[Candidate]]
) -> dict[int, float]:
    """Return the detector's score for each point of the path, zero where there is none.

    Scores are read before any demotion inside player boxes, so a ball about to be
    struck keeps its confidence.
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
    """Return the path with weak points and short confident runs hidden.

    Args:
        threshold: least detector score for a point to be shown.
        min_run: least number of consecutive confident frames to count as a trajectory.
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
