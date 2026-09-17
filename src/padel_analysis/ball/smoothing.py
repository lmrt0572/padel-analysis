"""Making the displayed ball trajectory readable, without rounding its bounces.

Two steps, for display. First, lone points that jump away from both neighbours are
dropped: on a chosen path they are a frame where the wrong candidate won. Then each
piece of trajectory is smoothed by a constant-velocity Kalman filter followed by a
Rauch-Tung-Striebel backward pass, which re-estimates every point from the frames on
both sides of it.

A global smoother would round every contact into a curve, which is why the trajectory
stage refused one. Here the smoothing is cut at each contact: straight between them,
sharp at them. That is also how the tennis reference project gets its fluid trails.
"""

import math

import numpy as np

Point = tuple[float, float]


def despike(path: dict[int, Point | None], max_deviation: float = 25.0) -> dict[int, Point | None]:
    """Drop points lying far from the midpoint of their two neighbours.

    Args:
        max_deviation: pixels a point may sit from that midpoint. A real ball between
            two frames moves along a near-straight line; a jump well beyond it is a
            wrong candidate, not a bounce - a bounce bends the path, it does not leave
            one point stranded.
    """
    kept = dict(path)

    def deviation(frame: int) -> float:
        point, before, after = kept.get(frame), kept.get(frame - 1), kept.get(frame + 1)
        if point is None or before is None or after is None:
            return 0.0
        return math.dist(point, ((before[0] + after[0]) / 2, (before[1] + after[1]) / 2))

    # Le pire d'abord : un point aberrant fausse aussi le jugement de ses deux voisins,
    # qui ne doivent etre juges qu'une fois qu'il a disparu.
    suspects = {f for f in kept if deviation(f) > max_deviation}
    while suspects:
        worst = max(suspects, key=deviation)
        if deviation(worst) <= max_deviation:
            break
        kept[worst] = None
        suspects.discard(worst)
        suspects = {f for f in suspects | {worst - 1, worst + 1} if deviation(f) > max_deviation}
    return kept


def smooth_path(
    path: dict[int, Point | None],
    cuts: list[int],
    max_gap: int = 0,
    process_noise: float = 100.0,
    measurement_noise: float = 9.0,
) -> dict[int, Point | None]:
    """Each piece of the path smoothed on its own, pieces split at contacts and long gaps.

    Args:
        cuts: contact frames; smoothing never runs across one.
        max_gap: missing frames a piece may bridge. Zero by default: measured on an
            annotated minute, bridging three frames invents positions and raises the
            wrong ones from 194 to 261.
        process_noise: how freely velocity may change between frames. At 100 the
            displayed jerk falls from 6.0 to 4.5 px with no loss of accuracy; at 4 it
            falls to 1.8 px, but wrong positions rise from 199 to 280, every undetected
            bend being rounded off.
        measurement_noise: variance of a detected position, in square pixels.
    """
    out: dict[int, Point | None] = {frame: None for frame in path}
    for piece in _pieces(path, set(cuts), max_gap):
        for frame, point in zip(range(piece[0], piece[-1] + 1),
                                _rts(path, piece[0], piece[-1], process_noise,
                                     measurement_noise)):
            out[frame] = point
    return out


def _pieces(path: dict[int, Point | None], cuts: set[int], max_gap: int) -> list[list[int]]:
    pieces: list[list[int]] = []
    current: list[int] = []
    for frame in sorted(f for f, p in path.items() if p is not None):
        if current and (frame - current[-1] - 1 > max_gap or frame in cuts):
            if frame in cuts and frame - current[-1] - 1 <= max_gap:
                current.append(frame)
            pieces.append(current)
            current = [frame] if frame in cuts else []
            if frame in cuts:
                continue
        current.append(frame)
    if current:
        pieces.append(current)
    return [p for p in pieces if len(p) >= 2]


def _rts(path, first, last, q, r):
    transition = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    observe = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    noise_q = q * np.diag([0.25, 0.25, 1.0, 1.0])
    noise_r = r * np.eye(2)

    start = path[first]
    state = np.array([start[0], start[1], 0.0, 0.0])
    cov = np.diag([r, r, 400.0, 400.0])
    means, covs, predicted_means, predicted_covs = [], [], [], []
    for frame in range(first, last + 1):
        pred_mean = transition @ state
        pred_cov = transition @ cov @ transition.T + noise_q
        predicted_means.append(pred_mean)
        predicted_covs.append(pred_cov)
        point = path.get(frame)
        if point is None:
            state, cov = pred_mean, pred_cov
        else:
            gain = pred_cov @ observe.T @ np.linalg.inv(observe @ pred_cov @ observe.T + noise_r)
            state = pred_mean + gain @ (np.array(point) - observe @ pred_mean)
            cov = (np.eye(4) - gain @ observe) @ pred_cov
        means.append(state)
        covs.append(cov)

    smoothed = [means[-1]]
    for k in range(len(means) - 2, -1, -1):
        gain = covs[k] @ transition.T @ np.linalg.inv(predicted_covs[k + 1])
        smoothed.insert(0, means[k] + gain @ (smoothed[0] - predicted_means[k + 1]))
    return [(float(s[0]), float(s[1])) for s in smoothed]
