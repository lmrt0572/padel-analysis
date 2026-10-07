"""Racket contacts read from the striker's gesture rather than from the ball alone.

A shot along the camera axis barely bends the ball in the image, and a ball bouncing
next to a still player looks like a shot to a distance rule. The wrist speed settles both.
"""

import math
from collections.abc import Mapping, Sequence

Point = tuple[float, float]
Players = Mapping[int, tuple[Sequence, Mapping[str, int]]]
"""Frame -> (detections, slot -> detection index)."""

WRISTS = (9, 10)


def gesture_near(players: Players, frame: int, ball: Point | None, reach: float = 80.0) -> float:
    """Return the fastest speed, in px per frame, of a wrist within `reach` of the ball.

    The player is followed through the tracker's slot, so a wrist is never compared
    with somebody else's.
    """
    here = players.get(frame)
    if here is None or ball is None:
        return 0.0
    people, slots = here
    best = 0.0
    for slot, index in slots.items():
        for wrist in WRISTS:
            now = people[index].keypoints[wrist]
            if now[2] <= 0.3 or math.dist(ball, now[:2]) > reach:
                continue
            for offset in (-2, 2):
                other = players.get(frame + offset)
                if other is None or slot not in other[1]:
                    continue
                then = other[0][other[1][slot]].keypoints[wrist]
                if then[2] > 0.3:
                    best = max(best, math.dist(now[:2], then[:2]) / 2)
    return best


def strikes(
    players: Players,
    ball: Mapping[int, Point | None],
    start: int,
    stop: int,
    min_speed: float = 10.0,
    reach: float = 80.0,
    suppression: int = 10,
    taken: Sequence[int] = (),
) -> list[int]:
    """Return the frames where a wrist near the ball moves fast enough to be striking it.

    The fastest frame of each swing is kept.

    Args:
        min_speed: least wrist speed, px per frame.
        reach: how far from the ball, in pixels, a wrist may be.
        suppression: frames around a kept swing that are dropped.
        taken: contacts already found another way; a swing near one is not a new contact.
    """
    speeds = {f: gesture_near(players, f, ball.get(f), reach) for f in range(start, stop + 1)}
    kept: list[int] = list(taken)
    for frame in sorted((f for f, s in speeds.items() if s >= min_speed), key=lambda f: -speeds[f]):
        if all(abs(frame - k) > suppression for k in kept):
            kept.append(frame)
    return sorted(f for f in kept if f not in set(taken))
