"""Racket contacts read from the striker's gesture rather than from the ball alone.

A shot along the camera axis barely bends the ball in the image, so the turn criterion
misses it; a ball bouncing next to a still player looks like a shot to a pure distance
rule. The wrist settles both. Measured against a complete hand marking: the wrist near
the ball moves at 26 px per frame in median at a marked shot, and at 5 px at instants
with no contact - and the shots the turn criterion missed move just as fast, at 29.
"""

import math
from collections.abc import Mapping, Sequence

Point = tuple[float, float]
Players = Mapping[int, tuple[Sequence, Mapping[str, int]]]
"""Frame -> (detections, slot -> detection index)."""

WRISTS = (9, 10)


def gesture_near(players: Players, frame: int, ball: Point | None, reach: float = 80.0) -> float:
    """The fastest speed, in px per frame, of a wrist within `reach` of the ball.

    Speed is measured over two frames each side, following the player through the
    tracker's slot so that a wrist is never compared with somebody else's.
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
    """Frames where a wrist near the ball moves fast enough to be striking it.

    One swing lasts several frames, so the fastest frame of each is kept and its
    neighbours within `suppression` are dropped.

    Args:
        min_speed: least wrist speed, px per frame. 10 was swept on the tuning match.
        reach: how far from the ball, in pixels, a wrist may be.
        taken: contacts already found another way. A swing near one of them is that
            contact, not a new one - and it must not suppress a genuine swing next to
            it, which is why they are excluded before the swings are merged.
    """
    speeds = {f: gesture_near(players, f, ball.get(f), reach) for f in range(start, stop + 1)}
    kept: list[int] = list(taken)
    for frame in sorted((f for f, s in speeds.items() if s >= min_speed), key=lambda f: -speeds[f]):
        if all(abs(frame - k) > suppression for k in kept):
            kept.append(frame)
    return sorted(f for f in kept if f not in set(taken))
