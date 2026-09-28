"""Cutting a stretch of match into rallies.

The video keeps the rallies and cuts out the dead time between them, so a new point
opens where the broadcast splices: measured against the serves annotated in the
dataset, a serve follows a splice within a second in nearly every case. Two things
were tried on top and dropped: requiring a strike after the splice changed nothing,
and opening a point after a long silence without contact added more false starts than
it caught serves filmed without a cut.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

RACKET = "raquette"


@dataclass(frozen=True)
class RallySpan:
    """One rally: the frames it covers, and the contacts detected in it."""

    start: int
    stop: int
    contacts: tuple[tuple[int, str], ...]  # (image, surface) dans l'ordre

    @property
    def strikes(self) -> int:
        return sum(1 for _, kind in self.contacts if kind == RACKET)

    def duration(self, fps: float) -> float:
        """Seconds from the first contact to the last - the ball in play."""
        if len(self.contacts) < 2:
            return 0.0
        return (self.contacts[-1][0] - self.contacts[0][0]) / fps


def rallies(
    splices: Sequence[int], contacts: Sequence[tuple[int, str]], start: int, stop: int
) -> list[RallySpan]:
    """The rallies between `start` and `stop`: one per splice, if the ball was struck.

    A stretch with no strike is not a rally - a replay, a crowd shot, a player walking
    back - and is left out. The first stretch starts with the clip, and may be the end
    of a point whose serve came before it.
    """
    bounds = [start] + sorted(f for f in splices if start < f <= stop) + [stop + 1]
    ordered = sorted(contacts)
    result = []
    for first, after in pairwise(bounds):
        inside = tuple((f, kind) for f, kind in ordered if first <= f < after)
        if any(kind == RACKET for _, kind in inside):
            result.append(RallySpan(first, after - 1, inside))
    return result
