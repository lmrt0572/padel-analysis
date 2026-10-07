"""Cutting a stretch of match into rallies.

The video cuts out the dead time between rallies, so a new point opens at each
broadcast splice.
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
    contacts: tuple[tuple[int, str], ...]  # (frame, surface) in order

    @property
    def strikes(self) -> int:
        return sum(1 for _, kind in self.contacts if kind == RACKET)

    def duration(self, fps: float) -> float:
        """Return the seconds from the first contact to the last."""
        if len(self.contacts) < 2:
            return 0.0
        return (self.contacts[-1][0] - self.contacts[0][0]) / fps


def rallies(
    splices: Sequence[int], contacts: Sequence[tuple[int, str]], start: int, stop: int
) -> list[RallySpan]:
    """Return the rallies between `start` and `stop`: one per splice with a strike.

    A stretch with no strike (a replay, a crowd shot) is left out.
    """
    bounds = [start] + sorted(f for f in splices if start < f <= stop) + [stop + 1]
    ordered = sorted(contacts)
    result = []
    for first, after in pairwise(bounds):
        inside = tuple((f, kind) for f, kind in ordered if first <= f < after)
        if any(kind == RACKET for _, kind in inside):
            result.append(RallySpan(first, after - 1, inside))
    return result
