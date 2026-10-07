"""The statistics of a whole match, by pair.

Given by pair because the tracker confuses partners: a pair's total stays right when
a single player's does not. Everything starts at the first scoreboard reading.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from ..io.scoreboard import ScoreState
from .basic_stats import distance_travelled, smooth_positions
from .net_control import NET_THRESHOLD
from .points import point_winner, side_of
from .segmentation import RACKET, RallySpan
from .sides import end_changes, half_of_top_row

WALLS = ("verre", "grillage")
ROWS = (1, 2)
LENGTHS = ((1, 3, "1 to 3 shots"), (4, 7, "4 to 7 shots"), (8, 1000, "8 shots or more"))


@dataclass
class PairStats:
    """One pair over the match: what it hit, how it moved, what it won."""

    row: int
    points_won: int = 0
    strikes: int = 0
    volleys: int = 0
    after_bounce: int = 0
    after_glass: int = 0
    distance_m: float = 0.0
    net_frames: int = 0
    located_frames: int = 0
    won_by_length: dict[str, int] = field(default_factory=dict)

    @property
    def net_share(self) -> float:
        return self.net_frames / self.located_frames if self.located_frames else float("nan")


@dataclass(frozen=True)
class Orientation:
    """Where the scoreboard's top row plays, change of ends after change of ends."""

    changes: tuple[int, ...]
    first_half: str
    votes_for: int
    votes_against: int

    def half_of(self, row: int, frame: int) -> str:
        top = half_of_top_row(frame, self.changes, self.first_half)
        if row == 1:
            return top
        return "far" if top == "near" else "near"

    def row_in(self, half: str, frame: int) -> int:
        return 1 if self.half_of(1, frame) == half else 2


def orient(readings: Sequence[tuple[int, ScoreState]],
           first_strikes: Mapping[int, str]) -> Orientation | None:
    """Return the changes of ends and where the top row starts, voted by the serves.

    Args:
        readings: (frame, state) at the start of each stretch read.
        first_strikes: per stretch start, the half of its first strike, the serve.
    """
    changes = end_changes([(f, s.set_number, *s.games) for f, s in readings])
    votes = Counter()
    for frame, state in readings:
        half = first_strikes.get(frame)
        if state.server is None or half is None:
            continue
        top = half if state.server == 1 else ("far" if half == "near" else "near")
        flips = sum(1 for change in changes if change <= frame)
        start = top if flips % 2 == 0 else ("far" if top == "near" else "near")
        votes[start] += 1
    if not votes:
        return None
    first, count = votes.most_common(1)[0]
    return Orientation(tuple(changes), first, count, sum(votes.values()) - count)


def stretch_points(readings: Sequence[tuple[int, ScoreState]]) -> dict[int, int]:
    """Return the pair that won the point played from each stretch start, when known."""
    winners = {}
    ordered = sorted(readings, key=lambda r: r[0])
    for (start, before), (_, after) in pairwise(ordered):
        if before == after:
            continue
        row = point_winner(before, after)
        if row is not None:
            winners[start] = row
    return winners


def point_stats(winners: Mapping[int, int], pairs: Mapping[int, PairStats]) -> None:
    """Return the points each pair won, from every stretch the board settles."""
    for row in winners.values():
        pairs[row].points_won += 1


def contact_stats(rallies: Sequence[RallySpan], strikers: Mapping[int, str],
                  orientation: Orientation, winners: Mapping[int, int],
                  pairs: Mapping[int, PairStats]) -> None:
    """Return strikes, volleys, shots after a bounce or the glass, and points by length."""
    for rally in rallies:
        bounced = glass = seen = False
        for frame, kind in rally.contacts:
            if kind == RACKET:
                slot = strikers.get(frame)
                if slot is not None:
                    pair = pairs[orientation.row_in(side_of(slot), frame)]
                    pair.strikes += 1
                    if seen:
                        pair.volleys += not bounced
                        pair.after_bounce += bounced
                        pair.after_glass += glass
                bounced = glass = False
                seen = True
            elif kind == "sol":
                bounced = True
            elif kind in WALLS:
                glass = True
        row = winners.get(rally.start)
        if row is not None:
            for low, high, name in LENGTHS:
                if low <= rally.strikes <= high:
                    won = pairs[row].won_by_length
                    won[name] = won.get(name, 0) + 1


def pair_positions(positions: Mapping[int, Mapping[str, tuple[float, float]]],
                   orientation: Orientation) -> dict[int, np.ndarray]:
    """Return every position of each pair's players, folded onto the near half."""
    folded: dict[int, list] = {1: [], 2: []}
    for frame, slots in positions.items():
        for slot, (x, y) in slots.items():
            row = orientation.row_in(side_of(slot), frame)
            folded[row].append((x, y) if y <= 0 else (-x, -y))
    return {row: np.array(points, dtype=np.float64).reshape(-1, 2)
            for row, points in folded.items()}


def movement_stats(positions: Mapping[int, Mapping[str, tuple[float, float]]],
                   breaks: Sequence[int], orientation: Orientation,
                   pairs: Mapping[int, PairStats], window: int = 9) -> None:
    """Return distance and time at the net per pair, stretch by stretch.

    A stretch ends at every broadcast splice and every change of ends.
    """
    frames = sorted(positions)
    if not frames:
        return
    cuts = sorted(set(breaks) | set(orientation.changes))
    bounds = [frames[0]] + [c for c in cuts if frames[0] < c <= frames[-1]] + [frames[-1] + 1]
    index = np.array(frames)
    for low, high in pairwise(bounds):
        chosen = index[(index >= low) & (index < high)]
        if chosen.size < 2:
            continue
        for slot in ("near_1", "near_2", "far_1", "far_2"):
            track = np.array([positions[f].get(slot, (np.nan, np.nan)) for f in chosen],
                             dtype=np.float64)
            located = ~np.isnan(track[:, 0])
            pair = pairs[orientation.row_in(side_of(slot), int(chosen[0]))]
            pair.distance_m += distance_travelled(chosen, smooth_positions(track, window))
            pair.located_frames += int(located.sum())
            pair.net_frames += int((np.abs(track[located, 1]) < NET_THRESHOLD).sum())
