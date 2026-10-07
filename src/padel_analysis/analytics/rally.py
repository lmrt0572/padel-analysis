"""What happened in one rally: its contacts, who struck, where the ball landed.

Computed from plain data, contacts already decided and players already placed, so
each statistic can be checked by hand. A rally inherits the errors of the contact model.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from .basic_stats import distance_travelled, smooth_positions
from .net_control import at_net_states

RACKET = "raquette"
WALLS = ("verre", "grillage")
RACKET_HEIGHT = 1.0
"""Height in metres a strike is placed at when estimating ball speed, by convention."""

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class RallyContact:
    """One contact of a rally, as the contact model decided it."""

    frame: int
    kind: str  # raquette, sol, verre, grillage or filet
    point: Point3 | None  # on the surface touched, in metres; None for a stroke
    player: str | None  # the striker's slot, for a stroke


@dataclass(frozen=True)
class Rally:
    """A stretch of play between two frames, with its contacts and players."""

    start: int
    stop: int
    fps: float
    contacts: tuple[RallyContact, ...]
    positions: dict[str, dict[int, Point2]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.contacts, key=lambda c: c.frame))
        object.__setattr__(self, "contacts", ordered)

    def time_of(self, contact: RallyContact) -> float:
        """Return the seconds from the start of the rally."""
        return (contact.frame - self.start) / self.fps

    @property
    def duration(self) -> float:
        return (self.stop - self.start + 1) / self.fps


def shots_by_player(rally: Rally) -> dict[str, int]:
    """Return how many times each player struck the ball."""
    return dict(Counter(c.player for c in rally.contacts if c.kind == RACKET and c.player))


def after_each_shot(rally: Rally) -> dict[str, dict[str, int]]:
    """Return, for each player, what the ball touched right after their shots.

    "fin" when nothing followed within the rally.
    """
    after: dict[str, Counter] = {}
    contacts = rally.contacts
    for index, contact in enumerate(contacts):
        if contact.kind != RACKET or contact.player is None:
            continue
        following = contacts[index + 1].kind if index + 1 < len(contacts) else "fin"
        after.setdefault(contact.player, Counter())[following] += 1
    return {player: dict(counts) for player, counts in after.items()}


def impacts(rally: Rally) -> list[RallyContact]:
    """Return the contacts that can be placed on the court: floor, walls and net."""
    return [c for c in rally.contacts if c.kind != RACKET and c.point is not None]


@dataclass(frozen=True)
class SpeedSegment:
    """The ball's mean speed between two consecutive contacts."""

    start: int
    stop: int
    metres_per_second: float
    estimated: bool  # one end is a stroke, placed by convention

    @property
    def kmh(self) -> float:
        return self.metres_per_second * 3.6


def ball_speeds(rally: Rally) -> list[SpeedSegment]:
    """Return the straight-line speed between consecutive placeable contacts.

    A lower bound: the ball flies a curve, and a missed bounce merges two flights.
    """
    segments = []
    contacts = rally.contacts
    for first, second in pairwise(contacts):
        a, b = _place(rally, first), _place(rally, second)
        seconds = (second.frame - first.frame) / rally.fps
        if a is None or b is None or seconds <= 0:
            continue
        estimated = RACKET in (first.kind, second.kind)
        segments.append(SpeedSegment(first.frame, second.frame, math.dist(a, b) / seconds,
                                     estimated))
    return segments


def _place(rally: Rally, contact: RallyContact) -> Point3 | None:
    if contact.kind != RACKET:
        return contact.point
    track = rally.positions.get(contact.player or "", {})
    if contact.frame not in track:
        return None
    x, y = track[contact.frame]
    return (x, y, RACKET_HEIGHT)


@dataclass(frozen=True)
class Movement:
    """How far a player went during the rally, and how much of it at the net."""

    path: np.ndarray  # (N, 2) smoothed positions, in metres
    distance: float
    net_share: float


def movements(rally: Rally) -> dict[str, Movement]:
    """Return each player's smoothed path, distance travelled and share at the net."""
    result = {}
    frames = np.arange(rally.start, rally.stop + 1)
    for player, track in rally.positions.items():
        raw = np.array([track.get(int(f), (np.nan, np.nan)) for f in frames], dtype=float)
        if np.isnan(raw[:, 0]).all():
            continue
        smooth = smooth_positions(raw)
        present = ~np.isnan(smooth[:, 0])
        depths = np.abs(smooth[:, 1])
        at_net = at_net_states(np.where(present, depths, np.nan))
        result[player] = Movement(
            path=smooth[present],
            distance=distance_travelled(frames, smooth),
            net_share=float(at_net[present].mean()),
        )
    return result


@dataclass(frozen=True)
class Summary:
    """The rally in four numbers."""

    duration: float
    shots: int
    walls: int
    last: str | None


def summary(rally: Rally) -> Summary:
    contacts = rally.contacts
    return Summary(
        duration=rally.duration,
        shots=sum(1 for c in contacts if c.kind == RACKET),
        walls=sum(1 for c in contacts if c.kind in WALLS),
        last=contacts[-1].kind if contacts else None,
    )
