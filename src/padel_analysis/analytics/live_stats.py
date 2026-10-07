"""Game and player statistics as they stand at each frame, for the video's side panel.

The video shows the match as it unfolds, so the numbers beside it must not know the
future: every figure here is counted from the start of the clip up to the frame being
drawn. The work is done once, frame by frame, so drawing a frame only reads a row.
"""

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from .basic_stats import smooth_positions
from .net_control import at_net_states
from .rally import RACKET, WALLS, Rally, ball_speeds
from .segmentation import rallies

SLOTS = ("near_1", "near_2", "far_1", "far_2")
PAIRS = {"proche": ("near_1", "near_2"), "fond": ("far_1", "far_2")}
MAX_STEP = 0.5
"""Metres in one frame, 54 km/h: a larger step is the tracker jumping, not a player
running, and it would set every running record on its own."""
TRAIL = 45  # frames of trail behind each player on the minimap, 1.5 s
SUSTAINED = 31  # frames: the top speed is held for one second


@dataclass(frozen=True)
class PlayerLine:
    slot: str
    shots: int
    volleys: int  # strokes with no floor bounce since the previous stroke
    after_bounce: int
    distance: float  # metres, smoothed
    top_speed: float  # km/h, the highest speed held for one second
    net_share: float  # share of the time spent at the net, over the frames where he is seen
    winners: int = 0  # points ended by a stroke of this player, won
    errors: int = 0  # points ended by a stroke of this player, lost


@dataclass(frozen=True)
class PointOutcome:
    """A point over: when it ended, the half that won it, and who it is credited to."""

    frame: int
    winner_side: str  # near or far
    player: str | None
    kind: str | None  # "gagnant" (winner) or "faute" (error)


@dataclass(frozen=True)
class LiveStats:
    """What the panel shows at one frame."""

    elapsed: float
    shots: int
    walls: int
    last: str | None  # the last contact, if it is less than a second old
    players: tuple[PlayerLine, ...]
    pair_shots: dict[str, int]
    pair_net: dict[str, float]
    last_shot_speed: float | None  # km/h, of the last shot whose arrival is known
    top_shot_speed: float | None
    positions: dict[str, list[tuple[float, float]]]  # the recent trail, the most recent last
    rally_number: int | None = None  # the rally in progress, counted from the start of the clip
    rally_shots: int = 0  # strokes of the rally in progress so far
    longest_rally: int = 0  # the most strokes in a single rally, so far
    pair_points: dict[str, int] | None = None  # points won by pair, if the score is read
    rally_pair_shots: dict[str, int] | None = None  # strokes of each pair in the rally


def volley_flags(rally: Rally) -> dict[int, bool | None]:
    """For each strike, whether it was a volley: no floor bounce since the previous strike.

    The first strike of a clip has no previous one to look back to, so it is left
    undecided. A floor bounce the contact model missed turns a shot after the bounce
    into a volley here: the split inherits the model's errors.
    """
    flags: dict[int, bool | None] = {}
    bounced, seen_strike = False, False
    for contact in rally.contacts:
        if contact.kind == RACKET:
            flags[contact.frame] = (not bounced) if seen_strike else None
            bounced, seen_strike = False, True
        elif contact.kind == "sol":
            bounced = True
    return flags


def _mean_share(shares) -> float:
    """The mean of the shares that are known, or NaN when none is."""
    known = [share for share in shares if not math.isnan(share)]
    return float(np.mean(known)) if known else math.nan


def _until(items, frame: int) -> list:
    """The items, in order, that have happened by `frame`."""
    return [item for item in items if item.frame <= frame]


def _by_pair(value, combine) -> dict:
    """For each pair, `combine` of `value(slot)` over its two players."""
    return {pair: combine(value(slot) for slot in slots) for pair, slots in PAIRS.items()}


class LiveTimeline:
    """Cumulative statistics for every frame of a rally, computed once."""

    def __init__(self, rally: Rally, recent: float = 1.0, splices=(), points=None) -> None:
        """Args:
            splices: where the broadcast splices, each opening a new rally.
            points: the points already decided, from the scoreboard; None when it
                was not read.
        """
        self.rally = rally
        self._points = None if points is None else sorted(points, key=lambda p: p.frame)
        self._spans = rallies(splices, [(c.frame, c.kind) for c in rally.contacts],
                              rally.start, rally.stop)
        self.recent = round(recent * rally.fps)
        frames = np.arange(rally.start, rally.stop + 1)
        self._volley = volley_flags(rally)
        self._speeds = [s for s in ball_speeds(rally)
                        if any(c.frame == s.start and c.kind == RACKET for c in rally.contacts)]
        self._distance: dict[str, np.ndarray] = {}
        self._top: dict[str, np.ndarray] = {}
        self._at_net: dict[str, np.ndarray] = {}
        self._seen: dict[str, np.ndarray] = {}
        self._smooth: dict[str, np.ndarray] = {}
        for slot in SLOTS:
            track = rally.positions.get(slot, {})
            raw = np.array([track.get(int(f), (np.nan, np.nan)) for f in frames], dtype=float)
            smooth = smooth_positions(raw)
            self._smooth[slot] = smooth
            present = ~np.isnan(smooth[:, 0])
            both = present[1:] & present[:-1]
            deltas = np.linalg.norm(np.diff(np.nan_to_num(smooth), axis=0), axis=1)
            steps = np.zeros(len(frames))
            steps[1:] = np.where(both & (deltas <= MAX_STEP), deltas, 0.0)
            self._distance[slot] = np.cumsum(steps)
            # The top speed is a speed held for one second. Any shorter, two artefacts
            # pass for sprints: a small jump lifts the ankles in the picture, and the
            # ground point moves back a metre at the far end of the court; an identity
            # swap between partners moves the position by several metres. Measured on two
            # rallies: 21 and 50 km/h with a median over 5 frames, 15 and 19 over 31.
            speed = median_filter(steps * rally.fps * 3.6, size=SUSTAINED, mode="nearest")
            self._top[slot] = np.maximum.accumulate(speed)
            at_net = at_net_states(np.where(present, np.abs(smooth[:, 1]), np.nan))
            self._at_net[slot] = np.cumsum(at_net & present)
            self._seen[slot] = np.cumsum(present)

    def at(self, frame: int) -> LiveStats:
        """The statistics counted from the start of the rally up to `frame` included."""
        rally = self.rally
        frame = min(max(frame, rally.start), rally.stop)
        index = frame - rally.start
        past = _until(rally.contacts, frame)
        strikes = [c for c in past if c.kind == RACKET]
        over = _until(self._points or (), frame)
        players = tuple(self._player_line(slot, index, strikes, over) for slot in SLOTS)
        by_slot = {line.slot: line for line in players}
        last_speed, top_speed = self._shot_speeds(frame)
        return LiveStats(
            elapsed=index / rally.fps,
            shots=len(strikes),
            walls=sum(c.kind in WALLS for c in past),
            last=self._recent_kind(past, frame),
            players=players,
            pair_shots=_by_pair(lambda slot: by_slot[slot].shots, sum),
            pair_net=_by_pair(lambda slot: by_slot[slot].net_share, _mean_share),
            last_shot_speed=last_speed,
            top_shot_speed=top_speed,
            positions={slot: self._trail(slot, index) for slot in SLOTS},
            pair_points=self._pair_points(over),
            **self._rally_at(frame),
        )

    def _recent_kind(self, past: list, frame: int) -> str | None:
        """What the last contact was, if it is recent enough to still be shown."""
        if past and frame - past[-1].frame <= self.recent:
            return past[-1].kind
        return None

    def _shot_speeds(self, frame: int) -> tuple[float | None, float | None]:
        """The speed of the last shot whose arrival is known by `frame`, and the fastest."""
        known = [s.kmh for s in self._speeds if s.stop <= frame]
        return (known[-1], max(known)) if known else (None, None)

    def _player_line(self, slot: str, index: int, strikes: list, over: list) -> PlayerLine:
        """One player's figures up to the frame at `index`."""
        mine = [c for c in strikes if c.player == slot]
        volley = Counter(self._volley.get(c.frame) for c in mine)
        ended = Counter(p.kind for p in over if p.player == slot)
        seen = int(self._seen[slot][index])
        return PlayerLine(
            slot=slot,
            shots=len(mine),
            volleys=volley[True],
            after_bounce=volley[False],
            distance=float(self._distance[slot][index]),
            top_speed=float(self._top[slot][index]),
            net_share=float(self._at_net[slot][index]) / seen if seen else math.nan,
            winners=ended["gagnant"],
            errors=ended["faute"],
        )

    def _trail(self, slot: str, index: int) -> list[tuple[float, float]]:
        """Where the player was over the last frames, the most recent last."""
        recent = self._smooth[slot][max(0, index - TRAIL):index + 1]
        return [(float(x), float(y)) for x, y in recent if not math.isnan(x)]

    def _pair_points(self, over: list) -> dict[str, int] | None:
        """The points each pair has won so far, or None when the score was not read."""
        if self._points is None:
            return None
        won = Counter(p.winner_side for p in over)
        return {"proche": won["near"], "fond": won["far"]}

    def _rally_at(self, frame: int) -> dict:
        begun = [span for span in self._spans if span.start <= frame]
        if not begun:
            return {}
        counts = [sum(1 for f, kind in span.contacts if kind == RACKET and f <= frame)
                  for span in begun]
        striker = {c.frame: c.player for c in self.rally.contacts if c.kind == RACKET}
        current = [striker.get(f) for f, kind in begun[-1].contacts
                   if kind == RACKET and f <= frame]
        return {"rally_number": len(begun), "rally_shots": counts[-1],
                "longest_rally": max(counts),
                "rally_pair_shots": {pair: sum(1 for slot in current if slot in slots)
                                     for pair, slots in PAIRS.items()}}
