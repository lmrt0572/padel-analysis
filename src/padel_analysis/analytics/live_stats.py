"""Game and player statistics as they stand at each frame, for the video's side panel.

The video shows the match as it unfolds, so the numbers beside it must not know the
future: every figure here is counted from the start of the clip up to the frame being
drawn. The work is done once, frame by frame, so drawing a frame only reads a row.
"""

import math
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
TRAIL = 45  # images de trace derriere chaque joueur sur la minimap, 1,5 s
SUSTAINED = 31  # images : la vitesse max est tenue pendant une seconde


@dataclass(frozen=True)
class PlayerLine:
    slot: str
    shots: int
    volleys: int  # frappes sans rebond au sol depuis la frappe precedente
    after_bounce: int
    distance: float  # metres, lisses
    top_speed: float  # km/h, la plus haute vitesse tenue pendant une seconde
    net_share: float  # part du temps passe au filet, sur les images ou il est vu


@dataclass(frozen=True)
class LiveStats:
    """What the panel shows at one frame."""

    elapsed: float
    shots: int
    walls: int
    last: str | None  # le dernier contact, s'il date de moins d'une seconde
    players: tuple[PlayerLine, ...]
    pair_shots: dict[str, int]
    pair_net: dict[str, float]
    last_shot_speed: float | None  # km/h, du dernier coup dont on connait l'arrivee
    top_shot_speed: float | None
    positions: dict[str, list[tuple[float, float]]]  # la trace recente, la plus recente a la fin
    rally_number: int | None = None  # l'echange en cours, compte depuis le debut de l'extrait
    rally_shots: int = 0  # frappes de l'echange en cours jusqu'ici
    longest_rally: int = 0  # le plus d'echanges de frappes en un echange, jusqu'ici


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


class LiveTimeline:
    """Cumulative statistics for every frame of a rally, computed once."""

    def __init__(self, rally: Rally, recent: float = 1.0, splices=()) -> None:
        """Args:
            splices: where the broadcast splices, each opening a new rally.
        """
        self.rally = rally
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
            # La vitesse max est une vitesse tenue une seconde. Plus court, deux artefacts
            # passent pour des sprints : un petit saut leve les chevilles dans l'image, et
            # le point au sol recule d'un metre au fond du court ; un echange d'identite
            # entre partenaires deplace la position de plusieurs metres. Mesure sur deux
            # echanges : 21 et 50 km/h avec une mediane sur 5 images, 15 et 19 sur 31.
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
        past = [c for c in rally.contacts if c.frame <= frame]
        strikes = [c for c in past if c.kind == RACKET]
        players = []
        for slot in SLOTS:
            mine = [c for c in strikes if c.player == slot]
            seen = int(self._seen[slot][index])
            players.append(PlayerLine(
                slot=slot,
                shots=len(mine),
                volleys=sum(1 for c in mine if self._volley.get(c.frame) is True),
                after_bounce=sum(1 for c in mine if self._volley.get(c.frame) is False),
                distance=float(self._distance[slot][index]),
                top_speed=float(self._top[slot][index]),
                net_share=float(self._at_net[slot][index]) / seen if seen else math.nan,
            ))
        by_slot = {line.slot: line for line in players}
        last = past[-1] if past and frame - past[-1].frame <= self.recent else None
        known = [s.kmh for s in self._speeds if s.stop <= frame]
        positions = {}
        for slot in SLOTS:
            recent = self._smooth[slot][max(0, index - TRAIL):index + 1]
            positions[slot] = [(float(x), float(y)) for x, y in recent if not math.isnan(x)]
        return LiveStats(
            elapsed=index / rally.fps,
            shots=len(strikes),
            walls=sum(1 for c in past if c.kind in WALLS),
            last=last.kind if last else None,
            players=tuple(players),
            pair_shots={pair: sum(by_slot[s].shots for s in slots)
                        for pair, slots in PAIRS.items()},
            pair_net={
                pair: float(np.nanmean([by_slot[s].net_share for s in slots]))
                if any(not math.isnan(by_slot[s].net_share) for s in slots) else math.nan
                for pair, slots in PAIRS.items()
            },
            last_shot_speed=known[-1] if known else None,
            top_shot_speed=max(known) if known else None,
            positions=positions,
            **self._rally_at(frame),
        )

    def _rally_at(self, frame: int) -> dict:
        begun = [span for span in self._spans if span.start <= frame]
        if not begun:
            return {}
        counts = [sum(1 for f, kind in span.contacts if kind == RACKET and f <= frame)
                  for span in begun]
        return {"rally_number": len(begun), "rally_shots": counts[-1],
                "longest_rally": max(counts)}
