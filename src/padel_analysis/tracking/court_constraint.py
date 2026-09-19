"""Assign observations to four fixed player slots, two per side of the net.

Padel is always played as two pairs, and players never cross the net during a game.
That turns identity into an assignment problem with a hard structural constraint,
which a generic tracker cannot exploit: partners cross constantly, and a purely
appearance- or motion-based tracker swaps their identifiers when they do.

The side of the net comes from the sign of the projected `y` coordinate, so the
calibration of the previous milestone feeds the tracker directly.

Partners wear the same kit, so colour cannot tell them apart, and at a broadcast cut
the players reappear elsewhere, so motion cannot either: measured on the women's final,
every identity switch left happened at a cut. What does hold is padel's own habit -
each partner keeps a side, the drive player on the right and the backhand player on
the left: 94.8 % of the time in the half second after a cut. Each slot learns its side
as play goes, and a cut is resolved by it.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

SLOT_NAMES: tuple[str, ...] = ("near_1", "near_2", "far_1", "far_2")
IMPOSSIBLE = 1e6


@dataclass(frozen=True)
class CourtObservation:
    """One detected player, already projected onto the court."""

    court_xy: np.ndarray  # (2,) in metres
    confidence: float
    appearance: np.ndarray  # colour histogram of the torso


@dataclass
class Slot:
    """One of the four player identities, with constant-velocity motion."""

    name: str
    side: int  # +1 for positive y, -1 for negative y
    position: np.ndarray | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    appearance: np.ndarray | None = None
    missing_frames: int = 0
    lateral: float = 0.0
    """Which side of the partner this slot keeps: towards +1 when it plays to the
    partner's right (larger x), towards -1 to the left, 0 while unknown."""

    def predict(self) -> np.ndarray | None:
        if self.position is None:
            return None
        return self.position + self.velocity

    def observe(self, observation: CourtObservation, smoothing: float = 0.5) -> None:
        if self.position is not None:
            measured_velocity = observation.court_xy - self.position
            self.velocity = (
                1 - smoothing
            ) * self.velocity + smoothing * measured_velocity
        self.position = observation.court_xy.copy()
        if self.appearance is None:
            self.appearance = observation.appearance.copy()
        else:
            self.appearance = 0.9 * self.appearance + 0.1 * observation.appearance
        self.missing_frames = 0

    def coast(self) -> None:
        """Carry on along the last known velocity while the player is not detected."""
        if self.position is not None:
            self.position = self.position + self.velocity
        self.missing_frames += 1


class CourtSlotTracker:
    """Keeps exactly four identities, two on each side of the net."""

    def __init__(
        self,
        appearance_weight: float = 1.0,
        confidence_weight: float = 2.0,
        max_missing_frames: int = 30,
        max_x: float = 9.0,
        max_y: float = 14.0,
        court_half_width: float = 5.0,
        court_half_length: float = 10.0,
        margin: float = 0.8,
        off_court_penalty: float = 20.0,
        cut_jump: float = 1.0,
        side_gap: float = 0.8,
        side_penalty: float = 5.0,
        side_memory: float = 0.05,
        cut_window: int = 15,
    ) -> None:
        """Args:
            max_x, max_y: half-extents beyond which an observation is refused, in
                metres. The court is 10 by 20, so these allow four metres of overrun
                on each side - padel players do leave through the side openings to
                return a lob - while refusing spectators in the stands. Without this
                bound, roughly one position in a hundred landed several metres past
                the glass.
            court_half_width, court_half_length, margin: the court, and the slack given
                to the ground point's projection error - 56 cm at the far baseline.
                Behind a back wall and within the court's width, nobody can be playing:
                the glass is in the way, and the only way out is the side openings.
                Seen on the women's final: a slot held for twenty seconds a person
                sitting 1.7 m behind the far glass, while the real player went untracked.
            off_court_penalty: metres-equivalent added to any observation off the
                court. A player may still leave through a side opening, but someone on
                the court is always preferred to someone beside it - an umpire or a
                ball boy sits there and never moves, so a slot that took them would
                keep them. Larger than the court is long, so that no distance to a
                player on the court can outweigh it.
            cut_jump: metres a player would have to cover in one frame, 30 m/s. When
                two players jump that far at once, the broadcast has cut to another
                shot, and the last positions no longer predict anything.
            side_gap: metres across the court two partners must be apart for their
                left-right order to mean anything - one at the net and one at the back
                can stand in line.
            side_penalty: metres-equivalent for putting a partner on the side it does
                not usually keep, at a cut.
            side_memory: how fast a slot's usual side follows play, per frame.
            cut_window: frames after a cut during which a half of the court stays
                undecided until both partners are seen - one of them often appears a
                few frames after the other, and settling on the first alone would be
                a coin toss.
        """
        self.slots = [
            Slot(name="near_1", side=-1),
            Slot(name="near_2", side=-1),
            Slot(name="far_1", side=+1),
            Slot(name="far_2", side=+1),
        ]
        self._appearance_weight = appearance_weight
        self._confidence_weight = confidence_weight
        self._max_missing_frames = max_missing_frames
        self._max_x = max_x
        self._max_y = max_y
        self._half_width = court_half_width
        self._half_length = court_half_length
        self._margin = margin
        self._off_court_penalty = off_court_penalty
        self._cut_jump = cut_jump
        self._side_penalty = side_penalty
        self._side_memory = side_memory
        self._cut_window = cut_window
        self._side_gap = side_gap
        self._since_cut = cut_window
        self._undecided: set[int] = set()

    def cost(self, slot: Slot, observation: CourtObservation, cut: bool = False,
             rank: int | None = None, sides: bool = False) -> float:
        """Cost of assigning `observation` to `slot`, in metres-equivalent.

        Args:
            cut: the broadcast has just cut: the last positions predict nothing.
            sides: this half of the court is still undecided since a cut: the
                partners' usual left-right order weighs in.
            rank: where the observation stands among those on its half of the court,
                counted from the right (larger x) - 0 for the rightmost. None when it
                stands alone there: a lone player's side of their partner is unknown.
        """
        x, y = float(observation.court_xy[0]), float(observation.court_xy[1])
        if abs(x) > self._max_x or abs(y) > self._max_y:
            return IMPOSSIBLE
        within_width = abs(x) <= self._half_width + self._margin
        behind_back_wall = abs(y) > self._half_length + self._margin
        if within_width and behind_back_wall:
            return IMPOSSIBLE
        off_court = not within_width or behind_back_wall

        observed_side = 1 if y >= 0 else -1
        if observed_side != slot.side:
            return IMPOSSIBLE

        predicted = slot.predict()
        distance = (
            0.0
            if predicted is None or cut
            else float(np.linalg.norm(observation.court_xy - predicted))
        )
        if sides and rank is not None and abs(slot.lateral) > 0.3:
            # A droite du partenaire d'habitude : la place la plus a droite du demi-court.
            expected = 0 if slot.lateral > 0 else 1
            distance += self._side_penalty * min(abs(rank - expected), 1)

        appearance = 0.0
        if slot.appearance is not None:
            appearance = float(np.linalg.norm(slot.appearance - observation.appearance))

        doubt = self._confidence_weight * (1.0 - observation.confidence)
        penalty = self._off_court_penalty if off_court else 0.0
        return distance + self._appearance_weight * appearance + doubt + penalty

    def update(self, observations: list[CourtObservation]) -> dict[str, int]:
        """Assign observations to slots. Returns {slot name: observation index}."""
        if not observations:
            for slot in self.slots:
                slot.coast()
            return {}

        matrix = np.array(
            [[self.cost(slot, obs) for obs in observations] for slot in self.slots]
        )
        rows, cols = linear_sum_assignment(matrix)
        cut = self._is_cut(rows, cols, matrix, observations)
        if cut:
            self._since_cut, self._undecided = 0, {-1, +1}
            for slot in self.slots:
                slot.velocity = np.zeros(2)
        else:
            self._since_cut += 1
        if self._since_cut >= self._cut_window:
            self._undecided = set()
        ranks = self._ranks(observations)
        if self._undecided:
            matrix = np.array([
                [self.cost(slot, obs, cut=cut, rank=ranks[j],
                           sides=slot.side in self._undecided)
                 for j, obs in enumerate(observations)]
                for slot in self.slots
            ])
            rows, cols = linear_sum_assignment(matrix)

        assignment: dict[str, int] = {}
        assigned_slots: set[str] = set()
        for row, col in zip(rows, cols):
            if matrix[row, col] >= IMPOSSIBLE:
                continue
            slot = self.slots[row]
            slot.observe(observations[col])
            assignment[slot.name] = int(col)
            assigned_slots.add(slot.name)

        for slot in self.slots:
            if slot.name not in assigned_slots:
                slot.coast()
                if slot.missing_frames > self._max_missing_frames:
                    slot.position = None
                    slot.velocity = np.zeros(2)
        for side in list(self._undecided):
            pair = [slot.name for slot in self.slots if slot.side == side]
            if all(name in assignment and ranks[assignment[name]] is not None for name in pair):
                self._undecided.discard(side)
        self._learn_sides(assignment, observations)
        return assignment

    def _is_cut(self, rows, cols, matrix, observations) -> bool:
        """Whether most players jumped further than anyone runs in one frame."""
        jumps = tracked = 0
        for row, col in zip(rows, cols):
            predicted = self.slots[row].predict()
            if predicted is None or matrix[row, col] >= IMPOSSIBLE:
                continue
            tracked += 1
            jumps += np.linalg.norm(observations[col].court_xy - predicted) > self._cut_jump
        return tracked >= 2 and jumps >= 2

    def _ranks(self, observations) -> list[int | None]:
        """Each observation's place from the right among the on-court ones of its half."""
        ranks = []
        for obs in observations:
            x, y = float(obs.court_xy[0]), float(obs.court_xy[1])
            same_half = [
                float(o.court_xy[0]) for o in observations
                if np.sign(float(o.court_xy[1]) or 1.0) == np.sign(y or 1.0)
                and abs(float(o.court_xy[0])) <= self._half_width + self._margin
                and abs(float(o.court_xy[1])) <= self._half_length + self._margin
            ]
            others = [other for other in same_half if other != x]
            unclear = not others or min(abs(other - x) for other in others) < self._side_gap
            ranks.append(None if unclear else sum(1 for other in same_half if other > x))
        return ranks

    def _learn_sides(self, assignment: dict[str, int], observations) -> None:
        """Update each slot's usual side from where the partners stand now."""
        by_name = {slot.name: slot for slot in self.slots}
        for first, second in (("near_1", "near_2"), ("far_1", "far_2")):
            if first not in assignment or second not in assignment:
                continue
            gap = float(observations[assignment[first]].court_xy[0]
                        - observations[assignment[second]].court_xy[0])
            if abs(gap) < 0.5:
                continue
            memory = self._side_memory
            for name, sign in ((first, np.sign(gap)), (second, -np.sign(gap))):
                slot = by_name[name]
                slot.lateral = (1 - memory) * slot.lateral + memory * sign
