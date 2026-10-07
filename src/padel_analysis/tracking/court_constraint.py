"""Assign observations to four fixed player slots, two per side of the net.

Players never cross the net during a game, so identity is an assignment problem under
a hard constraint; the side comes from the sign of the projected `y`. At a broadcast
splice the players reappear elsewhere, and the velocity it produces is dropped.
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
    """Tracker keeping exactly four identities, two on each side of the net."""

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
    ) -> None:
        """Set the tracker up.

        Args:
            max_x, max_y: half-extents beyond which an observation is refused, in
                metres; players do leave through the side openings.
            court_half_width, court_half_length, margin: the court, and the slack
                given to the ground point's projection error.
            off_court_penalty: metres-equivalent added to any observation off the
                court, larger than the court is long.
            cut_jump: metres a player would have to cover in one frame for it to
                count as a broadcast cut.
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

    def cost(self, slot: Slot, observation: CourtObservation) -> float:
        """Return the cost of assigning `observation` to `slot`, in metres-equivalent."""
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
            if predicted is None
            else float(np.linalg.norm(observation.court_xy - predicted))
        )

        appearance = 0.0
        if slot.appearance is not None:
            appearance = float(np.linalg.norm(slot.appearance - observation.appearance))

        doubt = self._confidence_weight * (1.0 - observation.confidence)
        penalty = self._off_court_penalty if off_court else 0.0
        return distance + self._appearance_weight * appearance + doubt + penalty

    def update(self, observations: list[CourtObservation]) -> dict[str, int]:
        """Assign observations to slots and return {slot name: observation index}."""
        if not observations:
            for slot in self.slots:
                slot.coast()
            return {}

        matrix = np.array(
            [[self.cost(slot, obs) for obs in observations] for slot in self.slots]
        )
        rows, cols = linear_sum_assignment(matrix)
        cut = self._is_cut(rows, cols, matrix, observations)

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
        if cut:
            # the velocity measured across a cut is that of a teleportation
            for slot in self.slots:
                slot.velocity = np.zeros(2)
        return assignment

    def _is_cut(self, rows, cols, matrix, observations) -> bool:
        """Return whether several players jumped further than anyone runs in one frame."""
        jumps = tracked = 0
        for row, col in zip(rows, cols):
            predicted = self.slots[row].predict()
            if predicted is None or matrix[row, col] >= IMPOSSIBLE:
                continue
            tracked += 1
            jumps += np.linalg.norm(observations[col].court_xy - predicted) > self._cut_jump
        return tracked >= 2 and jumps >= 2
