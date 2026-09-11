"""Assign observations to four fixed player slots, two per side of the net.

Padel is always played as two pairs, and players never cross the net during a game.
That turns identity into an assignment problem with a hard structural constraint,
which a generic tracker cannot exploit: partners cross constantly, and a purely
appearance- or motion-based tracker swaps their identifiers when they do.

The side of the net comes from the sign of the projected `y` coordinate, so the
calibration of the previous milestone feeds the tracker directly.
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
    """Keeps exactly four identities, two on each side of the net."""

    def __init__(
        self,
        appearance_weight: float = 1.0,
        confidence_weight: float = 2.0,
        max_missing_frames: int = 30,
    ) -> None:
        self.slots = [
            Slot(name="near_1", side=-1),
            Slot(name="near_2", side=-1),
            Slot(name="far_1", side=+1),
            Slot(name="far_2", side=+1),
        ]
        self._appearance_weight = appearance_weight
        self._confidence_weight = confidence_weight
        self._max_missing_frames = max_missing_frames

    def cost(self, slot: Slot, observation: CourtObservation) -> float:
        """Cost of assigning `observation` to `slot`, in metres-equivalent."""
        observed_side = 1 if observation.court_xy[1] >= 0 else -1
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
        return distance + self._appearance_weight * appearance + doubt

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
        return assignment
