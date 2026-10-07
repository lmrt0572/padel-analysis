"""Identity ground truth, which PadelTracker100 does not provide.

The machine assigns identity by nearest neighbour across the match, and a human
arbitrates only the moments where two partners come close or the broadcast cuts.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..io.atomic import write_json_atomically

SLOTS: tuple[str, ...] = ("near_1", "near_2", "far_1", "far_2")
NEAR_SLOTS = ("near_1", "near_2")
FAR_SLOTS = ("far_1", "far_2")


@dataclass(frozen=True)
class AmbiguousEpisode:
    """A stretch where two partners were close enough for identity to be in doubt."""

    start_frame: int
    end_frame: int
    slots: tuple[str, str]
    min_separation_m: float


@dataclass(frozen=True)
class CameraCut:
    """A shot change, and which pairs it puts at risk."""

    frame: int  # first frame of the new shot
    sides: tuple[str, ...]  # ("near",), ("far",) or both
    displacement_m: float  # smallest jump among the players at risk


@dataclass
class IdentityGroundTruth:
    """Per-frame slot assignments, plus the episodes a human still has to settle."""

    assignments: dict[int, dict[str, int]]
    episodes: list[AmbiguousEpisode]
    resolved: list[int] = field(default_factory=list)
    cuts: list[CameraCut] = field(default_factory=list)
    resolved_cuts: list[int] = field(default_factory=list)
    # frames where the teams changed ends: identity stops there and starts again
    boundaries: list[int] = field(default_factory=list)
    # what was answered at each clip, under "episode:<frame>" or "cut:<frame>",
    # so that an answer can be undone
    decisions: dict[str, str] = field(default_factory=dict)

    def apply_swap(self, from_frame: int, slots: tuple[str, str]) -> None:
        """Exchange two slots from `from_frame` onward."""
        first, second = slots
        for frame in sorted(self.assignments):
            if frame < from_frame:
                continue
            row = self.assignments[frame]
            if first in row and second in row:
                row[first], row[second] = row[second], row[first]

    def segment_of(self, frame: int) -> int:
        """Return the index of the stretch `frame` belongs to, counting from zero.

        Identity holds inside a stretch and makes no claim across one.
        """
        return sum(1 for boundary in self.boundaries if frame >= boundary)

    def save(self, path: Path) -> None:
        payload = {
            "assignments": {
                str(frame): row for frame, row in sorted(self.assignments.items())
            },
            "episodes": [
                {
                    "start_frame": e.start_frame,
                    "end_frame": e.end_frame,
                    "slots": list(e.slots),
                    "min_separation_m": e.min_separation_m,
                }
                for e in self.episodes
            ],
            "resolved": self.resolved,
            "cuts": [
                {
                    "frame": c.frame,
                    "sides": list(c.sides),
                    "displacement_m": c.displacement_m,
                }
                for c in self.cuts
            ],
            "resolved_cuts": self.resolved_cuts,
            "boundaries": self.boundaries,
            "decisions": self.decisions,
        }
        write_json_atomically(path, payload)

    @classmethod
    def load(cls, path: Path) -> "IdentityGroundTruth":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            assignments={
                int(frame): {k: int(v) for k, v in row.items()}
                for frame, row in payload["assignments"].items()
            },
            episodes=[
                AmbiguousEpisode(
                    start_frame=int(e["start_frame"]),
                    end_frame=int(e["end_frame"]),
                    slots=(e["slots"][0], e["slots"][1]),
                    min_separation_m=float(e["min_separation_m"]),
                )
                for e in payload["episodes"]
            ],
            resolved=[int(f) for f in payload["resolved"]],
            # fields added after the first ground truths: older files stay readable
            cuts=[
                CameraCut(
                    frame=int(c["frame"]),
                    sides=tuple(c["sides"]),
                    displacement_m=float(c["displacement_m"]),
                )
                for c in payload.get("cuts", [])
            ],
            resolved_cuts=[int(f) for f in payload.get("resolved_cuts", [])],
            boundaries=[int(f) for f in payload.get("boundaries", [])],
            decisions=dict(payload.get("decisions", {})),
        )


def _side_groups(positions: np.ndarray) -> tuple[list[int], list[int]]:
    near = [i for i, p in enumerate(positions) if p[1] < 0]
    far = [i for i, p in enumerate(positions) if p[1] >= 0]
    return near, far


def assign_by_proximity(
    positions_by_frame: dict[int, np.ndarray],
) -> dict[int, dict[str, int]]:
    """Assign the four slots frame by frame, following the nearest previous position.

    Frames that do not hold exactly two people per side are skipped.
    """
    assignments: dict[int, dict[str, int]] = {}
    previous: dict[str, np.ndarray] = {}

    for frame in sorted(positions_by_frame):
        positions = np.asarray(positions_by_frame[frame], dtype=np.float64)
        near, far = _side_groups(positions)
        if len(near) != 2 or len(far) != 2:
            continue

        row: dict[str, int] = {}
        for slots, indices in ((NEAR_SLOTS, near), (FAR_SLOTS, far)):
            known = [s for s in slots if s in previous]
            if len(known) < 2:
                # first usable frame: arbitrary but stable order, by x
                ordered = sorted(indices, key=lambda i: positions[i][0])
                for slot, index in zip(slots, ordered):
                    row[slot] = index
            else:
                cost = np.array(
                    [
                        [
                            float(np.linalg.norm(positions[i] - previous[s]))
                            for i in indices
                        ]
                        for s in slots
                    ]
                )
                fitted_rows, fitted_cols = linear_sum_assignment(cost)
                for r, c in zip(fitted_rows, fitted_cols):
                    row[slots[r]] = indices[c]

        assignments[frame] = row
        for slot, index in row.items():
            previous[slot] = positions[index]
    return assignments


def find_ambiguous_episodes(
    positions_by_frame: dict[int, np.ndarray], threshold: float = 1.5
) -> list[AmbiguousEpisode]:
    """Return the stretches where two partners came within `threshold` metres of each other."""
    episodes: list[AmbiguousEpisode] = []
    open_start: dict[tuple[str, str], int] = {}
    open_min: dict[tuple[str, str], float] = {}
    last_frame: dict[tuple[str, str], int] = {}

    for frame in sorted(positions_by_frame):
        positions = np.asarray(positions_by_frame[frame], dtype=np.float64)
        near, far = _side_groups(positions)

        for key, indices in ((NEAR_SLOTS, near), (FAR_SLOTS, far)):
            if len(indices) != 2:
                continue
            separation = float(
                np.linalg.norm(positions[indices[0]] - positions[indices[1]])
            )
            if separation < threshold:
                if key not in open_start:
                    open_start[key] = frame
                    open_min[key] = separation
                open_min[key] = min(open_min[key], separation)
                last_frame[key] = frame
            elif key in open_start:
                episodes.append(
                    AmbiguousEpisode(
                        start_frame=open_start.pop(key),
                        end_frame=last_frame[key],
                        slots=key,
                        min_separation_m=open_min.pop(key),
                    )
                )

    for key, start in open_start.items():
        episodes.append(
            AmbiguousEpisode(
                start_frame=start,
                end_frame=last_frame[key],
                slots=key,
                min_separation_m=open_min[key],
            )
        )
    return sorted(episodes, key=lambda e: e.start_frame)
