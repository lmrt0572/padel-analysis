"""Identity ground truth, which PadelTracker100 does not provide.

The annotations give four people per frame but never say which is which. Rebuilding
that is mostly free: over the whole match, two partners never come within half a
metre of each other, and only fourteen episodes bring them within a metre and a
half. Everywhere else, nearest-neighbour association is unambiguous.

So the machine assigns identity across the match, and a human arbitrates only those
fourteen moments.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

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

    def apply_swap(self, from_frame: int, slots: tuple[str, str]) -> None:
        """Exchange two slots from `from_frame` onward.

        Used when a human decides that two partners did cross, and the automatic
        association followed the wrong one afterwards.
        """
        first, second = slots
        for frame in sorted(self.assignments):
            if frame < from_frame:
                continue
            row = self.assignments[frame]
            if first in row and second in row:
                row[first], row[second] = row[second], row[first]

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
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Ecriture atomique. Le fichier est reecrit apres chaque episode arbitre, et
        # une ecriture directe le tronque avant de le remplir : une coupure a cet
        # instant laissait un fichier de zeros, effacant les quarante-cinq mille
        # frames d'assignation et tous les arbitrages deja rendus. Le nouveau
        # contenu n'est publie qu'une fois complet et sur le disque.
        descriptor, temporary = tempfile.mkstemp(
            dir=target.parent, prefix=f"{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

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
            # Les deux champs suivants sont apparus apres les premieres verites
            # terrain : un fichier qui les ignore reste lisible.
            cuts=[
                CameraCut(
                    frame=int(c["frame"]),
                    sides=tuple(c["sides"]),
                    displacement_m=float(c["displacement_m"]),
                )
                for c in payload.get("cuts", [])
            ],
            resolved_cuts=[int(f) for f in payload.get("resolved_cuts", [])],
        )


def _side_groups(positions: np.ndarray) -> tuple[list[int], list[int]]:
    near = [i for i, p in enumerate(positions) if p[1] < 0]
    far = [i for i, p in enumerate(positions) if p[1] >= 0]
    return near, far


def assign_by_proximity(
    positions_by_frame: dict[int, np.ndarray],
) -> dict[int, dict[str, int]]:
    """Assign the four slots frame by frame, following the nearest previous position.

    Frames that do not hold exactly two people per side are skipped rather than
    guessed at: a ground truth with an invented entry is worse than a shorter one.
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
                # Premiere frame utilisable : ordre arbitraire mais stable, par x.
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
    """Stretches where two partners came within `threshold` metres of each other.

    These, and only these, need a human decision: everywhere else the nearest
    previous position identifies a player without doubt.
    """
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
