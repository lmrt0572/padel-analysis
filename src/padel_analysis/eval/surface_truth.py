"""The surface ground truth: what a human said each contact hit.

Written atomically after every answer. The file carries no prediction, so the truth
does not share an assumption with what it judges.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..io.atomic import write_json_atomically

UNREADABLE = "x"
NO_CONTACT = "aucun"
ANSWERS = ("sol", "verre", "grillage", "filet", "raquette", NO_CONTACT, UNREADABLE)

NOT_A_SURFACE = (NO_CONTACT, UNREADABLE)
"""Answers that name no surface, excluded from the surface rates.

`x` is a non-measurement and `aucun` a false positive of the contact stage: they are
never merged."""

_CLASS_OF = {
    "sol": "sol",
    "verre": "mur",
    "grillage": "mur",
    "filet": "filet",
    "raquette": "raquette",
}


def class_of(answer: str) -> str | None:
    """Return the three-way class an answer belongs to, or None if it names no surface."""
    return _CLASS_OF.get(answer)


@dataclass(frozen=True)
class SurfaceTask:
    """One contact submitted to a human, and how sure the geometry was about it.

    The stratum says whether the rule hesitated, never what it concluded.
    """

    frame: int
    stratum: str


@dataclass
class SurfaceGroundTruth:
    """Every contact to judge on one match, and what has been answered so far."""

    video: str
    frame_range: tuple[int, int]
    parameters: dict[str, float]
    tasks: list[SurfaceTask]
    answers: dict[int, str] = field(default_factory=dict)

    def pending(self) -> list[SurfaceTask]:
        """Return the tasks still waiting for an answer, in listed order."""
        return [t for t in self.tasks if t.frame not in self.answers]

    def answer(self, frame: int, value: str) -> None:
        if value not in ANSWERS:
            raise ValueError(f"{value!r} is not one of {ANSWERS}")
        if all(t.frame != frame for t in self.tasks):
            raise KeyError(f"frame {frame} is not a task of this campaign")
        self.answers[frame] = value

    def undo(self, frame: int) -> str | None:
        """Forget the answer given for that frame and return it."""
        return self.answers.pop(frame, None)

    def save(self, path: Path) -> None:
        """Write the file atomically."""
        payload = {
            "video": self.video,
            "frame_range": list(self.frame_range),
            "parameters": self.parameters,
            "tasks": [{"frame": t.frame, "stratum": t.stratum} for t in self.tasks],
            "answers": {str(f): a for f, a in sorted(self.answers.items())},
        }
        write_json_atomically(path, payload, indent=1)

    @classmethod
    def load(cls, path: Path) -> "SurfaceGroundTruth":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            video=payload["video"],
            frame_range=tuple(payload["frame_range"]),
            parameters=payload.get("parameters", {}),
            tasks=[SurfaceTask(t["frame"], t["stratum"]) for t in payload["tasks"]],
            answers={int(f): a for f, a in payload.get("answers", {}).items()},
        )
