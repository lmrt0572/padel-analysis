"""The surface ground truth: what a human said each contact hit.

No public padel dataset labels contact surfaces - PadelTracker100 declares a `Wall`
category and never filled it - so this file is the measurement, not a convenience. It
is written atomically after every single answer, because a campaign on this project
has already been destroyed by a power cut leaving a half-written file.

The file deliberately carries no prediction. A ground truth built on the same
assumption as the thing it judges measures two errors agreeing, not accuracy: on
sub-project A, removing that flaw moved IDF1 from 0.956 to 0.819, and the flattering
number was the artefact.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

UNREADABLE = "x"
NO_CONTACT = "aucun"
ANSWERS = ("sol", "verre", "grillage", "raquette", NO_CONTACT, UNREADABLE)

NOT_A_SURFACE = (NO_CONTACT, UNREADABLE)
"""Answers that name no surface, and are therefore excluded from the surface rates.

They are excluded for opposite reasons and must never be merged. `x` is a
non-measurement: the annotator could not tell, and scoring it either way would
assert what the data does not support. `aucun` is a measurement: the trajectory ran
straight through, so the contact stage invented this event. That one is a false
positive of stage B.3, and the only way this project can measure its precision -
the shot annotation being too coarse to do it."""

_CLASS_OF = {
    "sol": "sol",
    "verre": "mur",
    "grillage": "mur",
    "raquette": "raquette",
}


def class_of(answer: str) -> str | None:
    """The three-way class an answer belongs to, or None if it names no surface.

    Glass and mesh are both walls. They are asked apart because the eye can tell
    them apart, which is what makes the geometric deduction testable rather than
    merely assumed.
    """
    return _CLASS_OF.get(answer)


@dataclass(frozen=True)
class SurfaceTask:
    """One contact submitted to a human, and how sure the geometry was about it.

    The stratum says whether the rule hesitated, never what it concluded. It is
    recorded so results can be split by difficulty, and it is never shown.
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
        """Tasks still waiting for an answer, in the order they were listed."""
        return [t for t in self.tasks if t.frame not in self.answers]

    def answer(self, frame: int, value: str) -> None:
        if value not in ANSWERS:
            raise ValueError(f"{value!r} is not one of {ANSWERS}")
        if all(t.frame != frame for t in self.tasks):
            raise KeyError(f"frame {frame} is not a task of this campaign")
        self.answers[frame] = value

    def undo(self, frame: int) -> str | None:
        """Forget the answer given for that frame, and return what it was."""
        return self.answers.pop(frame, None)

    def save(self, path: Path) -> None:
        """Write atomically: a crash costs the answer in progress, not the campaign."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video": self.video,
            "frame_range": list(self.frame_range),
            "parameters": self.parameters,
            "tasks": [{"frame": t.frame, "stratum": t.stratum} for t in self.tasks],
            "answers": {str(f): a for f, a in sorted(self.answers.items())},
        }
        descriptor, temporary = tempfile.mkstemp(
            dir=target.parent, prefix=f"{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=1)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

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
