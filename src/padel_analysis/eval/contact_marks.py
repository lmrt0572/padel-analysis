"""Every real contact of a stretch of match, marked by hand, and what it hit.

Judging the contacts a pipeline proposes measures its precision and nothing else: a
contact it never proposed is never judged, so a missed wall costs nothing. Marks made
while watching the whole stretch, without seeing any detection, measure both.

Written atomically after every mark, like every hand-made truth in this project.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..io.atomic import write_json_atomically

ANSWERS = ("sol", "verre", "grillage", "filet", "raquette")


@dataclass
class ContactMarks:
    """The contacts of one stretch, keyed by frame, and where the annotator stopped."""

    video: str
    frame_range: tuple[int, int]
    marks: dict[int, str] = field(default_factory=dict)
    position: int = 0
    order: list[int] = field(default_factory=list)

    def mark(self, frame: int, answer: str, merge: int = 0) -> None:
        """Record a contact; a mark within `merge` frames of another replaces it."""
        if answer not in ANSWERS:
            raise ValueError(f"{answer!r} is not one of {ANSWERS}")
        for existing in [f for f in self.marks if abs(f - frame) <= merge]:
            del self.marks[existing]
            self.order.remove(existing)
        self.marks[frame] = answer
        self.order.append(frame)

    def undo(self) -> tuple[int, str] | None:
        """Forget the most recent mark, and return it."""
        if not self.order:
            return None
        frame = self.order.pop()
        return frame, self.marks.pop(frame)

    def save(self, path: Path) -> None:
        payload = {
            "video": self.video,
            "frame_range": list(self.frame_range),
            "position": self.position,
            "marks": {str(f): a for f, a in sorted(self.marks.items())},
            "order": self.order,
        }
        write_json_atomically(path, payload, indent=1)

    @classmethod
    def load(cls, path: Path) -> "ContactMarks":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            video=payload["video"],
            frame_range=tuple(payload["frame_range"]),
            marks={int(f): a for f, a in payload["marks"].items()},
            position=payload.get("position", payload["frame_range"][0]),
            order=list(payload.get("order", [])),
        )


@dataclass(frozen=True)
class ContactMatch:
    """How detected contacts line up with the marked ones."""

    found: int
    missed: int
    invented: int
    right_surface: int

    @property
    def recall(self) -> float:
        return self.found / max(self.found + self.missed, 1)

    @property
    def precision(self) -> float:
        return self.found / max(self.found + self.invented, 1)


RIGHT, WRONG_SURFACE, INVENTED, MISSED = "juste", "surface fausse", "invente", "manque"


@dataclass(frozen=True)
class Pairing:
    """One line of the comparison: a detection, a mark, or both paired."""

    detected_frame: int | None
    marked_frame: int | None
    detected: str | None
    marked: str | None

    @property
    def status(self) -> str:
        if self.marked_frame is None:
            return INVENTED
        if self.detected_frame is None:
            return MISSED
        return RIGHT if self.detected == self.marked else WRONG_SURFACE


def pair_contacts(
    detected: dict[int, str], marks: dict[int, str], tolerance: int = 3
) -> list[Pairing]:
    """Pair each detection with the nearest free mark within `tolerance` frames.

    A mark is claimed at most once, so two detections of one bounce count as one
    found contact and one invented - the second is a false contact on the screen.
    Unclaimed marks come last, as missed contacts.
    """
    free = dict(marks)
    lines = []
    for frame, answer in sorted(detected.items()):
        near = [f for f in free if abs(f - frame) <= tolerance]
        if not near:
            lines.append(Pairing(frame, None, answer, None))
            continue
        nearest = min(near, key=lambda f: abs(f - frame))
        lines.append(Pairing(frame, nearest, answer, free.pop(nearest)))
    lines += [Pairing(None, frame, None, answer) for frame, answer in sorted(free.items())]
    return lines


def match_contacts(
    detected: dict[int, str], marks: dict[int, str], tolerance: int = 3
) -> ContactMatch:
    """The counts of `pair_contacts`: found, missed, invented, and right surface."""
    statuses = [line.status for line in pair_contacts(detected, marks, tolerance)]
    return ContactMatch(
        found=statuses.count(RIGHT) + statuses.count(WRONG_SURFACE),
        missed=statuses.count(MISSED),
        invented=statuses.count(INVENTED),
        right_surface=statuses.count(RIGHT),
    )
