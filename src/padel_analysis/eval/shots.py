"""Reader for the PadelTracker100 shot annotations.

The file marks an interval rather than an instant: a shot spans sixteen frames in
the women's final, eleven in the men's. It says "a shot happens in this half
second", never "the impact is on this frame" - so the accuracy of a detected
contact in time cannot be scored against it, only its membership of the interval.

The six categories name strokes, not surfaces: Serve, Forehand, Backhand, Smash,
Dropshot, Other.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

_FRAME_NUMBER = re.compile(r"(\d+)")


@dataclass(frozen=True)
class ShotEvent:
    """One annotated shot, as the interval the annotation actually gives."""

    start_frame: int
    end_frame: int
    category: str

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame


class ShotEvents:
    """Shot intervals of one match."""

    def __init__(self, events: list[ShotEvent], first: int, last: int) -> None:
        self._events = events
        self._first = first
        self._last = last

    @classmethod
    def load(cls, path: Path) -> "ShotEvents":
        marked: list[tuple[int, str]] = []
        frames: list[int] = []

        with Path(path).open(encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle, delimiter=";"):
                if not row or row[0] == "file_name":
                    continue
                match = _FRAME_NUMBER.search(row[0])
                if match is None:
                    raise ValueError(f"could not read a frame index from {row[0]!r}")
                frame = int(match.group(1))
                frames.append(frame)
                if int(row[1]):
                    marked.append((frame, row[2]))

        events: list[ShotEvent] = []
        for frame, category in marked:
            joins = (
                events
                and events[-1].category == category
                and events[-1].end_frame == frame - 1
            )
            if joins:
                events[-1] = ShotEvent(events[-1].start_frame, frame, category)
            else:
                events.append(ShotEvent(frame, frame, category))

        return cls(events, min(frames), max(frames))

    def covered_range(self) -> tuple[int, int]:
        """First and last frame the file says anything about."""
        return self._first, self._last

    def events(self, start: int = 0, stop: int | None = None) -> list[ShotEvent]:
        """Events whose interval starts within [start, stop]."""
        end = self._last if stop is None else stop
        return [e for e in self._events if start <= e.start_frame <= end]
