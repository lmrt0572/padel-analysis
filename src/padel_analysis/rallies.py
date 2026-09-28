"""The rallies chosen for the statistics page, and how each becomes a `Rally`.

Rallies are picked by hand in a small file: which match, which analysed minute, and
the first and last frame. Player names are optional - without them each player keeps
a neutral name, which is always right, whereas a real name is only as right as whoever
typed it.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analytics.rally import Rally, RallyContact

DEFAULT_NAMES = {
    "near_1": "Proche 1",
    "near_2": "Proche 2",
    "far_1": "Fond 1",
    "far_2": "Fond 2",
}
ANSWER_OF_LABEL = {"RAQUETTE": "raquette", "SOL": "sol", "VITRE": "verre",
                   "GRILLAGE": "grillage", "FILET": "filet"}
ALTERNATION_SPAN = 150
"""Three strikes within five seconds belong to one exchange, where the halves alternate."""


@dataclass(frozen=True)
class RallySpec:
    """One rally as written in the rallies file."""

    id: str
    match: str
    minute: int  # debut de la minute analysee qui contient l'echange
    start: int
    stop: int
    title: str
    players: dict[str, str] = field(default_factory=dict)

    def name_of(self, slot: str) -> str:
        return self.players.get(slot, DEFAULT_NAMES.get(slot, slot))


def load_specs(path: str | Path) -> list[RallySpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        RallySpec(
            id=item["id"], match=item["match"], minute=int(item["minute"]),
            start=int(item["start"]), stop=int(item["stop"]), title=item["title"],
            players=dict(item.get("players", {})),
        )
        for item in payload["rallies"]
    ]


def striker_slot(
    box: np.ndarray | None, people: Sequence, assignment: dict, ball=None,
    side: str | None = None,
) -> str | None:
    """The slot of the player who struck.

    The player whose box was lit, when the tracker holds them. The lit box can belong to
    a detection the tracker left out - a player half hidden, or a figure beside the
    court - and a strike still has a striker: then it is the tracked player whose box
    is nearest the ball. With `side`, only a player of that half can be the striker.
    """
    slot_of_index = {index: slot for slot, index in assignment.items()
                     if side is None or slot.startswith(side)}
    if box is not None:
        for index, person in enumerate(people):
            if np.array_equal(person.bbox, box) and index in slot_of_index:
                return slot_of_index[index]
    if ball is None:
        return None
    nearest, best = None, float("inf")
    for index, slot in slot_of_index.items():
        x1, y1, x2, y2 = people[index].bbox[:4]
        gap = np.hypot(max(x1 - ball[0], 0.0, ball[0] - x2), max(y1 - ball[1], 0.0, ball[1] - y2))
        if gap < best:
            nearest, best = slot, gap
    return nearest


def strikers(frames: dict, strikes: Sequence[tuple[int, np.ndarray | None, tuple]],
             ) -> dict[int, str | None]:
    """The striker of each (frame, lit box, ball) strike, the halves made to alternate.

    Three strikes in a row from one half cannot happen within an exchange: the middle
    one went to the wrong half, most often a lob or a smash near the camera, which rises
    in the picture beside the far players. It is given to the nearest player of the
    other half. Measured on the eleven training minutes, against the alternation of the
    hand-marked strikes: 12 of 403 strikes on the wrong half before, 6 after.
    """
    slots = {}
    for frame, box, ball in strikes:
        data = frames.get(frame, {})
        slots[frame] = striker_slot(box, data.get("people", []), data.get("assignment", {}),
                                    ball)
    ordered = sorted(slots)
    half = {frame: None if slot is None else slot.split("_")[0] for frame, slot in slots.items()}
    balls = {frame: ball for frame, _, ball in strikes}
    corrected = dict(slots)
    for first, middle, last in zip(ordered, ordered[1:], ordered[2:]):
        if last - first > ALTERNATION_SPAN or half[middle] is None:
            continue
        if half[first] == half[middle] == half[last]:
            data = frames.get(middle, {})
            other = "near" if half[middle] == "far" else "far"
            flipped = striker_slot(None, data.get("people", []), data.get("assignment", {}),
                                   balls[middle], side=other)
            if flipped is not None:
                corrected[middle] = flipped
    return corrected


def build_rally(analysis: dict, events: Sequence, spec: RallySpec, fps: float) -> Rally:
    """The rally's contacts and player positions, from a saved analysis and its events."""
    frames = analysis["frames"]
    chosen = [e for e in events if spec.start <= e.frame <= spec.stop]
    players = strikers(frames, [(e.frame, e.box, e.pixel) for e in chosen
                                if e.label == "RAQUETTE"])
    contacts = []
    for event in chosen:
        kind = ANSWER_OF_LABEL[event.label]
        point = None if event.point is None else tuple(float(v) for v in event.point)
        contacts.append(RallyContact(event.frame, kind, point, players.get(event.frame)))

    positions: dict[str, dict[int, tuple[float, float]]] = {}
    for frame in range(spec.start, spec.stop + 1):
        for slot, (x, y) in frames.get(frame, {}).get("positions", {}).items():
            positions.setdefault(slot, {})[frame] = (float(x), float(y))
    return Rally(spec.start, spec.stop, fps, tuple(contacts), positions)
