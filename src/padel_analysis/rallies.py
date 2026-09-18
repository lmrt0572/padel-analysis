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


def striker_slot(box: np.ndarray | None, people: Sequence, assignment: dict) -> str | None:
    """The slot of the player whose box was lit for a strike, if the tracker had one."""
    if box is None:
        return None
    slot_of_index = {index: slot for slot, index in assignment.items()}
    for index, person in enumerate(people):
        if np.array_equal(person.bbox, box):
            return slot_of_index.get(index)
    return None


def build_rally(analysis: dict, events: Sequence, spec: RallySpec, fps: float) -> Rally:
    """The rally's contacts and player positions, from a saved analysis and its events."""
    frames = analysis["frames"]
    contacts = []
    for event in events:
        if not spec.start <= event.frame <= spec.stop:
            continue
        kind = ANSWER_OF_LABEL[event.label]
        point = None if event.point is None else tuple(float(v) for v in event.point)
        player = None
        if kind == "raquette":
            frame = frames.get(event.frame, {})
            player = striker_slot(event.box, frame.get("people", []),
                                  frame.get("assignment", {}))
        contacts.append(RallyContact(event.frame, kind, point, player))

    positions: dict[str, dict[int, tuple[float, float]]] = {}
    for frame in range(spec.start, spec.stop + 1):
        for slot, (x, y) in frames.get(frame, {}).get("positions", {}).items():
            positions.setdefault(slot, {})[frame] = (float(x), float(y))
    return Rally(spec.start, spec.stop, fps, tuple(contacts), positions)
