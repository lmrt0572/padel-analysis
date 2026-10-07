"""On-disk store of per-frame player positions, in court metres.

Rendering and analytics are iterated on without paying for inference again.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FramePositions:
    """Court positions of the identified players on one frame."""

    frame: int
    positions: dict[str, tuple[float, float]] = field(default_factory=dict)


class PositionCache:
    """A sequence of per-frame positions, plus the inputs that produced them."""

    def __init__(self, video: str = "", calibration: str = "") -> None:
        self.video = video
        self.calibration = calibration
        self._by_frame: dict[int, FramePositions] = {}

    def add(self, entry: FramePositions) -> None:
        self._by_frame[entry.frame] = entry

    def frames(self) -> list[int]:
        return sorted(self._by_frame)

    def at(self, frame: int) -> FramePositions:
        return self._by_frame[frame]

    def save(self, path: Path, video: str, calibration: str) -> None:
        payload = {
            "video": video,
            "calibration": calibration,
            "frames": [
                {
                    "frame": entry.frame,
                    "positions": {k: list(v) for k, v in entry.positions.items()},
                }
                for entry in (self._by_frame[f] for f in self.frames())
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PositionCache":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cache = cls(video=payload["video"], calibration=payload["calibration"])
        for entry in payload["frames"]:
            cache.add(
                FramePositions(
                    frame=int(entry["frame"]),
                    positions={
                        k: (float(v[0]), float(v[1]))
                        for k, v in entry["positions"].items()
                    },
                )
            )
        return cache
