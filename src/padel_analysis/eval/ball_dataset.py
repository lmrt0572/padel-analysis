"""Reader for the PadelTracker100 ball annotations.

The file declares nine categories - Ball, Wall and seven event kinds - but only
Ball is ever used. Wall was planned by the authors and left empty, which is why
no contact surface can be scored against this dataset.

Coverage is uneven and must be respected rather than assumed: the women's final
is annotated across the whole match, the men's final stops at frame 21472. Past
that point a missing annotation does not mean the ball is absent.
"""

import json
import re
from pathlib import Path

_FRAME_NUMBER = re.compile(r"(\d+)")
BALL_CATEGORY = "Ball"


class BallAnnotations:
    """Ball centres in image pixels, indexed by frame number."""

    def __init__(self, by_frame: dict[int, list[tuple[float, float]]]) -> None:
        self._by_frame = by_frame

    @classmethod
    def load(cls, path: Path) -> "BallAnnotations":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

        ball_ids = {
            c["id"] for c in payload["categories"] if c["name"] == BALL_CATEGORY
        }

        frame_of_image: dict[int, int] = {}
        for image in payload["images"]:
            match = _FRAME_NUMBER.search(image["file_name"])
            if match is None:
                raise ValueError(
                    f"could not read a frame index from {image['file_name']!r}"
                )
            frame_of_image[image["id"]] = int(match.group(1))

        by_frame: dict[int, list[tuple[float, float]]] = {}
        for annotation in payload["annotations"]:
            if annotation["category_id"] not in ball_ids:
                continue
            x, y, w, h = annotation["bbox"]
            frame = frame_of_image[annotation["image_id"]]
            by_frame.setdefault(frame, []).append((x + w / 2, y + h / 2))
        return cls(by_frame)

    def frame_indices(self) -> list[int]:
        """Frames carrying at least one ball, in order."""
        return sorted(self._by_frame)

    def for_frame(self, index: int) -> list[tuple[float, float]]:
        return self._by_frame.get(index, [])

    def centres(self) -> dict[int, tuple[float, float]]:
        """One centre per annotated frame, for the frames carrying exactly one."""
        return {
            frame: balls[0]
            for frame, balls in self._by_frame.items()
            if len(balls) == 1
        }
