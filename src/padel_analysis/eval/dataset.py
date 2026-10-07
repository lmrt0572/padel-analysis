"""Reader for the PadelTracker100 pose annotations.

COCO-formatted, in image pixels, with no court coordinates and no track identity.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..perception.keypoints import LEFT_ANKLE, RIGHT_ANKLE, dataset_to_coco_order

_FRAME_NUMBER = re.compile(r"(\d+)")


@dataclass(frozen=True)
class AnnotatedPerson:
    """One annotated person on one frame, in image pixels."""

    bbox: np.ndarray  # (4,) x1 y1 x2 y2
    keypoints: np.ndarray  # (17, 3) x, y, visibility, in COCO order

    def ankle_midpoint(self) -> np.ndarray:
        """Return the midpoint of the two ankles, in image pixels."""
        return (self.keypoints[LEFT_ANKLE, :2] + self.keypoints[RIGHT_ANKLE, :2]) / 2


class PoseAnnotations:
    """All annotated people of a match, indexed by frame number."""

    def __init__(self, by_frame: dict[int, list[AnnotatedPerson]]) -> None:
        self._by_frame = by_frame

    @classmethod
    def load(cls, path: Path) -> "PoseAnnotations":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

        frame_of_image: dict[int, int] = {}
        for image in payload["images"]:
            match = _FRAME_NUMBER.search(image["file_name"])
            if match is None:
                raise ValueError(
                    f"could not read a frame index from {image['file_name']!r}"
                )
            frame_of_image[image["id"]] = int(match.group(1))

        by_frame: dict[int, list[AnnotatedPerson]] = {
            frame: [] for frame in frame_of_image.values()
        }
        for annotation in payload["annotations"]:
            x, y, w, h = annotation["bbox"]
            keypoints = np.asarray(annotation["keypoints"], dtype=np.float64)
            keypoints = dataset_to_coco_order(keypoints.reshape(17, 3))
            by_frame[frame_of_image[annotation["image_id"]]].append(
                AnnotatedPerson(
                    bbox=np.array([x, y, x + w, y + h], dtype=np.float64),
                    keypoints=keypoints,
                )
            )
        return cls(by_frame)

    def frame_indices(self) -> list[int]:
        return list(self._by_frame)

    def for_frame(self, index: int) -> list[AnnotatedPerson]:
        return self._by_frame.get(index, [])
