"""Random and sequential access to the frames of a video file.

Frames are always read from the video by index. The annotation files name frames
as `frame_000000.PNG`, but extracting fifty thousand full-HD frames to disk would
cost around a hundred gigabytes, so nothing is ever written out.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int


class VideoSource:
    """Reads frames of a video by index, or sequentially."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> Self:
        if not self._path.exists():
            raise FileNotFoundError(f"no such video: {self._path}")
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise OSError(f"could not open video: {self._path}")
        self._capture = capture
        return self

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def metadata(self) -> VideoMetadata:
        capture = self._require_open()
        return VideoMetadata(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def read(self, index: int) -> np.ndarray:
        """Return the frame at `index`. Seeking is slower than iterating."""
        capture = self._require_open()
        if index < 0 or index >= self.metadata.frame_count:
            raise IndexError(f"frame {index} out of range")
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise IndexError(f"could not read frame {index}")
        return frame

    def iter_frames(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (index, frame) sequentially. Much faster than repeated seeking."""
        capture = self._require_open()
        end = self.metadata.frame_count if stop is None else stop
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        while index < end:
            ok, frame = capture.read()
            if not ok:
                return
            if (index - start) % step == 0:
                yield index, frame
            index += 1

    def _require_open(self) -> cv2.VideoCapture:
        if self._capture is None:
            raise RuntimeError("video is not open; use `with VideoSource(path) as s:`")
        return self._capture
