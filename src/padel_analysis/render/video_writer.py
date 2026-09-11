"""Writes annotated frames to an MP4 file."""

from pathlib import Path
from typing import Self

import cv2
import numpy as np


class VideoWriter:
    """Context manager around `cv2.VideoWriter`."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        self._path = Path(path)
        self._fps = fps
        self._size = size
        self._writer: cv2.VideoWriter | None = None

    def __enter__(self) -> Self:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self._path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, self._size
        )
        if not self._writer.isOpened():
            raise OSError(f"could not open {self._path} for writing")
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("writer is not open")
        self._writer.write(frame)

    def __exit__(self, *exc_info: object) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
