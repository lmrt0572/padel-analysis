"""Court calibration: turning clicked correspondences into a validated projector.

Control points are excluded from the fit and only measure its accuracy.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .projector import CourtProjector, ReprojectionError


@dataclass(frozen=True)
class CalibrationPoint:
    """One correspondence between a known court location and a clicked pixel."""

    name: str
    court_xy: tuple[float, float]
    image_xy: tuple[float, float]
    is_control: bool = False
    height: float = 0.0
    """Metres above the ground; non-zero for the wall and net references a pose needs."""

    @property
    def court_xyz(self) -> tuple[float, float, float]:
        """The point in three dimensions."""
        return (self.court_xy[0], self.court_xy[1], self.height)


@dataclass(frozen=True)
class Calibration:
    """A fitted projector together with the evidence of its accuracy."""

    points: list[CalibrationPoint]
    projector: CourtProjector = field(compare=False)
    error: ReprojectionError | None = None
    n_control_points: int = 0

    def save(self, path: Path) -> None:
        payload = {
            "points": [
                {
                    "name": p.name,
                    "court_xy": list(p.court_xy),
                    "image_xy": list(p.image_xy),
                    "is_control": p.is_control,
                    "height": p.height,
                }
                for p in self.points
            ]
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        points = [
            CalibrationPoint(
                name=entry["name"],
                court_xy=tuple(entry["court_xy"]),
                image_xy=tuple(entry["image_xy"]),
                is_control=entry.get("is_control", False),
                height=entry.get("height", 0.0),
            )
            for entry in payload["points"]
        ]
        return calibration_from_points(points)


class Calibrator(Protocol):
    """Produce a calibration from a single video frame."""

    def calibrate(self, frame: np.ndarray) -> Calibration: ...


def calibration_from_points(points: list[CalibrationPoint]) -> Calibration:
    """Fit a projector on the non-control points and measure it on the control ones.

    Points above the ground take no part in either: a homography maps one plane only.
    """
    ground = [p for p in points if p.height == 0.0]
    fit = [p for p in ground if not p.is_control]
    control = [p for p in ground if p.is_control]

    if len(fit) < 4:
        raise ValueError("need at least 4 non-control points to fit a homography")

    projector = CourtProjector.from_correspondences(
        np.array([p.court_xy for p in fit], dtype=np.float64),
        np.array([p.image_xy for p in fit], dtype=np.float64),
    )

    error = None
    if control:
        error = projector.reprojection_error(
            np.array([p.court_xy for p in control], dtype=np.float64),
            np.array([p.image_xy for p in control], dtype=np.float64),
        )

    return Calibration(
        points=list(points),
        projector=projector,
        error=error,
        n_control_points=len(control),
    )
