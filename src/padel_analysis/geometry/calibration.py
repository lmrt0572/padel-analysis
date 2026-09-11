"""Court calibration: turning clicked correspondences into a validated projector.

A calibration is built from named point correspondences. Points flagged as control
points are excluded from the homography fit and used solely to measure accuracy, so
that the reported error is never optimistic.
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
            )
            for entry in payload["points"]
        ]
        return calibration_from_points(points)


class Calibrator(Protocol):
    """Produces a calibration from a single video frame.

    Only manual clicking exists today. The protocol is what allows a line-refinement
    calibrator to be added later as a new implementation rather than a rewrite, if
    the measured reprojection error justifies it.
    """

    def calibrate(self, frame: np.ndarray) -> Calibration: ...


def calibration_from_points(points: list[CalibrationPoint]) -> Calibration:
    """Fit a projector on the non-control points and measure it on the control ones."""
    fit = [p for p in points if not p.is_control]
    control = [p for p in points if p.is_control]

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
