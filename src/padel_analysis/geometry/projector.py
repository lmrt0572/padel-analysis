"""Bidirectional mapping between image pixels and court coordinates.

A plane-to-plane homography: only valid for points on the court surface.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ReprojectionError:
    """Calibration accuracy, measured on points excluded from the fit."""

    rmse_pixels: float
    median_pixels: float
    rmse_metres: float
    median_metres: float


class CourtProjector:
    """Map points between the image plane and the court plane."""

    def __init__(self, court_to_image_matrix: np.ndarray) -> None:
        self._forward = np.asarray(court_to_image_matrix, dtype=np.float64)
        try:
            self._inverse = np.linalg.inv(self._forward)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded upstream
            raise ValueError("homography is not invertible") from exc

    @classmethod
    def from_correspondences(
        cls, court_points: np.ndarray, image_points: np.ndarray
    ) -> "CourtProjector":
        """Fit a homography mapping court coordinates to image pixels.

        Args:
            court_points: array of shape (N, 2), N >= 4, in metres.
            image_points: array of shape (N, 2), the matching pixels.
        """
        court = np.asarray(court_points, dtype=np.float64)
        image = np.asarray(image_points, dtype=np.float64)
        if court.shape != image.shape or court.shape[0] < 4:
            raise ValueError("need at least 4 matching point pairs")

        matrix, _ = cv2.findHomography(court, image, method=0)
        if matrix is None or not np.isfinite(matrix).all():
            raise ValueError("homography could not be estimated from these points")
        if abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("homography is degenerate")
        return cls(matrix)

    def court_to_image(self, points: np.ndarray) -> np.ndarray:
        return self._apply(self._forward, points)

    def image_to_court(self, points: np.ndarray) -> np.ndarray:
        return self._apply(self._inverse, points)

    def reprojection_error(
        self, court_points: np.ndarray, image_points: np.ndarray
    ) -> ReprojectionError:
        """Measure accuracy on control points that were not used for the fit."""
        court = np.asarray(court_points, dtype=np.float64)
        image = np.asarray(image_points, dtype=np.float64)

        pixel_residuals = np.linalg.norm(self.court_to_image(court) - image, axis=1)
        metre_residuals = np.linalg.norm(self.image_to_court(image) - court, axis=1)

        return ReprojectionError(
            rmse_pixels=float(np.sqrt(np.mean(pixel_residuals**2))),
            median_pixels=float(np.median(pixel_residuals)),
            rmse_metres=float(np.sqrt(np.mean(metre_residuals**2))),
            median_metres=float(np.median(metre_residuals)),
        )

    @staticmethod
    def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, matrix)
        return out.reshape(-1, 2)
