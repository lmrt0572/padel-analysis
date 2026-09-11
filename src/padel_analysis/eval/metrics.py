"""Detection and localisation scores.

Errors are reported separately for each half of the court. A pixel is worth 1.51 cm
at the near baseline and 6.47 cm at the far one, so a single aggregated figure in
centimetres would hide which half it came from.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionScore:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class LocalisationError:
    """Distance between predicted and annotated points, in pixels."""

    median_px: float
    median_px_near: float
    median_px_far: float
    samples: int


def detection_score(matched: int, predicted: int, annotated: int) -> DetectionScore:
    """Precision, recall and F1 from raw counts."""
    precision = matched / predicted if predicted else 0.0
    recall = matched / annotated if annotated else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return DetectionScore(precision=precision, recall=recall, f1=f1)


def localisation_error(
    predicted: np.ndarray, annotated: np.ndarray, court_depths: np.ndarray
) -> LocalisationError:
    """Pixel distance between matched points, split by half of the court.

    Args:
        predicted: (N, 2) image points.
        annotated: (N, 2) the matching annotated points.
        court_depths: (N,) the signed `y` of each annotated point, in metres.
            Negative is the near half.
    """
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    annotated = np.asarray(annotated, dtype=np.float64).reshape(-1, 2)
    depths = np.asarray(court_depths, dtype=np.float64).reshape(-1)

    distances = np.linalg.norm(predicted - annotated, axis=1)
    near = distances[depths < 0]
    far = distances[depths >= 0]

    return LocalisationError(
        median_px=float(np.median(distances)) if distances.size else float("nan"),
        median_px_near=float(np.median(near)) if near.size else float("nan"),
        median_px_far=float(np.median(far)) if far.size else float("nan"),
        samples=int(distances.size),
    )
