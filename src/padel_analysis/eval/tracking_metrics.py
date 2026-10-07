"""MOTA and IDF1, the standard measures of a multi-object tracker.

The distance matrix is built here: `motmetrics.distances.iou_matrix` calls
`np.asfarray`, removed in NumPy 2.0.
"""

from dataclasses import dataclass

import numpy as np

from .matching import iou


@dataclass(frozen=True)
class TrackingScore:
    mota: float
    idf1: float
    id_switches: int
    frames: int


class TrackingAccumulator:
    """Collect per-frame correspondences, then score them."""

    def __init__(self, max_iou_distance: float = 0.5) -> None:
        import motmetrics as mm

        self._mm = mm
        self._accumulator = mm.MOTAccumulator(auto_id=True)
        self._max_distance = max_iou_distance
        self._frames = 0
        # pandas will not hold text and NaN in one column, so names are mapped to
        # stable integers
        self._truth_numbers: dict[str, int] = {}
        self._hypothesis_numbers: dict[str, int] = {}

    @staticmethod
    def _number(names: list[str], registry: dict[str, int]) -> list[int]:
        for name in names:
            if name not in registry:
                registry[name] = len(registry)
        return [registry[name] for name in names]

    def _distance_matrix(
        self, truth: list[np.ndarray], hypothesis: list[np.ndarray]
    ) -> np.ndarray:
        """Return 1 - IoU for each pair, NaN where the pair is too far to match."""
        if not truth or not hypothesis:
            return np.zeros((len(truth), len(hypothesis)))
        distances = np.array(
            [[1.0 - iou(t, h) for h in hypothesis] for t in truth], dtype=np.float64
        )
        distances[distances > self._max_distance] = np.nan
        return distances

    def add(
        self, truth: dict[str, np.ndarray], hypothesis: dict[str, np.ndarray]
    ) -> None:
        """Record one frame; boxes are xyxy and the keys are the identities."""
        truth_names = list(truth)
        hypothesis_names = list(hypothesis)
        distances = self._distance_matrix(
            [truth[name] for name in truth_names],
            [hypothesis[name] for name in hypothesis_names],
        )
        self._accumulator.update(
            self._number(truth_names, self._truth_numbers),
            self._number(hypothesis_names, self._hypothesis_numbers),
            distances,
        )
        self._frames += 1

    def score(self) -> TrackingScore:
        if self._frames == 0:
            return TrackingScore(
                mota=float("nan"), idf1=float("nan"), id_switches=0, frames=0
            )

        host = self._mm.metrics.create()
        summary = host.compute(
            self._accumulator,
            metrics=["mota", "idf1", "num_switches"],
            name="run",
        )
        return TrackingScore(
            mota=float(summary["mota"].iloc[0]),
            idf1=float(summary["idf1"].iloc[0]),
            id_switches=int(summary["num_switches"].iloc[0]),
            frames=self._frames,
        )
