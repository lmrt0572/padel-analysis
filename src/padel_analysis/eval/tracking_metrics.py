"""MOTA and IDF1, the standard measures of a multi-object tracker.

They need a ground truth identity on every frame, which PadelTracker100 does not
carry - hence the reconstruction in `identity.py`. MOTA charges false positives,
misses and identity switches against the number of ground truth objects; IDF1 asks
how well each ground truth identity is covered by a single hypothesis identity.

The distance matrix is built here rather than with `motmetrics.distances.iou_matrix`,
which calls `np.asfarray` - removed in NumPy 2.0, and motmetrics has not been
released since. Only that helper is affected; the accumulator and the metrics
themselves are fine, and the matrix is two lines of the IoU already written for
`matching.py`.
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
    """Collects per-frame correspondences, then scores them."""

    def __init__(self, max_iou_distance: float = 0.5) -> None:
        import motmetrics as mm

        self._mm = mm
        self._accumulator = mm.MOTAccumulator(auto_id=True)
        self._max_distance = max_iou_distance
        self._frames = 0
        # motmetrics stores unmatched entries as NaN alongside the identifiers, and
        # pandas will not hold text and NaN in the same column. Names are therefore
        # mapped to integers, stably, so an identity keeps its number across frames.
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
        """1 - IoU for each pair, with NaN where the pair is too far to match."""
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
        """Record one frame. Boxes are xyxy; the dictionary keys are the identities."""
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
