"""Runs the pipeline against the annotations and assembles every figure.

Both ablations are evaluated on the same frames within a single pass over the video,
so the comparison cannot be confounded by a difference of sample.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry.calibration import Calibration
from ..io.video_source import VideoSource
from ..perception.appearance import torso_histogram
from ..perception.ground_point import AnkleMidpoint
from ..perception.pose_detector import PoseDetector
from ..tracking.court_constraint import CourtObservation, CourtSlotTracker
from .bytetrack_baseline import ByteTrackBaseline
from .dataset import PoseAnnotations
from .identity import IdentityGroundTruth
from .matching import match_by_iou
from .metrics import detection_score, localisation_error
from .tracking_metrics import TrackingAccumulator


@dataclass
class Accumulator:
    """Counts and samples gathered frame by frame."""

    matched: int = 0
    predicted: int = 0
    annotated: int = 0
    ankle_predicted: list[np.ndarray] = field(default_factory=list)
    ankle_annotated: list[np.ndarray] = field(default_factory=list)
    bbox_predicted: list[np.ndarray] = field(default_factory=list)
    depths: list[float] = field(default_factory=list)
    constrained_tracks: list[int] = field(default_factory=list)
    bytetrack_tracks: list[int] = field(default_factory=list)


def run_campaign(
    video: Path,
    annotations_path: Path,
    calibration_path: Path,
    identity_path: Path,
    frames: int | None = None,
    start: int = 0,
    imgsz: int = 1600,
) -> dict[str, Any]:
    """Evaluate detection, localisation and both trackers over the same frames."""
    annotations = PoseAnnotations.load(annotations_path)
    truth_identity = IdentityGroundTruth.load(identity_path)
    projector = Calibration.load(calibration_path).projector
    detector = PoseDetector(imgsz=imgsz)
    ankle_strategy = AnkleMidpoint()
    constrained = CourtSlotTracker()
    baseline = ByteTrackBaseline()
    totals = Accumulator()
    constrained_score = TrackingAccumulator()
    bytetrack_score = TrackingAccumulator()

    with VideoSource(video) as source:
        meta = source.metadata
        remaining = meta.frame_count - start
        count = remaining if frames is None else min(frames, remaining)

        for index, frame in source.iter_frames(start=start, stop=start + count):
            truth = annotations.for_frame(index)
            detections = detector.detect(frame)

            totals.predicted += len(detections)
            totals.annotated += len(truth)

            pairs = match_by_iou(
                [d.bbox for d in detections], [p.bbox for p in truth], threshold=0.5
            )
            totals.matched += len(pairs)

            for prediction_index, truth_index in pairs:
                detection, annotated = detections[prediction_index], truth[truth_index]
                annotated_ankle = annotated.ankle_midpoint()
                court = projector.image_to_court(annotated_ankle.reshape(1, 2))[0]

                totals.ankle_predicted.append(detection.ankle_midpoint())
                totals.ankle_annotated.append(annotated_ankle)
                totals.bbox_predicted.append(detection.bbox_bottom_centre())
                totals.depths.append(float(court[1]))

            observations = []
            for detection in detections:
                point, confidence = ankle_strategy(detection)
                observations.append(
                    CourtObservation(
                        court_xy=projector.image_to_court(point.reshape(1, 2))[0],
                        confidence=confidence,
                        appearance=torso_histogram(frame, detection.bbox),
                    )
                )
            assignment = constrained.update(observations)
            tracked = baseline.update(detections)
            totals.constrained_tracks.append(len(assignment))
            totals.bytetrack_tracks.append(len(tracked))

            # MOTA et IDF1 : seulement sur les frames ou l'identite est connue.
            identity_row = truth_identity.assignments.get(index)
            if identity_row:
                # Les equipes changent de cote au cours du match, et les slots
                # designent une moitie de court : au-dela d'une frontiere, `near_1`
                # est quelqu'un d'autre. Le nom de reference porte donc le segment,
                # ce qui arrete l'identite au lieu de la prolonger a tort.
                segment = truth_identity.segment_of(index)
                reference = {
                    f"{slot}#{segment}": truth[annotation_index].bbox
                    for slot, annotation_index in identity_row.items()
                    if annotation_index < len(truth)
                }
                constrained_score.add(
                    truth=reference,
                    hypothesis={
                        slot: detections[i].bbox for slot, i in assignment.items()
                    },
                )
                bytetrack_score.add(
                    truth=reference,
                    hypothesis={str(t.track_id): t.bbox for t in tracked},
                )

    depths = np.array(totals.depths)
    ankle_error = localisation_error(
        np.array(totals.ankle_predicted), np.array(totals.ankle_annotated), depths
    )
    bbox_error = localisation_error(
        np.array(totals.bbox_predicted), np.array(totals.ankle_annotated), depths
    )
    score = detection_score(totals.matched, totals.predicted, totals.annotated)

    constrained_counts = np.array(totals.constrained_tracks)
    bytetrack_counts = np.array(totals.bytetrack_tracks)
    constrained_tracking = constrained_score.score()
    bytetrack_tracking = bytetrack_score.score()

    return {
        "frames": int(constrained_counts.size),
        "identity": {
            "cuts": len(truth_identity.cuts),
            "boundaries": len(truth_identity.boundaries),
            "arbitrated_episodes": len(truth_identity.resolved),
            "arbitrated_cuts": len(truth_identity.resolved_cuts),
        },
        "detection": {
            "precision": score.precision,
            "recall": score.recall,
            "f1": score.f1,
            "predicted": totals.predicted,
            "annotated": totals.annotated,
            "matched": totals.matched,
        },
        "ablation_ground_point": {
            "ankles_median_px": ankle_error.median_px,
            "ankles_median_px_near": ankle_error.median_px_near,
            "ankles_median_px_far": ankle_error.median_px_far,
            "bbox_median_px": bbox_error.median_px,
            "bbox_median_px_near": bbox_error.median_px_near,
            "bbox_median_px_far": bbox_error.median_px_far,
            "samples": ankle_error.samples,
        },
        "ablation_tracker": {
            "constrained_mean_tracks": float(constrained_counts.mean()),
            "constrained_frames_over_four": int((constrained_counts > 4).sum()),
            "constrained_mota": constrained_tracking.mota,
            "constrained_idf1": constrained_tracking.idf1,
            "constrained_id_switches": constrained_tracking.id_switches,
            "bytetrack_mean_tracks": float(bytetrack_counts.mean()),
            "bytetrack_frames_over_four": int((bytetrack_counts > 4).sum()),
            "bytetrack_mota": bytetrack_tracking.mota,
            "bytetrack_idf1": bytetrack_tracking.idf1,
            "bytetrack_id_switches": bytetrack_tracking.id_switches,
            "scored_frames": constrained_tracking.frames,
        },
    }
