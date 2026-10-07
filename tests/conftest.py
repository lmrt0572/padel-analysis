import cv2
import numpy as np
import pytest

from padel_analysis.geometry.camera import CameraPose
from padel_analysis.geometry.court import Court


@pytest.fixture
def synthetic_pose() -> tuple[CameraPose, np.ndarray, np.ndarray]:
    """A camera 7.6 m up, 6 m behind the negative baseline, looking down the court."""
    court = Court()
    image_size = (1920, 1080)
    focal = 1400.0
    intrinsics = np.array(
        [[focal, 0.0, image_size[0] / 2], [0.0, focal, image_size[1] / 2], [0.0, 0.0, 1.0]]
    )
    # 3D points: the 4 corners on the ground, plus the 2 top corners of the far back wall.
    h = court.back_wall_total_height
    corners = court.corners()
    object_points = np.array(
        [
            [corners[0][0], corners[0][1], 0.0],
            [corners[1][0], corners[1][1], 0.0],
            [corners[2][0], corners[2][1], 0.0],
            [corners[3][0], corners[3][1], 0.0],
            [corners[2][0], corners[2][1], h],
            [corners[3][0], corners[3][1], h],
        ]
    )
    # Rotation: camera looking towards positive y, tilted downwards.
    tilt = np.deg2rad(20.0)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -np.sin(tilt), -np.cos(tilt)],
            [0.0, np.cos(tilt), -np.sin(tilt)],
        ]
    )
    camera_centre = np.array([0.0, -court.half_length - 6.0, 7.6])
    translation = -rotation @ camera_centre

    rvec, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(object_points, rvec, translation, intrinsics, None)
    return (
        CameraPose(rotation=rotation, translation=translation, intrinsics=intrinsics),
        object_points,
        projected.reshape(-1, 2),
    )


SCENE_FRAMES = 30
STANDING = ((-2.5, -6.0), (2.5, -6.0), (-2.5, 6.0), (2.5, 6.0))
"""Where the four players of the synthetic scenes stand, in court metres."""


@pytest.fixture
def scene(tmp_path, synthetic_pose):
    """A short black video seen by the synthetic camera, its calibration and its players.

    Returns the video path, the calibration path and the detections a pose detector
    would give, on every frame, for four players standing still at `STANDING`.
    """
    from padel_analysis.geometry.calibration import CalibrationPoint, calibration_from_points
    from padel_analysis.perception.pose_detector import PersonDetection

    pose, object_points, image_points = synthetic_pose
    calibration = calibration_from_points([
        CalibrationPoint(f"point_{i}", (float(x), float(y)), (float(u), float(v)), height=float(z))
        for i, ((x, y, z), (u, v)) in enumerate(zip(object_points, image_points))
    ])
    calibration_path = tmp_path / "calibration.json"
    calibration.save(calibration_path)

    video = tmp_path / "scene.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1920, 1080))
    for _ in range(SCENE_FRAMES):
        writer.write(np.zeros((1080, 1920, 3), dtype=np.uint8))
    writer.release()

    detections = []
    for x, y in STANDING:
        u, v = pose.project([[x, y, 0.0]])[0]
        keypoints = np.zeros((17, 3))
        keypoints[:, :2] = (u, v - 60.0)
        keypoints[:, 2] = 0.9
        keypoints[15, :2], keypoints[16, :2] = (u - 4.0, v), (u + 4.0, v)
        detections.append(PersonDetection(
            bbox=np.array([u - 20.0, v - 120.0, u + 20.0, v]), confidence=0.9,
            keypoints=keypoints,
        ))
    return video, calibration_path, detections
