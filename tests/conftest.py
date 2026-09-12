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
    # Points 3D : les 4 coins au sol, plus les 2 coins hauts du mur de fond eloigne.
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
    # Rotation : camera regardant vers les y positifs, inclinee vers le bas.
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
