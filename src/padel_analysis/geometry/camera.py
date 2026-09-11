"""Camera pose and the fixed vertical planes of the court enclosure.

Unlike the players, the walls never move. Recovering the camera pose once per video
lets any 3D court point be projected into the image, which is what contact-surface
classification needs in order to tell a glass contact from a mesh contact.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .court import Court


@dataclass(frozen=True)
class WallPlane:
    """A vertical plane of the enclosure, in court coordinates.

    The plane is the set of points satisfying ``normal . p == offset``, bounded
    below by the ground and above by ``height``.
    """

    name: str
    normal: np.ndarray
    offset: float
    height: float
    glass_height: float


@dataclass(frozen=True)
class CameraPose:
    """Position and orientation of the camera in court coordinates."""

    rotation: np.ndarray
    translation: np.ndarray
    intrinsics: np.ndarray

    @property
    def camera_centre(self) -> np.ndarray:
        """Camera position in court coordinates, in metres."""
        return -self.rotation.T @ self.translation

    @classmethod
    def from_correspondences(
        cls,
        object_points: np.ndarray,
        image_points: np.ndarray,
        intrinsics: np.ndarray,
    ) -> "CameraPose":
        """Recover the pose from 3D court points and their pixels.

        Include points off the ground plane - the top corners of the back wall -
        otherwise the vertical direction is only weakly constrained.
        """
        objects = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
        images = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
        if objects.shape[0] < 6:
            raise ValueError("need at least 6 point pairs, including off-ground points")

        ok, rvec, tvec = cv2.solvePnP(
            objects,
            images,
            np.asarray(intrinsics, dtype=np.float64),
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise ValueError("camera pose could not be estimated from these points")

        rotation, _ = cv2.Rodrigues(rvec)
        return cls(
            rotation=rotation,
            translation=tvec.reshape(3),
            intrinsics=np.asarray(intrinsics, dtype=np.float64),
        )

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project court-space 3D points to image pixels."""
        pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 1, 3)
        rvec, _ = cv2.Rodrigues(self.rotation)
        projected, _ = cv2.projectPoints(pts, rvec, self.translation, self.intrinsics, None)
        return projected.reshape(-1, 2)


def back_wall_planes(court: Court) -> list[WallPlane]:
    """The two back walls, as vertical planes at each baseline.

    The stepped side walls depend on dimensions still to be confirmed against the
    FIP rulebook, and are added once those are settled.
    """
    return [
        WallPlane(
            name="back_wall_negative_y",
            normal=np.array([0.0, 1.0, 0.0]),
            offset=-court.half_length,
            height=court.back_wall_total_height,
            glass_height=court.back_wall_glass_height,
        ),
        WallPlane(
            name="back_wall_positive_y",
            normal=np.array([0.0, 1.0, 0.0]),
            offset=court.half_length,
            height=court.back_wall_total_height,
            glass_height=court.back_wall_glass_height,
        ),
    ]
