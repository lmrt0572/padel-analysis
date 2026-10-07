"""Camera pose, and the surfaces a ball can bounce off.

The pose, recovered once per video, projects any 3D court point into the image and
casts a ray through any pixel.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .court import Court

GLASS = "verre"
MESH = "grillage"


@dataclass(frozen=True)
class Surface:
    """One bounded plane of the court: the floor, a wall or the net.

    The plane is the set of points satisfying ``normal . p == offset``; `contains`
    bounds it to the real extent of the surface.
    """

    name: str
    normal: np.ndarray
    offset: float
    bounds: tuple[tuple[float, float], ...]
    """(min, max) per axis, x then y then z. Equal bounds mean the surface is flat
    along that axis."""
    glass_height: float | None = None
    """Back walls: glass at or below this height, mesh above."""
    glass_from_ends: float | None = None
    """Side walls: glass where |y| reaches this, mesh between the two."""

    def intersect(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> np.ndarray | None:
        """Return where the ray meets this plane, or None if it never does in front."""
        along = float(self.normal @ direction)
        if abs(along) < 1e-9:
            return None
        distance = (self.offset - float(self.normal @ origin)) / along
        if distance <= 0.0:
            return None
        return origin + distance * direction

    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Return whether the point lies within the real extent of the surface.

        Args:
            point: a court-space point, in metres.
            margin: slack in metres, to absorb the camera pose error.
        """
        for value, (low, high) in zip(point, self.bounds):
            if low == high:
                # the flat axis is guaranteed by the plane equation; checking it would fail
                # on a rounding error
                continue
            if not low - margin <= value <= high + margin:
                return False
        return True

    def material_at(self, point: np.ndarray) -> str | None:
        """Return glass or mesh at that point, or None for the floor."""
        if self.glass_height is not None:
            return GLASS if point[2] <= self.glass_height else MESH
        if self.glass_from_ends is not None:
            return GLASS if abs(point[1]) >= self.glass_from_ends else MESH
        return None


@dataclass(frozen=True)
class CameraPose:
    """Position and orientation of the camera in court coordinates."""

    rotation: np.ndarray
    translation: np.ndarray
    intrinsics: np.ndarray

    @property
    def camera_centre(self) -> np.ndarray:
        """The camera position in court coordinates, in metres."""
        return -self.rotation.T @ self.translation

    @classmethod
    def from_correspondences(
        cls,
        object_points: np.ndarray,
        image_points: np.ndarray,
        intrinsics: np.ndarray,
    ) -> "CameraPose":
        """Recover the pose from 3D court points and their pixels.

        Points off the ground plane are needed to constrain the vertical direction.
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

    def ray(self, pixel: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """Return the camera centre and a unit direction through `pixel`, in court coordinates."""
        homogeneous = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
        direction = self.rotation.T @ (np.linalg.inv(self.intrinsics) @ homogeneous)
        return self.camera_centre, direction / np.linalg.norm(direction)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project court-space 3D points to image pixels."""
        pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 1, 3)
        rvec, _ = cv2.Rodrigues(self.rotation)
        projected, _ = cv2.projectPoints(pts, rvec, self.translation, self.intrinsics, None)
        return projected.reshape(-1, 2)


def court_surfaces(court: Court) -> list[Surface]:
    """Return the six surfaces a ball can bounce off, the floor first and the net last.

    When a ray admits more than one surface, the caller takes the first.
    """
    half_width, half_length = court.half_width, court.half_length
    glass_from_ends = half_length - court.side_wall_glass_length

    surfaces = [
        Surface(
            name="floor",
            normal=np.array([0.0, 0.0, 1.0]),
            offset=0.0,
            bounds=((-half_width, half_width), (-half_length, half_length), (0.0, 0.0)),
        )
    ]
    for sign, side in ((-1.0, "negative"), (1.0, "positive")):
        surfaces.append(
            Surface(
                name=f"back_wall_{side}_y",
                normal=np.array([0.0, 1.0, 0.0]),
                offset=sign * half_length,
                bounds=(
                    (-half_width, half_width),
                    (sign * half_length, sign * half_length),
                    (0.0, court.back_wall_total_height),
                ),
                glass_height=court.back_wall_glass_height,
            )
        )
    for sign, side in ((-1.0, "negative"), (1.0, "positive")):
        surfaces.append(
            Surface(
                name=f"side_wall_{side}_x",
                normal=np.array([1.0, 0.0, 0.0]),
                offset=sign * half_width,
                bounds=(
                    (sign * half_width, sign * half_width),
                    (-half_length, half_length),
                    (0.0, court.side_wall_total_height),
                ),
                glass_from_ends=glass_from_ends,
            )
        )
    # the net last: the smallest and rarest target, it only wins when alone
    surfaces.append(
        Surface(
            name="net",
            normal=np.array([0.0, 1.0, 0.0]),
            offset=0.0,
            bounds=(
                (-half_width, half_width),
                (0.0, 0.0),
                (0.0, court.net_height_posts),
            ),
        )
    )
    return surfaces


def estimate_intrinsics(
    object_points: np.ndarray,
    image_points: np.ndarray,
    image_size: tuple[int, int],
    focal_range: range = range(700, 4001, 5),
) -> np.ndarray:
    """Return the intrinsics whose focal length reprojects these correspondences best.

    The principal point is assumed at the image centre and distortion ignored.

    Args:
        object_points: 3D court points, including some off the ground.
        image_points: the matching pixels.
        image_size: (width, height), to place the principal point.
        focal_range: focal lengths to try, in pixels.
    """
    width, height = image_size
    best: tuple[float, np.ndarray] | None = None
    for focal in focal_range:
        intrinsics = np.array(
            [
                [float(focal), 0.0, width / 2],
                [0.0, float(focal), height / 2],
                [0.0, 0.0, 1.0],
            ]
        )
        try:
            pose = CameraPose.from_correspondences(
                object_points, image_points, intrinsics
            )
        except ValueError:
            continue
        gaps = pose.project(object_points) - np.asarray(image_points, dtype=np.float64)
        error = float(np.sqrt((gaps**2).sum(axis=1).mean()))
        if best is None or error < best[0]:
            best = (error, intrinsics)
    if best is None:
        raise ValueError("no focal length fitted these correspondences")
    return best[1]


def pose_from_calibration(points: Sequence, image_size: tuple[int, int]) -> CameraPose:
    """Return the camera pose fitted on a calibration's non-control points.

    Raises:
        ValueError: when every fitting point sits on the ground.
    """
    fit = [p for p in points if not p.is_control]
    objects = np.array([p.court_xyz for p in fit], dtype=np.float64)
    pixels = np.array([p.image_xy for p in fit], dtype=np.float64)
    if not (objects[:, 2] > 0).any():
        raise ValueError("a camera pose needs fitting points above the ground")
    return CameraPose.from_correspondences(
        objects, pixels, estimate_intrinsics(objects, pixels, image_size)
    )
