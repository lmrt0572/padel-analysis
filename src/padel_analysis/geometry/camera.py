"""Camera pose, and the five surfaces a ball can bounce off.

Unlike the players, the walls never move. Recovering the camera pose once per video
lets any 3D court point be projected into the image, which is what contact-surface
classification needs in order to tell a glass contact from a mesh contact.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .court import Court

GLASS = "verre"
MESH = "grillage"


@dataclass(frozen=True)
class Surface:
    """One bounded plane of the court: the floor, or one of the four walls.

    The plane is the set of points satisfying ``normal . p == offset``. `contains`
    bounds it to the real extent of the surface, and that bound is what makes an
    intersection admissible or not - without it every ray meets every plane
    somewhere, and nothing is decided.
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
        """Where the ray meets this plane, or None if it never does in front."""
        along = float(self.normal @ direction)
        if abs(along) < 1e-9:
            return None
        distance = (self.offset - float(self.normal @ origin)) / along
        if distance <= 0.0:
            return None
        return origin + distance * direction

    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Whether the point lies within the real extent of the surface.

        Args:
            point: a court-space point, in metres.
            margin: slack in metres, to absorb the camera pose error. Metres and not
                pixels: a pixel is 1.51 cm near the camera and 6.47 cm at the far
                baseline, so a pixel margin would be four times looser at depth.
        """
        return all(
            low - margin <= value <= high + margin
            for value, (low, high) in zip(point, self.bounds)
        )

    def material_at(self, point: np.ndarray) -> str | None:
        """Glass or mesh at that point, or None for the floor."""
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

    def ray(self, pixel: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """Camera centre and a unit direction, in court coordinates, through `pixel`.

        The ball is somewhere on this ray and nothing says where - which is why a
        height cannot be read off a single image in flight. At a contact it can, the
        ball being then on a surface, and a ray meets a plane once.
        """
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
    """The five surfaces a ball can bounce off, the floor first.

    The order matters: when a ray admits more than one surface, the caller takes the
    first. The floor leads because a floor bounce is at zero height, so its position
    is exact, whereas an admissible wall point may be nothing but an effect of the
    margin.
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
    return surfaces
