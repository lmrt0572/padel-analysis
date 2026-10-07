"""Keypoint orders, and the conversion between them.

PadelTracker100 swaps left and right for every paired joint except the ears. The
ankle midpoint is unaffected, so a naive check would not catch it.
"""

import numpy as np

# order used by Ultralytics and the COCO convention
COCO_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# order found in the `*_pose.json` files of PadelTracker100
DATASET_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "right_eye",
    "left_eye",
    "left_ear",
    "right_ear",
    "right_shoulder",
    "left_shoulder",
    "right_elbow",
    "left_elbow",
    "right_wrist",
    "left_wrist",
    "right_hip",
    "left_hip",
    "right_knee",
    "left_knee",
    "right_ankle",
    "left_ankle",
)

LEFT_ANKLE = COCO_KEYPOINTS.index("left_ankle")
RIGHT_ANKLE = COCO_KEYPOINTS.index("right_ankle")

_DATASET_TO_COCO = np.array(
    [DATASET_KEYPOINTS.index(name) for name in COCO_KEYPOINTS], dtype=np.intp
)


def dataset_to_coco_order(keypoints: np.ndarray) -> np.ndarray:
    """Reorder dataset keypoints into COCO order.

    Args:
        keypoints: array of shape (17, C) or (N, 17, C).

    Returns:
        An array of the same shape, reordered along the joint axis.
    """
    array = np.asarray(keypoints)
    if array.shape[-2] != 17:
        raise ValueError(f"expected 17 keypoints, got {array.shape[-2]}")
    return array[..., _DATASET_TO_COCO, :]
