import numpy as np

from padel_analysis.perception.keypoints import (
    COCO_KEYPOINTS,
    DATASET_KEYPOINTS,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    dataset_to_coco_order,
)


def test_both_orders_name_the_same_seventeen_joints():
    assert len(COCO_KEYPOINTS) == 17
    assert len(DATASET_KEYPOINTS) == 17
    assert set(COCO_KEYPOINTS) == set(DATASET_KEYPOINTS)


def test_the_two_orders_actually_differ():
    """Si un jour ils coincident, le remapping devient inutile et ce test le dira."""
    assert COCO_KEYPOINTS != DATASET_KEYPOINTS


def test_ankle_indices_follow_the_coco_convention():
    assert COCO_KEYPOINTS[LEFT_ANKLE] == "left_ankle"
    assert COCO_KEYPOINTS[RIGHT_ANKLE] == "right_ankle"


def test_remapping_puts_each_joint_at_its_coco_index():
    # Un tableau ou chaque ligne porte l'indice de sa position dans l'ordre dataset.
    values = np.arange(17 * 3, dtype=np.float64).reshape(17, 3)
    remapped = dataset_to_coco_order(values)
    for coco_index, name in enumerate(COCO_KEYPOINTS):
        dataset_index = DATASET_KEYPOINTS.index(name)
        np.testing.assert_array_equal(remapped[coco_index], values[dataset_index])


def test_remapping_swaps_left_and_right_ankles():
    """Le cas concret qui produirait un bug silencieux si on l'oubliait."""
    values = np.zeros((17, 3))
    values[DATASET_KEYPOINTS.index("left_ankle")] = [100.0, 200.0, 1.0]
    values[DATASET_KEYPOINTS.index("right_ankle")] = [300.0, 400.0, 1.0]
    remapped = dataset_to_coco_order(values)
    np.testing.assert_array_equal(remapped[LEFT_ANKLE], [100.0, 200.0, 1.0])
    np.testing.assert_array_equal(remapped[RIGHT_ANKLE], [300.0, 400.0, 1.0])


def test_remapping_preserves_the_ankle_midpoint():
    """L'inversion s'annule pour le milieu : utile a savoir, dangereux a supposer."""
    values = np.zeros((17, 3))
    values[DATASET_KEYPOINTS.index("left_ankle")] = [100.0, 200.0, 1.0]
    values[DATASET_KEYPOINTS.index("right_ankle")] = [300.0, 400.0, 1.0]
    remapped = dataset_to_coco_order(values)
    before = (values[15, :2] + values[16, :2]) / 2
    after = (remapped[LEFT_ANKLE, :2] + remapped[RIGHT_ANKLE, :2]) / 2
    np.testing.assert_allclose(before, after)


def test_remapping_accepts_a_batch():
    values = np.arange(4 * 17 * 3, dtype=np.float64).reshape(4, 17, 3)
    remapped = dataset_to_coco_order(values)
    assert remapped.shape == (4, 17, 3)
