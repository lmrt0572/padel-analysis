import numpy as np

from padel_analysis.io.splices import frame_changes, splices


def _frame(value, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    image = np.full((90, 160, 3), value, dtype=np.float32)
    image += rng.normal(0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def test_the_first_frame_has_no_change():
    assert frame_changes([(10, _frame(100))]) == {10: 0.0}


def test_a_new_shot_stands_out_from_play():
    frames = [(i, _frame(100, noise=2, seed=i)) for i in range(5)]
    frames += [(i, _frame(160, noise=2, seed=i)) for i in range(5, 10)]
    changes = frame_changes(frames)
    assert splices(changes) == [5]


def test_small_movement_is_not_a_splice():
    frames = []
    for i in range(10):
        image = _frame(100)
        image[40:50, 10 + 5 * i:20 + 5 * i] = 255  # un joueur qui se deplace
        frames.append((i, image))
    assert splices(frame_changes(frames)) == []
