import pickle

import numpy as np
import pytest
import torch

from padel_analysis.ball.candidates import Candidate
from padel_analysis.contact.learned import (
    CLASSES,
    GLASS_WALL,
    IGNORED,
    MESH_WALL,
    ContactModel,
    ContactNet,
    decode,
    frame_features,
    frame_labels,
    train,
)
from padel_analysis.geometry.camera import court_surfaces
from padel_analysis.geometry.court import Court
from padel_analysis.render.ball_overlay import ContactEvent


def _probabilities(count, peaks):
    """Frame -> (contact probability, class index) for the given frames, none elsewhere."""
    probs = np.zeros((count, len(CLASSES)))
    probs[:, 0] = 1.0
    for frame, (contact, kind) in peaks.items():
        probs[frame] = 0.0
        probs[frame, 0] = 1.0 - contact
        probs[frame, kind] = contact
    return probs


def test_a_mark_labels_its_frame_and_both_neighbours():
    labels = frame_labels({10: "sol"}, start=0, count=20)
    assert list(labels[9:12]) == [2, 2, 2]
    assert labels[8] == IGNORED and labels[12] == IGNORED
    assert labels[5] == 0 and labels[15] == 0


def test_glass_and_mesh_share_the_wall_class():
    labels = frame_labels({5: "verre", 15: "grillage"}, start=0, count=20)
    assert labels[5] == labels[15] == CLASSES.index("mur")


def test_the_ignored_margin_never_hides_a_neighbouring_contact():
    labels = frame_labels({10: "sol", 12: "raquette"}, start=0, count=20)
    assert labels[11] == CLASSES.index("raquette")
    assert labels[12] == CLASSES.index("raquette")
    assert labels[10] == CLASSES.index("sol")


def test_labels_are_placed_relative_to_the_range_start():
    labels = frame_labels({1005: "filet"}, start=1000, count=20)
    assert labels[5] == CLASSES.index("filet")


def test_decode_keeps_a_confident_peak_with_its_class():
    probs = _probabilities(30, {12: (0.9, CLASSES.index("sol"))})
    assert decode(probs, np.zeros(30), start=100) == {112: "sol"}


def test_decode_ignores_a_peak_under_the_threshold():
    probs = _probabilities(30, {12: (0.5, CLASSES.index("sol"))})
    assert decode(probs, np.zeros(30), start=0, threshold=0.7) == {}


def test_decode_keeps_one_contact_per_neighbourhood_the_strongest():
    probs = _probabilities(30, {10: (0.8, 1), 12: (0.95, 2), 25: (0.9, 1)})
    assert decode(probs, np.zeros(30), start=0, radius=4) == {12: "sol", 25: "raquette"}


def test_a_wall_contact_takes_its_material_from_the_geometry():
    probs = _probabilities(30, {5: (0.9, 3), 20: (0.9, 3)})
    material = np.zeros(30)
    material[5], material[20] = GLASS_WALL, MESH_WALL
    assert decode(probs, material, start=0) == {5: "verre", 20: "grillage"}


def test_the_network_labels_every_frame():
    net = ContactNet(cues=7).eval()
    out = net(torch.zeros(2, 7, 50))
    assert out.shape == (2, len(CLASSES), 50)


def test_a_saved_model_predicts_the_same(tmp_path):
    model = ContactModel([ContactNet(4).eval()], np.zeros(4), np.ones(4))
    features = np.random.default_rng(0).normal(size=(40, 4)).astype(np.float32)
    model.save(tmp_path / "m.pt")
    again = ContactModel.load(tmp_path / "m.pt")
    np.testing.assert_allclose(again.probabilities(features), model.probabilities(features),
                               atol=1e-6)


def test_a_model_saved_with_numpy_arrays_still_loads(tmp_path):
    """The weights written before the save as tensors keep their normalisation as NumPy."""
    net = ContactNet(4).eval()
    torch.save({"cues": 4, "mean": np.zeros(4, np.float32), "std": np.ones(4, np.float32),
                "nets": [net.state_dict()]}, tmp_path / "old.pt")
    model = ContactModel.load(tmp_path / "old.pt")
    features = np.random.default_rng(0).normal(size=(40, 4)).astype(np.float32)
    np.testing.assert_allclose(model.probabilities(features),
                               ContactModel([net], np.zeros(4), np.ones(4)).probabilities(features),
                               atol=1e-6)


class _Payload:
    def __reduce__(self):
        return (print, ("code execute au chargement",))


def test_a_model_file_cannot_run_code_when_loaded(tmp_path):
    torch.save({"cues": 4, "mean": _Payload(), "std": np.ones(4, np.float32), "nets": []},
               tmp_path / "piege.pt")
    with pytest.raises(pickle.UnpicklingError):
        ContactModel.load(tmp_path / "piege.pt")


def test_training_learns_a_cue_that_marks_the_contacts():
    rng = np.random.default_rng(0)
    sequences = []
    for _ in range(2):
        features = rng.normal(0.0, 0.1, size=(300, 3)).astype(np.float32)
        marks = {f: "sol" for f in range(20, 300, 40)}
        for frame in marks:
            features[frame, 0] = 3.0
        sequences.append((features, frame_labels(marks, 0, 300)))
    model = train(sequences, seeds=(0,), steps=150, crop=64, batch=8)
    found = decode(model.probabilities(sequences[0][0]), np.zeros(300), start=0, threshold=0.5)
    # Each mark also teaches its two neighbours: the peak can fall one frame off.
    assert len(found) == len(range(20, 300, 40))
    assert all(min(abs(f - m) for m in range(20, 300, 40)) <= 1 for f in found)
    assert set(found.values()) == {"sol"}


def _analysis(count, ball):
    frames = {
        f: {"people": [], "assignment": {}, "wrists": [],
            "raw": [Candidate(*ball[f], 0.9)] if ball.get(f) else []}
        for f in range(count)
    }
    return {"start": 0, "stop": count - 1, "size": (1920, 1080), "frames": frames}


def test_features_give_one_row_per_frame_and_mark_a_missing_ball(synthetic_pose):
    pose, _, _ = synthetic_pose
    floor_pixel = tuple(pose.project(np.array([[1.0, -2.0, 0.0]]))[0])
    ball = {f: (floor_pixel if f < 10 else None) for f in range(20)}
    analysis = _analysis(20, ball)
    events = [ContactEvent(5, "SOL", floor_pixel)]
    features, material = frame_features(
        analysis, ball, ball, events, pose, court_surfaces(Court())
    )
    assert features.shape[0] == 20 and material.shape == (20,)
    assert features[3, 0] == 1.0 and features[15, 0] == 0.0
    # A frame without a ball is described the same way every time, and differently
    # from a frame where the ball is there.
    np.testing.assert_array_equal(features[15], features[16])
    assert (features[3] != features[15]).any()
    assert features[5, -5:].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert not features[6, -5:].any()


def test_a_contact_of_another_kind_may_follow_closely_when_sure():
    sol, wall = CLASSES.index("sol"), CLASSES.index("mur")
    probs = _probabilities(30, {10: (0.95, sol), 14: (0.9, wall)})
    assert decode(probs, np.full(30, GLASS_WALL), start=0) == {10: "sol", 14: "verre"}


def test_a_close_contact_of_another_kind_needs_more_confidence():
    sol, wall = CLASSES.index("sol"), CLASSES.index("mur")
    probs = _probabilities(30, {10: (0.95, sol), 14: (0.75, wall)})
    assert decode(probs, np.full(30, GLASS_WALL), start=0) == {10: "sol"}


def test_two_contacts_of_the_same_kind_stay_apart():
    sol = CLASSES.index("sol")
    probs = _probabilities(30, {10: (0.95, sol), 14: (0.9, sol)})
    assert decode(probs, np.zeros(30), start=0) == {10: "sol"}
