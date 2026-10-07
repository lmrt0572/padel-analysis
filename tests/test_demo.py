import pickle

import cv2
import numpy as np
import pytest

from padel_analysis import demo
from padel_analysis.ball.candidates import Candidate
from padel_analysis.demo import ball_near
from padel_analysis.geometry.calibration import Calibration
from padel_analysis.geometry.camera import court_surfaces
from padel_analysis.geometry.court import Court
from padel_analysis.perception.ground_point import AnkleMidpoint
from padel_analysis.render.ball_overlay import ContactEvent
from padel_analysis.render.court_zones import zone_of

from .conftest import SCENE_FRAMES, STANDING

BOUNCE = (0.0, 5.0, 0.0)


def _analysis(pose, detections):
    """A saved analysis of the scene: four players, and a ball that bounces mid-way."""
    frames = {}
    half = SCENE_FRAMES // 2
    for f in range(SCENE_FRAMES):
        if f <= half:
            u = f / half
            point = np.array([2.0, -6.0, 1.0]) * (1 - u) + np.array(BOUNCE) * u
            point[2] += 8.0 * u * (1 - u)
        else:
            u = (f - half) / half
            point = np.array(BOUNCE) * (1 - u) + np.array([-0.5, 9.0, 1.6]) * u
        x, y = pose.project([point])[0]
        ball = [Candidate(float(x), float(y), 0.95)]
        frames[f] = {"people": detections, "assignment": {}, "positions": {}, "wrists": [],
                     "raw": ball, "candidates": ball}
    return {"start": 0, "stop": SCENE_FRAMES - 1, "size": (1920, 1080), "frames": frames}


def _frames_of(path):
    video = cv2.VideoCapture(str(path))
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video.release()
    return count, size


def test_the_ball_at_the_contact_frame_is_taken_first():
    assert ball_near(10, {10: (5.0, 5.0), 11: (6.0, 6.0)}, {}, 2) == (5.0, 5.0)


def test_a_ball_dropped_at_the_turn_is_taken_from_the_nearest_frame():
    shown = {7: (1.0, 1.0), 11: (2.0, 2.0)}
    assert ball_near(10, shown, {}, 2) == (2.0, 2.0)


def test_the_raw_path_fills_in_before_a_farther_displayed_ball():
    assert ball_near(10, {12: (2.0, 2.0)}, {10: (3.0, 3.0)}, 2) == (3.0, 3.0)


def test_no_ball_within_reach_leaves_the_contact_undrawn():
    assert ball_near(10, {13: (2.0, 2.0)}, {}, 2) is None


def test_a_floor_contact_is_placed_where_the_ray_meets_the_floor(synthetic_pose):
    pose, _, _ = synthetic_pose
    pixel = tuple(pose.project([BOUNCE])[0])
    verdict = demo._verdict_on("SOL", pixel, pose, court_surfaces(Court()))
    assert verdict.surface == "floor"
    assert verdict.point == pytest.approx(BOUNCE, abs=0.01)


def test_a_wall_contact_is_placed_on_the_first_wall_the_ray_reaches(synthetic_pose):
    pose, _, _ = synthetic_pose
    pixel = tuple(pose.project([[1.0, 10.0, 1.5]])[0])
    verdict = demo._verdict_on("VITRE", pixel, pose, court_surfaces(Court()))
    assert verdict.surface == "back_wall_positive_y" and verdict.material == "verre"
    assert verdict.point == pytest.approx((1.0, 10.0, 1.5), abs=0.01)


def test_a_ray_that_meets_no_wall_gives_no_verdict(synthetic_pose):
    pose, _, _ = synthetic_pose
    pixel = tuple(pose.project([BOUNCE])[0])
    assert demo._verdict_on("VITRE", pixel, pose, court_surfaces(Court())) is None


def test_detected_people_are_placed_on_the_court(scene):
    _, calibration, detections = scene
    picture = np.zeros((1080, 1920, 3), dtype=np.uint8)
    observed = demo.observe(detections, picture, Calibration.load(calibration), AnkleMidpoint())
    np.testing.assert_allclose([o.court_xy for o in observed], STANDING, atol=0.01)
    assert all(o.confidence == pytest.approx(0.9) for o in observed)


def test_retracking_keeps_the_detections_and_names_the_four_players(scene, synthetic_pose):
    video, calibration, detections = scene
    analysis = _analysis(synthetic_pose[0], detections)
    again = demo.retrack(analysis, video, Calibration.load(calibration))
    assert analysis["frames"][3]["assignment"] == {}  # the original analysis is not touched
    for frame in again["frames"].values():
        assert frame["people"] is detections
        assert sorted(frame["assignment"]) == ["far_1", "far_2", "near_1", "near_2"]
        assert all((y < 0) == slot.startswith("near") for slot, (_, y) in frame["positions"].items())


def test_the_rendered_video_has_every_frame_and_the_side_panel(tmp_path, scene, synthetic_pose):
    video, calibration, detections = scene
    pose = synthetic_pose[0]
    analysis = demo.retrack(_analysis(pose, detections), video, Calibration.load(calibration))
    pixel = tuple(pose.project([BOUNCE])[0])
    verdict = demo._verdict_on("SOL", pixel, pose, court_surfaces(Court()))
    events = [ContactEvent(10, "SOL", pixel, zone_of(verdict, Court())),
              ContactEvent(12, "RAQUETTE", pixel, None, detections[0].bbox)]
    drawn = {f: (v["raw"][0].x, v["raw"][0].y) for f, v in analysis["frames"].items()}
    out = tmp_path / "demo.mp4"
    demo.render(video, analysis, events, pose, drawn, out, 0, SCENE_FRAMES - 1,
                side=lambda frame: np.full((1080, 100, 3), 40, dtype=np.uint8))
    assert _frames_of(out) == (SCENE_FRAMES, (2020, 1080))


def test_a_replay_is_drawn_on_a_backdrop_instead_of_the_footage(tmp_path, scene, synthetic_pose):
    video, calibration, detections = scene
    analysis = demo.retrack(_analysis(synthetic_pose[0], detections), video,
                            Calibration.load(calibration))
    backdrop = np.full((1080, 1920, 3), 90, dtype=np.uint8)
    out = tmp_path / "replay.mp4"
    demo.render(video, analysis, [], synthetic_pose[0], {}, out, 0, 4, minimap=False,
                backdrop=backdrop)
    assert _frames_of(out) == (5, (1920, 1080))
    replay = cv2.VideoCapture(str(out))
    _, first = replay.read()
    replay.release()
    assert first[5, 5].mean() == pytest.approx(90, abs=4)  # the backdrop, not the black frame


def test_the_demo_is_rendered_again_from_a_saved_analysis(tmp_path, monkeypatch, capsys, scene,
                                                         synthetic_pose):
    video, calibration, detections = scene
    out = tmp_path / "demo.mp4"
    analysis = _analysis(synthetic_pose[0], detections)
    out.with_name("demo_analysis.pkl").write_bytes(pickle.dumps(analysis))
    monkeypatch.setattr("sys.argv", ["demo", "--video", str(video), "--calibration",
                                     str(calibration), "--weights", "unused.pt", "--reuse",
                                     "--out", str(out)])
    demo.main()
    assert _frames_of(out) == (SCENE_FRAMES, (1920, 1080))
    assert "contacts shown" in capsys.readouterr().out
