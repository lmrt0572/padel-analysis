import cv2
import pytest

from padel_analysis import cli
from padel_analysis.pipeline.cache import PositionCache

from .conftest import SCENE_FRAMES, STANDING


def test_the_parser_reads_every_frame_from_the_start_by_default():
    args = cli.build_parser().parse_args(["--video", "v.mp4", "--calibration", "c.json",
                                          "--cache", "cache.json"])
    assert (args.start, args.frames, args.ground_point) == (0, None, "ankles")
    assert not args.no_video and args.out is None


def test_a_video_is_asked_for_without_a_place_to_write_it(monkeypatch):
    monkeypatch.setattr("sys.argv", ["cli", "--video", "v.mp4", "--calibration", "c.json",
                                     "--cache", "cache.json"])
    with pytest.raises(SystemExit, match="--out"):
        cli.main()


def test_the_pipeline_places_four_players_and_writes_the_video(tmp_path, monkeypatch, scene):
    video, calibration, detections = scene

    class Detector:
        def __init__(self, imgsz):
            self.imgsz = imgsz

        def detect(self, frame):
            return detections

    monkeypatch.setattr(cli, "PoseDetector", Detector)
    out, cache_path = tmp_path / "out" / "annotated.mp4", tmp_path / "cache.json"
    monkeypatch.setattr("sys.argv", ["cli", "--video", str(video), "--calibration",
                                     str(calibration), "--out", str(out), "--cache",
                                     str(cache_path), "--start", "1", "--frames", "4"])
    cli.main()

    cache = PositionCache.load(cache_path)
    assert cache.frames() == [1, 2, 3, 4]
    for frame in cache.frames():
        positions = cache.at(frame).positions
        assert sorted(positions) == ["far_1", "far_2", "near_1", "near_2"]
        found = sorted((round(x, 1), round(y, 1)) for x, y in positions.values())
        assert found == sorted(STANDING)
        assert all((y < 0) == slot.startswith("near") for slot, (_, y) in positions.items())
    written = cv2.VideoCapture(str(out))
    assert int(written.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    written.release()


def test_the_cache_alone_is_written_when_no_video_is_wanted(tmp_path, monkeypatch, scene):
    video, calibration, detections = scene

    class Detector:
        def __init__(self, imgsz):
            pass

        def detect(self, frame):
            return detections[:2]

    monkeypatch.setattr(cli, "PoseDetector", Detector)
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr("sys.argv", ["cli", "--video", str(video), "--calibration",
                                     str(calibration), "--cache", str(cache_path),
                                     "--no-video", "--ground-point", "bbox"])
    cli.main()

    cache = PositionCache.load(cache_path)
    assert cache.frames() == list(range(SCENE_FRAMES))
    assert all(len(cache.at(f).positions) == 2 for f in cache.frames())
    assert [p.name for p in tmp_path.glob("*.mp4")] == ["scene.mp4"]
