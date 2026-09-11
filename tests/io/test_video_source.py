import numpy as np
import pytest

from padel_analysis.io.video_source import VideoMetadata, VideoSource


@pytest.fixture
def tiny_video(tmp_path):
    """Une video synthetique de 10 frames, chacune remplie de sa valeur d'index."""
    import cv2

    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    return path


def test_metadata_reports_size_and_rate(tiny_video):
    with VideoSource(tiny_video) as source:
        meta = source.metadata
    assert isinstance(meta, VideoMetadata)
    assert meta.width == 64
    assert meta.height == 48
    assert meta.fps == pytest.approx(30.0, abs=0.5)
    assert meta.frame_count == 10


def test_read_returns_a_frame_of_the_declared_size(tiny_video):
    with VideoSource(tiny_video) as source:
        frame = source.read(3)
    assert frame.shape == (48, 64, 3)


def test_reading_the_same_index_twice_gives_the_same_frame(tiny_video):
    with VideoSource(tiny_video) as source:
        a = source.read(5)
        b = source.read(5)
    np.testing.assert_array_equal(a, b)


def test_iteration_yields_every_frame_in_order(tiny_video):
    with VideoSource(tiny_video) as source:
        indices = [index for index, _ in source.iter_frames()]
    assert indices == list(range(10))


def test_iteration_can_start_and_stop_at_given_indices(tiny_video):
    with VideoSource(tiny_video) as source:
        indices = [index for index, _ in source.iter_frames(start=2, stop=5)]
    assert indices == [2, 3, 4]


def test_reading_past_the_end_raises(tiny_video):
    with VideoSource(tiny_video) as source:
        with pytest.raises(IndexError):
            source.read(999)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoSource(tmp_path / "absent.mp4").open()
