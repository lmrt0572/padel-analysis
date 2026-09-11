import numpy as np
import pytest

from padel_analysis.pipeline.cache import FramePositions, PositionCache


def test_cache_roundtrips_through_disk(tmp_path):
    cache = PositionCache()
    cache.add(FramePositions(frame=0, positions={"near_1": (1.0, -5.0)}))
    cache.add(FramePositions(frame=1, positions={"near_1": (1.1, -5.1)}))
    path = tmp_path / "positions.json"
    cache.save(path, video="match.mp4", calibration="cal.json")

    loaded = PositionCache.load(path)
    assert loaded.frames() == [0, 1]
    np.testing.assert_allclose(loaded.at(1).positions["near_1"], (1.1, -5.1))


def test_cache_records_its_provenance(tmp_path):
    cache = PositionCache()
    cache.add(FramePositions(frame=0, positions={}))
    path = tmp_path / "positions.json"
    cache.save(path, video="match.mp4", calibration="cal.json")

    loaded = PositionCache.load(path)
    assert loaded.video == "match.mp4"
    assert loaded.calibration == "cal.json"


def test_a_frame_may_hold_fewer_than_four_players(tmp_path):
    cache = PositionCache()
    cache.add(FramePositions(frame=0, positions={"near_1": (0.0, -1.0)}))
    path = tmp_path / "positions.json"
    cache.save(path, video="v", calibration="c")
    assert len(PositionCache.load(path).at(0).positions) == 1


def test_asking_for_an_absent_frame_raises(tmp_path):
    cache = PositionCache()
    cache.add(FramePositions(frame=0, positions={}))
    with pytest.raises(KeyError):
        cache.at(42)
