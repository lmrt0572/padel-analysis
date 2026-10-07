import json

import numpy as np
import pytest
import torch

from padel_analysis.ball.heatmap_net import (
    BallHeatmapNet,
    NetCandidates,
    meta_path,
    peaks_of,
)


def test_the_net_answers_at_the_input_resolution():
    out = BallHeatmapNet(width=8)(torch.zeros(2, 9, 72, 128))
    assert out.shape == (2, 1, 72, 128)


def test_the_net_stays_within_its_measured_size():
    """Width 16 fits in 1.82 GB; beyond that the probe had saturated the card."""
    assert sum(p.numel() for p in BallHeatmapNet(width=16).parameters()) < 1_000_000


def test_the_same_input_gives_the_same_answer():
    net = BallHeatmapNet(width=8).eval()
    x = torch.randn(1, 9, 72, 128)
    with torch.no_grad():
        assert torch.allclose(net(x), net(x))


def test_the_net_reads_three_stacked_colour_frames():
    net = BallHeatmapNet(width=8, stacked=3)
    assert net(torch.zeros(1, 9, 72, 128)).shape[1] == 1


def test_peaks_are_ranked_strongest_first():
    heatmap = np.zeros((32, 64), dtype=np.float32)
    heatmap[10, 20] = 0.9
    heatmap[25, 50] = 0.5
    found = peaks_of(heatmap, threshold=0.1, suppression=3)
    assert [(c.x, c.y) for c in found] == [(20.0, 10.0), (50.0, 25.0)]
    assert found[0].score > found[1].score


def test_a_flat_heatmap_holds_no_peak():
    assert peaks_of(np.zeros((32, 64), dtype=np.float32), threshold=0.1) == []


def test_a_peak_below_the_threshold_is_dropped():
    heatmap = np.zeros((32, 64), dtype=np.float32)
    heatmap[10, 20] = 0.05
    assert peaks_of(heatmap, threshold=0.1) == []


def test_one_blob_yields_one_candidate():
    """A wide blob must not return thirty neighbouring candidates."""
    heatmap = np.zeros((32, 64), dtype=np.float32)
    heatmap[10:14, 20:24] = 0.8
    assert len(peaks_of(heatmap, threshold=0.1, suppression=6)) == 1


def test_peaks_are_reported_in_the_original_scale():
    heatmap = np.zeros((32, 64), dtype=np.float32)
    heatmap[10, 20] = 0.9
    found = peaks_of(heatmap, threshold=0.1, scale=(3.0, 3.0))
    assert (found[0].x, found[0].y) == (60.0, 30.0)


def test_the_number_of_candidates_is_capped():
    rng = np.random.default_rng(0)
    heatmap = rng.random((64, 128)).astype(np.float32)
    assert len(peaks_of(heatmap, threshold=0.1, suppression=1, limit=5)) == 5


def _weights(tmp_path, width, meta=None):
    path = tmp_path / "net.pt"
    torch.save(BallHeatmapNet(width=width).state_dict(), path)
    if meta is not None:
        meta_path(path).write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_the_best_weights_find_the_settings_of_their_run(tmp_path):
    assert meta_path(tmp_path / "net_best.pt") == meta_path(tmp_path / "net.pt")


def test_weights_are_read_at_the_resolution_they_were_trained(tmp_path):
    """A 720p model read at 360p would return wrong candidates without saying anything."""
    path = _weights(tmp_path, 8, {"size": [1280, 720], "width": 8, "spacing": 3})
    finder = NetCandidates(path, spacing=3, device="cpu")
    assert finder.size == (1280, 720)


def test_weights_refuse_a_spacing_they_were_not_trained_with(tmp_path):
    path = _weights(tmp_path, 8, {"size": [640, 360], "width": 8, "spacing": 3})
    with pytest.raises(ValueError):
        NetCandidates(path, spacing=2, device="cpu")


def test_weights_without_settings_keep_the_first_resolution(tmp_path):
    finder = NetCandidates(_weights(tmp_path, 16), spacing=3, device="cpu")
    assert finder.size == (640, 360)


def test_a_size_the_network_cannot_halve_three_times_is_refused(tmp_path):
    path = _weights(tmp_path, 8, {"size": [960, 540], "width": 8, "spacing": 3})
    with pytest.raises(ValueError):
        NetCandidates(path, spacing=3, device="cpu")


def test_suppression_scales_with_the_resolution(tmp_path):
    small = NetCandidates(_weights(tmp_path, 16), spacing=3, device="cpu")
    path = _weights(tmp_path, 8, {"size": [1280, 720], "width": 8, "spacing": 3})
    large = NetCandidates(path, spacing=3, device="cpu")
    assert large.suppression == 2 * small.suppression
