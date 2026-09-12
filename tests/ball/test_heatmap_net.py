import numpy as np
import torch

from padel_analysis.ball.heatmap_net import BallHeatmapNet, peaks_of


def test_the_net_answers_at_the_input_resolution():
    out = BallHeatmapNet(width=8)(torch.zeros(2, 9, 72, 128))
    assert out.shape == (2, 1, 72, 128)


def test_the_net_stays_within_its_measured_size():
    """La largeur 16 tient dans 1,82 Go ; au-dela la sonde avait sature la carte."""
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
    """Une tache large ne doit pas rendre trente candidats voisins."""
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
