import numpy as np
import pytest

from padel_analysis.analytics.heatmap import occupancy_grid
from padel_analysis.geometry.court import Court


def test_grid_covers_the_court_at_the_requested_resolution():
    grid, extent = occupancy_grid(np.zeros((0, 2)), Court(), cell_size=0.5)
    assert grid.shape == (40, 20)  # 20 m by 10 m, cells of 0.5 m
    assert extent == pytest.approx((-5.0, 5.0, -10.0, 10.0))


def test_a_single_position_lands_in_one_cell():
    positions = np.array([[0.0, 0.0]])
    grid, _ = occupancy_grid(positions, Court(), cell_size=0.5)
    assert grid.sum() == pytest.approx(1.0)


def test_the_grid_is_normalised_to_sum_to_one():
    positions = np.array([[0.0, 0.0], [1.0, 1.0], [-2.0, -3.0]])
    grid, _ = occupancy_grid(positions, Court(), cell_size=0.5, normalise=True)
    assert grid.sum() == pytest.approx(1.0)


def test_absent_positions_are_dropped():
    positions = np.array([[0.0, 0.0], [np.nan, np.nan]])
    grid, _ = occupancy_grid(positions, Court(), cell_size=0.5, normalise=False)
    assert grid.sum() == pytest.approx(1.0)


def test_positions_outside_the_court_are_dropped_not_clamped():
    """Exits through the side openings must not pile up on an edge."""
    positions = np.array([[0.0, 0.0], [-8.0, -13.0]])
    grid, _ = occupancy_grid(positions, Court(), cell_size=0.5, normalise=False)
    assert grid.sum() == pytest.approx(1.0)


def test_a_player_at_the_far_end_lands_in_the_far_rows():
    positions = np.array([[0.0, 9.0]])
    grid, _ = occupancy_grid(positions, Court(), cell_size=0.5, normalise=False)
    rows = np.nonzero(grid.sum(axis=1))[0]
    assert rows[0] > grid.shape[0] * 0.8


def test_an_empty_input_gives_an_empty_grid_rather_than_nan():
    grid, _ = occupancy_grid(np.zeros((0, 2)), Court(), cell_size=0.5, normalise=True)
    assert grid.sum() == pytest.approx(0.0)
