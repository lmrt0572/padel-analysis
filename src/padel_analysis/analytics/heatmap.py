"""Court occupancy, as a grid of time spent per cell.

Row 0 is the negative-y baseline, so the grid is indexed the way the court is
described rather than the way an image is drawn; the renderer flips it.
"""

import numpy as np

from ..geometry.court import Court


def occupancy_grid(
    positions: np.ndarray,
    court: Court,
    cell_size: float = 0.5,
    normalise: bool = True,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Histogram of court positions.

    Positions outside the court are dropped rather than clamped: padel players do
    leave through the side openings, and piling those frames onto an edge cell
    would invent an occupancy that never happened.

    Returns:
        The grid, shaped (rows along y, columns along x), and its extent as
        (x_min, x_max, y_min, y_max) in metres.
    """
    extent = (
        -court.half_width,
        court.half_width,
        -court.half_length,
        court.half_length,
    )
    columns = round(court.width / cell_size)
    rows = round(court.length / cell_size)

    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    if positions.size:
        positions = positions[~np.isnan(positions[:, 0])]

    ys = positions[:, 1] if positions.size else np.zeros(0)
    xs = positions[:, 0] if positions.size else np.zeros(0)

    grid, _, _ = np.histogram2d(
        ys,
        xs,
        bins=(rows, columns),
        range=((extent[2], extent[3]), (extent[0], extent[1])),
    )

    if normalise:
        total = grid.sum()
        if total > 0:
            grid = grid / total
    return grid, extent
