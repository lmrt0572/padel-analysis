import numpy as np

from padel_analysis.geometry.court import Court
from padel_analysis.render.minimap import Minimap


def test_minimap_has_the_requested_width_and_a_court_aspect_ratio():
    minimap = Minimap(Court(), width=300)
    image = minimap.draw({})
    assert image.shape[1] == 300
    # 20 m long for 10 m wide: the image must be about twice as tall.
    assert 1.7 < image.shape[0] / image.shape[1] < 2.5


def test_the_net_centre_lands_in_the_middle_of_the_image():
    minimap = Minimap(Court(), width=300)
    x, y = minimap.to_pixels(np.array([0.0, 0.0]))
    image = minimap.draw({})
    assert abs(x - image.shape[1] / 2) < 2
    assert abs(y - image.shape[0] / 2) < 2


def test_positive_y_is_drawn_towards_the_top():
    """Same orientation trap as in milestone 1: a round trip does not detect it."""
    minimap = Minimap(Court(), width=300)
    _, near = minimap.to_pixels(np.array([0.0, -10.0]))
    _, far = minimap.to_pixels(np.array([0.0, 10.0]))
    assert far < near


def test_positive_x_is_drawn_to_the_right():
    minimap = Minimap(Court(), width=300)
    left, _ = minimap.to_pixels(np.array([-5.0, 0.0]))
    right, _ = minimap.to_pixels(np.array([5.0, 0.0]))
    assert right > left


def test_drawing_players_changes_the_image():
    minimap = Minimap(Court(), width=300)
    empty = minimap.draw({})
    populated = minimap.draw({"near_1": (0.0, -5.0), "far_1": (0.0, 5.0)})
    assert not np.array_equal(empty, populated)


def test_a_player_outside_the_court_does_not_crash():
    """Padel players sometimes leave the court through the side openings."""
    minimap = Minimap(Court(), width=300)
    image = minimap.draw({"near_1": (-8.0, -13.0)})
    assert image.shape[1] == 300


def test_a_lit_zone_is_drawn_on_the_minimap():
    from padel_analysis.render.court_zones import FLOOR, Zone

    minimap = Minimap(Court())
    zone = Zone("sol", "area", ((0.0, 0.0), (5.0, 0.0), (5.0, 6.95), (0.0, 6.95)), FLOOR)
    assert not np.array_equal(minimap.draw({}), minimap.draw({}, lit=[(zone, 1.0)]))


def test_a_faded_zone_is_fainter_than_a_fresh_one():
    from padel_analysis.render.court_zones import GLASS, Zone

    minimap = Minimap(Court())
    wall = Zone("fond", "line", ((-5.0, 10.0), (5.0, 10.0)), GLASS)
    bare = minimap.draw({}).astype(int)
    fresh = np.abs(minimap.draw({}, lit=[(wall, 1.0)]).astype(int) - bare).sum()
    faded = np.abs(minimap.draw({}, lit=[(wall, 0.2)]).astype(int) - bare).sum()
    assert fresh > faded > 0


def test_the_minimap_draws_the_same_with_nothing_lit():
    minimap = Minimap(Court())
    assert np.array_equal(minimap.draw({}), minimap.draw({}, lit=[]))
