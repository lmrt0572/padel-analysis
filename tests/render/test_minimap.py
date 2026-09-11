import numpy as np

from padel_analysis.geometry.court import Court
from padel_analysis.render.minimap import Minimap


def test_minimap_has_the_requested_width_and_a_court_aspect_ratio():
    minimap = Minimap(Court(), width=300)
    image = minimap.draw({})
    assert image.shape[1] == 300
    # 20 m de long pour 10 m de large : l'image doit etre environ deux fois plus haute.
    assert 1.7 < image.shape[0] / image.shape[1] < 2.5


def test_the_net_centre_lands_in_the_middle_of_the_image():
    minimap = Minimap(Court(), width=300)
    x, y = minimap.to_pixels(np.array([0.0, 0.0]))
    image = minimap.draw({})
    assert abs(x - image.shape[1] / 2) < 2
    assert abs(y - image.shape[0] / 2) < 2


def test_positive_y_is_drawn_towards_the_top():
    """Meme piege d'orientation qu'au jalon 1 : un aller-retour ne le detecte pas."""
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
    """Les joueurs de padel sortent parfois du court par les ouvertures laterales."""
    minimap = Minimap(Court(), width=300)
    image = minimap.draw({"near_1": (-8.0, -13.0)})
    assert image.shape[1] == 300
