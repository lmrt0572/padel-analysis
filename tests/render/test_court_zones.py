import numpy as np

from padel_analysis.contact.surfaces import RACKET, Verdict
from padel_analysis.geometry.court import Court
from padel_analysis.render.court_zones import zone_of


def _v(surface, point, material=None):
    return Verdict(surface, np.array(point, dtype=float), material, candidates=1)


def test_a_floor_bounce_lights_its_service_box():
    zone = zone_of(_v("floor", (2.0, 3.0, 0.0)), Court())
    assert zone.name == "sol_eloigne_service_droite"


def test_a_floor_bounce_behind_the_service_line_lights_the_back_zone():
    assert zone_of(_v("floor", (-2.0, -8.5, 0.0)), Court()).name == "sol_proche_fond"


def test_a_back_wall_contact_lights_that_wall_in_its_material():
    zone = zone_of(_v("back_wall_positive_y", (1.0, 10.0, 3.5), "grillage"), Court())
    assert zone.name == "fond_eloigne_grillage"


def test_a_side_wall_contact_lights_the_panel_it_hit():
    court = Court()
    near_end = zone_of(_v("side_wall_negative_x", (-5.0, -8.0, 1.0), "verre"), court)
    middle = zone_of(_v("side_wall_negative_x", (-5.0, 1.0, 1.0), "grillage"), court)
    assert near_end.name == "cote_gauche_proche_verre"
    assert middle.name == "cote_gauche_grillage"


def test_the_net_lights_the_net():
    assert zone_of(_v("net", (0.0, 0.0, 0.5)), Court()).name == "filet"


def test_a_racket_lights_no_zone():
    """Une raquette n'est pas une surface du court."""
    assert zone_of(Verdict(RACKET, None, None, candidates=0), Court()) is None


def test_an_undecided_contact_lights_no_zone():
    assert zone_of(Verdict(None, None, None, candidates=0), Court()) is None
