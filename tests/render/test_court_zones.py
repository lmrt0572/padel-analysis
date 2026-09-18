import numpy as np

from padel_analysis.contact.surfaces import RACKET, Verdict
from padel_analysis.geometry.court import Court
from padel_analysis.render.court_zones import impact_patch, zone_of


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


def test_a_floor_zone_lies_on_the_ground():
    zone = zone_of(_v("floor", (2.0, 3.0, 0.0)), Court())
    assert all(corner[2] == 0.0 for corner in zone.corners)


def test_a_back_wall_mesh_panel_sits_above_the_glass():
    zone = zone_of(_v("back_wall_positive_y", (1.0, 10.0, 3.5), "grillage"), Court())
    assert sorted({corner[2] for corner in zone.corners}) == [3.0, 4.0]
    assert all(corner[1] == 10.0 for corner in zone.corners)


def test_a_side_glass_panel_spans_the_end_of_its_wall():
    zone = zone_of(_v("side_wall_negative_x", (-5.0, -8.0, 1.0), "verre"), Court())
    assert {corner[0] for corner in zone.corners} == {-5.0}
    assert sorted({corner[1] for corner in zone.corners}) == [-10.0, -5.9]


def test_lighting_a_zone_draws_on_a_copy(synthetic_pose):
    from padel_analysis.render.court_zones import draw_zone

    pose, _, _ = synthetic_pose
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    zone = zone_of(_v("floor", (2.0, 3.0, 0.0)), Court())
    lit = draw_zone(frame, zone, pose, strength=1.0)
    assert frame.sum() == 0
    assert lit.sum() > 0


def test_a_faded_zone_lights_less(synthetic_pose):
    from padel_analysis.render.court_zones import draw_zone

    pose, _, _ = synthetic_pose
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    zone = zone_of(_v("floor", (2.0, 3.0, 0.0)), Court())
    assert draw_zone(frame, zone, pose, 1.0).sum() > draw_zone(frame, zone, pose, 0.3).sum()


def test_the_near_back_wall_lights_only_its_foot():
    """La camera est derriere : le panneau entier couvrirait la moitie de l'image."""
    zone = zone_of(_v("back_wall_negative_y", (0.0, -10.0, 1.5), "verre"), Court())
    assert max(corner[2] for corner in zone.corners) <= 0.5


def test_an_impact_patch_is_a_square_centred_on_the_bounce():
    patch = impact_patch(_v("floor", (2.0, 3.0, 0.0)), Court(), size=1.5)
    xs = [corner[0] for corner in patch.corners]
    ys = [corner[1] for corner in patch.corners]
    assert (min(xs), max(xs)) == (1.25, 2.75)
    assert (min(ys), max(ys)) == (2.25, 3.75)
    assert {corner[2] for corner in patch.corners} == {0.0}


def test_an_impact_patch_never_leaves_the_court():
    patch = impact_patch(_v("floor", (4.9, -9.8, 0.0)), Court(), size=1.5)
    court = Court()
    assert max(corner[0] for corner in patch.corners) == court.half_width
    assert min(corner[1] for corner in patch.corners) == -court.half_length


def test_a_wall_patch_stands_on_the_wall_at_the_impact_height():
    court = Court()
    patch = impact_patch(_v("side_wall_positive_x", (5.0, 2.0, 2.0)), court, size=1.5)
    assert {corner[0] for corner in patch.corners} == {court.half_width}
    assert min(corner[2] for corner in patch.corners) == 1.25
    assert max(corner[2] for corner in patch.corners) == 2.75


def test_a_wall_patch_is_kept_under_the_top_of_the_wall():
    court = Court()
    patch = impact_patch(_v("back_wall_positive_y", (0.0, 10.0, 3.9), "grillage"), court)
    assert max(corner[2] for corner in patch.corners) == court.back_wall_total_height


def test_a_racket_contact_lights_no_patch():
    assert impact_patch(Verdict(RACKET, None, None, 0), Court()) is None
