import pytest

from padel_analysis.ball.contacts import find_contacts, sharpness_of, turn_of, velocities


def test_velocities_reads_both_sides_of_a_frame():
    path = {0: (0.0, 0.0), 2: (10.0, 0.0), 4: (30.0, 0.0)}
    incoming, outgoing = velocities(path, 2, span=2)
    assert incoming == pytest.approx((10.0, 0.0))
    assert outgoing == pytest.approx((20.0, 0.0))


def test_velocities_needs_all_three_points():
    path = {0: (0.0, 0.0), 2: (10.0, 0.0)}
    assert velocities(path, 2, span=2) is None


def test_velocities_refuses_an_absent_neighbour():
    """Une frame sans position ne peut pas porter une vitesse."""
    path = {0: None, 2: (10.0, 0.0), 4: (30.0, 0.0)}
    assert velocities(path, 2, span=2) is None


def test_a_straight_line_does_not_turn():
    assert turn_of((10.0, 0.0), (10.0, 0.0)) == pytest.approx(0.0)


def test_turn_is_the_length_of_the_velocity_change():
    assert turn_of((10.0, 0.0), (10.0, 6.0)) == pytest.approx(6.0)


def test_a_full_reversal_turns_by_both_speeds():
    assert turn_of((10.0, 0.0), (-10.0, 0.0)) == pytest.approx(20.0)


def test_sharpness_is_scale_free():
    """Le meme demi-tour vaut pareil a vitesse lente et rapide."""
    slow = sharpness_of((5.0, 0.0), (-5.0, 0.0))
    fast = sharpness_of((50.0, 0.0), (-50.0, 0.0))
    assert slow == pytest.approx(fast)
    assert slow == pytest.approx(1.0)


def test_sharpness_of_a_straight_line_is_zero():
    assert sharpness_of((10.0, 0.0), (10.0, 0.0)) == pytest.approx(0.0)


def test_sharpness_refuses_to_divide_by_nothing():
    assert sharpness_of((0.0, 0.0), (0.0, 0.0)) == pytest.approx(0.0)


def _bounce():
    """Une balle qui part a droite, rebondit a la frame 20, et revient.

    Le pas de 30 px est choisi pour que les frames voisines du sommet passent elles
    aussi le plancher ajoute a la tache 3 - sans quoi la suppression ne serait plus
    testee par la suite.
    """
    path = {f: (f * 30.0, 0.0) for f in range(0, 21)}
    for f in range(21, 41):
        path[f] = (600.0 - (f - 20) * 30.0, 0.0)
    return path


def _two_bounces():
    """La meme, qui rebondit une seconde fois a la frame 40."""
    path = _bounce()
    for f in range(41, 61):
        path[f] = ((f - 40) * 30.0, 0.0)
    return path


def test_a_straight_path_holds_no_contact():
    assert find_contacts({f: (f * 30.0, 0.0) for f in range(0, 40)}) == []


def test_a_bounce_is_found_at_its_frame():
    assert [c.frame for c in find_contacts(_bounce())] == [20]


def test_a_contact_carries_both_directions():
    contact = find_contacts(_bounce())[0]
    assert contact.incoming[0] > 0.0
    assert contact.outgoing[0] < 0.0


def test_a_contact_reports_the_size_of_its_bend():
    contact = find_contacts(_bounce())[0]
    assert contact.turn == pytest.approx(120.0)
    assert contact.sharpness == pytest.approx(1.0)


def test_only_the_strongest_of_a_burst_survives():
    """Deux contacts reels ne peuvent pas etre a une frame d'ecart."""
    assert len(find_contacts(_bounce(), suppression=5)) == 1


def test_suppression_can_be_widened_to_keep_fewer():
    assert [c.frame for c in find_contacts(_two_bounces(), suppression=5)] == [20, 40]
    assert len(find_contacts(_two_bounces(), suppression=100)) == 1


def test_a_gentler_bend_needs_a_lower_threshold():
    path = {0: (0.0, 0.0), 2: (60.0, 0.0), 4: (120.0, 30.0)}
    assert find_contacts(path, sharpness=0.9) == []
    assert len(find_contacts(path, sharpness=0.05)) == 1
