import pytest

from padel_analysis.ball.contacts import (
    Contact,
    find_contacts,
    sharpness_of,
    turn_of,
    velocities,
)


def test_velocities_reads_both_sides_of_a_frame():
    path = {0: (0.0, 0.0), 2: (10.0, 0.0), 4: (30.0, 0.0)}
    incoming, outgoing = velocities(path, 2, span=2)
    assert incoming == pytest.approx((10.0, 0.0))
    assert outgoing == pytest.approx((20.0, 0.0))


def test_velocities_needs_all_three_points():
    path = {0: (0.0, 0.0), 2: (10.0, 0.0)}
    assert velocities(path, 2, span=2) is None


def test_velocities_refuses_an_absent_neighbour():
    """A frame without a position cannot carry a velocity."""
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
    """A ball that goes right, bounces at frame 20, and comes back.

    The step of 30 px is chosen so that the frames next to the vertex also pass the
    floor added in task 3, without which the suppression would no longer be tested
    afterwards.
    """
    path = {f: (f * 30.0, 0.0) for f in range(21)}
    for f in range(21, 41):
        path[f] = (600.0 - (f - 20) * 30.0, 0.0)
    return path


def _two_bounces():
    """The same one, bouncing a second time at frame 40."""
    path = _bounce()
    for f in range(41, 61):
        path[f] = ((f - 40) * 30.0, 0.0)
    return path


def test_a_straight_path_holds_no_contact():
    assert find_contacts({f: (f * 30.0, 0.0) for f in range(40)}) == []


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
    """Two real contacts cannot be one frame apart."""
    assert len(find_contacts(_bounce(), suppression=5)) == 1


def test_suppression_can_be_widened_to_keep_fewer():
    assert [c.frame for c in find_contacts(_two_bounces(), suppression=5)] == [20, 40]
    assert len(find_contacts(_two_bounces(), suppression=100)) == 1


def test_a_gentler_bend_needs_a_lower_threshold():
    path = {0: (0.0, 0.0), 2: (60.0, 0.0), 4: (120.0, 30.0)}
    assert find_contacts(path, sharpness=0.9) == []
    assert len(find_contacts(path, sharpness=0.05)) == 1


def test_a_microscopic_wobble_is_not_a_contact():
    """A turn of 2 px is noise, even if it reverses the velocity."""
    path = {0: (0.0, 0.0), 2: (1.0, 0.0), 4: (0.0, 0.0)}
    assert find_contacts(path, floor=25.0) == []


def test_the_floor_can_be_lowered_to_accept_it():
    path = {0: (0.0, 0.0), 2: (1.0, 0.0), 4: (0.0, 0.0)}
    assert len(find_contacts(path, floor=0.5)) == 1


def test_an_impossible_jump_is_not_a_contact():
    """A ball does not jump 600 px in two frames: it is the path that failed."""
    path = {0: (0.0, 0.0), 2: (600.0, 0.0), 4: (0.0, 0.0)}
    assert find_contacts(path, ceiling=300.0) == []


def test_the_ceiling_can_be_raised_to_accept_it():
    path = {0: (0.0, 0.0), 2: (600.0, 0.0), 4: (0.0, 0.0)}
    assert len(find_contacts(path, ceiling=10_000.0)) == 1


def test_a_real_bounce_sits_inside_the_bracket():
    path = {0: (0.0, 0.0), 2: (60.0, 0.0), 4: (0.0, 0.0)}
    assert [c.frame for c in find_contacts(path, floor=25.0, ceiling=300.0)] == [2]


def test_a_contact_on_a_splice_is_dropped():
    assert find_contacts(_bounce(), cuts=[20]) == []


def test_a_splice_only_shadows_its_own_window():
    assert [c.frame for c in find_contacts(_bounce(), cuts=[200])] == [20]


def test_the_shadow_covers_the_velocity_window():
    """A cut contaminates the frames whose velocity window straddles it.

    A cut at 22 straddles the window of frame 20, which goes from 18 to 22. The vertex
    of the bounce therefore disappears; its flank at 19, whose window stops at 21, stays.
    """
    assert 20 not in [c.frame for c in find_contacts(_bounce(), span=2, cuts=[22])]
    assert [c.frame for c in find_contacts(_bounce(), span=2, cuts=[23])] == [20]


def test_a_contact_is_the_type_it_claims_to_be():
    assert all(isinstance(c, Contact) for c in find_contacts(_two_bounces()))
