import numpy as np

from padel_analysis.contact.surfaces import RACKET, Verdict
from padel_analysis.perception.pose_detector import PersonDetection
from padel_analysis.render.ball_overlay import (
    ContactEvent,
    contact_label,
    draw_ball,
    draw_hitter,
    following_box,
    hitter_box,
    trail,
    visible_events,
)


def _verdict(surface, material=None):
    return Verdict(surface, None, material, candidates=1)


def test_each_surface_gets_its_label():
    assert contact_label(_verdict(RACKET)) == "RAQUETTE"
    assert contact_label(_verdict("floor")) == "SOL"
    assert contact_label(_verdict("net")) == "FILET"
    assert contact_label(_verdict("back_wall_negative_y", "verre")) == "VITRE"
    assert contact_label(_verdict("side_wall_positive_x", "grillage")) == "GRILLAGE"


def test_an_undecided_contact_gets_no_label():
    """Un contact sans surface admissible n'est pas affiche plutot qu'affiche au hasard."""
    assert contact_label(_verdict(None)) is None


def test_the_trail_ends_on_the_current_frame():
    path = {f: (float(f), 0.0) for f in range(10)}
    assert trail(path, 9, length=3) == [(7.0, 0.0), (8.0, 0.0), (9.0, 0.0)]


def test_the_trail_skips_frames_without_a_ball():
    path = {7: (7.0, 0.0), 8: None, 9: (9.0, 0.0)}
    assert trail(path, 9, length=3) == [(7.0, 0.0), (9.0, 0.0)]


def test_the_trail_never_reaches_into_the_future():
    path = {f: (float(f), 0.0) for f in range(10)}
    assert (9.0, 0.0) not in trail(path, 5, length=3)


def test_a_contact_stays_on_screen_for_its_hold():
    event = ContactEvent(frame=100, label="SOL", pixel=(1.0, 1.0))
    assert visible_events([event], 99, hold=20) == []
    assert visible_events([event], 100, hold=20) == [event]
    assert visible_events([event], 119, hold=20) == [event]
    assert visible_events([event], 120, hold=20) == []


def test_drawing_leaves_the_original_frame_untouched():
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    event = ContactEvent(frame=5, label="VITRE", pixel=(40.0, 40.0))
    drawn = draw_ball(frame, [(10.0, 10.0), (20.0, 20.0)], [event])
    assert frame.sum() == 0
    assert drawn.sum() > 0


def _person(box, wrist):
    keypoints = np.zeros((17, 3))
    keypoints[9] = (wrist[0], wrist[1], 0.9)
    return PersonDetection(bbox=np.array(box, dtype=float), confidence=0.9, keypoints=keypoints)


def test_the_hitter_is_the_player_whose_wrist_is_nearest():
    near = _person((0, 0, 10, 10), (100.0, 100.0))
    far = _person((50, 50, 60, 60), (400.0, 400.0))
    assert tuple(hitter_box((105.0, 100.0), [far, near])) == (0, 0, 10, 10)


def test_no_visible_wrist_names_no_hitter():
    hidden = _person((0, 0, 10, 10), (100.0, 100.0))
    hidden.keypoints[9, 2] = 0.0
    assert hitter_box((100.0, 100.0), [hidden]) is None


def test_a_fresh_hit_lights_the_box_more_than_a_fading_one():
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    box = np.array([20.0, 20.0, 80.0, 70.0])
    assert frame.sum() == 0
    assert draw_hitter(frame, box, 1.0).sum() > draw_hitter(frame, box, 0.2).sum() > 0


def test_the_lit_box_follows_the_player_who_moved():
    box = np.array([100.0, 100.0, 150.0, 200.0])
    moved = _person((130.0, 100.0, 180.0, 200.0), (140.0, 150.0))
    assert following_box(box, [moved]) is moved.bbox


def test_the_lit_box_stays_put_when_nobody_is_near():
    box = np.array([100.0, 100.0, 150.0, 200.0])
    far = _person((900.0, 100.0, 950.0, 200.0), (910.0, 150.0))
    np.testing.assert_array_equal(following_box(box, [far]), box)


def test_the_lit_box_stays_put_without_any_detection():
    box = np.array([100.0, 100.0, 150.0, 200.0])
    np.testing.assert_array_equal(following_box(box, []), box)
