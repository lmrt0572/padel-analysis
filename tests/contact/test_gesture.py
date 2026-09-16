import numpy as np

from padel_analysis.contact.gesture import gesture_near, strikes
from padel_analysis.perception.pose_detector import PersonDetection


def _person(wrist, box=(0.0, 0.0, 50.0, 100.0)):
    keypoints = np.zeros((17, 3))
    keypoints[9] = (wrist[0], wrist[1], 0.9)
    return PersonDetection(bbox=np.array(box), confidence=0.9, keypoints=keypoints)


def _players(wrist_x_of_frame):
    """Un seul joueur, emplacement near_1, dont le poignet se deplace le long de x."""
    return {f: ([_person((x, 50.0))], {"near_1": 0}) for f, x in wrist_x_of_frame.items()}


def test_a_still_wrist_makes_no_gesture():
    players = _players({f: 100.0 for f in range(10)})
    assert gesture_near(players, 5, (100.0, 50.0), reach=80.0) == 0.0


def test_a_moving_wrist_near_the_ball_makes_a_gesture():
    players = _players({f: 100.0 + 20.0 * f for f in range(10)})
    assert gesture_near(players, 5, (200.0, 50.0), reach=80.0) == 20.0


def test_a_moving_wrist_far_from_the_ball_makes_no_gesture():
    players = _players({f: 100.0 + 20.0 * f for f in range(10)})
    assert gesture_near(players, 5, (900.0, 50.0), reach=80.0) == 0.0


def test_no_ball_makes_no_gesture():
    players = _players({f: 100.0 + 20.0 * f for f in range(10)})
    assert gesture_near(players, 5, None, reach=80.0) == 0.0


def test_a_swing_near_the_ball_is_a_strike():
    xs = {f: 100.0 for f in range(40)}
    xs.update({f: 100.0 + 25.0 * (f - 18) for f in range(18, 23)})
    xs.update({f: 200.0 for f in range(23, 40)})
    players = _players(xs)
    ball = {f: (xs[f], 50.0) for f in range(40)}
    found = strikes(players, ball, 0, 39, min_speed=10.0, reach=80.0, suppression=10)
    assert len(found) == 1
    assert 17 <= found[0] <= 23


def test_two_swings_close_together_count_once():
    xs = {f: 100.0 + 25.0 * f for f in range(9)}
    players = _players(xs)
    ball = {f: (xs[f], 50.0) for f in range(9)}
    assert len(strikes(players, ball, 0, 8, min_speed=10.0, reach=80.0, suppression=10)) == 1


def test_a_swing_at_a_contact_already_found_is_not_added_again():
    xs = {f: 100.0 for f in range(40)}
    xs.update({f: 100.0 + 25.0 * (f - 18) for f in range(18, 23)})
    xs.update({f: 200.0 for f in range(23, 40)})
    players = _players(xs)
    ball = {f: (xs[f], 50.0) for f in range(40)}
    assert strikes(players, ball, 0, 39, min_speed=10.0, reach=80.0, suppression=10, taken=[20]) == []


def test_a_swing_far_from_any_found_contact_is_kept():
    xs = {f: 100.0 for f in range(40)}
    xs.update({f: 100.0 + 25.0 * (f - 18) for f in range(18, 23)})
    xs.update({f: 200.0 for f in range(23, 40)})
    players = _players(xs)
    ball = {f: (xs[f], 50.0) for f in range(40)}
    found = strikes(players, ball, 0, 39, min_speed=10.0, reach=80.0, suppression=10, taken=[2])
    assert len(found) == 1
    assert 17 <= found[0] <= 23
