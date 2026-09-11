import numpy as np

from padel_analysis.tracking.court_constraint import (
    SLOT_NAMES,
    CourtObservation,
    CourtSlotTracker,
)


def _observation(x, y, confidence=0.9, appearance=None):
    return CourtObservation(
        court_xy=np.array([x, y], dtype=np.float64),
        confidence=confidence,
        appearance=np.zeros(8) if appearance is None else np.asarray(appearance),
    )


def test_four_slots_two_on_each_side():
    tracker = CourtSlotTracker()
    assert len(SLOT_NAMES) == 4
    assert sum(1 for s in tracker.slots if s.side > 0) == 2
    assert sum(1 for s in tracker.slots if s.side < 0) == 2


def test_first_frame_assigns_every_observation_to_its_own_side():
    tracker = CourtSlotTracker()
    assignment = tracker.update(
        [_observation(-2, -5), _observation(2, -5), _observation(-2, 5), _observation(2, 5)]
    )
    assert len(assignment) == 4
    for slot_name, observation_index in assignment.items():
        slot = next(s for s in tracker.slots if s.name == slot_name)
        observed_y = [-5, -5, 5, 5][observation_index]
        assert np.sign(observed_y) == slot.side


def test_identity_is_kept_when_players_move_slightly():
    tracker = CourtSlotTracker()
    first = tracker.update(
        [_observation(-2, -5), _observation(2, -5), _observation(-2, 5), _observation(2, 5)]
    )
    second = tracker.update(
        [
            _observation(-2.1, -5.1), _observation(2.1, -5.1),
            _observation(-2.1, 5.1), _observation(2.1, 5.1),
        ]
    )
    assert first == second


def test_partners_crossing_do_not_swap_identities():
    """Le cas qui fait sauter les identifiants avec un tracker generique."""
    tracker = CourtSlotTracker()
    first = tracker.update([_observation(-3, -5), _observation(3, -5),
                            _observation(-3, 5), _observation(3, 5)])
    # Les deux partenaires du cote negatif convergent puis se croisent.
    for left_x, right_x in ((-1.5, 1.5), (-0.5, 0.5), (0.5, -0.5), (1.5, -1.5)):
        assignment = tracker.update(
            [_observation(left_x, -5), _observation(right_x, -5),
             _observation(-3, 5), _observation(3, 5)]
        )
    # Les observations 0 et 1 ont echange leurs positions. Chaque slot doit
    # continuer a suivre la meme observation qu'au depart.
    assert assignment["near_1"] == first["near_1"]
    assert assignment["near_2"] == first["near_2"]


def test_an_observation_on_the_wrong_side_is_never_assigned():
    tracker = CourtSlotTracker()
    assignment = tracker.update([_observation(0, -5), _observation(0, -4)])
    assert set(assignment) <= {"near_1", "near_2"}
    assert "far_1" not in assignment
    assert "far_2" not in assignment


def test_a_missing_player_keeps_its_slot_coasting():
    tracker = CourtSlotTracker()
    tracker.update([_observation(-2, -5), _observation(2, -5),
                    _observation(-2, 5), _observation(2, 5)])
    tracker.update([_observation(-2, -5), _observation(2, -5), _observation(-2, 5)])
    far_slots = [s for s in tracker.slots if s.side > 0]
    coasting = [s for s in far_slots if s.missing_frames == 1]
    assert len(coasting) == 1
    assert coasting[0].position is not None
    assert all(s.missing_frames == 0 for s in tracker.slots if s.side < 0)


def test_a_fifth_person_is_ignored():
    """Ramasseurs de balle et arbitre ne doivent pas voler un slot."""
    tracker = CourtSlotTracker()
    tracker.update([_observation(-2, -5), _observation(2, -5),
                    _observation(-2, 5), _observation(2, 5)])
    assignment = tracker.update(
        [_observation(-2, -5), _observation(2, -5), _observation(-2, 5),
         _observation(2, 5), _observation(4.8, 9.5)]
    )
    assert len(assignment) == 4
    assert 4 not in assignment.values()


def test_appearance_breaks_a_positional_tie():
    tracker = CourtSlotTracker(appearance_weight=10.0)
    red, black = np.array([1.0] + [0.0] * 7), np.array([0.0] * 7 + [1.0])
    first = tracker.update([
        _observation(-2, -5, appearance=red), _observation(2, -5, appearance=black),
        _observation(-2, 5, appearance=red), _observation(2, 5, appearance=black),
    ])
    # Les deux partenaires echangent leurs positions exactes, sans changer de tenue.
    # Sans apparence, la position seule ferait sauter les identites.
    assignment = tracker.update([
        _observation(2, -5, appearance=red), _observation(-2, -5, appearance=black),
        _observation(-2, 5, appearance=red), _observation(2, 5, appearance=black),
    ])
    assert assignment["near_1"] == first["near_1"]
    assert assignment["near_2"] == first["near_2"]


def test_low_confidence_observations_cost_more():
    tracker = CourtSlotTracker()
    tracker.update([_observation(-2, -5), _observation(2, -5),
                    _observation(-2, 5), _observation(2, 5)])
    confident = tracker.cost(tracker.slots[0], _observation(-2, -5, confidence=0.9))
    doubtful = tracker.cost(tracker.slots[0], _observation(-2, -5, confidence=0.1))
    assert doubtful > confident


def test_a_spectator_in_the_stands_is_refused():
    """Sans borne, un spectateur du bon cote peut voler un slot."""
    tracker = CourtSlotTracker()
    tracker.update([_observation(-2, -5), _observation(2, -5),
                    _observation(-2, 5), _observation(2, 5)])
    assignment = tracker.update(
        [_observation(-2, -5), _observation(2, -5), _observation(-2, 5),
         _observation(2, 5), _observation(11.0, 6.0)]
    )
    assert len(assignment) == 4
    assert 4 not in assignment.values()


def test_a_player_leaving_through_the_side_opening_is_still_accepted():
    """Au padel on sort du court pour rattraper un lob : la borne doit etre large."""
    tracker = CourtSlotTracker()
    tracker.update([_observation(-2, -5), _observation(2, -5),
                    _observation(-2, 5), _observation(2, 5)])
    assignment = tracker.update(
        [_observation(-7.5, -12.0), _observation(2, -5),
         _observation(-2, 5), _observation(2, 5)]
    )
    assert len(assignment) == 4
    assert 0 in assignment.values()
