import json

import numpy as np

from padel_analysis.perception.pose_detector import PersonDetection
from padel_analysis.rallies import (
    DEFAULT_NAMES,
    RallySpec,
    build_rally,
    load_specs,
    striker_slot,
    strikers,
)
from padel_analysis.render.ball_overlay import ContactEvent


def _person(box):
    return PersonDetection(bbox=np.array(box, dtype=float), confidence=0.9,
                           keypoints=np.zeros((17, 3)))


def _analysis():
    near, far = _person((100, 500, 160, 700)), _person((900, 100, 930, 180))
    frames = {
        f: {
            "people": [near, far],
            "assignment": {"near_1": 0, "far_1": 1},
            "positions": {"near_1": (1.0, -6.0), "far_1": (-2.0, 7.0)},
        }
        for f in range(100, 200)
    }
    return {"start": 100, "stop": 199, "frames": frames}, near


def _spec(start=110, stop=180):
    return RallySpec("r1", "FinalF", 16000, start, stop, "Un echange", {})


def test_specs_are_read_with_their_optional_names(tmp_path):
    path = tmp_path / "rallies.json"
    path.write_text(json.dumps({"rallies": [
        {"id": "a", "match": "FinalF", "minute": 16000, "start": 1, "stop": 9, "title": "A"},
        {"id": "b", "match": "FinalM", "minute": 8000, "start": 2, "stop": 8, "title": "B",
         "players": {"near_1": "Alice"}},
    ]}), encoding="utf-8")
    first, second = load_specs(path)
    assert first.players == {} and second.players == {"near_1": "Alice"}


def test_a_player_without_a_given_name_keeps_a_neutral_one():
    spec = RallySpec("r", "FinalF", 0, 0, 1, "t", {"near_1": "Alice"})
    assert spec.name_of("near_1") == "Alice"
    assert spec.name_of("far_2") == DEFAULT_NAMES["far_2"]


def test_the_striker_is_the_slot_whose_box_was_lit():
    analysis, near = _analysis()
    frame = analysis["frames"][120]
    assert striker_slot(near.bbox, frame["people"], frame["assignment"]) == "near_1"


def test_a_box_that_belongs_to_nobody_has_no_slot():
    analysis, _ = _analysis()
    frame = analysis["frames"][120]
    assert striker_slot(np.array([0.0, 0.0, 1.0, 1.0]), frame["people"],
                        frame["assignment"]) is None


def test_a_rally_keeps_only_the_contacts_between_its_bounds():
    analysis, near = _analysis()
    events = [
        ContactEvent(105, "SOL", (0.0, 0.0), point=np.array([0.0, 5.0, 0.0])),
        ContactEvent(120, "RAQUETTE", (0.0, 0.0), box=near.bbox),
        ContactEvent(140, "VITRE", (0.0, 0.0), point=np.array([5.0, 3.0, 1.5])),
        ContactEvent(190, "SOL", (0.0, 0.0), point=np.array([0.0, 5.0, 0.0])),
    ]
    rally = build_rally(analysis, events, _spec(), fps=30.0)
    assert [(c.frame, c.kind, c.player) for c in rally.contacts] == [
        (120, "raquette", "near_1"), (140, "verre", None),
    ]
    assert rally.contacts[1].point == (5.0, 3.0, 1.5)


def test_a_rally_carries_the_players_positions_over_its_frames_only():
    analysis, _ = _analysis()
    rally = build_rally(analysis, [], _spec(110, 180), fps=30.0)
    assert set(rally.positions) == {"near_1", "far_1"}
    assert min(rally.positions["near_1"]) == 110 and max(rally.positions["near_1"]) == 180
    assert rally.positions["far_1"][150] == (-2.0, 7.0)


def test_a_lit_box_the_tracker_left_out_falls_back_on_the_nearest_tracked_player():
    analysis, _ = _analysis()
    frame = analysis["frames"][120]
    stranger = np.array([150.0, 520.0, 200.0, 690.0])
    assert striker_slot(stranger, frame["people"], frame["assignment"],
                        ball=(165.0, 560.0)) == "near_1"


def test_without_a_ball_an_unknown_box_still_has_no_slot():
    analysis, _ = _analysis()
    frame = analysis["frames"][120]
    assert striker_slot(np.array([0.0, 0.0, 1.0, 1.0]), frame["people"],
                        frame["assignment"]) is None


def test_a_player_of_the_asked_half_is_the_striker_even_when_another_is_nearer():
    analysis, _ = _analysis()
    frame = analysis["frames"][120]
    assert striker_slot(None, frame["people"], frame["assignment"], ball=(130.0, 480.0),
                        side="far") == "far_1"


def test_the_middle_of_three_strikes_from_one_half_goes_to_the_other_half():
    analysis, _ = _analysis()
    far_box = analysis["frames"][120]["people"][1].bbox
    ball = (915.0, 90.0)  # un lob du joueur proche, haut dans l'image, pres du fond
    strikes = [(110, far_box, ball), (130, far_box, ball), (150, far_box, ball)]
    assert strikers(analysis["frames"], strikes) == {110: "far_1", 130: "near_1", 150: "far_1"}


def test_strikes_far_apart_are_not_made_to_alternate():
    analysis, _ = _analysis()
    frames = {**analysis["frames"], 400: analysis["frames"][120]}
    far_box = frames[120]["people"][1].bbox
    ball = (915.0, 90.0)
    strikes = [(110, far_box, ball), (150, far_box, ball), (400, far_box, ball)]
    assert strikers(frames, strikes)[150] == "far_1"
