import numpy as np
import pytest

from padel_analysis.eval import identity
from padel_analysis.eval.identity import (
    AmbiguousEpisode,
    IdentityGroundTruth,
    assign_by_proximity,
    find_ambiguous_episodes,
)


def _court_positions(rows: list[list[tuple[float, float]]]) -> dict[int, np.ndarray]:
    """Positions court par frame, dans l'ordre des annotations."""
    return {i: np.array(row, dtype=np.float64) for i, row in enumerate(rows)}


def _steady(n: int) -> dict[int, np.ndarray]:
    row = [(-2.0, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)]
    return _court_positions([row] * n)


def test_each_frame_gets_the_four_slots():
    assignments = assign_by_proximity(_steady(5))
    assert set(assignments[0]) == {"near_1", "near_2", "far_1", "far_2"}


def test_a_slot_points_at_an_annotation_index():
    assignments = assign_by_proximity(_steady(3))
    assert sorted(assignments[0].values()) == [0, 1, 2, 3]


def test_slots_keep_their_annotation_when_nobody_moves():
    assignments = assign_by_proximity(_steady(5))
    assert assignments[0] == assignments[4]


def test_slots_follow_a_player_who_moves():
    rows = []
    for step in range(5):
        rows.append([(-2.0 + step, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)])
    assignments = assign_by_proximity(_court_positions(rows))
    # L'annotation 0 bouge mais reste la meme personne : meme slot du debut a la fin.
    assert assignments[0] == assignments[4]


def test_a_slot_never_takes_a_player_from_the_other_side():
    assignments = assign_by_proximity(_steady(3))
    positions = _steady(3)[0]
    for slot, index in assignments[0].items():
        expected = -1 if slot.startswith("near") else 1
        assert np.sign(positions[index][1]) == expected


def test_a_frame_without_four_people_is_skipped():
    rows = {
        0: np.array([(-2.0, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)]),
        1: np.array([(-2.0, -5.0), (2.0, -5.0)]),
    }
    assignments = assign_by_proximity(rows)
    assert 0 in assignments
    assert 1 not in assignments


def test_no_episode_when_partners_stay_apart():
    episodes = find_ambiguous_episodes(_steady(100), threshold=1.5)
    assert episodes == []


def test_an_episode_is_found_when_partners_come_close():
    rows = []
    for frame in range(20):
        gap = 4.0 if frame < 8 or frame > 12 else 0.5
        rows.append([(-gap / 2, -5.0), (gap / 2, -5.0), (-2.0, 5.0), (2.0, 5.0)])
    episodes = find_ambiguous_episodes(_court_positions(rows), threshold=1.5)
    assert len(episodes) == 1
    assert isinstance(episodes[0], AmbiguousEpisode)
    assert episodes[0].start_frame <= 8
    assert episodes[0].end_frame >= 12


def test_an_episode_records_the_closest_approach():
    rows = []
    for frame in range(20):
        gap = 4.0 if frame != 10 else 0.4
        rows.append([(-gap / 2, -5.0), (gap / 2, -5.0), (-2.0, 5.0), (2.0, 5.0)])
    episodes = find_ambiguous_episodes(_court_positions(rows), threshold=1.5)
    assert episodes[0].min_separation_m == pytest.approx(0.4, abs=1e-6)


def test_two_separate_approaches_give_two_episodes():
    rows = []
    for frame in range(40):
        close = frame in range(5, 8) or frame in range(25, 28)
        gap = 0.5 if close else 4.0
        rows.append([(-gap / 2, -5.0), (gap / 2, -5.0), (-2.0, 5.0), (2.0, 5.0)])
    episodes = find_ambiguous_episodes(_court_positions(rows), threshold=1.5)
    assert len(episodes) == 2


def test_ground_truth_roundtrips_through_json(tmp_path):
    truth = IdentityGroundTruth(
        assignments=assign_by_proximity(_steady(3)),
        episodes=[
            AmbiguousEpisode(
                start_frame=1, end_frame=2,
                slots=("near_1", "near_2"), min_separation_m=0.8,
            )
        ],
        resolved=[],
    )
    path = tmp_path / "identity.json"
    truth.save(path)
    loaded = IdentityGroundTruth.load(path)
    assert loaded.assignments[0] == truth.assignments[0]
    assert loaded.episodes[0].min_separation_m == pytest.approx(0.8)


def test_applying_a_swap_flips_two_slots_from_that_frame_onward():
    truth = IdentityGroundTruth(
        assignments=assign_by_proximity(_steady(6)), episodes=[], resolved=[]
    )
    before = dict(truth.assignments[0])
    truth.apply_swap(from_frame=3, slots=("near_1", "near_2"))
    assert truth.assignments[0] == before
    assert truth.assignments[3]["near_1"] == before["near_2"]
    assert truth.assignments[3]["near_2"] == before["near_1"]
    assert truth.assignments[5]["near_1"] == before["near_2"]


def test_a_save_interrupted_at_the_last_moment_keeps_the_previous_file(
    tmp_path, monkeypatch
):
    """Une coupure pendant l'ecriture ne doit pas detruire l'arbitrage deja rendu."""
    path = tmp_path / "truth.json"
    IdentityGroundTruth(
        assignments={0: {"near_1": 0}}, episodes=[], resolved=[7]
    ).save(path)

    def interrupted(*args: object, **kwargs: object) -> None:
        raise OSError("coupure de courant")

    monkeypatch.setattr("padel_analysis.io.atomic.os.replace", interrupted)
    with pytest.raises(OSError):
        IdentityGroundTruth(
            assignments={1: {"near_1": 1}}, episodes=[], resolved=[]
        ).save(path)

    survivor = IdentityGroundTruth.load(path)
    assert survivor.resolved == [7]
    assert survivor.assignments == {0: {"near_1": 0}}


def test_a_save_leaves_no_temporary_file_behind(tmp_path):
    path = tmp_path / "truth.json"
    IdentityGroundTruth(
        assignments={0: {"near_1": 0}}, episodes=[], resolved=[]
    ).save(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["truth.json"]


def test_a_failed_save_leaves_no_temporary_file_behind(tmp_path, monkeypatch):
    path = tmp_path / "truth.json"
    IdentityGroundTruth(
        assignments={0: {"near_1": 0}}, episodes=[], resolved=[]
    ).save(path)

    def interrupted(*args: object, **kwargs: object) -> None:
        raise OSError("coupure de courant")

    monkeypatch.setattr("padel_analysis.io.atomic.os.replace", interrupted)
    with pytest.raises(OSError):
        IdentityGroundTruth(
            assignments={1: {"near_1": 1}}, episodes=[], resolved=[]
        ).save(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["truth.json"]


def test_a_ground_truth_written_before_cuts_existed_still_loads(tmp_path):
    """Les premieres verites terrain n'ont ni coupures ni arbitrage de coupures."""
    path = tmp_path / "truth.json"
    path.write_text(
        '{"assignments": {"0": {"near_1": 0}}, "episodes": [], "resolved": [0]}',
        encoding="utf-8",
    )
    truth = IdentityGroundTruth.load(path)
    assert truth.resolved == [0]
    assert truth.cuts == []
    assert truth.resolved_cuts == []


def test_cuts_survive_a_save_and_a_load(tmp_path):
    path = tmp_path / "truth.json"
    IdentityGroundTruth(
        assignments={0: {"near_1": 0}},
        episodes=[],
        cuts=[identity.CameraCut(frame=12, sides=("near", "far"), displacement_m=3.5)],
        resolved_cuts=[12],
    ).save(path)
    loaded = IdentityGroundTruth.load(path)
    assert loaded.cuts == [
        identity.CameraCut(frame=12, sides=("near", "far"), displacement_m=3.5)
    ]
    assert loaded.resolved_cuts == [12]


def test_without_a_boundary_every_frame_is_in_the_first_segment():
    truth = IdentityGroundTruth(assignments={}, episodes=[])
    assert truth.segment_of(0) == 0
    assert truth.segment_of(50_000) == 0


def test_a_boundary_starts_a_new_segment_from_its_own_frame():
    """Un changement de cote n'echange rien : il arrete l'identite et la relance."""
    truth = IdentityGroundTruth(assignments={}, episodes=[], boundaries=[100, 500])
    assert truth.segment_of(99) == 0
    assert truth.segment_of(100) == 1
    assert truth.segment_of(499) == 1
    assert truth.segment_of(500) == 2


def test_boundaries_survive_a_save_and_a_load(tmp_path):
    path = tmp_path / "truth.json"
    IdentityGroundTruth(
        assignments={0: {"near_1": 0}}, episodes=[], boundaries=[7, 9]
    ).save(path)
    assert IdentityGroundTruth.load(path).boundaries == [7, 9]


def test_decisions_survive_a_save_and_a_load(tmp_path):
    path = tmp_path / "truth.json"
    IdentityGroundTruth(
        assignments={0: {"near_1": 0}},
        episodes=[],
        decisions={"cut:120": "p", "cut:300": "c", "episode:7": "n"},
    ).save(path)
    assert IdentityGroundTruth.load(path).decisions == {
        "cut:120": "p",
        "cut:300": "c",
        "episode:7": "n",
    }


def test_a_ground_truth_written_before_decisions_existed_still_loads(tmp_path):
    path = tmp_path / "truth.json"
    path.write_text(
        '{"assignments": {}, "episodes": [], "resolved": []}', encoding="utf-8"
    )
    assert IdentityGroundTruth.load(path).decisions == {}


def test_a_swap_applied_twice_returns_to_the_start():
    """C'est ce qui rend l'annulation possible : l'echange est son propre inverse."""
    truth = IdentityGroundTruth(
        assignments={5: {"near_1": 0, "near_2": 1}, 6: {"near_1": 0, "near_2": 1}},
        episodes=[],
    )
    truth.apply_swap(5, ("near_1", "near_2"))
    assert truth.assignments[5] == {"near_1": 1, "near_2": 0}
    truth.apply_swap(5, ("near_1", "near_2"))
    assert truth.assignments[5] == {"near_1": 0, "near_2": 1}
