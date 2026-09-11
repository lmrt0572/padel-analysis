import numpy as np

from padel_analysis.eval.cuts import CameraCut, find_camera_cuts


def _positions(rows: dict[int, list[tuple[float, float]]]) -> dict[int, dict]:
    """near_1, near_2, far_1, far_2 - dans cet ordre."""
    names = ("near_1", "near_2", "far_1", "far_2")
    return {
        frame: {
            name: np.array(xy, dtype=np.float64) for name, xy in zip(names, points)
        }
        for frame, points in rows.items()
    }


def _still(frames: range) -> dict[int, dict]:
    return _positions(
        {f: [(-2.0, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)] for f in frames}
    )


def test_a_still_scene_holds_no_cut():
    assert find_camera_cuts(_still(range(50))) == []


def test_ordinary_play_holds_no_cut():
    """Un joueur rapide parcourt quelques centimetres entre deux frames."""
    rows = {
        f: [
            (-2.0 + 0.05 * f, -5.0),
            (2.0, -5.0 + 0.05 * f),
            (-2.0, 5.0 - 0.04 * f),
            (2.0 - 0.05 * f, 5.0),
        ]
        for f in range(50)
    }
    assert find_camera_cuts(_positions(rows)) == []


def test_both_players_of_one_side_jumping_is_a_cut():
    rows = {0: [(-2.0, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)],
            1: [(3.0, -8.0), (-3.0, -2.0), (-2.0, 5.0), (2.0, 5.0)]}
    cuts = find_camera_cuts(_positions(rows))
    assert cuts == [CameraCut(frame=1, sides=("near",), displacement_m=cuts[0].displacement_m)]
    assert cuts[0].sides == ("near",)


def test_a_cut_can_put_both_sides_at_risk():
    rows = {0: [(-2.0, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)],
            1: [(3.0, -8.0), (-3.0, -2.0), (3.0, 8.0), (-3.0, 2.0)]}
    assert find_camera_cuts(_positions(rows))[0].sides == ("near", "far")


def test_one_player_moving_alone_is_enough():
    """L'association efface le deplacement d'une paire qui permute : un seul suffit."""
    rows = {0: [(-2.0, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)],
            1: [(4.0, -8.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)]}
    assert find_camera_cuts(_positions(rows))[0].sides == ("near",)


def test_a_hole_in_the_annotations_is_not_a_cut():
    """Entre deux frames eloignees, les joueurs ont eu le temps de se deplacer."""
    rows = {0: [(-2.0, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)],
            60: [(3.0, -8.0), (-3.0, -2.0), (3.0, 8.0), (-3.0, 2.0)]}
    assert find_camera_cuts(_positions(rows)) == []


def test_the_reported_displacement_is_the_largest_seen():
    """C'est le plus grand saut qui atteste du raccord, pas le plus petit."""
    rows = {0: [(0.0, -5.0), (0.0, -5.0), (-2.0, 5.0), (2.0, 5.0)],
            1: [(0.0, -8.0), (0.0, -7.0), (-2.0, 5.0), (2.0, 5.0)]}
    assert find_camera_cuts(_positions(rows))[0].displacement_m == 3.0


def test_the_threshold_can_be_raised():
    rows = {0: [(-2.0, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)],
            1: [(-2.0, -6.2), (2.0, -6.2), (-2.0, 5.0), (2.0, 5.0)]}
    assert find_camera_cuts(_positions(rows), threshold=1.0)
    assert find_camera_cuts(_positions(rows), threshold=1.5) == []


def test_an_absent_player_does_not_prevent_reading_the_others():
    """Le joueur qui reste suffit a attester le raccord de son cote."""
    rows = _positions({0: [(-2.0, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)],
                       1: [(3.0, -8.0), (-3.0, -2.0), (3.0, 8.0), (-3.0, 2.0)]})
    del rows[1]["far_2"]
    assert find_camera_cuts(rows)[0].sides == ("near", "far")


def test_a_side_with_no_annotated_player_is_left_out():
    rows = _positions({0: [(-2.0, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)],
                       1: [(3.0, -8.0), (-3.0, -2.0), (3.0, 8.0), (-3.0, 2.0)]})
    del rows[1]["far_1"]
    del rows[1]["far_2"]
    assert find_camera_cuts(rows)[0].sides == ("near",)


def test_a_teleport_across_a_short_hole_is_a_cut():
    """Huit metres en trois frames absentes : personne ne court aussi vite."""
    rows = {0: [(-2.0, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)],
            3: [(-2.0, 3.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)]}
    assert find_camera_cuts(_positions(rows))[0].frame == 3


def test_running_across_a_long_hole_is_not_a_cut():
    """Sept metres en trente-six frames absentes : un joueur qui court."""
    rows = {0: [(-2.0, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)],
            36: [(-2.0, 2.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)]}
    assert find_camera_cuts(_positions(rows)) == []


def test_a_very_long_hole_excuses_any_displacement():
    rows = {0: [(-2.0, -5.0), (2.0, -5.0), (-2.0, 5.0), (2.0, 5.0)],
            600: [(4.0, 9.0), (-4.0, 9.0), (4.0, -9.0), (-4.0, -9.0)]}
    assert find_camera_cuts(_positions(rows)) == []
