import numpy as np

from padel_analysis.contact.surfaces import RACKET, classify
from padel_analysis.geometry.camera import court_surfaces
from padel_analysis.geometry.court import Court


def _surfaces():
    return court_surfaces(Court())


def test_a_contact_at_a_wrist_is_a_racket(synthetic_pose):
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[0.0, -5.0, 1.0]]))[0]
    wrist = (ball[0] + 10.0, ball[1] + 5.0)
    verdict = classify(ball, [wrist], pose, _surfaces(), wrist_distance=80.0)
    assert verdict.surface == RACKET
    assert verdict.point is None


def test_a_far_wrist_does_not_make_a_racket(synthetic_pose):
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[0.0, 0.0, 0.0]]))[0]
    verdict = classify(ball, [(10.0, 10.0)], pose, _surfaces(), wrist_distance=80.0)
    assert verdict.surface != RACKET


def test_no_wrist_at_all_is_not_a_racket(synthetic_pose):
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[1.0, -2.0, 0.0]]))[0]
    assert classify(ball, [], pose, _surfaces()).surface != RACKET


def test_a_ball_on_the_floor_is_read_as_the_floor(synthetic_pose):
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[1.0, -2.0, 0.0]]))[0]
    verdict = classify(ball, [], pose, _surfaces())
    assert verdict.surface == "floor"
    np.testing.assert_allclose(verdict.point, [1.0, -2.0, 0.0], atol=0.05)


def test_a_ball_on_the_far_back_wall_is_read_as_that_wall(synthetic_pose):
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[1.0, 10.0, 2.0]]))[0]
    verdict = classify(ball, [], pose, _surfaces(), margin=0.0)
    assert verdict.surface == "back_wall_positive_y"
    assert verdict.material == "verre"


def test_a_high_contact_on_a_back_wall_is_mesh(synthetic_pose):
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[1.0, 10.0, 3.6]]))[0]
    verdict = classify(ball, [], pose, _surfaces(), margin=0.0)
    assert verdict.material == "grillage"


def test_a_contact_with_no_admissible_surface_is_undecided(synthetic_pose):
    pose, _, _ = synthetic_pose
    verdict = classify((5.0, 5.0), [], pose, _surfaces())
    assert verdict.surface is None
    assert verdict.candidates == 0


def test_the_verdict_counts_the_admissible_surfaces(synthetic_pose):
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[1.0, -2.0, 0.0]]))[0]
    assert classify(ball, [], pose, _surfaces()).candidates >= 1


def test_a_wider_margin_can_only_admit_more(synthetic_pose):
    """La marge absorbe l'erreur de pose : elle elargit, elle ne restreint jamais."""
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[4.9, 9.8, 0.0]]))[0]
    tight = classify(ball, [], pose, _surfaces(), margin=0.0).candidates
    loose = classify(ball, [], pose, _surfaces(), margin=0.5).candidates
    assert loose >= tight


def test_the_nearest_wrist_decides_not_the_first(synthetic_pose):
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[0.0, -5.0, 1.0]]))[0]
    far, near = (ball[0] + 500.0, ball[1]), (ball[0] + 10.0, ball[1])
    assert classify(ball, [far, near], pose, _surfaces(), wrist_distance=80.0).surface == RACKET


def test_a_racket_verdict_names_no_material(synthetic_pose):
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[0.0, -5.0, 1.0]]))[0]
    wrist = (ball[0], ball[1])
    assert classify(ball, [wrist], pose, _surfaces()).material is None


def test_a_deep_floor_candidate_yields_to_a_wall(synthetic_pose):
    """Un point-sol trop profond trahit un contact bas contre la vitre proche."""
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[0.0, -10.0, 0.6]]))[0]
    without = classify(ball, [], pose, _surfaces(), depth_cut=None)
    with_cut = classify(ball, [], pose, _surfaces(), depth_cut=-7.5)
    assert without.surface == "floor"
    assert with_cut.surface == "back_wall_negative_y"


def test_a_shallow_floor_candidate_keeps_the_floor(synthetic_pose):
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[0.0, 2.0, 0.0]]))[0]
    assert classify(ball, [], pose, _surfaces(), depth_cut=-7.5).surface == "floor"


def test_the_cut_needs_another_candidate_to_yield_to(synthetic_pose):
    """Sans second candidat, le sol reste : on ne remplace pas une reponse par rien."""
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[0.0, -9.5, 0.0]]))[0]
    verdict = classify(ball, [], pose, _surfaces(), depth_cut=-7.5, margin=0.0)
    assert verdict.surface is not None


def test_a_racket_is_decided_before_any_depth_question(synthetic_pose):
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[0.0, -10.0, 0.6]]))[0]
    verdict = classify(ball, [tuple(ball)], pose, _surfaces(), depth_cut=-7.5)
    assert verdict.surface == RACKET


def test_the_cut_leaves_the_candidate_count_alone(synthetic_pose):
    """L'arbitrage choisit parmi les admissibles ; il n'en ajoute ni n'en retire."""
    pose, _, _ = synthetic_pose
    ball = pose.project(np.array([[0.0, -10.0, 0.6]]))[0]
    assert (
        classify(ball, [], pose, _surfaces(), depth_cut=-7.5).candidates
        == classify(ball, [], pose, _surfaces(), depth_cut=None).candidates
    )
