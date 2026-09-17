import pytest

from padel_analysis.ball.candidates import Candidate
from padel_analysis.ball.path import acceleration_cost, best_path, emission_cost


def test_a_constant_velocity_costs_nothing():
    cost = acceleration_cost((0.0, 0.0), (10.0, 0.0), (20.0, 0.0), ceiling=80.0)
    assert cost == pytest.approx(0.0)


def test_a_bend_costs_what_it_deviates():
    cost = acceleration_cost((0.0, 0.0), (10.0, 0.0), (20.0, 6.0), ceiling=80.0)
    assert cost == pytest.approx(6.0)


def test_the_cost_of_a_bounce_is_capped():
    """Un rebond doit rester possible : son cout est borne, pas infini."""
    cost = acceleration_cost((0.0, 0.0), (50.0, 0.0), (0.0, 0.0), ceiling=80.0)
    assert cost == pytest.approx(80.0)


def test_the_ceiling_is_what_lets_a_contact_through():
    reversal = ((0.0, 0.0), (50.0, 0.0), (0.0, 0.0))
    assert acceleration_cost(*reversal, ceiling=40.0) == pytest.approx(40.0)
    assert acceleration_cost(*reversal, ceiling=200.0) == pytest.approx(100.0)


def test_the_best_candidate_of_a_frame_emits_nothing():
    frame = [Candidate(0.0, 0.0, 200.0), Candidate(1.0, 1.0, 100.0)]
    assert emission_cost(frame[0], frame, weight=30.0) == pytest.approx(0.0)


def test_a_weaker_candidate_costs_in_proportion():
    frame = [Candidate(0.0, 0.0, 200.0), Candidate(1.0, 1.0, 100.0)]
    assert emission_cost(frame[1], frame, weight=30.0) == pytest.approx(15.0)


def test_a_zero_weight_makes_every_candidate_equal():
    frame = [Candidate(0.0, 0.0, 200.0), Candidate(1.0, 1.0, 1.0)]
    assert emission_cost(frame[1], frame, weight=0.0) == pytest.approx(0.0)


def test_an_empty_frame_emits_nothing():
    assert emission_cost(Candidate(0.0, 0.0, 10.0), [], weight=30.0) == pytest.approx(
        0.0
    )


def test_a_clean_straight_line_is_followed():
    candidates = {f: [Candidate(100.0 + 20.0 * f, 100.0, 100.0)] for f in range(10)}
    found = best_path(candidates, start=0, stop=9)
    assert found[5] == pytest.approx((200.0, 100.0))
    assert all(found[f] is not None for f in range(10))


def test_a_decoy_that_breaks_the_line_is_refused():
    """Le chemin global prefere la continuite a un score isole."""
    candidates = {f: [Candidate(100.0 + 20.0 * f, 100.0, 100.0)] for f in range(10)}
    candidates[5] = [Candidate(800.0, 700.0, 900.0)] + candidates[5]
    found = best_path(candidates, start=0, stop=9)
    assert found[5] == pytest.approx((200.0, 100.0))


def test_this_is_what_the_greedy_version_could_not_do():
    """Une amorce sur le leurre condamnait le segment glouton ; ici non."""
    candidates = {f: [Candidate(100.0 + 20.0 * f, 100.0, 100.0)] for f in range(12)}
    for f in (0, 1):
        candidates[f] = [Candidate(700.0 + 3.0 * f, 700.0, 900.0)] + candidates[f]
    found = best_path(candidates, start=0, stop=11)
    assert found[8] == pytest.approx((260.0, 100.0))


def test_a_frame_without_candidates_comes_back_absent():
    candidates = {f: [Candidate(100.0 + 20.0 * f, 100.0, 100.0)] for f in range(10)}
    del candidates[4]
    found = best_path(candidates, start=0, stop=9)
    assert found[4] is None


def test_a_cheaper_absence_makes_the_path_give_up_more_often():
    """Une frame absente blanchit un saut : elle ne contraint la vitesse ni en
    entrant ni en sortant. C'est bon marche, donc le prix de l'absence doit rester
    au-dessus de celui d'un rebond."""
    candidates = {f: [Candidate(500.0 * (f % 2), 700.0, 100.0)] for f in range(10)}
    cheap = best_path(candidates, start=0, stop=9, absent_cost=1.0)
    dear = best_path(candidates, start=0, stop=9, absent_cost=10000.0)
    assert sum(1 for p in cheap.values() if p is None) > sum(
        1 for p in dear.values() if p is None
    )


def test_an_expensive_absence_keeps_the_path_committed():
    candidates = {f: [Candidate(100.0 + 20.0 * f, 100.0, 100.0)] for f in range(10)}
    found = best_path(candidates, start=0, stop=9, absent_cost=10000.0)
    assert all(p is not None for p in found.values())


def test_every_requested_frame_is_answered():
    found = best_path({}, start=3, stop=7)
    assert sorted(found) == [3, 4, 5, 6, 7]
    assert all(p is None for p in found.values())


def test_the_width_bounds_what_is_considered():
    """Le cout croit comme le cube du nombre de candidats retenus, et la balle est
    dans les dix premiers 96 % du temps. Ce qui est au-dela n'est pas vu."""
    candidates = {
        f: [Candidate(9000.0, 9000.0, 1.0)] * 8
        + [Candidate(100.0 + 20.0 * f, 100.0, 100.0)]
        for f in range(10)
    }
    found = best_path(candidates, start=0, stop=9, width=8)
    assert found[5] == pytest.approx((9000.0, 9000.0))


def test_the_path_follows_a_bounce_rather_than_dropping_out():
    """Disparaitre ne doit pas etre moins cher que suivre un rebond, sinon la balle
    s'evanouirait a chaque contact - et il y en a un toutes les quinze frames."""
    candidates = {}
    for f in range(6):
        candidates[f] = [Candidate(100.0 + 20.0 * f, 100.0, 100.0)]
    for f in range(6, 12):
        candidates[f] = [Candidate(200.0 - 20.0 * (f - 6), 100.0, 100.0)]
    found = best_path(candidates, start=0, stop=11)
    assert all(found[f] is not None for f in range(12))


def test_a_bounce_is_allowed_by_the_capped_cost():
    """La balle repart en sens inverse : le chemin doit la suivre malgre tout."""
    candidates = {}
    for f in range(6):
        candidates[f] = [Candidate(100.0 + 20.0 * f, 100.0, 100.0)]
    for f in range(6, 12):
        candidates[f] = [Candidate(200.0 - 20.0 * (f - 6), 100.0, 100.0)]
    found = best_path(candidates, start=0, stop=11)
    assert found[8] == pytest.approx((160.0, 100.0))


def test_the_default_costs_still_follow_a_bounce():
    """Un rebond franc doit rester payable au tarif par defaut."""
    candidates = {}
    for f in range(6):
        candidates[f] = [Candidate(100.0 + 20.0 * f, 100.0, 100.0)]
    for f in range(6, 12):
        candidates[f] = [Candidate(200.0 - 20.0 * (f - 6), 100.0, 100.0)]
    found = best_path(candidates, start=0, stop=11)
    assert all(found[f] is not None for f in range(12))


def test_the_default_weight_does_not_let_a_strong_decoy_win():
    """Le poids d'emission a un optimum interieur ; trop haut, le score l'emporte
    sur la continuite."""
    candidates = {f: [Candidate(100.0 + 20.0 * f, 100.0, 100.0)] for f in range(10)}
    candidates[5] = [Candidate(800.0, 700.0, 900.0)] + candidates[5]
    found = best_path(candidates, start=0, stop=9)
    assert found[5] == pytest.approx((200.0, 100.0))


def test_an_absolute_score_makes_a_weak_best_candidate_costly():
    """En relatif le meilleur d'une image ne coute rien, meme tres faible."""
    weak = [Candidate(0.0, 0.0, 0.05)]
    assert emission_cost(weak[0], weak, weight=100.0) == pytest.approx(0.0)
    assert emission_cost(weak[0], weak, weight=100.0, absolute=True) == pytest.approx(95.0)


def test_an_absolute_path_gives_up_where_only_weak_candidates_remain():
    strong = {f: [Candidate(10.0 * f, 0.0, 0.95)] for f in range(10)}
    weak = {f: [Candidate(500.0 - 37.0 * f, 300.0, 0.02)] for f in range(10, 20)}
    candidates = {**strong, **weak}
    relative = best_path(candidates, 0, 19, weight=240.0, absent_cost=100.0)
    absolute = best_path(candidates, 0, 19, weight=240.0, absent_cost=100.0, absolute=True)
    assert all(relative[f] is not None for f in range(10, 20))
    assert all(absolute[f] is None for f in range(12, 20))
    assert all(absolute[f] is not None for f in range(8))
