import pytest

from padel_analysis.ball.candidates import Candidate
from padel_analysis.ball.path import acceleration_cost, emission_cost


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
