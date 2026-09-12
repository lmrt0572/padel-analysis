import math

import pytest

from padel_analysis.eval.surface_metrics import per_class, weighted_accuracy


def test_a_perfect_prediction_scores_one():
    score = per_class(["sol", "mur"], ["sol", "mur"])
    assert score["sol"].precision == 1.0
    assert score["sol"].recall == 1.0
    assert score["sol"].f1 == 1.0


def test_a_missed_class_has_no_recall():
    score = per_class(["sol", "sol"], ["sol", "mur"])
    assert score["sol"].recall == pytest.approx(1.0)
    assert score["mur"].recall == 0.0


def test_a_class_never_predicted_has_no_precision():
    score = per_class(["sol", "sol"], ["sol", "mur"])
    assert math.isnan(score["mur"].precision)
    assert math.isnan(score["mur"].f1)


def test_the_support_is_reported_beside_every_rate():
    """Un taux sur trois exemples n'est pas un taux : l'effectif doit suivre."""
    assert per_class(["sol"] * 3, ["sol"] * 3)["sol"].support == 3


def test_an_unreadable_truth_is_excluded_not_counted_wrong():
    score = per_class(["sol", "mur"], ["sol", "x"])
    assert score["sol"].support == 1
    assert score["sol"].precision == 1.0
    assert "x" not in score


def test_lengths_must_match():
    with pytest.raises(ValueError):
        per_class(["sol"], ["sol", "mur"])


def test_weighted_accuracy_follows_the_real_population():
    """Annoter surtout les cas difficiles ne doit pas deformer le chiffre global."""
    value = weighted_accuracy(
        {
            "settled": (["sol"] * 10, ["sol"] * 9 + ["mur"]),
            "ambiguous": (["mur"] * 5, ["sol"] * 5),
        },
        weights={"settled": 0.70, "ambiguous": 0.30},
    )
    assert value == pytest.approx(0.9 * 0.70)


def test_a_stratum_of_only_unreadable_contacts_is_skipped():
    value = weighted_accuracy(
        {"a": (["sol"], ["sol"]), "b": (["sol"], ["x"])},
        weights={"a": 0.5, "b": 0.5},
    )
    assert value == pytest.approx(0.5)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        weighted_accuracy({"a": (["sol"], ["sol"])}, weights={"a": 0.5})


def test_every_stratum_needs_a_weight():
    with pytest.raises(KeyError):
        weighted_accuracy(
            {"a": (["sol"], ["sol"]), "b": (["sol"], ["sol"])}, weights={"a": 1.0}
        )
