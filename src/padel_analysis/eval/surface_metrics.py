"""Scoring surface predictions without letting the easy cases carry the figure.

The annotation campaign oversamples the hard contacts on purpose - they are where a
human judgement is worth having, two agreeing indices already settling seventy percent
of the rest. That makes any raw overall rate meaningless, the sample no longer looking
like the population it is drawn from.

So every rate is reported per class and per stratum, and the single overall figure is
reweighted to the real mix with the weights declared. A weighted figure whose weights
are not stated is a figure whose sampling has been hidden.

An unreadable contact - the annotator could not tell - is excluded rather than scored.
It is neither a success nor a failure of the rule, and counting it either way would
assert something the data does not support. Its rate is reported separately.
"""

import math
from dataclasses import dataclass

UNREADABLE = "x"


@dataclass(frozen=True)
class ClassScore:
    """Precision, recall, and the count they rest on."""

    precision: float
    recall: float
    support: int
    """How many truths carried this class. A rate without it invites over-reading -
    mesh contacts are expected to be few, and three of them cannot carry a figure."""

    @property
    def f1(self) -> float:
        if math.isnan(self.precision) or math.isnan(self.recall):
            return math.nan
        if self.precision + self.recall == 0.0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


def _readable(predicted: list[str], truth: list[str]) -> list[tuple[str, str]]:
    if len(predicted) != len(truth):
        raise ValueError("predicted and truth must have the same length")
    return [(p, t) for p, t in zip(predicted, truth) if t != UNREADABLE]


def per_class(predicted: list[str], truth: list[str]) -> dict[str, ClassScore]:
    """One score per class present in the truth, unreadable contacts dropped."""
    pairs = _readable(predicted, truth)
    scores: dict[str, ClassScore] = {}
    for label in sorted({t for _, t in pairs}):
        hits = sum(1 for p, t in pairs if p == label and t == label)
        claimed = sum(1 for p, _ in pairs if p == label)
        actual = sum(1 for _, t in pairs if t == label)
        scores[label] = ClassScore(
            precision=hits / claimed if claimed else math.nan,
            recall=hits / actual if actual else math.nan,
            support=actual,
        )
    return scores


def unreadable_rate(truth: list[str]) -> float:
    """Share of contacts the annotator could not settle."""
    if not truth:
        return math.nan
    return sum(1 for t in truth if t == UNREADABLE) / len(truth)


def weighted_accuracy(
    strata: dict[str, tuple[list[str], list[str]]], weights: dict[str, float]
) -> float:
    """Accuracy of each stratum, recombined at the population's real proportions.

    Args:
        strata: name -> (predicted, truth) for that stratum.
        weights: name -> share of the real population. Must sum to one, since a
            weighted figure whose weights do not is an average of nothing.

    Raises:
        KeyError: a stratum carries no weight, which would silently drop it.
    """
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("weights must sum to one")

    total = 0.0
    for name, (predicted, truth) in strata.items():
        weight = weights[name]
        pairs = _readable(predicted, truth)
        if not pairs:
            continue
        total += weight * sum(1 for p, t in pairs if p == t) / len(pairs)
    return total
