"""The hand-marked minutes, and their role.

The women's minutes are used for tuning. Minutes marked as judges are scored only once,
for a verdict. FinalM 5000 has served for several decisions, so it is neither. `EXTRA`
holds minutes marked later to train the contact model.
"""

TUNING = [("FinalF", 16000), ("FinalF", 17800), ("FinalF", 30000), ("FinalF", 40000)]
USED = [("FinalM", 5000)]
EXTRA = [("FinalM", 8000), ("FinalM", 18000), ("FinalM", 31000), ("FinalM", 47000),
         ("FinalF", 21000), ("FinalM", 21500)]
JUDGE = [("FinalM", 12000), ("FinalM", 25000), ("FinalM", 40000)]
# the first judge's minutes guided later choices, hence a second one from both matches
JUDGE_2 = [("FinalF", 8000), ("FinalF", 25000), ("FinalM", 35000)]
# a judge serves only once: deciding anything from it uses it up
JUDGE_3 = [("FinalF", 12000), ("FinalF", 43000), ("FinalM", 50000)]
# the first three scored the chain twice; the fourth scores the next change, once
JUDGE_4 = [("FinalF", 32000), ("FinalM", 16000), ("FinalM", 45000)]
# retired judges now train the contact model; a judge that has not scored yet
# must never enter it
RETIRED_JUDGES = JUDGE + JUDGE_2 + JUDGE_3 + JUDGE_4
# the fifth judge scores the model trained on the 23 minutes, once
JUDGE_5 = [("FinalF", 5000), ("FinalF", 23000), ("FinalM", 33000)]
PENDING_JUDGES = JUDGE_5
CONTACT_TRAINING = TUNING + USED + EXTRA + RETIRED_JUDGES
# identity judge: the identity truth covers both matches in full, so these
# minutes need no marking, only to be analysed
IDENTITY_JUDGE = [("FinalF", 3000), ("FinalF", 27000), ("FinalF", 34000), ("FinalF", 37000),
                  ("FinalM", 1000), ("FinalM", 14000), ("FinalM", 28000), ("FinalM", 43000)]
FRAMES = 1800


def video(match: str) -> str:
    return f"data/padeltracker100/extracted/2022_BCN_{match}_1.mp4"


def calibration(match: str) -> str:
    return f"ground_truth/calibrations/{match}.json"


def marks(match: str, start: int) -> str:
    return f"ground_truth/contact_marks/{match}_{start}.json"


def analysis(match: str, start: int, tag: str) -> str:
    return f"outputs/analysis/{match}_{start}_{tag}.pkl"
