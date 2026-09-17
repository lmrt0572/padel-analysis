"""Les minutes pointees a la main, et leur role.

Les minutes du match feminin servent a regler. Celles du match masculin marquees
`juge` ne servent qu'une fois, pour le verdict final : aucune decision ne doit etre
prise en les regardant. FinalM 5000 a deja servi a plusieurs decisions, elle est donc
rangee a part, ni reglage ni juge. `EXTRA` : minutes pointees plus tard pour
entrainer le modele de contacts.
"""

TUNING = [("FinalF", 16000), ("FinalF", 17800), ("FinalF", 30000), ("FinalF", 40000)]
USED = [("FinalM", 5000)]
EXTRA = [("FinalM", 8000), ("FinalM", 18000), ("FinalM", 31000), ("FinalM", 47000)]
JUDGE = [("FinalM", 12000), ("FinalM", 25000), ("FinalM", 40000)]
FRAMES = 1800


def video(match: str) -> str:
    return f"data/padeltracker100/extracted/2022_BCN_{match}_1.mp4"


def calibration(match: str) -> str:
    return f"ground_truth/calibrations/{match}.json"


def marks(match: str, start: int) -> str:
    return f"ground_truth/contact_marks/{match}_{start}.json"


def analysis(match: str, start: int, tag: str) -> str:
    return f"outputs/analysis/{match}_{start}_{tag}.pkl"
