"""Les minutes pointees a la main, et leur role.

Les minutes du match feminin servent a regler. Celles du match masculin marquees
`juge` ne servent qu'une fois, pour le verdict final : aucune decision ne doit etre
prise en les regardant. FinalM 5000 a deja servi a plusieurs decisions, elle est donc
rangee a part, ni reglage ni juge. `EXTRA` : minutes pointees plus tard pour
entrainer le modele de contacts.
"""

TUNING = [("FinalF", 16000), ("FinalF", 17800), ("FinalF", 30000), ("FinalF", 40000)]
USED = [("FinalM", 5000)]
EXTRA = [("FinalM", 8000), ("FinalM", 18000), ("FinalM", 31000), ("FinalM", 47000),
         ("FinalF", 21000), ("FinalM", 21500)]
JUDGE = [("FinalM", 12000), ("FinalM", 25000), ("FinalM", 40000)]
# Le premier juge a rendu son verdict, et ses minutes ont servi a choisir la suite : un
# second juge, pris dans les deux matchs, note le modele qui en sort.
JUDGE_2 = [("FinalF", 8000), ("FinalF", 25000), ("FinalM", 35000)]
# Chaque juge ne sert qu'une fois : en decider quoi que ce soit le consomme.
JUDGE_3 = [("FinalF", 12000), ("FinalF", 43000), ("FinalM", 50000)]
# Juge du suivi d'identite : la verite d'identite couvre les deux matchs en entier, ces
# minutes n'ont donc besoin d'aucun pointage, seulement d'etre analysees.
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
