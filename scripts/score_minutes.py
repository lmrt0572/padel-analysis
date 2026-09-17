"""Note la chaine de demonstration sur toutes les minutes pointees d'un ensemble.

Les comptes sont additionnes sur les minutes, pas les taux : une minute a 90 contacts
pese plus qu'une a 70.

Les minutes de juge ne sont notees qu'avec --juge, pour que le verdict final ne puisse
pas etre regarde par megarde pendant qu'on regle encore.

Usage:
    python scripts/score_minutes.py --tag 360
    python scripts/score_minutes.py --tag 360 --juge
    python scripts/score_minutes.py --tag 360 --contact-model weights/contact_net.pt --juge
"""

import argparse
import pickle
from pathlib import Path

import minutes

from padel_analysis.contact.learned import ContactModel
from padel_analysis.demo import build_events, learned_events
from padel_analysis.eval.contact_marks import ContactMarks, match_contacts
from padel_analysis.geometry.calibration import Calibration

ANSWER_OF_LABEL = {"SOL": "sol", "VITRE": "verre", "GRILLAGE": "grillage", "FILET": "filet",
                   "RAQUETTE": "raquette"}


def score(chosen, tag, model=None):
    found = missed = invented = right = shown = real = 0
    for match, start in chosen:
        analysis = pickle.loads(Path(minutes.analysis(match, start, tag)).read_bytes())
        points = Calibration.load(minutes.calibration(match)).points
        if model is None:
            events = build_events(analysis, points)[2]
        else:
            events = learned_events(analysis, points, model)[2]
        detected = {e.frame: ANSWER_OF_LABEL[e.label] for e in events}
        truth = ContactMarks.load(minutes.marks(match, start)).marks
        m = match_contacts(detected, truth, 3)
        print(f"  {match} {start:5}  affiches {len(detected):3}  reels {len(truth):3}  "
              f"justes {m.right_surface:3}")
        found += m.found
        missed += m.missed
        invented += m.invented
        right += m.right_surface
        shown += len(detected)
        real += len(truth)
    print(f"  total : {real} contacts reels, {shown} affiches")
    print(f"  precision {found / max(found + invented, 1):.1%}   rappel {found / max(real, 1):.1%}   "
          f"justes {right} ({right / max(real, 1):.1%})   score {2 * right / max(shown + real, 1):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--contact-model", type=Path, help="noter le modele appris")
    parser.add_argument("--juge", action="store_true", help="noter les minutes de juge : une seule fois")
    args = parser.parse_args()
    model = ContactModel.load(args.contact_model) if args.contact_model else None
    if args.juge:
        print("=== JUGE : match masculin, minutes jamais regardees ===")
        score(minutes.JUDGE, args.tag, model)
    else:
        print("=== reglage : match feminin ===")
        score(minutes.TUNING, args.tag, model)


if __name__ == "__main__":
    main()
