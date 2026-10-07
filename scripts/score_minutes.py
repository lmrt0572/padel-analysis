"""Score the demonstration chain on every marked minute of a set.

Counts are added up over the minutes, not rates. The judge minutes are only scored with
--juge; the first four judges now train the model, so scoring them with the shipped
model no longer measures anything.

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
        print(f"  {match} {start:5}  shown {len(detected):3}  real {len(truth):3}  "
              f"right {m.right_surface:3}")
        found += m.found
        missed += m.missed
        invented += m.invented
        right += m.right_surface
        shown += len(detected)
        real += len(truth)
    print(f"  total: {real} real contacts, {shown} shown")
    print(f"  precision {found / max(found + invented, 1):.1%}   recall {found / max(real, 1):.1%}   "
          f"right {right} ({right / max(real, 1):.1%})   score {2 * right / max(shown + real, 1):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--contact-model", type=Path, help="score the learned model")
    parser.add_argument("--juge-5", action="store_true",
                        help="score the fifth judge: only once")
    parser.add_argument("--juge-4", action="store_true",
                        help="score the fourth judge: only once")
    parser.add_argument("--juge-3", action="store_true",
                        help="score the third judge: only once")
    parser.add_argument("--juge-2", action="store_true", help="score the second judge: only once")
    parser.add_argument("--juge", action="store_true", help="score the judge minutes: only once")
    args = parser.parse_args()
    model = ContactModel.load(args.contact_model) if args.contact_model else None
    if args.juge_5:
        print("=== FIFTH JUDGE: minutes never looked at ===")
        score(minutes.JUDGE_5, args.tag, model)
    elif args.juge_4:
        print("=== FOURTH JUDGE: minutes never looked at ===")
        score(minutes.JUDGE_4, args.tag, model)
    elif args.juge_3:
        print("=== THIRD JUDGE: minutes never looked at ===")
        score(minutes.JUDGE_3, args.tag, model)
    elif args.juge_2:
        print("=== SECOND JUDGE: minutes never looked at ===")
        score(minutes.JUDGE_2, args.tag, model)
    elif args.juge:
        print("=== JUDGE: men's match, minutes never looked at ===")
        score(minutes.JUDGE, args.tag, model)
    else:
        print("=== tuning: women's match ===")
        score(minutes.TUNING, args.tag, model)


if __name__ == "__main__":
    main()
