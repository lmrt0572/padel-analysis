"""Train the contact model on the marked minutes, never on the judge ones.

With --cv, each minute is first predicted by a model trained on the others, then the
final model is trained on all the minutes. With --curve, the learning curve. With
--results, these figures are written as JSON, for the figures of the README.

Usage:
    python scripts/train_contact_model.py --cv --out weights/contact_net.pt
"""

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import minutes
import torch

from padel_analysis.contact.learned import decode, frame_features, frame_labels, train
from padel_analysis.demo import LABEL_OF_ANSWER, build_events
from padel_analysis.eval.contact_marks import ContactMarks, match_contacts, pair_contacts
from padel_analysis.geometry.calibration import Calibration
from padel_analysis.geometry.camera import court_surfaces
from padel_analysis.geometry.court import Court

ANSWER_OF_LABEL = {label: answer for answer, label in LABEL_OF_ANSWER.items()}


def load(match: str, start: int, tag: str) -> dict:
    analysis = pickle.loads(Path(minutes.analysis(match, start, tag)).read_bytes())
    path, shown, events, pose = build_events(
        analysis, Calibration.load(minutes.calibration(match)).points
    )
    features, material = frame_features(
        analysis, path, shown, events, pose, court_surfaces(Court())
    )
    marks = ContactMarks.load(minutes.marks(match, start)).marks
    return {
        "features": features,
        "material": material,
        "labels": frame_labels(marks, analysis["start"], len(features)),
        "start": analysis["start"],
        "marks": marks,
        "rules": {e.frame: ANSWER_OF_LABEL[e.label] for e in events},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="360")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--curve", action="store_true", help="learning curve")
    parser.add_argument("--results", type=Path, help="write the figures as JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    keys = minutes.CONTACT_TRAINING
    if set(keys) & set(minutes.PENDING_JUDGES):
        raise SystemExit("a judge minute is in the training set")
    data = {key: load(*key, args.tag) for key in keys}
    results: dict = {}

    if args.cv:
        totals = {"regles": [0, 0, 0], "modele": [0, 0, 0]}
        confusion: Counter = Counter()
        for key in keys:
            others = [(data[k]["features"], data[k]["labels"]) for k in keys if k != key]
            model = train(others, device=device)
            item = data[key]
            answers = decode(
                model.probabilities(item["features"]), item["material"], item["start"],
                args.threshold,
            )
            line = f"  {key[0]} {key[1]:5}"
            for name, detected in (("regles", item["rules"]), ("modele", answers)):
                m = match_contacts(detected, item["marks"], 3)
                total = totals[name]
                total[0] += m.right_surface
                total[1] += len(detected)
                total[2] += len(item["marks"])
                line += f"   {name} {m.right_surface:3}/{len(item['marks'])} aff {len(detected):3}"
            for pairing in pair_contacts(answers, item["marks"]):
                confusion[(pairing.marked or "aucun", pairing.detected or "aucun")] += 1
            print(line, flush=True)
        for name, (right, shown, real) in totals.items():
            print(
                f"{name:7} right {right}/{real} ({right / real:.1%})  shown {shown}  "
                f"score {2 * right / (shown + real):.3f}"
            )
        results["cv"] = {
            name: {"right": right, "shown": shown, "real": real}
            for name, (right, shown, real) in totals.items()
        }
        results["confusion"] = [
            {"marked": marked, "detected": detected, "count": count}
            for (marked, detected), count in sorted(confusion.items())
        ]

    if args.curve:
        curve = []
        for size in range(2, len(keys), 2):
            right = real = shown = 0
            for key in keys:
                others = random.Random(f"{key}-{size}").sample(
                    [k for k in keys if k != key], size
                )
                model = train([(data[k]["features"], data[k]["labels"]) for k in others],
                              seeds=(0,), device=device)
                item = data[key]
                answers = decode(model.probabilities(item["features"]), item["material"],
                                 item["start"], args.threshold)
                right += match_contacts(answers, item["marks"], 3).right_surface
                shown += len(answers)
                real += len(item["marks"])
            curve.append({"minutes": size, "right": right, "shown": shown, "real": real})
            print(f"  {size:2} training minutes: right {right / real:.1%}", flush=True)
        results["curve"] = curve

    if args.results is not None:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"figures written: {args.results}")

    model = train([(item["features"], item["labels"]) for item in data.values()], device=device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"model written: {args.out}")


if __name__ == "__main__":
    main()
