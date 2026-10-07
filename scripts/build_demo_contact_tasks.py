"""Draws up the list of contacts found by the demonstration chain, to judge them.

The earlier judgements were on contacts detected on the ANNOTATED ball. The
demonstration video, for its part, works on the reconstructed trajectory: that is where
what is seen has to be measured. Each instant is judged once, for what really happened
there; any variant of the chain is then scored on these judgements by matching its
contacts to the judged instants, to within two frames.

To cover several variants at once, the list is the union of the contacts found with and
without the display filters. The computed positions are written alongside, so that the
arbitration tool shows what the system claims.

Usage:
    python scripts/build_demo_contact_tasks.py --analysis outputs/demo_FinalF_analysis.pkl \
        --video FinalF --out ground_truth/demo_contacts/FinalF_16000.json
"""

import argparse
import json
import pickle
from pathlib import Path

from padel_analysis.ball.confidence import confident_path, path_scores
from padel_analysis.ball.contacts import find_contacts
from padel_analysis.ball.path import best_path
from padel_analysis.ball.smoothing import despike, smooth_path
from padel_analysis.eval.surface_truth import SurfaceGroundTruth, SurfaceTask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--merge", type=int, default=2, help="frames within which two contacts count as one")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    analysis = pickle.loads(args.analysis.read_bytes())
    start, stop = analysis["start"], analysis["stop"]
    raw = {f: v["raw"] for f, v in analysis["frames"].items()}
    path = best_path(raw, start, stop, weight=960.0, absent_cost=150.0, absolute=True)
    shown = despike(confident_path(path, path_scores(path, raw), 0.7, 8))

    variants = {
        "chemin_brut": find_contacts(path),
        "affichage": find_contacts(smooth_path(shown, cuts=[], process_noise=100.0)),
    }
    frames: list[int] = []
    for contacts in variants.values():
        for contact in contacts:
            if all(abs(contact.frame - f) > args.merge for f in frames):
                frames.append(contact.frame)
    frames.sort()

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists: the judgements given would be lost")
    SurfaceGroundTruth(
        video=args.video,
        frame_range=(start, stop),
        parameters={"weight": 960.0, "absent_cost": 150.0, "threshold": 0.7, "min_run": 8},
        tasks=[SurfaceTask(frame=f, stratum="demo") for f in frames],
    ).save(args.out)
    positions = args.out.with_name(args.out.stem + "_positions.json")
    positions.write_text(
        json.dumps({str(f): list(p) for f, p in path.items() if p is not None}),
        encoding="utf-8",
    )
    counts = ", ".join(f"{name} {len(c)}" for name, c in variants.items())
    print(f"{len(frames)} instants to judge ({counts}) -> {args.out}")


if __name__ == "__main__":
    main()
