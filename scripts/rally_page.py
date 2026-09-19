"""Construit la page de statistiques des echanges choisis dans config/rallies.json.

Pour chaque echange : les contacts decides par le modele, les statistiques, un extrait
video rendu avec les incrustations de la demonstration puis reencode en H.264 pour
les navigateurs, et, si la minute a ete pointee a la main, la comparaison avec ce
pointage pour le mode verite. Tout est ecrit dans --out, hors du depot.

Demande l'analyse sauvegardee de chaque minute (scripts/analyse_minutes.py) et
ffmpeg dans le PATH.

Usage:
    python scripts/rally_page.py --contact-model weights/contact_net.pt
"""

import argparse
import json
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

import minutes

from padel_analysis.ball.smoothing import smooth_path
from padel_analysis.contact.learned import ContactModel
from padel_analysis.demo import learned_events, render
from padel_analysis.eval.contact_marks import ContactMarks, pair_contacts
from padel_analysis.geometry.calibration import Calibration
from padel_analysis.rallies import ANSWER_OF_LABEL, DEFAULT_NAMES, build_rally, load_specs
from padel_analysis.render.rally_page import page, rally_payload

FPS = 30.0


def to_h264(source: Path, target: Path) -> None:
    """Reencode for browsers: OpenCV writes MPEG-4 part 2, which they do not play."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg est introuvable dans le PATH")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(target)],
        check=True,
    )


def truth_for(spec, events) -> list | None:
    """The line-by-line comparison with the hand marks, if this minute was marked."""
    path = Path(minutes.marks(spec.match, spec.minute))
    if not path.exists():
        return None
    marks = {f: a for f, a in ContactMarks.load(path).marks.items()
             if spec.start <= f <= spec.stop}
    detected = {e.frame: ANSWER_OF_LABEL[e.label] for e in events
                if spec.start <= e.frame <= spec.stop}
    return pair_contacts(detected, marks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rallies", type=Path, default=Path("config/rallies.json"))
    parser.add_argument("--contact-model", type=Path, required=True)
    parser.add_argument("--tag", default="360")
    parser.add_argument("--out", type=Path, default=Path("outputs/rallies"))
    parser.add_argument("--no-video", action="store_true",
                        help="ne pas refaire les extraits video, seulement la page")
    args = parser.parse_args()

    model = ContactModel.load(args.contact_model)
    args.out.mkdir(parents=True, exist_ok=True)
    payloads = []
    for spec in load_specs(args.rallies):
        analysis = pickle.loads(
            Path(minutes.analysis(spec.match, spec.minute, args.tag)).read_bytes()
        )
        if not analysis["start"] <= spec.start <= spec.stop <= analysis["stop"]:
            raise SystemExit(f"{spec.id} deborde de la minute analysee {spec.minute}")
        points = Calibration.load(minutes.calibration(spec.match)).points
        _, shown, events, pose = learned_events(analysis, points, model)
        rally = build_rally(analysis, events, spec, FPS)
        names = {slot: spec.name_of(slot) for slot in DEFAULT_NAMES}
        clip = f"{spec.id}.mp4"

        if not args.no_video:
            print(f"{spec.id} : rendu de l'extrait", flush=True)
            drawn = smooth_path(shown, cuts=[e.frame for e in events])
            with tempfile.TemporaryDirectory() as scratch:
                raw = Path(scratch) / "brut.mp4"
                render(Path(minutes.video(spec.match)), analysis, events, pose, drawn, raw,
                       spec.start, spec.stop)
                to_h264(raw, args.out / clip)

        payloads.append(rally_payload(rally, spec.id, spec.title, names, clip,
                                      truth_for(spec, events)))
        print(f"{spec.id} : {len(rally.contacts)} contacts", flush=True)

    target = args.out / "index.html"
    target.write_text(page(payloads), encoding="utf-8")
    # Les memes donnees, pour les figures du README (scripts/make_figures.py).
    (args.out / "data.json").write_text(json.dumps(payloads, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"ecrit {target}")


if __name__ == "__main__":
    main()
