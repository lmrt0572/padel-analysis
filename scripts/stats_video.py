"""Video de demonstration avec un panneau de statistiques a droite de l'image.

Les statistiques - coups, vitres, frappes, distance et temps au filet par joueur et par
paire - sont cumulees depuis le debut de l'extrait et avancent avec la video. L'image
de diffusion n'est pas recouverte : le panneau s'ajoute a cote.

Demande l'analyse sauvegardee de la minute (scripts/analyse_minutes.py) et ffmpeg.

Usage:
    python scripts/stats_video.py --match FinalF --minute 12000 --start 12888 \
        --stop 13799 --contact-model weights/contact_net.pt --out outputs/stats.mp4
"""

import argparse
import pickle
import tempfile
from pathlib import Path

import minutes
from rally_page import to_h264

from padel_analysis.analytics.live_stats import LiveTimeline
from padel_analysis.ball.smoothing import smooth_path
from padel_analysis.contact.learned import ContactModel
from padel_analysis.demo import learned_events, render
from padel_analysis.geometry.calibration import Calibration
from padel_analysis.rallies import DEFAULT_NAMES, RallySpec, build_rally
from padel_analysis.render.figure_style import PLAYER
from padel_analysis.render.stats_panel import StatsPanel

FPS = 30.0
PANEL_WIDTH = 480
# Les couleurs des joueurs du panneau, en BGR pour OpenCV : une joueuse a la meme
# couleur sur l'image et dans les chiffres.
PLAYER_BGR = {slot: tuple(int(c[i:i + 2], 16) for i in (5, 3, 1)) for slot, c in PLAYER.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True)
    parser.add_argument("--minute", type=int, required=True)
    parser.add_argument("--start", type=int, help="par defaut, le debut de la minute")
    parser.add_argument("--stop", type=int, help="par defaut, la fin de la minute")
    parser.add_argument("--contact-model", type=Path, required=True)
    parser.add_argument("--tag", default="360")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    analysis = pickle.loads(
        Path(minutes.analysis(args.match, args.minute, args.tag)).read_bytes()
    )
    start = analysis["start"] if args.start is None else args.start
    stop = analysis["stop"] if args.stop is None else args.stop
    if not analysis["start"] <= start <= stop <= analysis["stop"]:
        raise SystemExit("l'extrait deborde de la minute analysee")

    points = Calibration.load(minutes.calibration(args.match)).points
    _, shown, events, pose = learned_events(analysis, points,
                                            ContactModel.load(args.contact_model))
    spec = RallySpec("extrait", args.match, args.minute, start, stop, "")
    timeline = LiveTimeline(build_rally(analysis, events, spec, FPS))
    panel = StatsPanel(PANEL_WIDTH, analysis["size"][1], dict(DEFAULT_NAMES))
    drawn = smooth_path(shown, cuts=[e.frame for e in events])

    with tempfile.TemporaryDirectory() as scratch:
        raw = Path(scratch) / "brut.mp4"
        render(Path(minutes.video(args.match)), analysis, events, pose, drawn, raw, start,
               stop, side=lambda frame: panel.draw(timeline.at(frame)), minimap=False,
               labels=dict(DEFAULT_NAMES), colours=PLAYER_BGR)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        to_h264(raw, args.out)
    print(f"ecrit {args.out}")


if __name__ == "__main__":
    main()
