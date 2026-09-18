"""The rally statistics page: one self-contained HTML file, the video next to its data.

Everything the page shows is computed here, in Python, and embedded as JSON; the page
itself only draws and keeps the panels in step with the video. No library, no server:
the file opens from disk, next to the rally clips it plays.
"""

import html
import json
from collections.abc import Sequence
from pathlib import Path

from ..analytics.rally import (
    Rally,
    after_each_shot,
    ball_speeds,
    impacts,
    movements,
    shots_by_player,
    summary,
)
from ..eval.contact_marks import Pairing

PATH_STEP = 3  # une position de joueur toutes les 3 images suffit au plan anime
MAX_STEP = 2.0  # metres entre deux points d'une trace : au-dela, c'est un saut du suivi
COURT_MARGIN = (5.5, 10.5)

RELIABILITY = {
    "frise": "Contacts et surfaces : 77,7 % des contacts réels donnés avec la bonne "
             "surface, et plus de 90 % des contacts affichés sont réels (716 contacts "
             "pointés à la main, trois juges).",
    "plan": "Position d'un impact : 13 cm près de la caméra, 56 cm au fond du court "
            "(erreur de la pose de caméra). Elle hérite aussi des erreurs de surface.",
    "frappes": "Une frappe est attribuée au joueur dont le poignet est le plus proche de "
               "la balle ; elle hérite des erreurs du modèle de contacts.",
    "vitesse": "Vitesse moyenne en ligne droite entre deux contacts : c'est une borne "
               "basse. Hachurée quand un bout est une frappe, placée par convention au "
               "joueur à 1 m de hauteur.",
    "deplacements": "Distance lissée sur 9 images : le bruit de position ajouterait "
                    "20 à 30 % de mètres que personne n'a courus. Filet : à moins de "
                    "5,85 m, le seuil mesuré sur un match entier.",
    "chiffres": "Comptés sur les contacts détectés : un contact manqué ou inventé "
                "change ces nombres.",
}


def rally_payload(
    rally: Rally, rally_id: str, title: str, names: dict[str, str], video: str,
    truth: Sequence[Pairing] | None = None,
) -> dict:
    """Everything the page needs about one rally, in plain JSON types."""
    shots = shots_by_player(rally)
    after = after_each_shot(rally)
    moves = movements(rally)
    players = {}
    for slot in sorted(set(rally.positions) | set(shots)):
        track = rally.positions.get(slot, {})
        path = player_path(rally, track)
        move = moves.get(slot)
        players[slot] = {
            "name": names.get(slot, slot),
            "path": path,
            "shots": shots.get(slot, 0),
            "after": after.get(slot, {}),
            "distance": round(move.distance, 1) if move else None,
            "net_share": round(move.net_share, 3) if move else None,
        }

    result = summary(rally)
    return {
        "id": rally_id,
        "title": title,
        "video": video,
        "fps": rally.fps,
        "start": rally.start,
        "duration": round(rally.duration, 2),
        "contacts": [
            {
                "t": round(rally.time_of(c), 3),
                "kind": c.kind,
                "player": c.player,
                "point": None if c.point is None else [round(v, 2) for v in c.point],
            }
            for c in rally.contacts
        ],
        "impacts": [
            {"t": round(rally.time_of(c), 3), "kind": c.kind,
             "point": [round(v, 2) for v in c.point]}
            for c in impacts(rally)
        ],
        "speeds": [
            {"t0": round((s.start - rally.start) / rally.fps, 3),
             "t1": round((s.stop - rally.start) / rally.fps, 3),
             "kmh": round(s.kmh, 1), "estimated": s.estimated}
            for s in ball_speeds(rally)
        ],
        "players": players,
        "summary": {"duration": round(result.duration, 1), "shots": result.shots,
                    "walls": result.walls, "last": result.last},
        "truth": None if truth is None else [
            {
                "t_detected": None if line.detected_frame is None
                else round((line.detected_frame - rally.start) / rally.fps, 3),
                "t_marked": None if line.marked_frame is None
                else round((line.marked_frame - rally.start) / rally.fps, 3),
                "detected": line.detected,
                "marked": line.marked,
                "status": line.status,
            }
            for line in truth
        ],
    }


def player_path(rally: Rally, track: dict) -> list[list[float] | None]:
    """A player's trace for the page: [seconds, x, y] points, None where it breaks.

    A position far outside the court, or one that jumps further than a player can run
    between two points, is a tracking slip. Joined to the rest, it would draw a line
    across the court that nobody ran; the trace is cut there instead.
    """
    points: list[list[float] | None] = []
    last = None
    for frame, (x, y) in sorted(track.items()):
        if (frame - rally.start) % PATH_STEP:
            continue
        if abs(x) > COURT_MARGIN[0] or abs(y) > COURT_MARGIN[1]:
            if points and points[-1] is not None:
                points.append(None)
            last = None
            continue
        if last is not None and ((x - last[0]) ** 2 + (y - last[1]) ** 2) ** 0.5 > MAX_STEP:
            points.append(None)
        points.append([round((frame - rally.start) / rally.fps, 3), round(x, 2), round(y, 2)])
        last = (x, y)
    return points


def page(payloads: Sequence[dict], title: str = "Padel Analysis — échanges") -> str:
    """The HTML page, with every rally's data embedded."""
    template = (Path(__file__).parent / "rally_page.html").read_text(encoding="utf-8")
    data = json.dumps({"rallies": list(payloads), "reliability": RELIABILITY},
                      ensure_ascii=False)
    # Rien dans les donnees ne doit pouvoir fermer la balise script qui les porte.
    data = data.replace("</", "<\\/")
    return template.replace("{{TITLE}}", html.escape(title)).replace("{{DATA}}", data)
