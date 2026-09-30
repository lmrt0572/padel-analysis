"""The statistics panel drawn beside each video frame.

Drawn with Pillow rather than OpenCV: OpenCV's built-in fonts have no accents and read
poorly at small sizes, and the panel is mostly text. The panel is a fixed layout that
only changes its numbers, so a viewer's eye learns where to look within seconds.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..analytics.live_stats import LiveStats
from ..geometry.court import Court
from . import figure_style as style

FONTS = Path("C:/Windows/Fonts")
LABELS = {"raquette": "FRAPPE", "sol": "SOL", "verre": "VITRE", "grillage": "GRILLAGE",
          "filet": "FILET"}
PAIR_NAMES = {"proche": "Paire proche", "fond": "Paire du fond"}
PAIR_COLOUR = {"proche": style.PLAYER["near_1"], "fond": style.PLAYER["far_1"]}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    for candidate in (FONTS / name, Path(name), Path("DejaVuSans-Bold.ttf" if bold else
                                                      "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    value = hex_colour.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


class StatsPanel:
    """Draws `LiveStats` as a column of the given size, returned as a BGR image."""

    def __init__(self, width: int, height: int, names: dict[str, str]) -> None:
        self.width, self.height = width, height
        self.names = names
        self.court = Court()
        self.scale = height / 1080
        s = self.scale
        self.fonts = {
            "title": _font(round(26 * s), bold=True),
            "big": _font(round(46 * s), bold=True),
            "label": _font(round(17 * s)),
            "name": _font(round(21 * s), bold=True),
            "value": _font(round(21 * s)),
            "small": _font(round(14 * s)),
            "chip": _font(round(20 * s), bold=True),
        }

    def draw(self, stats: LiveStats) -> np.ndarray:
        image = Image.new("RGB", (self.width, self.height), _rgb(style.BACKGROUND))
        pen = ImageDraw.Draw(image)
        s, pad = self.scale, round(26 * self.scale)
        x0, x1 = pad, self.width - pad

        minutes, seconds = divmod(int(stats.elapsed), 60)
        pen.text((x0, pad), "STATISTIQUES", font=self.fonts["title"], fill=_rgb(style.TEXT))
        pen.text((x1, pad), f"{minutes:02d}:{seconds:02d}", font=self.fonts["title"],
                 fill=_rgb(style.MUTED), anchor="ra")
        top = pad + round(56 * s)

        # En haut : la minimap a gauche, le jeu a droite.
        map_height = round(360 * s)
        map_width = map_height // 2
        self._minimap(pen, stats, x0, top, map_width, map_height)
        cx = x0 + map_width + round(26 * s)
        y = top
        if stats.rally_number is None:
            first = (str(stats.shots), "coups joués")
        else:
            first = (str(stats.rally_shots), f"coups, échange n° {stats.rally_number}")
        for value, label in (first, (str(stats.walls), "vitres et grillages")):
            pen.text((cx, y), value, font=self.fonts["big"], fill=_rgb(style.TEXT))
            pen.text((cx, y + round(54 * s)), label, font=self.fonts["label"],
                     fill=_rgb(style.MUTED))
            y += round(88 * s)
        if stats.rally_number is not None:
            pen.text((cx, y - round(8 * s)), f"plus long échange : {stats.longest_rally} coups",
                     font=self.fonts["small"], fill=_rgb(style.MUTED))
            y += round(22 * s)
        if stats.pair_points is not None:
            won = stats.pair_points
            pen.text((cx, y - round(8 * s)),
                     f"points : proche {won['proche']} · fond {won['fond']}",
                     font=self.fonts["small"], fill=_rgb(style.TEXT))
            y += round(22 * s)
        pen.text((cx, y), "balle, dernier coup", font=self.fonts["label"],
                 fill=_rgb(style.MUTED))
        speed = stats.last_shot_speed
        pen.text((cx, y + round(22 * s)), "—" if speed is None else f"≈ {speed:.0f} km/h",
                 font=self.fonts["name"], fill=_rgb(style.TEXT))
        best = stats.top_shot_speed
        pen.text((cx, y + round(50 * s)), "" if best is None else f"max ≈ {best:.0f} km/h",
                 font=self.fonts["small"], fill=_rgb(style.MUTED))
        y += round(88 * s)
        if stats.last:
            colour = _rgb(style.KIND.get(stats.last, style.TEXT))
            text = LABELS.get(stats.last, stats.last.upper())
            box = pen.textbbox((0, 0), text, font=self.fonts["chip"])
            w, h = box[2] - box[0] + round(28 * s), box[3] - box[1] + round(16 * s)
            pen.rounded_rectangle((cx, y, cx + w, y + h), radius=round(8 * s), fill=colour)
            pen.text((cx + w / 2, y + h / 2), text, font=self.fonts["chip"],
                     fill=_rgb(style.BACKGROUND), anchor="mm")

        # Les joueurs : une carte chacun, frappes puis course.
        y = self._section(pen, "LES JOUEURS", x0, x1, top + map_height + round(22 * s))
        card = round(114 * s)
        column = (x1 - x0 - round(20 * s)) / 4
        for line in stats.players:
            colour = _rgb(style.PLAYER.get(line.slot, style.TEXT))
            left = x0 + round(20 * s)
            self._card(pen, line, colour, x0, x1, y, y + card - round(12 * s), left)
            shots = f"{line.shots} frappe" + ("s" if line.shots > 1 else "")
            pen.text((x1 - round(12 * s), y + round(17 * s)), shots, font=self.fonts["value"],
                     fill=_rgb(style.BACKGROUND), anchor="rm")
            volleys = f"{line.volleys} volée" + ("s" if line.volleys > 1 else "")
            detail = f"{volleys} · {line.after_bounce} après rebond"
            pen.text((x1 - round(12 * s), y + round(38 * s)), detail, font=self.fonts["small"],
                     fill=_rgb(style.MUTED), anchor="ra")
            net = "—" if math.isnan(line.net_share) else f"{100 * line.net_share:.0f} %"
            values = [(f"{line.distance:.0f} m", "parcourus"),
                      (f"≈ {line.top_speed:.0f}", "km/h max"),
                      (net, "au filet")]
            if stats.pair_points is not None:
                values.append((f"{line.winners} · {line.errors}", "gagnés · fautes"))
            for i, (value, label) in enumerate(values):
                vx = left + i * column
                pen.text((vx, y + round(56 * s)), value, font=self.fonts["value"],
                         fill=_rgb(style.TEXT))
                pen.text((vx, y + round(82 * s)), label, font=self.fonts["small"],
                         fill=_rgb(style.MUTED))
            y += card

        footer = ["Cumulé depuis le début de l'extrait. Contacts : 82 % justes, mesuré",
                  "à la main. Vitesse max : tenue pendant une seconde.",
                  "Vitesse de balle : estimation en ligne droite."]
        if stats.pair_points is not None:
            footer.append("Points : lus au tableau, crédités au dernier frappeur.")
        for i, text in enumerate(reversed(footer)):
            pen.text((x0, self.height - pad - i * round(22 * s)), text,
                     font=self.fonts["small"], fill=_rgb(style.MUTED), anchor="ld")
        return np.asarray(image)[:, :, ::-1].copy()

    def _minimap(self, pen, stats: LiveStats, x: int, y: int, width: int, height: int) -> None:
        """The court from above, camera at the bottom, as it is built.

        Walls on the outline - glass across each back wall and along the first metres of
        each side, mesh in between - and the white lines where they are painted: the
        service lines, the centre line from one to the other, and the net.
        """
        court = self.court
        half_w, half_l = court.half_width, court.half_length
        service, glass = court.service_line_distance, court.side_wall_glass_length

        def point(cx: float, cy: float) -> tuple[float, float]:
            return (x + (cx + half_w) / (2 * half_w) * width,
                    y + (half_l - cy) / (2 * half_l) * height)

        def segment(a, b, colour, thickness):
            pen.line((*point(*a), *point(*b)), fill=colour, width=max(1, round(thickness)))

        s = self.scale
        white, glass_colour = (235, 238, 240), _rgb(style.KIND["verre"])
        mesh_colour = _rgb(style.KIND["grillage"])
        pen.rectangle((*point(-half_w, half_l), *point(half_w, -half_l)),
                      fill=_rgb(style.COURT_BLUE))
        for depth in (-service, service):
            segment((-half_w, depth), (half_w, depth), white, 2 * s)
        segment((0, -service - 0.2), (0, service + 0.2), white, 2 * s)
        for sign in (-1, 1):
            segment((-half_w, sign * half_l), (half_w, sign * half_l), glass_colour, 4 * s)
            for side in (-half_w, half_w):
                segment((side, sign * half_l), (side, sign * (half_l - glass)), glass_colour,
                        4 * s)
        for side in (-half_w, half_w):
            segment((side, -(half_l - glass)), (side, half_l - glass), mesh_colour, 3 * s)
        segment((-half_w - 0.3, 0), (half_w + 0.3, 0), white, 4 * s)

        radius = round(7 * s)
        for slot, trail in stats.positions.items():
            if not trail:
                continue
            colour = _rgb(style.PLAYER.get(slot, style.TEXT))
            inside = [p for p in trail if abs(p[0]) <= half_w + 0.5 and abs(p[1]) <= half_l + 0.5]
            if len(inside) > 1:
                pen.line([point(*p) for p in inside], fill=colour, width=max(1, round(2 * s)))
            if inside:
                px, py = point(*inside[-1])
                pen.ellipse((px - radius, py - radius, px + radius, py + radius), fill=colour,
                            outline=_rgb(style.BACKGROUND), width=2)

    def _card(self, pen, line, colour, x0, x1, top, bottom, left) -> None:
        """The frame of one player's card: the name in a full band, as a broadcast does."""
        s, name = self.scale, self.names.get(line.slot, line.slot)
        radius, band = round(10 * s), top + round(34 * s)
        pen.rounded_rectangle((x0, top, x1, bottom), radius=radius, fill=_rgb(style.PANEL))
        pen.rounded_rectangle((x0, top, x1, band), radius=radius, fill=colour,
                              corners=(True, True, False, False))
        pen.text((left, top + round(17 * s)), name.upper(), font=self.fonts["name"],
                 fill=_rgb(style.BACKGROUND), anchor="lm")

    def _section(self, pen, title: str, x0: int, x1: int, y: int) -> int:
        pen.text((x0, y), title, font=self.fonts["label"], fill=_rgb(style.MUTED))
        y += round(28 * self.scale)
        pen.line((x0, y, x1, y), fill=_rgb(style.LINE), width=max(1, round(2 * self.scale)))
        return y + round(16 * self.scale)

    def _bar(self, pen, x0: int, x1: int, y: int, share: float, colour) -> None:
        h = round(10 * self.scale)
        pen.rounded_rectangle((x0, y, x1, y + h), radius=h // 2, fill=_rgb(style.LINE))
        if share > 0:
            pen.rounded_rectangle((x0, y, x0 + max(h, (x1 - x0) * min(share, 1.0)), y + h),
                                  radius=h // 2, fill=colour)
