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
        for value, label in ((str(stats.shots), "coups joués"),
                             (str(stats.walls), "vitres et grillages")):
            pen.text((cx, y), value, font=self.fonts["big"], fill=_rgb(style.TEXT))
            pen.text((cx, y + round(54 * s)), label, font=self.fonts["label"],
                     fill=_rgb(style.MUTED))
            y += round(88 * s)
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
        column = (x1 - x0 - round(20 * s)) / 3
        for line in stats.players:
            colour = _rgb(style.PLAYER.get(line.slot, style.TEXT))
            bottom = y + card - round(12 * s)
            pen.rounded_rectangle((x0, y, x1, bottom), radius=round(10 * s),
                                  fill=_rgb(style.PANEL))
            pen.rectangle((x0, y, x0 + round(6 * s), bottom), fill=colour)
            left = x0 + round(20 * s)
            pen.text((left, y + round(8 * s)), self.names.get(line.slot, line.slot),
                     font=self.fonts["name"], fill=colour)
            shots = f"{line.shots} frappe" + ("s" if line.shots > 1 else "")
            pen.text((x1 - round(12 * s), y + round(10 * s)), shots, font=self.fonts["value"],
                     fill=_rgb(style.TEXT), anchor="ra")
            detail = f"{line.volleys} volées · {line.after_bounce} après rebond"
            pen.text((x1 - round(12 * s), y + round(38 * s)), detail, font=self.fonts["small"],
                     fill=_rgb(style.MUTED), anchor="ra")
            net = "—" if math.isnan(line.net_share) else f"{100 * line.net_share:.0f} %"
            values = [(f"{line.distance:.0f} m", "parcourus"),
                      (f"{line.top_speed:.0f} km/h", "pointe"),
                      (net, "au filet")]
            for i, (value, label) in enumerate(values):
                vx = left + i * column
                pen.text((vx, y + round(62 * s)), value, font=self.fonts["value"],
                         fill=_rgb(style.TEXT))
                pen.text((vx + round(6 * s) + pen.textlength(value, font=self.fonts["value"]),
                          y + round(67 * s)), label, font=self.fonts["small"],
                         fill=_rgb(style.MUTED))
            y += card

        footer = ("Cumulé depuis le début de l'extrait. Contacts : 78 % justes, mesuré",
                  "à la main. Vitesse de balle : estimation en ligne droite.")
        for i, text in enumerate(reversed(footer)):
            pen.text((x0, self.height - pad - i * round(22 * s)), text,
                     font=self.fonts["small"], fill=_rgb(style.MUTED), anchor="ld")
        return np.asarray(image)[:, :, ::-1].copy()

    def _minimap(self, pen, stats: LiveStats, x: int, y: int, width: int, height: int) -> None:
        """The court from above, camera at the bottom: each player and their last steps."""
        half_w, half_l, service = 5.0, 10.0, 6.95

        def point(cx: float, cy: float) -> tuple[float, float]:
            return (x + (cx + half_w) / (2 * half_w) * width,
                    y + (half_l - cy) / (2 * half_l) * height)

        green, line = _rgb(style.KIND["sol"]), _rgb(style.LINE)
        pen.rectangle((x, y, x + width, y + height), fill=_rgb(style.COURT), outline=green,
                      width=max(1, round(2 * self.scale)))
        for depth in (-service, service):
            pen.line((*point(-half_w, depth), *point(half_w, depth)), fill=line, width=1)
        pen.line((*point(0, -service), *point(0, service)), fill=line, width=1)
        pen.line((*point(-half_w - 0.4, 0), *point(half_w + 0.4, 0)), fill=_rgb(style.TEXT),
                 width=max(2, round(3 * self.scale)))
        radius = round(7 * self.scale)
        for slot, trail in stats.positions.items():
            if not trail:
                continue
            colour = _rgb(style.PLAYER.get(slot, style.TEXT))
            inside = [p for p in trail if abs(p[0]) <= half_w + 0.5 and abs(p[1]) <= half_l + 0.5]
            if len(inside) > 1:
                pen.line([point(*p) for p in inside], fill=colour, width=max(1, round(2 * self.scale)))
            if inside:
                px, py = point(*inside[-1])
                pen.ellipse((px - radius, py - radius, px + radius, py + radius), fill=colour,
                            outline=_rgb(style.BACKGROUND), width=2)

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
