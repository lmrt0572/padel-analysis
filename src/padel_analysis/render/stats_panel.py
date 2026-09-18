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
        s, pad = self.scale, round(28 * self.scale)
        x0, x1 = pad, self.width - pad

        minutes, seconds = divmod(int(stats.elapsed), 60)
        pen.text((x0, pad), "STATISTIQUES", font=self.fonts["title"], fill=_rgb(style.TEXT))
        pen.text((x1, pad), f"{minutes:02d}:{seconds:02d}", font=self.fonts["title"],
                 fill=_rgb(style.MUTED), anchor="ra")
        y = pad + round(52 * s)

        # Le jeu : trois nombres, et le dernier contact tant qu'il est recent.
        y = self._section(pen, "LE JEU", x0, x1, y)
        cells = [(stats.shots, "coups joués"), (stats.walls, "vitres et grillages")]
        cell_width = (x1 - x0) / 2
        for i, (value, label) in enumerate(cells):
            cx = x0 + i * cell_width
            pen.text((cx, y), str(value), font=self.fonts["big"], fill=_rgb(style.TEXT))
            pen.text((cx, y + round(56 * s)), label, font=self.fonts["label"],
                     fill=_rgb(style.MUTED))
        y += round(96 * s)
        if stats.last:
            colour = _rgb(style.KIND.get(stats.last, style.TEXT))
            text = LABELS.get(stats.last, stats.last.upper())
            box = pen.textbbox((0, 0), text, font=self.fonts["chip"])
            w, h = box[2] - box[0] + round(28 * s), box[3] - box[1] + round(16 * s)
            pen.rounded_rectangle((x0, y, x0 + w, y + h), radius=round(8 * s), fill=colour)
            pen.text((x0 + w / 2, y + h / 2), text, font=self.fonts["chip"],
                     fill=_rgb(style.BACKGROUND), anchor="mm")
        y += round(58 * s)

        # Les paires : qui frappe le plus, qui tient le filet.
        y = self._section(pen, "LES PAIRES", x0, x1, y)
        total = max(sum(stats.pair_shots.values()), 1)
        for pair, name in PAIR_NAMES.items():
            colour = _rgb(PAIR_COLOUR[pair])
            pen.text((x0, y), name, font=self.fonts["name"], fill=colour)
            net = stats.pair_net[pair]
            pen.text((x1, y), f"{stats.pair_shots[pair]} frappes", font=self.fonts["value"],
                     fill=_rgb(style.TEXT), anchor="ra")
            y += round(34 * s)
            self._bar(pen, x0, x1, y, stats.pair_shots[pair] / total, colour)
            y += round(20 * s)
            label = "au filet —" if math.isnan(net) else f"au filet {100 * net:.0f} %"
            pen.text((x0, y), label, font=self.fonts["small"], fill=_rgb(style.MUTED))
            y += round(34 * s)
        y += round(8 * s)

        # Les joueurs : une carte chacun.
        y = self._section(pen, "LES JOUEURS", x0, x1, y)
        card = round(92 * s)
        for line in stats.players:
            colour = _rgb(style.PLAYER.get(line.slot, style.TEXT))
            pen.rounded_rectangle((x0, y, x1, y + card - round(12 * s)), radius=round(10 * s),
                                  fill=_rgb(style.PANEL))
            pen.rectangle((x0, y, x0 + round(6 * s), y + card - round(12 * s)), fill=colour)
            pen.text((x0 + round(20 * s), y + round(8 * s)), self.names.get(line.slot, line.slot),
                     font=self.fonts["name"], fill=colour)
            net = "—" if math.isnan(line.net_share) else f"{100 * line.net_share:.0f} %"
            values = [(f"{line.shots}", "frappes"),
                      (f"{line.distance:.0f} m".replace(".", ","), "parcourus"),
                      (net, "au filet")]
            column = (x1 - x0 - round(20 * s)) / 3
            for i, (value, label) in enumerate(values):
                vx = x0 + round(20 * s) + i * column
                pen.text((vx, y + round(40 * s)), value, font=self.fonts["value"],
                         fill=_rgb(style.TEXT))
                pen.text((vx + round(6 * s) + pen.textlength(value, font=self.fonts["value"]),
                          y + round(45 * s)), label, font=self.fonts["small"],
                         fill=_rgb(style.MUTED))
            y += card

        footer = ("Cumulé depuis le début de l'extrait.",
                  "Contacts : 78 % justes, mesuré à la main.")
        for i, text in enumerate(reversed(footer)):
            pen.text((x0, self.height - pad - i * round(22 * s)), text,
                     font=self.fonts["small"], fill=_rgb(style.MUTED), anchor="ld")
        return np.asarray(image)[:, :, ::-1].copy()

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
