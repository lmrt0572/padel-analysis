"""The statistics panel drawn beside each video frame.

Drawn with Pillow rather than OpenCV: OpenCV's built-in fonts have no accents and read
poorly at small sizes, and the panel is mostly text. The panel is a fixed layout that
only changes its numbers, so a viewer's eye learns where to look within seconds.

Its look is a printed statistics table rather than an app: one condensed typeface,
hairline rules instead of boxes, a small square of each player's colour. The four
players sit as they stand on the court - the far pair above, the near pair below - so
the grid reads like the minimap beside it.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..analytics.live_stats import LiveStats, PlayerLine
from ..geometry.court import Court
from . import figure_style as style

FONTS = Path("C:/Windows/Fonts")
LABELS = {"raquette": "FRAPPE", "sol": "SOL", "verre": "VITRE", "grillage": "GRILLAGE",
          "filet": "FILET"}
PAIR_NAMES = {"proche": "Paire proche", "fond": "Paire du fond"}
PAIR_COLOUR = {"proche": style.PLAYER["near_1"], "fond": style.PLAYER["far_1"]}
GRID = {"far_1": (0, 0), "far_2": (1, 0), "near_2": (0, 1), "near_1": (1, 1)}
"""Where each player's cell sits: the far pair on top, as on the minimap."""


def _font(size: int, weight: str = "Regular") -> ImageFont.ImageFont:
    """Bahnschrift at the given weight, or Segoe UI, or what the system has."""
    try:
        font = ImageFont.truetype(str(FONTS / "bahnschrift.ttf"), size)
        font.set_variation_by_name(weight)
        return font
    except (OSError, ValueError):
        pass
    bold = weight not in ("Regular", "Light")
    for name in ("segoeuib.ttf" if bold else "segoeui.ttf",
                 "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        for candidate in (FONTS / name, Path(name)):
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    value = hex_colour.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("s" if count > 1 else "")


class StatsPanel:
    """Draws `LiveStats` as a column of the given size, returned as a BGR image.

    Args:
        names: each player's name, by slot.
        pair_names: each pair's name, "proche" and "fond", as the scoreboard writes it.
    """

    def __init__(self, width: int, height: int, names: dict[str, str],
                 pair_names: dict[str, str] | None = None) -> None:
        self.width, self.height = width, height
        self.names = names
        self.pair_names = pair_names or PAIR_NAMES
        self.court = Court()
        self.scale = height / 1080
        s = self.scale
        self.fonts = {
            "title": _font(round(26 * s), "SemiBold"),
            "big": _font(round(48 * s), "SemiBold"),
            "label": _font(round(17 * s), "Light"),
            "name": _font(round(21 * s), "SemiBold"),
            "value": _font(round(21 * s)),
            "small": _font(round(14 * s), "Light"),
            "chip": _font(round(19 * s), "SemiBold"),
        }
        # Bahnschrift n'a pas le signe environ egal.
        self.about = "~"

    def draw(self, stats: LiveStats) -> np.ndarray:
        image = Image.new("RGB", (self.width, self.height), _rgb(style.BACKGROUND))
        pen = ImageDraw.Draw(image)
        s, pad = self.scale, round(26 * self.scale)
        x0, x1 = pad, self.width - pad
        text, muted = _rgb(style.TEXT), _rgb(style.MUTED)

        minutes, seconds = divmod(int(stats.elapsed), 60)
        pen.text((x0, pad), "STATISTIQUES", font=self.fonts["title"], fill=text)
        pen.text((x1, pad), f"{minutes:02d}:{seconds:02d}", font=self.fonts["title"],
                 fill=muted, anchor="ra")
        top = pad + round(56 * s)

        # En haut : la minimap a gauche, l'echange en cours a droite.
        map_height = round(360 * s)
        map_width = map_height // 2
        self._minimap(pen, stats, x0, top, map_width, map_height)
        cx, y = x0 + map_width + round(26 * s), top
        shots = stats.shots if stats.rally_number is None else stats.rally_shots
        label = ("coups joués" if stats.rally_number is None
                 else f"coups, échange n° {stats.rally_number}")
        pen.text((cx, y), str(shots), font=self.fonts["big"], fill=text)
        pen.text((cx, y + round(56 * s)), label, font=self.fonts["label"], fill=muted)
        y += round(86 * s)
        if stats.rally_number is not None:
            pen.text((cx, y), f"plus long : {stats.longest_rally} coups",
                     font=self.fonts["small"], fill=muted)
        y += round(40 * s)
        y = self._duel(pen, stats, cx, x1, y)
        pen.text((cx, y), "balle, dernier coup", font=self.fonts["label"], fill=muted)
        speed = stats.last_shot_speed
        pen.text((cx, y + round(22 * s)),
                 "—" if speed is None else f"{self.about} {speed:.0f} km/h",
                 font=self.fonts["name"], fill=text)
        best = stats.top_shot_speed
        pen.text((cx, y + round(50 * s)),
                 "" if best is None else f"max {self.about} {best:.0f} km/h",
                 font=self.fonts["small"], fill=muted)
        y += round(86 * s)
        if stats.last:
            colour = _rgb(style.KIND.get(stats.last, style.TEXT))
            word = LABELS.get(stats.last, stats.last.upper())
            box = pen.textbbox((0, 0), word, font=self.fonts["chip"])
            w, h = box[2] - box[0] + round(28 * s), box[3] - box[1] + round(16 * s)
            pen.rounded_rectangle((cx, y, cx + w, y + h), radius=round(6 * s), fill=colour)
            pen.text((cx + w / 2, y + h / 2), word, font=self.fonts["chip"],
                     fill=_rgb(style.BACKGROUND), anchor="mm")

        self._players(pen, stats, x0, x1, top + map_height + round(36 * s))
        return np.asarray(image)[:, :, ::-1].copy()

    def _duel(self, pen, stats: LiveStats, x0: int, x1: int, y: int) -> int:
        """The strikes of each pair in the rally in play, as one split bar."""
        s = self.scale
        shots = stats.rally_pair_shots or {"proche": 0, "fond": 0}
        near, far = shots["proche"], shots["fond"]
        pen.text((x0, y), "frappes dans l'échange", font=self.fonts["label"],
                 fill=_rgb(style.MUTED))
        y += round(28 * s)
        height = round(14 * s)
        split = x0 + (x1 - x0) * (near / (near + far) if near + far else 0.5)
        if near:
            pen.rounded_rectangle((x0, y, max(x0 + height, split - 2), y + height),
                                  radius=height // 2, fill=_rgb(PAIR_COLOUR["proche"]))
        if far:
            pen.rounded_rectangle((min(x1 - height, split + 2), y, x1, y + height),
                                  radius=height // 2, fill=_rgb(PAIR_COLOUR["fond"]))
        if not near and not far:
            pen.rounded_rectangle((x0, y, x1, y + height), radius=height // 2,
                                  fill=_rgb(style.LINE))
        y += height + round(8 * s)
        pen.text((x0, y), str(near), font=self.fonts["name"], fill=_rgb(style.TEXT))
        pen.text((x1, y), str(far), font=self.fonts["name"], fill=_rgb(style.TEXT), anchor="ra")
        y += round(28 * s)
        pen.text((x0, y), self.pair_names["proche"], font=self.fonts["small"],
                 fill=_rgb(style.MUTED))
        pen.text((x1, y), self.pair_names["fond"], font=self.fonts["small"],
                 fill=_rgb(style.MUTED), anchor="ra")
        return y + round(36 * s)

    def _players(self, pen, stats: LiveStats, x0: int, x1: int, y: int) -> None:
        """A 2 x 2 grid of the players, with hairlines between the cells."""
        s = self.scale
        text, muted, rule = _rgb(style.TEXT), _rgb(style.MUTED), _rgb(style.LINE)
        pen.text((x0, y), "JOUEURS", font=self.fonts["label"], fill=muted)
        y += round(32 * s)
        pen.line((x0, y, x1, y), fill=text, width=max(1, round(2 * s)))
        gap, height = round(18 * s), round(236 * s)
        width = (x1 - x0 - gap) / 2
        middle = x0 + width + gap / 2
        pen.line((middle, y + round(12 * s), middle, y + 2 * height - round(12 * s)),
                 fill=rule, width=1)
        pen.line((x0, y + height, x1, y + height), fill=rule, width=1)
        for line in stats.players:
            column, row = GRID.get(line.slot, (0, 0))
            bx = x0 + column * (width + gap)
            by = y + row * height + round(16 * s)
            marker = round(11 * s)
            mid = by + round(12 * s)
            pen.rectangle((bx, mid - marker / 2, bx + marker, mid + marker / 2),
                          fill=_rgb(style.PLAYER.get(line.slot, style.TEXT)))
            pen.text((bx + round(20 * s), mid), self.names.get(line.slot, line.slot),
                     font=self.fonts["name"], fill=text, anchor="lm")
            pen.text((bx, by + round(36 * s)),
                     f"{_plural(line.shots, 'frappe')} · {line.volleys} vol. · "
                     f"{line.after_bounce} reb.", font=self.fonts["small"], fill=muted)
            for i, (value, label) in enumerate(self._values(line, stats)):
                vx = bx + (i % 2) * (width / 2)
                vy = by + round(70 * s) + (i // 2) * round(70 * s)
                pen.text((vx, vy), value, font=self.fonts["value"], fill=text)
                pen.text((vx, vy + round(26 * s)), label, font=self.fonts["small"], fill=muted)

    def _values(self, line: PlayerLine, stats: LiveStats) -> list[tuple[str, str]]:
        net = "—" if math.isnan(line.net_share) else f"{100 * line.net_share:.0f} %"
        values = [(f"{line.distance:.0f} m", "parcourus"),
                  (f"{self.about} {line.top_speed:.0f}", "km/h max"),
                  (net, "au filet")]
        if stats.pair_points is not None:
            values.append((f"{line.winners} · {line.errors}", "gagnés · fautes"))
        return values

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
