"""Reading the broadcast scoreboard: games, points, set and serving pair.

The points cell is the only light one, and it moves right by one column at each new
set. No OCR: the font never changes, so each cell is compared with templates of known
values, rebuilt from the frames listed in `ground_truth/scoreboard/templates.json`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROWS = ((86, 126), (135, 175))
BAND_X = (240, 520)
CELL_WIDTH = 50
FIRST_SET_RIGHT = 363  # right edge of the points cell in the first set
MATCH_DISTANCE = 0.08
POINTS = ("0", "15", "30", "40")


@dataclass(frozen=True)
class ScoreState:
    """What the scoreboard says at one frame."""

    set_number: int
    games: tuple[int, int]
    points: tuple[str, str]
    server: int | None  # 1 or 2: the row of the serving pair


def light_cell(image: np.ndarray) -> tuple[int, int] | None:
    """Return the x-range of the points cell, found by its light background, or None."""
    x0, x1 = BAND_X
    band = cv2.cvtColor(image[ROWS[0][0]:ROWS[1][1], x0:x1], cv2.COLOR_BGR2GRAY)
    # share of light pixels per column, which the dark digits do not erase
    light = list((band > 170).mean(axis=0) > 0.45) + [False]
    start = None
    for i, value in enumerate(light):
        if value and start is None:
            start = i
        if not value and start is not None:
            if 20 <= i - start <= 80:
                right = x0 + i
                return right - CELL_WIDTH, right
            start = None
    return None


def glyph(cell: np.ndarray, dark_text: bool) -> np.ndarray:
    """Return a cell as a 30x24 contrast-normalised picture of its characters."""
    grey = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    grey = cv2.resize(grey, (30, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    if dark_text:
        grey = 255 - grey
    low, high = np.percentile(grey, 3), np.percentile(grey, 97)
    return np.clip((grey - low) / max(high - low, 1.0), 0.0, 1.0)


def cells(image: np.ndarray) -> dict[str, np.ndarray] | None:
    """Return the four cells of the current set, points and games of each row, as glyphs."""
    found = light_cell(image)
    if found is None:
        return None
    x0, x1 = found
    result = {}
    for row, (y0, y1) in enumerate(ROWS, 1):
        result[f"points_{row}"] = glyph(image[y0:y1, x0 + 3:x1 - 3], dark_text=True)
        result[f"jeux_{row}"] = glyph(image[y0:y1, x0 - 47:x0 - 5], dark_text=False)
    return result


def serving_row(image: np.ndarray, light: tuple[int, int]) -> int | None:
    """Return which row carries the yellow serve marker, in the names part."""
    scores = []
    for y0, y1 in ROWS:
        names = cv2.cvtColor(image[y0:y1, 60:light[0] - 60], cv2.COLOR_BGR2HSV)
        yellow = (names[..., 0] > 20) & (names[..., 0] < 40) & (names[..., 1] > 120) & (
            names[..., 2] > 150)
        scores.append(float(yellow.mean()))
    if max(scores) < 0.02:
        return None
    return 1 if scores[0] > scores[1] else 2


class Scoreboard:
    """Reader of a frame's score, comparing each cell with templates of known values."""

    def __init__(self, templates: dict[str, list[tuple[str, np.ndarray]]]) -> None:
        self.templates = templates  # "points" / "jeux" -> [(value, glyph)]

    @classmethod
    def from_examples(cls, examples_path: Path, video_of) -> "Scoreboard":
        """Cut the templates from the listed frames; `video_of(match)` opens that match."""
        examples = json.loads(Path(examples_path).read_text(encoding="utf-8"))["examples"]
        templates: dict[str, list[tuple[str, np.ndarray]]] = {"points": [], "jeux": []}
        for example in examples:
            image = video_of(example["match"], example["frame"])
            found = cells(image)
            if found is None:
                raise ValueError(f"no scoreboard at {example['match']} {example['frame']}")
            glyph_ = found[f"{example['cell']}_{example['row']}"]
            templates[example["cell"]].append((example["value"], glyph_))
        return cls(templates)

    def _value(self, kind: str, picture: np.ndarray) -> str | None:
        best, distance = None, MATCH_DISTANCE
        for value, template in self.templates[kind]:
            gap = float(np.abs(picture - template).mean())
            if gap < distance:
                best, distance = value, gap
        return best

    def read(self, image: np.ndarray) -> ScoreState | None:
        """Return the score on this frame, or None where the scoreboard is hidden or unclear."""
        light = light_cell(image)
        found = cells(image)
        if light is None or found is None:
            return None
        values = {name: self._value(name.split("_")[0], picture)
                  for name, picture in found.items()}
        if any(value is None for value in values.values()):
            return None
        return ScoreState(
            set_number=1 + round((light[1] - FIRST_SET_RIGHT) / CELL_WIDTH),
            games=(int(values["jeux_1"]), int(values["jeux_2"])),
            points=(values["points_1"], values["points_2"]),
            server=serving_row(image, light),
        )
