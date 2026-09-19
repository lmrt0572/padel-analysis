import numpy as np

from padel_analysis.io.scoreboard import ROWS, Scoreboard, light_cell


def _board(right=363, light=250):
    image = np.full((1080, 1920, 3), 30, dtype=np.uint8)
    image[ROWS[0][0]:ROWS[1][1], right - 50:right] = light
    return image


def test_the_points_cell_is_found_by_its_light_background():
    assert light_cell(_board()) == (313, 363)


def test_the_cell_moves_right_by_one_column_at_the_second_set():
    assert light_cell(_board(right=413)) == (363, 413)


def test_no_light_cell_means_no_scoreboard():
    assert light_cell(_board(light=30)) is None


def test_a_cell_is_read_as_its_nearest_template_or_not_at_all():
    blank = np.zeros((24, 30), dtype=np.float32)
    bar = blank.copy()
    bar[:, 12:18] = 1.0
    board = Scoreboard({"points": [("0", blank), ("15", bar)], "jeux": []})
    assert board._value("points", bar) == "15"
    assert board._value("points", 1.0 - blank) is None
