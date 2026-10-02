from padel_analysis.analytics.sides import end_changes, half_of_top_row


def test_ends_change_after_each_odd_game_of_a_set():
    readings = [(0, 1, 0, 0), (100, 1, 1, 0), (200, 1, 1, 1), (300, 1, 2, 1)]
    assert end_changes(readings) == [100, 300]


def test_a_set_ended_on_an_odd_game_is_counted_once_then_the_first_game_of_the_next():
    readings = [(0, 1, 5, 3), (100, 1, 6, 3), (200, 2, 0, 1)]
    assert end_changes(readings) == [100, 200]


def test_after_a_set_of_even_games_the_first_game_of_the_next_set_changes_ends():
    readings = [(0, 1, 5, 4), (100, 1, 6, 4), (200, 2, 1, 0)]
    assert end_changes(readings) == [200]


def test_two_odd_games_missed_between_readings_cancel_out():
    readings = [(0, 1, 0, 0), (100, 1, 2, 1)]
    assert end_changes(readings) == []


def test_the_top_row_swaps_halves_at_each_change():
    assert half_of_top_row(50, [100, 300], "near") == "near"
    assert half_of_top_row(150, [100, 300], "near") == "far"
    assert half_of_top_row(300, [100, 300], "near") == "near"
