"""Who won each point, from the scoreboard before and after it.

Padel counts points as tennis does - 0, 15, 30, 40, game - with a golden point at
40-40 on this tour: the next point takes the game. So between two readings one point
apart, exactly one pair's score moves up by one step, or one pair wins a game and both
points go back to 0. Any other change means a point was missed between the readings,
or a cell was misread: it is left undecided rather than guessed, and counting those
cases is how the reading itself is checked.
"""

from ..io.scoreboard import POINTS, ScoreState

Winner = int | None


def point_winner(before: ScoreState, after: ScoreState) -> Winner:
    """1 or 2 for the pair that won the point between two readings, None if unclear."""
    if after.set_number != before.set_number:
        # A set won: the winner is the one who was leading, the games start again from zero.
        if after.set_number == before.set_number + 1 and before.games[0] != before.games[1]:
            return 1 if before.games[0] > before.games[1] else 2
        return None
    game_steps = (after.games[0] - before.games[0], after.games[1] - before.games[1])
    if game_steps in ((1, 0), (0, 1)):
        winner = 1 if game_steps == (1, 0) else 2
        return winner if after.points == ("0", "0") else None
    if game_steps != (0, 0):
        return None
    steps = tuple(POINTS.index(a) - POINTS.index(b)
                  for a, b in zip(after.points, before.points, strict=True))
    if steps == (1, 0):
        return 1
    if steps == (0, 1):
        return 2
    return None


WINNER, ERROR = "gagnant", "faute"


def side_of(slot: str) -> str:
    """The half of the court a player slot stands in: "near" or "far"."""
    return "near" if slot.startswith("near") else "far"


def serving_side(server_row: int | None, strikes: list[tuple[int, str]]) -> dict[int, str]:
    """Which half each scoreboard row plays in, for one rally.

    The scoreboard marks the serving pair's row, and the rally's first strike is the
    serve: the server's half is that pair's half, and the other row has the other one.
    Empty when either is unknown.
    """
    if server_row is None or not strikes:
        return {}
    serving = side_of(min(strikes)[1])
    other = "far" if serving == "near" else "near"
    return {server_row: serving, 3 - server_row: other}


def credit(winner_side: str, strikes: list[tuple[int, str]]) -> tuple[str, str] | None:
    """Who the point goes to, and how: the rally's last striker.

    Struck by the winning pair, the last shot is a winner; by the other pair, the ball
    never came back and the point is that player's error. A strike the contact model
    missed hands the point to the wrong player - the split is only as good as it is.
    """
    if not strikes:
        return None
    _, slot = max(strikes)
    return slot, WINNER if side_of(slot) == winner_side else ERROR
