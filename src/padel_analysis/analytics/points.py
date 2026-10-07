"""Who won each point, from the scoreboard before and after it.

Between two readings one point apart, one pair moves up one step or wins a game.
Any other change is left undecided rather than guessed.
"""

from ..io.scoreboard import POINTS, ScoreState

Winner = int | None


def point_winner(before: ScoreState, after: ScoreState) -> Winner:
    """Return 1 or 2 for the pair that won the point between two readings, None if unclear."""
    if after.set_number != before.set_number:
        # a set won: the leader takes it and the games start again from zero
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
    """Return the half of the court a player slot stands in: "near" or "far"."""
    return "near" if slot.startswith("near") else "far"


def serving_side(server_row: int | None, strikes: list[tuple[int, str]]) -> dict[int, str]:
    """Return which half each scoreboard row plays in for one rally, from the serve.

    Empty when the server or the first strike is unknown.
    """
    if server_row is None or not strikes:
        return {}
    serving = side_of(min(strikes)[1])
    other = "far" if serving == "near" else "near"
    return {server_row: serving, 3 - server_row: other}


def credit(winner_side: str, strikes: list[tuple[int, str]]) -> tuple[str, str] | None:
    """Return who the point goes to, and how: the rally's last striker.

    A last shot by the winning pair is a winner; by the other pair, an error.
    """
    if not strikes:
        return None
    _, slot = max(strikes)
    return slot, WINNER if side_of(slot) == winner_side else ERROR
