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
        # Un set gagne : le vainqueur est celui qui menait, les jeux repartent de zero.
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
