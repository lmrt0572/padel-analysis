"""When the two pairs change ends, read off the scoreboard.

The tracker's slots name halves of the court, not people: after a change of ends,
`near_1` is someone else. The rules say when it happens - after the first, third and
every other odd game of a set, and at the end of a set whose games add up to an odd
number - and the scoreboard says how many games have been played. Measured on both
finals against the identity ground truth: every change after the first reading of
the scoreboard is found on the very frame, and one the ground truth had missed.
"""

from collections.abc import Sequence

Reading = tuple[int, int, int, int]
"""Frame, set number, games of the top row, games of the bottom row."""


def end_changes(readings: Sequence[Reading]) -> list[int]:
    """The first frame read after each change of ends."""
    changes = []
    previous = None
    for frame, set_number, top, bottom in sorted(readings):
        games = top + bottom
        if previous is not None:
            last_set, last_games = previous
            if set_number == last_set:
                crossed = [k for k in range(last_games + 1, games + 1) if k % 2 == 1]
                if len(crossed) % 2 == 1:
                    changes.append(frame)
            elif set_number > last_set:
                # Le tableau garde le score du set fini jusqu'au premier point du suivant :
                # son dernier jeu, impair ou non, a deja ete compte. Restent les jeux
                # impairs du nouveau set ; deux changements s'annulent.
                flips = sum(1 for k in range(1, games + 1) if k % 2 == 1)
                if flips % 2 == 1:
                    changes.append(frame)
        previous = (set_number, games)
    return changes


def half_of_top_row(frame: int, changes: Sequence[int], first_half: str) -> str:
    """Which half the scoreboard's top row plays in at `frame`.

    Args:
        first_half: the half it plays in before the first change.
    """
    flips = sum(1 for change in changes if change <= frame)
    if flips % 2 == 0:
        return first_half
    return "far" if first_half == "near" else "near"
