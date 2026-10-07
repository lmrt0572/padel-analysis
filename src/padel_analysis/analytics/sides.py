"""When the two pairs change ends, read off the scoreboard.

The tracker's slots name halves of the court, not people. The pairs change after
every odd game of a set, and the scoreboard says how many games have been played.
"""

from collections.abc import Sequence

Reading = tuple[int, int, int, int]
"""Frame, set number, games of the top row, games of the bottom row."""


def end_changes(readings: Sequence[Reading]) -> list[int]:
    """Return the first frame read after each change of ends."""
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
                # the board keeps the finished set until the next point: only the odd games of
                # the new set are left to count
                flips = sum(1 for k in range(1, games + 1) if k % 2 == 1)
                if flips % 2 == 1:
                    changes.append(frame)
        previous = (set_number, games)
    return changes


def half_of_top_row(frame: int, changes: Sequence[int], first_half: str) -> str:
    """Return which half the scoreboard's top row plays in at `frame`.

    Args:
        first_half: the half it plays in before the first change.
    """
    flips = sum(1 for change in changes if change <= frame)
    if flips % 2 == 0:
        return first_half
    return "far" if first_half == "near" else "near"
