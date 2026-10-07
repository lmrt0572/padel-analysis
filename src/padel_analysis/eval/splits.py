"""Which frames are used for what, declared once.

Shots exist only on frames 0 to 20099 and the men's ball annotation stops at 21472,
so the evaluation slices sit inside both. Bounds are inclusive.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameRange:
    start: int
    stop: int

    def contains(self, frame: int) -> bool:
        return self.start <= frame <= self.stop

    @property
    def length(self) -> int:
        return self.stop - self.start + 1


SPLITS: dict[str, FrameRange | tuple[FrameRange, ...]] = {
    "finalf_train": (FrameRange(0, 15_999), FrameRange(20_100, 45_933)),
    "finalf_eval": FrameRange(16_000, 20_099),
    "finalm_heldout": FrameRange(0, 20_099),
    "finalm_heldout_ball_only": FrameRange(20_100, 21_472),
}
