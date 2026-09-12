"""Which frames are used for what, declared once.

Three annotations cover three different ranges and the split has to satisfy all
three at once. Shots exist only on frames 0 to 20099, so an evaluation slice taken
outside that range would allow no contact metric at all - which is why the women's
evaluation slice sits inside the shot range and training happens on either side of
it. The men's ball annotation stops at 21472; evaluating past it would count every
correct detection as a false positive.

Bounds are inclusive.
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
