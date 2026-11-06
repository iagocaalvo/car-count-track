"""Zone-based directional vehicle counter.

Setup (all in pixel coordinates of the processed frame):

* **Two lines** bound the **counting zone** (the area between them). Vehicles
  are only detected and tracked inside the zone (plus a small margin).
* **IN point** - a point beyond one of the lines (outside the zone). That line
  is the **IN line**; the other one is the **OUT line**.

A vehicle is counted once, as soon as it **leaves the zone** through a line:

* leaves through the IN line  -> IN
* leaves through the OUT line -> OUT

Entering the zone through a line is not counted. The category of a track
(e.g. car / truck) is the majority vote of the per-frame detector classes, so a
truck briefly detected as a car is still counted as a truck.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .geometry import Line, Point

IN = "IN"
OUT = "OUT"


@dataclass(frozen=True)
class CountingSetup:
    line_1: Line
    line_2: Line
    in_point: Point

    def __post_init__(self) -> None:
        if self.line_1.length == 0 or self.line_2.length == 0:
            raise ValueError("Lines must have two distinct points.")
        if self.line_1.side_sign(self.line_2.midpoint) == 0 or self.line_2.side_sign(self.line_1.midpoint) == 0:
            raise ValueError("The two lines must not lie on top of each other.")
        beyond_1 = self.line_1.side_sign(self.in_point) == -self._inside(self.line_1, self.line_2)
        beyond_2 = self.line_2.side_sign(self.in_point) == -self._inside(self.line_2, self.line_1)
        if beyond_1 == beyond_2:
            raise ValueError("The IN point must lie outside the zone, beyond exactly one of the lines.")

    @staticmethod
    def _inside(line: Line, other: Line) -> int:
        """Side sign of ``line`` that faces the zone (where the other line is)."""
        return line.side_sign(other.midpoint)

    @property
    def in_line(self) -> Line:
        beyond_1 = self.line_1.side_sign(self.in_point) == -self._inside(self.line_1, self.line_2)
        return self.line_1 if beyond_1 else self.line_2

    @property
    def out_line(self) -> Line:
        return self.line_2 if self.in_line is self.line_1 else self.line_1

    @property
    def in_inside(self) -> int:
        """Zone side sign of the IN line."""
        return self._inside(self.in_line, self.out_line)

    @property
    def out_inside(self) -> int:
        """Zone side sign of the OUT line."""
        return self._inside(self.out_line, self.in_line)

    def in_zone(self, p: Point, margin: float = 0.0) -> bool:
        """True if ``p`` lies between the two lines (or within ``margin`` px outside them)."""
        return (
            self.in_line.signed_distance(p) * self.in_inside >= -margin
            and self.out_line.signed_distance(p) * self.out_inside >= -margin
        )

    def scaled(self, sx: float, sy: float) -> "CountingSetup":
        return CountingSetup(self.line_1.scaled(sx, sy), self.line_2.scaled(sx, sy), (self.in_point[0] * sx, self.in_point[1] * sy))

    def extended_to(self, width: float, height: float) -> "CountingSetup":
        """Both lines extended to the frame edges, so vehicles count up to the borders of the image.

        Raises ValueError if the extended lines cross inside the frame (the zone would be ill-defined).
        """
        l1, l2 = self.line_1.extended_to(width, height), self.line_2.extended_to(width, height)
        if l1.intersects(l2):
            raise ValueError("The two lines cross inside the frame once extended; draw them so they don't meet.")
        return CountingSetup(l1, l2, self.in_point)


@dataclass
class CountEvent:
    track_id: int
    direction: str
    frame: int
    time_s: float
    category: str


@dataclass
class _LineState:
    pos: Point | None = None  # last position strictly off the line
    side: int = 0


@dataclass
class _TrackState:
    last_frame: int
    votes: Counter = field(default_factory=Counter)
    in_line: _LineState = field(default_factory=_LineState)
    out_line: _LineState = field(default_factory=_LineState)
    counted: str | None = None
    category: str | None = None  # frozen once counted

    def voted_category(self) -> str:
        return self.category or self.votes.most_common(1)[0][0]


def _exits(line: Line, inside: int, st: _LineState, p: Point) -> bool:
    """Update the line state with ``p``; True if the move crossed ``line`` from the zone side outwards."""
    s = line.side_sign(p)
    if s == 0:
        return False
    exited = st.side == inside and s == -inside and line.crossed_by(st.pos, p)
    st.pos, st.side = p, s
    return exited


class LineCounter:
    """Stateful counter fed with tracked anchor points frame by frame."""

    def __init__(self, setup: CountingSetup, categories: Sequence[str] = (), max_age_frames: int = 300) -> None:
        self.setup = setup
        self.max_age_frames = max_age_frames
        self.counts: dict[str, int] = {IN: 0, OUT: 0}
        self.counts_by_category: dict[str, dict[str, int]] = {c: {IN: 0, OUT: 0} for c in categories}
        self.events: list[CountEvent] = []
        self._tracks: dict[int, _TrackState] = {}

    @property
    def total(self) -> int:
        return self.counts[IN] + self.counts[OUT]

    def status(self, track_id: int) -> str | None:
        """Direction a track was counted with, or None if not (yet) counted."""
        st = self._tracks.get(track_id)
        return st.counted if st else None

    def category(self, track_id: int) -> str | None:
        """Current majority-vote category of a track."""
        st = self._tracks.get(track_id)
        return st.voted_category() if st and st.votes else None

    def update(self, frame_idx: int, time_s: float, tracks: Iterable[tuple[int, Point, str]]) -> list[CountEvent]:
        """Feed ``(track_id, anchor_point, category)`` for the current frame.

        Returns the count events produced in this frame.
        """
        s = self.setup
        in_line, out_line, in_inside, out_inside = s.in_line, s.out_line, s.in_inside, s.out_inside
        new_events: list[CountEvent] = []

        for track_id, p, category in tracks:
            st = self._tracks.get(track_id)
            if st is None:
                st = self._tracks[track_id] = _TrackState(last_frame=frame_idx)
            st.last_frame = frame_idx
            st.votes[category] += 1

            exit_in = _exits(in_line, in_inside, st.in_line, p)
            exit_out = _exits(out_line, out_inside, st.out_line, p)
            if st.counted is None and (exit_in or exit_out):
                st.counted = IN if exit_in else OUT
                st.category = st.voted_category()
                self.counts[st.counted] += 1
                self.counts_by_category.setdefault(st.category, {IN: 0, OUT: 0})[st.counted] += 1
                ev = CountEvent(track_id, st.counted, frame_idx, time_s, st.category)
                self.events.append(ev)
                new_events.append(ev)

        self._prune(frame_idx)
        return new_events

    def _prune(self, frame_idx: int) -> None:
        stale = [tid for tid, st in self._tracks.items() if frame_idx - st.last_frame > self.max_age_frames]
        for tid in stale:
            del self._tracks[tid]

    def summary(self) -> dict:
        return {
            "in": self.counts[IN],
            "out": self.counts[OUT],
            "total": self.total,
            "by_category": {k: dict(v) for k, v in self.counts_by_category.items()},
        }
