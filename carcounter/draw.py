"""Frame annotation: lines, ignored region, boxes, trails and counters."""

from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np

from .counter import IN, OUT, CountingSetup, LineCounter
from .lines import IN_COLOR, OUT_COLOR

WHITE = (255, 255, 255)
PALETTE = [
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
    (0, 114, 178), (213, 94, 0), (204, 121, 167), (255, 255, 255),
]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _ipt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _half_plane(line, inside: int, w: int, h: int, margin: float) -> np.ndarray:
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    side = (line.b[0] - line.a[0]) * (ys - line.a[1]) - (line.b[1] - line.a[1]) * (xs - line.a[0])
    return (side / line.length) * inside >= -margin


def ignored_mask(setup: CountingSetup, frame_size: tuple[int, int], margin: float) -> np.ndarray:
    """Boolean HxW mask of everything outside the zone between the two lines (+ margin)."""
    w, h = frame_size
    zone = _half_plane(setup.in_line, setup.in_inside, w, h, margin) & _half_plane(setup.out_line, setup.out_inside, w, h, margin)
    return ~zone


def _outward_arrow(line, inside: int, length: float):
    """Arrow from the line's midpoint pointing out of the zone."""
    mx, my = line.midpoint
    nx, ny = -(line.b[1] - line.a[1]) / line.length, (line.b[0] - line.a[0]) / line.length
    if line.side_sign((mx + nx, my + ny)) == inside:
        nx, ny = -nx, -ny
    tip = (mx + nx * length, my + ny * length)
    label = (mx + nx * length * 1.3, my + ny * length * 1.3)
    return _ipt((mx, my)), _ipt(tip), _ipt(label)


class Annotator:
    def __init__(self, setup: CountingSetup, ignored: np.ndarray, categories: list[str], trail_len: int = 30):
        self.setup = setup
        h, w = ignored.shape
        self.t = max(1, round(w / 960))  # thickness/font scale
        self.categories = categories
        self.trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_len))
        self.ignored = ignored[..., None]  # HxWx1, broadcasts over BGR
        self.has_ignored = bool(ignored.any())

        # Arrows pointing out of the zone: leaving through the IN line = IN, through the OUT line = OUT.
        L = 0.05 * w
        self.arrow_in = _outward_arrow(setup.in_line, setup.in_inside, L)
        self.arrow_out = _outward_arrow(setup.out_line, setup.out_inside, L)

    def _text(self, img, text, org, color, scale=0.6, thick=None, bg=True):
        thick = thick or self.t
        scale *= self.t
        x, y = org
        if bg:
            (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
            cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

    def draw(self, frame: np.ndarray, xyxy: np.ndarray, track_ids: np.ndarray, anchors: np.ndarray, counter: LineCounter) -> np.ndarray:
        """Draw in place on ``frame`` and return it."""
        t = self.t
        out = frame
        if self.has_ignored:
            np.copyto(out, (out >> 1) + (out >> 3), where=self.ignored)  # ~0.6x brightness

        for line, (a, b, lab), color, name in (
            (self.setup.in_line, self.arrow_in, IN_COLOR, "IN"),
            (self.setup.out_line, self.arrow_out, OUT_COLOR, "OUT"),
        ):
            cv2.line(out, _ipt(line.a), _ipt(line.b), color, 3 * t, cv2.LINE_AA)
            cv2.arrowedLine(out, a, b, color, 2 * t, cv2.LINE_AA, tipLength=0.25)
            self._text(out, name, lab, color, 0.6)

        live = set()
        for box, tid, anc in zip(xyxy, track_ids, anchors):
            tid = int(tid)
            live.add(tid)
            status = counter.status(tid)
            color = IN_COLOR if status == IN else OUT_COLOR if status == OUT else PALETTE[tid % len(PALETTE)]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, (2 if status else 1) * t, cv2.LINE_AA)
            label = f"#{tid} {counter.category(tid) or ''}" + (f" {status}" if status else "")
            self._text(out, label, (x1, max(y1 - 4, 12)), color, 0.4)
            trail = self.trails[tid]
            trail.append(_ipt(anc))
            if len(trail) > 1:
                cv2.polylines(out, [np.array(trail, dtype=np.int32)], False, color, t, cv2.LINE_AA)
        for tid in [k for k in self.trails if k not in live]:
            del self.trails[tid]

        self._draw_panel(out, counter)
        return out

    def _draw_panel(self, out: np.ndarray, counter: LineCounter) -> None:
        """Table: one row per category plus TOTAL, columns IN / OUT."""
        t = self.t
        rows = [(c.upper(), counter.counts_by_category.get(c, {IN: 0, OUT: 0})) for c in self.categories]
        rows.append(("TOTAL", counter.counts))
        row_h, pad = 28 * t, 10 * t
        col_in, col_out = 120 * t, 190 * t
        width, height = 250 * t, pad * 2 + row_h * (len(rows) + 1)
        cv2.rectangle(out, (0, 0), (width, height), (0, 0, 0), -1)
        y = pad + row_h - 8 * t
        self._text(out, "IN", (col_in, y), IN_COLOR, 0.65, 2 * t, bg=False)
        self._text(out, "OUT", (col_out, y), OUT_COLOR, 0.65, 2 * t, bg=False)
        for i, (name, c) in enumerate(rows, start=1):
            y = pad + row_h * (i + 1) - 8 * t
            if name == "TOTAL":
                cv2.line(out, (pad, y - row_h + 6 * t), (width - pad, y - row_h + 6 * t), (120, 120, 120), 1)
            self._text(out, name, (pad, y), WHITE, 0.65, 2 * t, bg=False)
            self._text(out, str(c[IN]), (col_in, y), IN_COLOR, 0.65, 2 * t, bg=False)
            self._text(out, str(c[OUT]), (col_out, y), OUT_COLOR, 0.65, 2 * t, bg=False)
