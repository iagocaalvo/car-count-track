"""2D geometry helpers for line-crossing detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True)
class Line:
    """A line segment from ``a`` to ``b`` in pixel coordinates."""

    a: Point
    b: Point

    @property
    def length(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])

    @property
    def midpoint(self) -> Point:
        return ((self.a[0] + self.b[0]) / 2, (self.a[1] + self.b[1]) / 2)

    def side(self, p: Point) -> float:
        """Signed area (cross product) of ``p`` relative to the infinite line a->b.

        The sign tells which side of the line ``p`` lies on; zero means on the line.
        """
        return (self.b[0] - self.a[0]) * (p[1] - self.a[1]) - (self.b[1] - self.a[1]) * (p[0] - self.a[0])

    def side_sign(self, p: Point) -> int:
        s = self.side(p)
        return (s > 0) - (s < 0)

    def signed_distance(self, p: Point) -> float:
        """Perpendicular signed distance (pixels) from ``p`` to the infinite line."""
        return self.side(p) / self.length

    def crossed_by(self, p: Point, q: Point) -> bool:
        """True if the movement p -> q crosses this *segment* (strict side change).

        Endpoints exactly on the line do not count as a crossing; callers should
        keep the last position that was strictly on one side.
        """
        sp, sq = self.side(p), self.side(q)
        if sp * sq >= 0:
            return False
        # Intersection point of p->q with the infinite line.
        t = sp / (sp - sq)
        x = (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
        # Project onto the segment to check the crossing lies within a..b.
        ex, ey = self.b[0] - self.a[0], self.b[1] - self.a[1]
        u = ((x[0] - self.a[0]) * ex + (x[1] - self.a[1]) * ey) / (ex * ex + ey * ey)
        return 0.0 <= u <= 1.0

    def scaled(self, sx: float, sy: float) -> "Line":
        return Line((self.a[0] * sx, self.a[1] * sy), (self.b[0] * sx, self.b[1] * sy))

    def extended_to(self, width: float, height: float) -> "Line":
        """The infinite line through a and b, clipped to the frame [0, width] x [0, height]."""
        (ax, ay), (bx, by) = self.a, self.b
        dx, dy = bx - ax, by - ay
        ts = []
        if dx:
            ts += [(0 - ax) / dx, (width - ax) / dx]
        if dy:
            ts += [(0 - ay) / dy, (height - ay) / dy]
        eps = 1e-6
        inside = [t for t in ts if -eps <= ax + t * dx <= width + eps and -eps <= ay + t * dy <= height + eps]
        if len(inside) < 2:
            return self  # the line does not cross the frame; keep the clicked segment
        t0, t1 = min(inside), max(inside)
        return Line((ax + t0 * dx, ay + t0 * dy), (ax + t1 * dx, ay + t1 * dy))

    def intersects(self, other: "Line") -> bool:
        """True if the two segments cross (strictly)."""
        return other.crossed_by(self.a, self.b)
