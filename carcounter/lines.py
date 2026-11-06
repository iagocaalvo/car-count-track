"""Interactive line selection (5 clicks) and JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .counter import CountingSetup
from .geometry import Line

STEPS = [
    "LINE 1: click point 1",
    "LINE 1: click point 2",
    "LINE 2: click point 1",
    "LINE 2: click point 2",
    "Click a point OUTSIDE the zone, beyond the line that counts as IN",
]
HELP = "[u] undo  [r] reset  [enter/s] save  [q/esc] cancel"

LINE_COLOR = (0, 215, 255)  # BGR yellow (before the IN side is chosen)
IN_COLOR = (80, 200, 80)  # BGR green
OUT_COLOR = (60, 60, 230)  # BGR red


# --------------------------------------------------------------------------- IO
def save_setup(path: Path, setup: CountingSetup, frame_size: tuple[int, int], video_name: str) -> None:
    data = {
        "video": video_name,
        "frame_size": list(frame_size),
        "line_1": [list(setup.line_1.a), list(setup.line_1.b)],
        "line_2": [list(setup.line_2.a), list(setup.line_2.b)],
        "in_point": list(setup.in_point),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_setup(path: Path, target_size: tuple[int, int] | None = None) -> CountingSetup:
    """Load a setup and rescale it from the stored frame size to ``target_size`` (w, h).

    Files saved by earlier versions (``detection_line`` / ``counting_line``) are accepted.
    """
    data = json.loads(Path(path).read_text())
    l1 = data.get("line_1", data.get("detection_line"))
    l2 = data.get("line_2", data.get("counting_line"))
    setup = CountingSetup(Line(tuple(l1[0]), tuple(l1[1])), Line(tuple(l2[0]), tuple(l2[1])), tuple(data["in_point"]))
    if target_size is not None:
        w0, h0 = data["frame_size"]
        setup = setup.scaled(target_size[0] / w0, target_size[1] / h0)
    return setup


def read_frame(video: Path, time_s: float = 0.0) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read a frame at {time_s}s from {video}")
    return frame


# --------------------------------------------------------------- interactive UI
def _label(img, text, org, color, t):
    cv2.putText(img, text, (org[0] + 6, org[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, color, 2 * t, cv2.LINE_AA)


def _draw_overlay(img: np.ndarray, pts: list[tuple[int, int]], message: str | None) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    t = max(1, round(w / 800))
    colors = [LINE_COLOR, LINE_COLOR]
    if len(pts) == 5:  # colour the lines once the IN side is known
        try:
            setup = CountingSetup(Line(pts[0], pts[1]), Line(pts[2], pts[3]), pts[4])
            colors = [IN_COLOR, OUT_COLOR] if setup.in_line is setup.line_1 else [OUT_COLOR, IN_COLOR]
        except ValueError:
            pass
    for i, p in enumerate(pts[:4]):
        cv2.circle(out, p, 4 * t, colors[i // 2], -1, cv2.LINE_AA)
    for k in range(min(len(pts), 4) // 2):
        a, b = pts[2 * k], pts[2 * k + 1]
        if a != b:  # the line extends to the frame edges (thin), the clicked segment is thick
            ext = Line(a, b).extended_to(w - 1, h - 1)
            cv2.line(out, tuple(map(round, ext.a)), tuple(map(round, ext.b)), colors[k], t, cv2.LINE_AA)
        cv2.line(out, a, b, colors[k], 2 * t, cv2.LINE_AA)
        name = f"LINE {k + 1}"
        if len(pts) == 5 and colors[k] != LINE_COLOR:
            name = "IN" if colors[k] == IN_COLOR else "OUT"
        _label(out, name, a, colors[k], t)
    if len(pts) == 5:
        cv2.circle(out, pts[4], 6 * t, IN_COLOR, -1, cv2.LINE_AA)
        _label(out, "IN side", pts[4], IN_COLOR, t)

    title = STEPS[len(pts)] if len(pts) < len(STEPS) else "Done: press ENTER/s to save"
    lines = [title, HELP] + ([message] if message else [])
    box_h = 26 * t * len(lines) + 10
    cv2.rectangle(out, (0, 0), (w, box_h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        color = (0, 215, 255) if (message and text == message) else (255, 255, 255)
        cv2.putText(out, text, (10, 22 * t + i * 26 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, color, t, cv2.LINE_AA)
    return out


def select_setup(
    frame: np.ndarray,
    window: str = "Line setup",
    initial: CountingSetup | None = None,
    max_display_width: int = 1600,
) -> CountingSetup | None:
    """Open a window on ``frame`` and let the user click the 5 points.

    ``initial`` (in frame pixels) pre-loads previously saved lines: ENTER reuses
    them, ``r`` clears them to draw new ones.
    Returns the setup in the *frame's* pixel coordinates, or None if cancelled.
    """
    h, w = frame.shape[:2]
    scale = min(1.0, max_display_width / w)
    disp = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale < 1 else frame
    pts: list[tuple[int, int]] = []
    state = {"message": None}
    if initial is not None:
        raw = [initial.line_1.a, initial.line_1.b, initial.line_2.a, initial.line_2.b, initial.in_point]
        pts = [(round(x * scale), round(y * scale)) for x, y in raw]
        state["message"] = "Saved lines loaded: ENTER to reuse, r to redraw"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < len(STEPS):
            pts.append((x, y))
            state["message"] = None

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            cv2.imshow(window, _draw_overlay(disp, pts, state["message"]))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key == ord("u") and pts:
                pts.pop()
                state["message"] = None
            elif key == ord("r"):
                pts.clear()
                state["message"] = None
            elif key in (13, 10, ord("s")) and len(pts) == len(STEPS):
                p = [(x / scale, y / scale) for x, y in pts]
                try:
                    setup = CountingSetup(Line(p[0], p[1]), Line(p[2], p[3]), p[4])
                except ValueError as e:  # wrong IN point (or overlapping lines): click it again
                    state["message"] = f"Invalid setup: {e}"
                    pts.pop()
                    continue
                try:
                    setup.extended_to(w, h)  # extended lines must not cross inside the frame
                except ValueError:
                    state["message"] = "The lines cross inside the frame once extended: redraw them."
                    pts.clear()
                    continue
                return setup
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)
