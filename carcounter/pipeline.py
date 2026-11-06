"""Detection -> tracking -> two-line counting -> annotated video.

Per processed frame:
  1. a background thread decodes the video (skipped frames are only grabbed, not
     decoded to images) and resizes it to ``process_width``,
  2. the frame is cropped to the bounding box of the zone between the two lines
     and everything outside the zone is blacked out, so the detector only sees
     the zone (and runs on fewer pixels),
  3. YOLO + ByteTrack (ultralytics) give tracked boxes, mapped to categories,
  4. the two-line counter updates and the frame is annotated and written.
"""

from __future__ import annotations

import csv
import json
import queue
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from .counter import LineCounter
from .draw import Annotator, ignored_mask
from .lines import load_setup
from .paths import resolve_weights

ANCHORS = ("bottom_center", "center")


# ----------------------------------------------------------------------- config
@dataclass
class TrackerConfig:
    """Ultralytics ByteTrack settings (see ultralytics/cfg/trackers/bytetrack.yaml)."""

    track_high_thresh: float = 0.25  # first association stage / confident detections
    track_low_thresh: float = 0.1  # low-score detections used to recover tracks
    new_track_thresh: float = 0.25  # minimum score to start a new track
    track_buffer: int = 30  # frames a lost track is kept alive
    match_thresh: float = 0.8
    fuse_score: bool = True

    def write_yaml(self, path: Path) -> Path:
        path.write_text(yaml.safe_dump({"tracker_type": "bytetrack", **asdict(self)}, sort_keys=False))
        return path


@dataclass
class ExperimentConfig:
    name: str
    weights: str = "yolo11s.pt"
    # model class name -> counted category; model classes not listed are ignored.
    classes: dict[str, str] = field(default_factory=lambda: {"car": "car", "truck": "truck", "bus": "truck"})
    imgsz: int = 1280
    iou: float = 0.5
    anchor: str = "bottom_center"
    process_width: int = 1920
    target_fps: float = 25
    region_margin_px: float = 40
    device: str = "auto"
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        tracker = TrackerConfig(**(data.pop("tracker", None) or {}))
        cfg = cls(**data, tracker=tracker)
        if cfg.anchor not in ANCHORS:
            raise ValueError(f"anchor must be one of {ANCHORS}")
        if not isinstance(cfg.classes, dict) or not cfg.classes:
            raise ValueError("classes must be a non-empty mapping {model_class: category}")
        return cfg

    @property
    def categories(self) -> list[str]:
        return list(dict.fromkeys(self.classes.values()))  # unique, in config order


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def class_mapping(model_names: dict[int, str], classes: dict[str, str]) -> dict[int, str]:
    """Model class id -> category, for the model classes listed in the config."""
    mapping = {i: classes[n] for i, n in model_names.items() if n in classes}
    if not mapping:
        raise ValueError(f"None of the configured classes {list(classes)} exist in the model ({list(model_names.values())}).")
    return mapping


def anchor_points(xyxy: np.ndarray, anchor: str) -> np.ndarray:
    """Point of each box that is tested against the lines."""
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    y = xyxy[:, 3] if anchor == "bottom_center" else (xyxy[:, 1] + xyxy[:, 3]) / 2
    return np.stack([cx, y], axis=1)


# ------------------------------------------------------------------- video I/O
class FrameReader:
    """Decode + resize frames in a background thread (OpenCV releases the GIL)."""

    _END = object()

    def __init__(self, cap: cv2.VideoCapture, stride: int, size: tuple[int, int], max_frames: int | None, buffer: int = 8):
        self.cap, self.stride, self.size, self.max_frames = cap, stride, size, max_frames
        self.q: queue.Queue = queue.Queue(maxsize=buffer)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        src_idx, emitted = 0, 0
        try:
            while not self.stop.is_set():
                if self.max_frames and emitted >= self.max_frames:
                    break
                if src_idx % self.stride:
                    if not self.cap.grab():  # advance without decoding to an image
                        break
                    src_idx += 1
                    continue
                ok, frame = self.cap.read()
                if not ok:
                    break
                if frame.shape[1::-1] != self.size:
                    frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
                self._put((src_idx, frame))
                src_idx += 1
                emitted += 1
        finally:
            self._put(self._END)

    def _put(self, item) -> None:
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is self._END:
                return
            yield item

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    for fourcc in ("avc1", "mp4v"):  # H.264 if the OpenCV build supports it
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Could not open a video writer for {path}")


# --------------------------------------------------------------------- pipeline
class RegionDetector:
    """YOLO + ByteTrack restricted to the counting zone of the frame."""

    def __init__(self, cfg: ExperimentConfig, weights: str, ignored: np.ndarray):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(weights)
        self.device = resolve_device(cfg.device)
        self.mapping = class_mapping(self.model.names, cfg.classes)
        self.class_ids = sorted(self.mapping)

        self._tmp = tempfile.TemporaryDirectory()
        self.tracker_yaml = str(cfg.tracker.write_yaml(Path(self._tmp.name) / "bytetrack.yaml"))
        # Detections below the tracker's low threshold are never used, so that is the detector floor.
        self.conf = cfg.tracker.track_low_thresh

        # Crop to the bounding box of the zone and keep the object scale:
        # the inference size shrinks with the crop instead of upscaling it.
        h, w = ignored.shape
        ys, xs = np.nonzero(~ignored)
        if len(xs) == 0:
            raise ValueError("The counting zone is empty: check the two lines.")
        self.x0, self.y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        self.crop = (slice(self.y0, y1), slice(self.x0, x1))
        self.active = np.where(ignored[self.crop], 0, 255).astype(np.uint8)
        self.needs_mask = bool(ignored[self.crop].any())
        ratio = max(x1 - self.x0, y1 - self.y0) / max(w, h)
        self.imgsz = max(320, int(round(cfg.imgsz * ratio / 32)) * 32)

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        roi = frame[self.crop]
        if self.needs_mask:
            roi = cv2.bitwise_and(roi, roi, mask=self.active)
        r = self.model.track(
            roi, persist=True, tracker=self.tracker_yaml, imgsz=self.imgsz, conf=self.conf,
            iou=self.cfg.iou, classes=self.class_ids, device=self.device, verbose=False,
        )[0]
        b = r.boxes
        if b is None or b.id is None or not len(b):
            return np.empty((0, 4), np.float32), np.empty((0,), int), []
        xyxy = b.xyxy.cpu().numpy()
        xyxy[:, [0, 2]] += self.x0
        xyxy[:, [1, 3]] += self.y0
        ids = b.id.cpu().numpy().astype(int)
        cats = [self.mapping[int(c)] for c in b.cls.cpu().numpy()]
        return xyxy, ids, cats

    def close(self) -> None:
        self._tmp.cleanup()


def process_video(
    video: Path,
    lines_json: Path,
    cfg: ExperimentConfig,
    out_dir: Path,
    weights: str | None = None,
    show: bool = False,
    max_frames: int | None = None,
) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(src_fps / cfg.target_fps))
    out_fps = src_fps / stride
    scale = min(1.0, cfg.process_width / src_w)
    size = (round(src_w * scale) // 2 * 2, round(src_h * scale) // 2 * 2)

    setup = load_setup(lines_json, size).extended_to(*size)  # lines reach the frame edges
    ignored = ignored_mask(setup, size, cfg.region_margin_px)
    weights = resolve_weights(weights or cfg.weights)
    detector = RegionDetector(cfg, weights, ignored)
    counter = LineCounter(setup, cfg.categories, max_age_frames=int(src_fps * 10))  # frames = source frame indices
    annotator = Annotator(setup, ignored, cfg.categories)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    out_video = out_dir / f"{stem}_annotated.mp4"
    writer = open_writer(out_video, out_fps, size)
    total = -(-n_frames // stride) if n_frames > 0 else None
    if max_frames:
        total = min(total or max_frames, max_frames)

    reader = FrameReader(cap, stride, size, max_frames)
    window = f"{stem} (q to stop)"
    t0 = time.time()
    processed, stopped_early = 0, False
    tracked_boxes, track_ids_seen = 0, set()
    try:
        with tqdm(total=total, desc=stem, unit="frame") as bar:
            for src_idx, frame in reader:
                xyxy, ids, cats = detector(frame)
                tracked_boxes += len(ids)
                track_ids_seen.update(ids.tolist())
                anchors = anchor_points(xyxy, cfg.anchor)
                tracks = [(int(i), (float(a[0]), float(a[1])), c) for i, a, c in zip(ids, anchors, cats)]
                if counter.update(src_idx, src_idx / src_fps, tracks):
                    bar.set_postfix(IN=counter.counts["IN"], OUT=counter.counts["OUT"], refresh=False)
                writer.write(annotator.draw(frame, xyxy, ids, anchors, counter))
                processed += 1
                bar.update(1)
                if show:
                    preview = frame if size[0] <= 1280 else cv2.resize(frame, (1280, round(size[1] * 1280 / size[0])))
                    cv2.imshow(window, preview)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        stopped_early = True
                        break
    except KeyboardInterrupt:  # Ctrl+C: keep what was processed, like q
        stopped_early = True
        print(f"\n[{stem}] interrupted; saving partial results.")
    finally:
        reader.close()
        cap.release()
        writer.release()
        detector.close()
        if show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    elapsed = time.time() - t0

    events_csv = out_dir / f"{stem}_events.csv"
    with events_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "direction", "category", "frame", "time_s"])
        for e in counter.events:
            w.writerow([e.track_id, e.direction, e.category, e.frame, f"{e.time_s:.2f}"])

    summary = {
        "experiment": cfg.name,
        "video": video.name,
        "weights": Path(weights).name,
        "device": detector.device,
        "source_fps": round(src_fps, 3),
        "processed_fps": round(out_fps, 3),
        "frame_size": list(size),
        "inference_imgsz": detector.imgsz,
        "frames_processed": processed,
        "stopped_early": stopped_early,
        "processing_seconds": round(elapsed, 1),
        "speed_fps": round(processed / elapsed, 2) if elapsed else None,
        "tracking": {
            "mean_tracked_per_frame": round(tracked_boxes / processed, 2) if processed else 0,
            "unique_track_ids": len(track_ids_seen),
        },
        "counts": counter.summary(),
        "outputs": {"video": out_video.name, "events": events_csv.name},
    }
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
