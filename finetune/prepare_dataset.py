"""Build an auto-labelled YOLO dataset of aerial vehicles from a video.

Steps:
1. sample every N-th frame and downscale to --width (same scale used at inference),
2. cut each frame into overlapping square tiles (small vehicles stay large enough),
3. label every tile with a large "teacher" model (COCO car -> 'car', truck/bus -> 'truck'),
   using test-time augmentation for better recall,
4. split train/val *by time* (the last part of the video is validation) to avoid near-duplicate leakage,
5. write a YOLO dataset + data.yaml.

The labels are a starting point: review/correct them (e.g. Label Studio or CVAT)
before trusting the fine-tuned model.

    python finetune/prepare_dataset.py \
        --video data/finetune/aerial-view/148317-793718152.mp4 \
        --out datasets/aerial-vehicles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from carcounter.paths import resolve_weights  # noqa: E402
from carcounter.pipeline import resolve_device  # noqa: E402

# COCO teacher class -> dataset class id. Matches the categories counted by the experiments.
TEACHER_TO_DATASET = {"car": 0, "truck": 1, "bus": 1}
DATASET_NAMES = {0: "car", 1: "truck"}


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    origins = list(range(0, length - tile, step))
    origins.append(length - tile)
    return origins


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", default="datasets/aerial-vehicles", type=Path)
    ap.add_argument("--every", type=int, default=6, help="Keep one frame every N source frames.")
    ap.add_argument("--width", type=int, default=1920, help="Downscale frames to this width (match process_width).")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--teacher", default="yolo11x.pt", help="Large model used for auto-labelling.")
    ap.add_argument("--teacher-imgsz", type=int, default=1280, help="Tiles are upscaled to this size for the teacher.")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--val-frac", type=float, default=0.2, help="Fraction of the video (at the end) used for validation.")
    ap.add_argument("--keep-empty", action="store_true", help="Also keep tiles without any vehicle.")
    ap.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation of the teacher.")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from ultralytics import YOLO

    teacher = YOLO(resolve_weights(args.teacher))
    to_dataset = {i: TEACHER_TO_DATASET[n] for i, n in teacher.names.items() if n in TEACHER_TO_DATASET}
    class_ids = sorted(to_dataset)
    device = resolve_device(args.device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    val_start = int(n * (1 - args.val_frac))
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": [0, 0, 0], "val": [0, 0, 0]}  # tiles, cars, trucks
    idx = -1
    with tqdm(total=n // args.every, desc="frames") as bar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % args.every:
                continue
            bar.update(1)
            h, w = frame.shape[:2]
            if w > args.width:
                frame = cv2.resize(frame, (args.width, round(h * args.width / w)), interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
            split = "val" if idx >= val_start else "train"
            for y0 in tile_origins(h, args.tile, args.overlap):
                for x0 in tile_origins(w, args.tile, args.overlap):
                    crop = frame[y0 : y0 + args.tile, x0 : x0 + args.tile]
                    ch, cw = crop.shape[:2]
                    res = teacher.predict(
                        crop, imgsz=args.teacher_imgsz, conf=args.conf, classes=class_ids,
                        augment=not args.no_tta, device=device, verbose=False,
                    )[0]
                    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
                    labels = [to_dataset[int(c)] for c in res.boxes.cls.cpu().numpy()] if res.boxes is not None else []
                    if len(boxes) == 0 and not args.keep_empty:
                        continue
                    name = f"{args.video.stem}_f{idx:06d}_x{x0}_y{y0}"
                    cv2.imwrite(str(args.out / "images" / split / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    lines = []
                    for (x1, y1, x2, y2), cls in zip(boxes, labels):
                        cx, cy, bw, bh = (x1 + x2) / 2 / cw, (y1 + y2) / 2 / ch, (x2 - x1) / cw, (y2 - y1) / ch
                        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    (args.out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                    stats[split][0] += 1
                    stats[split][1] += labels.count(0)
                    stats[split][2] += labels.count(1)
    cap.release()

    data = {"path": str(args.out.resolve()), "train": "images/train", "val": "images/val", "names": DATASET_NAMES}
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    for split, (tiles, cars, trucks) in stats.items():
        print(f"{split}: {tiles} tiles, {cars} cars, {trucks} trucks")
    print(f"Dataset written to {args.out} (review labels before training for best results).")


if __name__ == "__main__":
    main()
