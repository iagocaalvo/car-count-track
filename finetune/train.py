"""Fine-tune a COCO-pretrained YOLO model on the aerial vehicle dataset.

    python finetune/train.py --data datasets/aerial-vehicles/data.yaml

The best checkpoint is copied to weights/aerial-vehicle.pt. Use it with:

    python -m carcounter -e aerial-view -w weights/aerial-vehicle.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from carcounter.paths import RUNS, WEIGHTS, resolve_weights  # noqa: E402
from carcounter.pipeline import resolve_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/aerial-vehicles/data.yaml")
    ap.add_argument("--model", default="yolo11s.pt", help="Starting checkpoint (COCO-pretrained).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640, help="Should match the tile size used in prepare_dataset.py.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--freeze", type=int, default=10, help="Freeze the first N layers (10 = backbone in YOLO11; 0 = train all).")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default="aerial-vehicle")
    ap.add_argument("--out", default=str(WEIGHTS / "aerial-vehicle.pt"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(resolve_weights(args.model))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=resolve_device(args.device),
        project=str(RUNS / "finetune"),
        name=args.name,
        exist_ok=True,
        freeze=args.freeze or None,
        # Top-down views: vertical flips are valid vehicle appearances.
        fliplr=0.5,
        flipud=0.5,
    )
    best = Path(model.trainer.best)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)
    print(f"Best weights: {best} -> {args.out}")


if __name__ == "__main__":
    main()
