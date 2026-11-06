"""Add VisDrone truck tiles to the aerial dataset (human-labelled trucks from drones).

The auto-labelled dataset from prepare_dataset.py has few lorries, so the
fine-tuned model misses some trucks. VisDrone2019-DET has hand-labelled drone
images with many trucks and buses. This script:

1. downloads VisDrone2019-DET train + val (~1.5 GB) unless already present,
2. maps its classes to ours: car/van -> car (0), truck/bus -> truck (1); others dropped,
3. greys out VisDrone's "ignored regions" (unlabelled crowds/far areas) so they
   don't teach the model that vehicles there are background,
4. cuts 640 px tiles (same as prepare_dataset.py) and keeps the tiles that
   contain at least one truck,
5. writes a merged dataset: the base dataset (repeated --base-repeat times, so it
   keeps its weight) + a random sample of --max-train-tiles VisDrone train tiles
   (train) and the VisDrone val tiles (val).

    python finetune/add_visdrone.py --base datasets/aerial-vehicles --out datasets/aerial-vehicles-trucks

VisDrone is licensed for academic/research use only; its images are not redistributed.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import DATASET_NAMES, tile_origins  # noqa: E402

# VisDrone category ids (annotation column 6) -> our class ids
VISDRONE_TO_DATASET = {4: 0, 5: 0, 6: 1, 9: 1}  # car, van -> car ; truck, bus -> truck
SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val"}


def download(root: Path) -> None:
    missing = [name for name in SPLITS.values() if not (root / name / "annotations").exists()]
    if not missing:
        return
    from ultralytics.utils import ASSETS_URL
    from ultralytics.utils.downloads import download as ul_download

    root.mkdir(parents=True, exist_ok=True)
    ul_download([f"{ASSETS_URL}/{name}.zip" for name in missing], dir=root, threads=2)


def read_annotations(path: Path):
    """Yield (x1, y1, x2, y2, category, ignored) from a VisDrone annotation file."""
    for row in path.read_text().strip().splitlines():
        v = row.split(",")
        if len(v) < 6:
            continue
        x, y, w, h, score, cat = (int(float(t)) for t in v[:6])
        ignored = score == 0 or cat in (0, 11)  # ignored regions / "others"
        yield x, y, x + w, y + h, cat, ignored


def process_split(src: Path, out: Path, split: str, args) -> tuple[int, int, int]:
    img_out, lbl_out = out / "images" / split, out / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n_tiles = n_cars = n_trucks = 0
    for ann in tqdm(sorted((src / "annotations").glob("*.txt")), desc=f"VisDrone {split}"):
        boxes = list(read_annotations(ann))
        if not any(VISDRONE_TO_DATASET.get(c) == 1 and not ig for *_, c, ig in boxes):
            continue  # no truck in the whole image
        img = cv2.imread(str(src / "images" / f"{ann.stem}.jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(1.0, args.width / w)
        if scale < 1:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        mean = img.reshape(-1, 3).mean(0).astype(int).tolist()
        objs = []
        for x1, y1, x2, y2, c, ig in boxes:
            x1, y1, x2, y2 = (round(v * scale) for v in (x1, y1, x2, y2))
            cls = VISDRONE_TO_DATASET.get(c)
            if ig:
                cv2.rectangle(img, (x1, y1), (x2, y2), mean, -1)  # grey out ignored regions
            elif cls is not None:
                objs.append((x1, y1, x2, y2, cls))

        for y0 in tile_origins(h, args.tile, args.overlap):
            for x0 in tile_origins(w, args.tile, args.overlap):
                tw, th = min(args.tile, w - x0), min(args.tile, h - y0)
                lines, classes = [], []
                for x1, y1, x2, y2, cls in objs:
                    cx1, cy1, cx2, cy2 = max(x1, x0), max(y1, y0), min(x2, x0 + tw), min(y2, y0 + th)
                    area, vis = (x2 - x1) * (y2 - y1), max(0, cx2 - cx1) * max(0, cy2 - cy1)
                    if area <= 0 or vis / area < 0.5:  # keep boxes at least half inside the tile
                        continue
                    bx, by = (cx1 + cx2) / 2 - x0, (cy1 + cy2) / 2 - y0
                    lines.append(f"{cls} {bx / tw:.6f} {by / th:.6f} {(cx2 - cx1) / tw:.6f} {(cy2 - cy1) / th:.6f}")
                    classes.append(cls)
                if 1 not in classes:
                    continue  # keep only tiles with a truck
                name = f"visdrone_{ann.stem}_x{x0}_y{y0}"
                cv2.imwrite(str(img_out / f"{name}.jpg"), img[y0 : y0 + th, x0 : x0 + tw], [cv2.IMWRITE_JPEG_QUALITY, 95])
                (lbl_out / f"{name}.txt").write_text("\n".join(lines) + "\n")
                n_tiles += 1
                n_cars += classes.count(0)
                n_trucks += classes.count(1)
    return n_tiles, n_cars, n_trucks


def _link(src: Path, dst: Path) -> None:
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def link_base(base: Path, out: Path, repeat: int) -> None:
    """Hard-link (or copy) the base dataset into the merged one; train tiles ``repeat`` times."""
    for split in ("train", "val"):
        copies = repeat if split == "train" else 1
        for sub in ("images", "labels"):
            dst = out / sub / split
            dst.mkdir(parents=True, exist_ok=True)
            for f in (base / sub / split).iterdir():
                for k in range(copies):
                    _link(f, dst / (f.name if k == 0 else f"{f.stem}_rep{k}{f.suffix}"))


def limit_tiles(out: Path, split: str, max_tiles: int, seed: int = 0) -> int:
    """Keep a random sample of ``max_tiles`` VisDrone tiles in ``split``; delete the rest."""
    tiles = sorted((out / "images" / split).glob("visdrone_*.jpg"))
    if max_tiles <= 0 or len(tiles) <= max_tiles:
        return len(tiles)
    keep = set(random.Random(seed).sample(tiles, max_tiles))
    for img in tiles:
        if img not in keep:
            img.unlink()
            (out / "labels" / split / f"{img.stem}.txt").unlink(missing_ok=True)
    return max_tiles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("datasets/visdrone-raw"), help="Where VisDrone is downloaded.")
    ap.add_argument("--base", type=Path, default=Path("datasets/aerial-vehicles"))
    ap.add_argument("--out", type=Path, default=Path("datasets/aerial-vehicles-trucks"))
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--width", type=int, default=1920, help="Downscale wider images to this width.")
    ap.add_argument("--max-train-tiles", type=int, default=2500, help="Random sample of VisDrone train tiles to keep (0 = all).")
    ap.add_argument("--base-repeat", type=int, default=2, help="Repeat the base train tiles to keep their weight.")
    args = ap.parse_args()

    download(args.root)
    link_base(args.base, args.out, args.base_repeat)
    for split, folder in SPLITS.items():
        tiles, cars, trucks = process_split(args.root / folder, args.out, split, args)
        print(f"VisDrone {split}: {tiles} tiles with trucks ({cars} cars, {trucks} trucks)")
    kept = limit_tiles(args.out, "train", args.max_train_tiles)
    print(f"VisDrone train: kept a random sample of {kept} tiles")
    data = {"path": str(args.out.resolve()), "train": "images/train", "val": "images/val", "names": DATASET_NAMES}
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    for split in ("train", "val"):
        print(f"{split}: {len(list((args.out / 'images' / split).glob('*.jpg')))} tiles in total")
    print(f"Merged dataset written to {args.out}")


if __name__ == "__main__":
    main()
