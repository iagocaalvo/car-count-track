"""Compare detectors on the aerial experiment videos (same lines, same settings).

    python finetune/compare.py --models yolo11s.pt weights/aerial-vehicle-v1.pt weights/aerial-vehicle.pt

For each model and video it runs the full pipeline (no preview) into
results/aerial-view/compare/<model>/ and prints a Markdown table with counts,
tracked vehicles per frame (detection recall proxy), unique track IDs
(fragmentation proxy) and speed. With --snapshot it also saves a side-by-side
frame of all the models to results/<experiment>/compare/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from carcounter.paths import EXPERIMENTS, RESULTS, ROOT, resolve_weights  # noqa: E402
from carcounter.pipeline import ExperimentConfig, process_video  # noqa: E402


def snapshot(videos: list[Path], labels: list[str], time_s: float, out: Path) -> None:
    frames = []
    for v, label in zip(videos, labels):
        cap = cv2.VideoCapture(str(v))
        cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
        ok, f = cap.read()
        cap.release()
        if not ok:
            return
        f = cv2.resize(f, (960, round(f.shape[0] * 960 / f.shape[1])))
        cv2.rectangle(f, (0, f.shape[0] - 40), (f.shape[1], f.shape[0]), (0, 0, 0), -1)
        cv2.putText(f, label, (12, f.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(f)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(frames), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"snapshot: {out.resolve().relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["yolo11s.pt", "weights/aerial-vehicle-v1.pt", "weights/aerial-vehicle.pt"])
    ap.add_argument("--experiment", default="aerial-view")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--snapshot", nargs=2, metavar=("VIDEO_STEM", "SECOND"), help="Save a side-by-side frame to results/<experiment>/compare/.")
    args = ap.parse_args()

    exp = EXPERIMENTS / args.experiment
    out_root = RESULTS / args.experiment / "compare"
    cfg = ExperimentConfig.load(exp / "experiment.yaml")
    videos = sorted(p for p in (exp / "videos").glob("*.mp4") if (exp / "lines" / f"{p.stem}.json").exists())
    if not videos:
        sys.exit(f"No videos with saved lines in {exp}. Run `python -m carcounter -e {args.experiment}` first.")

    rows = []
    for model in args.models:
        weights = resolve_weights(model)
        name = Path(model).stem
        for video in videos:
            s = process_video(
                video, exp / "lines" / f"{video.stem}.json", cfg,
                out_root / name, weights=weights, max_frames=args.max_frames,
            )
            c = s["counts"]["by_category"]
            rows.append((
                video.stem, name,
                f"{c['car']['IN']} / {c['car']['OUT']}", f"{c['truck']['IN']} / {c['truck']['OUT']}",
                s["tracking"]["mean_tracked_per_frame"], s["tracking"]["unique_track_ids"], s["speed_fps"],
            ))

    print("\n| Video | Model | Cars IN / OUT | Trucks IN / OUT | Tracked per frame | Unique track IDs | Speed (fps) |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(rows):
        print("| " + " | ".join(str(x) for x in r) + " |")

    if args.snapshot and len(args.models) >= 2:
        stem, sec = args.snapshot
        names = [Path(m).stem for m in args.models]
        vids = [out_root / n / f"{stem}_annotated.mp4" for n in names]
        snapshot(vids, names, float(sec), out_root / f"compare_{stem}.jpg")


if __name__ == "__main__":
    main()
