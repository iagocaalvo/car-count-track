"""Single entry point: set lines -> detect, track and count -> save annotated video.

    python -m carcounter -e front-view                  # every video in the experiment
    python -m carcounter -e aerial-view -v my_video.mp4  # one video

For each video:
  1. the first frame opens; click line 1, line 2 and a point beyond the IN line
     (saved lines are pre-loaded: ENTER reuses them, r redraws),
  2. the video is processed with a live preview (q stops early),
  3. results are written to results/<experiment>/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .paths import CONFIGS, EXPERIMENTS, RESULTS, ROOT, resolve_weights

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def experiment_dir(name: str) -> Path:
    """experiments/<name>/, created from the preset configs/<name>.yaml on first use."""
    d = EXPERIMENTS / name
    if not (d / "experiment.yaml").exists():
        preset = CONFIGS / f"{name}.yaml"
        if not preset.exists():
            available = sorted(p.stem for p in CONFIGS.glob("*.yaml"))
            sys.exit(f"Unknown experiment '{name}'. Presets available in configs/: {', '.join(available)}")
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(preset, d / "experiment.yaml")
        print(f"Created {d.relative_to(ROOT)}/ from {preset.relative_to(ROOT)}.")
    for sub in ("videos", "lines"):
        (d / sub).mkdir(exist_ok=True)
    return d


def list_videos(exp: Path, only: str | None) -> list[Path]:
    if only:
        p = Path(only)
        if not p.exists():
            p = exp / "videos" / only
        if not p.exists():
            sys.exit(f"Video not found: {only}")
        return [p]
    videos = sorted(p for p in (exp / "videos").iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        sys.exit(f"No videos in {(exp / 'videos').relative_to(ROOT)}/ - add your videos there and run again.")
    return videos


def lines_path(exp: Path, video: Path) -> Path:
    return exp / "lines" / f"{video.stem}.json"


def set_lines(exp: Path, video: Path, frame_time: float) -> bool:
    """Open the reference frame, let the user (re)draw the lines and save them."""
    from .lines import load_setup, read_frame, save_setup, select_setup

    frame = read_frame(video, frame_time)
    size = (frame.shape[1], frame.shape[0])
    path = lines_path(exp, video)
    initial = None
    if path.exists():
        try:
            initial = load_setup(path, size)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[{video.name}] saved zone ignored ({e}); draw a new one.")
    print(f"[{video.name}] set the zone: line 1 (2 clicks), line 2 (2), point beyond the IN line (1); ENTER to start.")
    setup = select_setup(frame, window=f"Lines - {video.name}", initial=initial)
    if setup is None:
        print(f"[{video.name}] cancelled, skipping.")
        return False
    save_setup(path, setup, size, video.name)
    return True


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="carcounter",
        description="Set a counting zone (two lines), then detect, track and count vehicles leaving it; save the annotated video.",
    )
    p.add_argument("-e", "--experiment", required=True, help="Experiment / preset name (configs/<name>.yaml), e.g. front-view or aerial-view.")
    p.add_argument("-v", "--video", help="Process a single video (file name in the experiment or a path).")
    p.add_argument("-w", "--weights", help="Override the model weights of the experiment config.")
    p.add_argument("--reuse-lines", action="store_true", help="Skip the line window when lines are already saved.")
    p.add_argument("--frame-time", type=float, default=0.0, help="Second of the video shown to set the lines (default: first frame).")
    p.add_argument("--no-preview", action="store_true", help="Do not show the live preview while processing.")
    p.add_argument("--max-frames", type=int, help="Stop after N processed frames (quick tests).")
    args = p.parse_args(argv)

    from .pipeline import ExperimentConfig, process_video

    exp = experiment_dir(args.experiment)
    cfg = ExperimentConfig.load(exp / "experiment.yaml")
    out_dir = RESULTS / args.experiment
    try:
        weights = resolve_weights(args.weights or cfg.weights)
    except FileNotFoundError as e:
        sys.exit(str(e))

    for video in list_videos(exp, args.video):
        if not (args.reuse_lines and lines_path(exp, video).exists()):
            if not set_lines(exp, video, args.frame_time):
                continue
        summary = process_video(
            video, lines_path(exp, video), cfg, out_dir,
            weights=weights, show=not args.no_preview, max_frames=args.max_frames,
        )
        c = summary["counts"]
        print(f"[{video.name}] IN={c['in']} OUT={c['out']} TOTAL={c['total']}  {json.dumps(c['by_category'])}")
        print(f"[{video.name}] annotated video: {(out_dir / summary['outputs']['video']).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
