"""Project paths and model-weight resolution, shared by the CLI and the fine-tuning scripts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"
RUNS = ROOT / "runs"
DATASETS = ROOT / "datasets"


def resolve_weights(weights: str | Path) -> str:
    """Return a usable path for ``weights``.

    * an existing path (absolute, relative to the repo root or to the cwd) is used as is;
    * a bare official name such as ``yolo11s.pt`` maps to ``weights/yolo11s.pt``, where
      ultralytics downloads it on first use (instead of cluttering the working directory);
    * anything else must exist, otherwise FileNotFoundError is raised.
    """
    p = Path(weights)
    for candidate in (p, ROOT / p):
        if candidate.exists():
            return str(candidate.resolve())
    if p.parent == Path("."):
        WEIGHTS.mkdir(exist_ok=True)
        return str(WEIGHTS / p.name)
    raise FileNotFoundError(f"Weights not found: {weights} (see finetune/ to train them).")
