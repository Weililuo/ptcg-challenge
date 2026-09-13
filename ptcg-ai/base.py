from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from base_data import file_sha256, iter_batches, read_manifest, split_shards
from common import ModelConfig
from test import evaluate_model
from train import TrainingConfig, train


SPLITS = ("train", "validation", "test")


def load_config(name: str, path: Path | None) -> dict[str, Any]:
    """Load an optional base model configuration."""
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    default = ROOT / "configs" / "base" / f"{name}.json"
    return json.loads(default.read_text(encoding="utf-8")) if default.exists() else {}


def dataset_info(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build compact base dataset information for the model record."""
    return {
        "directory": "datasets/processed/base",
        "manifest": "datasets/manifests/base.json",
        "dataset_sha256": manifest["dataset_sha256"],
        "splits": {
            split: {
                "episodes": manifest["splits"][split]["episodes"],
                "decisions": manifest["splits"][split]["decisions"],
            }
            for split in SPLITS
        },
    }


def requirements_info() -> dict[str, Any]:
    """Build environment information from requirements.txt."""
    path = ROOT / "requirements.txt"
    packages = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return {
        "path": "requirements.txt",
        "sha256": file_sha256(path),
        "packages": packages,
    }


def run(name: str, config: dict[str, Any], resume: bool = False) -> None:
    """Train and test one base model."""
    manifest = read_manifest()
    paths = {split: split_shards(split, manifest) for split in SPLITS}

    def batches(split: str, batch_size: int, seed: int, shuffle: bool):
        """Yield batches from one base dataset split."""
        return iter_batches(paths[split], batch_size, seed, shuffle)

    output_dir = ROOT / "models" / "base" / name
    device = config.get("device")
    train(
        name,
        batches,
        dataset_info(manifest),
        output_dir,
        ROOT / "workspace" / "train" / "base" / name,
        config=TrainingConfig(**config.get("training", {})),
        model_config=ModelConfig(**config.get("model", {})),
        resume=resume,
        device=device,
        info={
            "model_type": "base",
            "environment": {"requirements": requirements_info()},
        },
    )
    evaluate_model(
        output_dir,
        batches,
        **{"device": device, **config.get("test", {})},
    )


def parse_args() -> argparse.Namespace:
    """Parse the base model launch settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Launch base model training."""
    args = parse_args()
    try:
        run(args.name, load_config(args.name, args.config), args.resume)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
