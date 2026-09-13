from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from base_data import file_sha256
from expert_data import (
    EXPERT_MANIFESTS_DIR,
    build_expert_dataset,
    card_requirement,
    iter_batches,
    read_expert_manifest,
)
from test import evaluate_model
from train import TrainingConfig, train


SPLITS = ("train", "validation", "test")
EXPERT_TRAINING_DEFAULTS = {
    "epochs": 1,
    "learning_rate": 5e-5,
    "evaluate_every": 500,
    "validation_batches": 512,
}


def load_config(name: str, path: Path | None) -> dict[str, Any]:
    """Load an optional expert model configuration."""
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    default = ROOT / "configs" / "experts" / f"{name}.json"
    return json.loads(default.read_text(encoding="utf-8")) if default.exists() else {}


def dataset_info(name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Build compact expert dataset information for the model record."""
    return {
        "directory": f"datasets/processed/experts/{name}",
        "manifest": f"datasets/manifests/experts/{name}.json",
        "dataset_sha256": manifest["dataset_sha256"],
        "base_dataset_sha256": manifest["base_dataset_sha256"],
        "filter": manifest["filter"],
        "matched_decks": len(manifest["matched_decks"]),
        "splits": {
            split: {"decisions": manifest["splits"][split]["decisions"]}
            for split in SPLITS
        },
    }


def run(
    name: str,
    config: dict[str, Any],
    deck: str | None = None,
    base: str | None = None,
    data: str | None = None,
    cards: list[tuple[int, int]] | None = None,
    resume: bool = False,
) -> None:
    """Train and test one expert model."""
    deck_value = deck or config.get("deck")
    base_name = base or config.get("base")
    data_name = data or config.get("data", name)
    required_cards = (
        dict(cards)
        if cards
        else {int(card_id): int(count) for card_id, count in config.get("cards", {}).items()}
    )
    if not deck_value or not base_name:
        raise ValueError("expert training requires a deck and base model")

    manifest_path = EXPERT_MANIFESTS_DIR / f"{data_name}.json"
    if not manifest_path.exists():
        build_expert_dataset(data_name, required_cards)
    manifest = read_expert_manifest(data_name)
    manifest_cards = {
        card["id"]: card["minimum_count"]
        for card in manifest["filter"]["cards"]
    }
    if required_cards and required_cards != manifest_cards:
        raise ValueError("expert cards do not match the existing data view")

    def batches(split: str, batch_size: int, seed: int, shuffle: bool):
        """Yield batches from one expert dataset split."""
        return iter_batches(manifest, split, batch_size, seed, shuffle)

    deck_path = ROOT / deck_value
    base_model = ROOT / "models" / "base" / base_name / "model.pt"
    output_dir = ROOT / "models" / "experts" / name
    device = config.get("device")
    training = {**EXPERT_TRAINING_DEFAULTS, **config.get("training", {})}
    train(
        name,
        batches,
        dataset_info(data_name, manifest),
        output_dir,
        ROOT / "workspace" / "train" / "experts" / name,
        config=TrainingConfig(**training),
        initial_model=None if resume else base_model,
        resume=resume,
        device=device,
        info={
            "model_type": "expert",
            "deck": {
                "path": Path(deck_value).as_posix(),
                "sha256": file_sha256(deck_path),
            },
            "base_model": {
                "name": base_name,
                "path": base_model.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(base_model),
            },
        },
    )
    evaluate_model(
        output_dir,
        batches,
        **{"device": device, **config.get("test", {})},
    )


def parse_args() -> argparse.Namespace:
    """Parse the expert model launch settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--deck")
    parser.add_argument("--base")
    parser.add_argument("--data")
    parser.add_argument("--card", type=card_requirement, action="append")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Launch expert model training."""
    args = parse_args()
    try:
        run(
            args.name,
            load_config(args.name, args.config),
            args.deck,
            args.base,
            args.data,
            args.card,
            args.resume,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
