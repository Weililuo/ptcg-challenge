import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

from base_data import (
    DECKS_PATH,
    MANIFESTS_DIR,
    PROCESSED_DIR,
    ROOT,
    DecisionBatch,
    batch_from_shard,
    file_sha256,
    load_shard,
    read_manifest as read_base_manifest,
    split_shards,
    value_sha256,
)


EXPERT_DATA_DIR = PROCESSED_DIR / "experts"
EXPERT_MANIFESTS_DIR = MANIFESTS_DIR / "experts"
WORK_DIR = ROOT / "workspace" / "expert_data"
SPLITS = ("train", "validation", "test")
FORMAT = "expert"


def ensure_directories() -> None:
    """Create the fixed expert data directories."""
    for path in (EXPERT_DATA_DIR, EXPERT_MANIFESTS_DIR, WORK_DIR):
        path.mkdir(parents=True, exist_ok=True)


def card_requirement(value: str) -> tuple[int, int]:
    """Parse CARD_ID or CARD_ID:COUNT."""
    card_id, separator, count = value.partition(":")
    try:
        result = (int(card_id), int(count) if separator else 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("card must be CARD_ID or CARD_ID:COUNT") from error
    if min(result) <= 0:
        raise argparse.ArgumentTypeError("card ID and count must be positive")
    return result


def load_decks() -> list[dict[str, Any]]:
    """Load the base deck catalog."""
    with DECKS_PATH.open(encoding="utf-8") as file:
        return json.load(file)["decks"]


def matching_decks(
    required_cards: dict[int, int],
    decks: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return decks containing every required card count."""
    matches = []
    catalog = load_decks() if decks is None else decks
    for deck in catalog:
        counts = {card["id"]: card["count"] for card in deck["cards"]}
        if all(counts.get(card_id, 0) >= count for card_id, count in required_cards.items()):
            matches.append(deck)
    return sorted(matches, key=lambda deck: deck["key"])


def _write_json(value: Any, path: Path) -> None:
    """Write readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_index(offsets: list[int], rows: list[torch.Tensor], path: Path) -> None:
    """Write one split's compact row index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    index = {
        "shard_offsets": torch.tensor(offsets, dtype=torch.int64),
        "rows": torch.cat(rows) if rows else torch.empty(0, dtype=torch.int32),
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(index, temporary)
    temporary.replace(path)


def _selected_rows(path: Path, deck_keys: torch.Tensor) -> torch.Tensor:
    """Return rows whose winner deck matches an expert deck."""
    shard = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return torch.isin(shard["deck_key"].long(), deck_keys).nonzero().flatten().to(torch.int32)


def _build_split_index(
    split: str,
    base_manifest: dict[str, Any],
    deck_keys: torch.Tensor,
    path: Path,
) -> tuple[int, int]:
    """Build one split index and return shard and decision counts."""
    paths = split_shards(split, base_manifest)
    offsets = [0]
    rows = []
    for shard_path in paths:
        selected = _selected_rows(shard_path, deck_keys)
        if len(selected):
            rows.append(selected)
        offsets.append(offsets[-1] + len(selected))
    _write_index(offsets, rows, path)
    return len(paths), offsets[-1]


def build_expert_dataset(
    name: str,
    required_cards: dict[int, int],
) -> dict[str, Any]:
    """Build a lightweight expert view over the base dataset."""
    if not name or Path(name).name != name:
        raise ValueError("expert name must be one directory name")
    if not required_cards:
        raise ValueError("at least one core card is required")

    ensure_directories()
    output_dir = EXPERT_DATA_DIR / name
    manifest_path = EXPERT_MANIFESTS_DIR / f"{name}.json"
    if output_dir.exists() or manifest_path.exists():
        raise FileExistsError(f"expert dataset already exists: {name}")

    base_manifest = read_base_manifest()
    matches = matching_decks(required_cards)
    if not matches:
        raise ValueError("no base decks match the core cards")

    stage_root = Path(tempfile.mkdtemp(prefix="build-", dir=WORK_DIR))
    stage_data = stage_root / name
    stage_manifest = stage_root / f"{name}.json"
    deck_keys = torch.tensor([deck["key"] for deck in matches], dtype=torch.int64)
    split_data = {}

    for split in SPLITS:
        stage_index = stage_data / f"{split}.index.pt"
        shard_count, decision_count = _build_split_index(
            split,
            base_manifest,
            deck_keys,
            stage_index,
        )
        final_index = output_dir / stage_index.name
        split_data[split] = {
            "index": final_index.relative_to(ROOT).as_posix(),
            "index_sha256": file_sha256(stage_index),
            "bytes": stage_index.stat().st_size,
            "base_shards": shard_count,
            "decisions": decision_count,
        }

        print(
            json.dumps(
                {
                    "expert": name,
                    "split": split,
                    "decisions": decision_count,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )


    if not split_data["train"]["decisions"] or not split_data["validation"]["decisions"]:
        raise ValueError("expert view has no train or validation decisions")

    card_filter = [
        {"id": card_id, "minimum_count": count}
        for card_id, count in sorted(required_cards.items())
    ]
    identity = {
        "format": FORMAT,
        "name": name,
        "base_dataset_sha256": base_manifest["dataset_sha256"],
        "token_schema_sha256": base_manifest["token_schema_sha256"],
        "filter": {
            "mode": "contains_all",
            "cards": card_filter,
        },
        "matched_decks": [
            {"key": deck["key"], "sha256": deck["sha256"]}
            for deck in matches
        ],
        "splits": split_data,
    }
    manifest = {
        **identity,
        "dataset_sha256": value_sha256(identity),
        "base_manifest": "datasets/manifests/base.json",
        "deck_catalog_sha256": base_manifest["deck_catalog"]["sha256"],
        "learning_decisions": sum(
            split_data[split]["decisions"]
            for split in SPLITS
        ),
    }
    _write_json(manifest, stage_manifest)
    stage_data.replace(output_dir)
    stage_manifest.replace(manifest_path)
    stage_root.rmdir()
    return manifest


def read_expert_manifest(name: str) -> dict[str, Any]:
    """Read an expert view manifest for the current base dataset."""
    path = EXPERT_MANIFESTS_DIR / f"{name}.json"
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    base_manifest = read_base_manifest()
    if manifest["base_dataset_sha256"] != base_manifest["dataset_sha256"]:
        raise ValueError("expert view belongs to a different base dataset")
    return manifest


def split_selection(
    manifest: dict[str, Any],
    split: str,
) -> Iterator[tuple[Path, torch.Tensor]]:
    """Yield base shards and their selected row indices."""
    base_manifest = read_base_manifest()
    paths = split_shards(split, base_manifest)
    index_path = ROOT / manifest["splits"][split]["index"]
    index = torch.load(index_path, map_location="cpu", weights_only=True, mmap=True)
    offsets = index["shard_offsets"]
    rows = index["rows"]
    for shard_index, path in enumerate(paths):
        start = int(offsets[shard_index])
        end = int(offsets[shard_index + 1])
        if start < end:
            yield path, rows[start:end].long()


def iter_batches(
    manifest: dict[str, Any],
    split: str,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> Iterator[DecisionBatch]:
    """Yield batches from an expert view without copying base shards."""
    selections = list(split_selection(manifest, split))
    if shuffle:
        random.Random(seed).shuffle(selections)
    for path, indices in selections:
        shard = load_shard(path)
        if shuffle:
            path_seed = int(hashlib.sha256(path.name.encode()).hexdigest()[:8], 16)
            generator = torch.Generator().manual_seed(seed ^ path_seed)
            indices = indices[torch.randperm(len(indices), generator=generator)]
        for start in range(0, len(indices), batch_size):
            yield batch_from_shard(shard, indices[start : start + batch_size])


def parse_args() -> argparse.Namespace:
    """Parse the expert name and core card requirements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--card", type=card_requirement, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    """Build one expert dataset view."""
    args = parse_args()
    requirements = dict(args.card)
    try:
        manifest = build_expert_dataset(args.name, requirements)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "name": manifest["name"],
                "dataset_sha256": manifest["dataset_sha256"],
                "matched_decks": len(manifest["matched_decks"]),
                "learning_decisions": manifest["learning_decisions"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

