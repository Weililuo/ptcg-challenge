from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

BASE_DIR = Path(__file__).resolve().parent

AI_PY = BASE_DIR / "ai.py"
MAIN_DECK_CSV = BASE_DIR / "deck.csv"
HISTORY_CSV = BASE_DIR / "train_features.csv"

# The benchmark asset library synced from the upstream repo (currently on the v4
# branch). It is the source of truth for both the environment deck list and the
# official torch expert models used as rollout opponents.
OWEN_ROOT = BASE_DIR / "ptcg_owen"
EXPERT_DECKS_DIR = OWEN_ROOT / "decks"
EXPERT_MODELS_DIR = OWEN_ROOT / "models" / "experts"

# Pre-owen location, kept only so old runs can still be reproduced explicitly.
LEGACY_EXPERT_DECKS_DIR = BASE_DIR / "ptcg-ai" / "decks"

DEPLOYED_MODEL = BASE_DIR / "xgb_model.json"
DEPLOYED_MAPPING = BASE_DIR / "action_label_mapping.json"
CANDIDATE_MODEL = BASE_DIR / "candidate_xgb_model.json"
CANDIDATE_MAPPING = BASE_DIR / "candidate_action_label_mapping.json"

SELFPLAY_DIR = BASE_DIR / "selfplay"

ENV_MODEL = "PTCG_MODEL_PATH"
ENV_MAPPING = "PTCG_MAPPING_PATH"

FEATURE_NAMES = [
    "turn",
    "turn_action_count",
    "hand_size",
    "hand_grass_cnt",
    "hand_lightning_cnt",
    "hand_fight_cnt",
    "is_active_raging_bolt",
    "active_hp_ratio",
    "active_energy_cnt",
    "has_latias_ex",
    "bench_ogerpon_cnt",
    "total_board_energies",
    "my_prizes_remaining",
    "opp_prizes_remaining",
    "prize_diff",
    "opp_active_immune_ex",
    "options_count",
    "my_bench_lowest_hp",
    "opp_has_crustle",
    "opp_has_dragapult",
    "opp_has_froslass",
]

NEW_FEATURE_COLS = ["opp_tier_expert", "opp_deck_id"]

# Metadata carried alongside a rollout row but deliberately NOT fed to the
# model: at inference time the opponent's identity is unknown, so making it an
# input would train on a signal that is constant (unknown) at serve time.
# These columns are for sampling stratification, archiving and diagnostics.
# `opp_model_version` / `is_torch_expert` record which checkpoint actually
# piloted the opponent so a silent fallback to the XGB agent stays visible.
METADATA_COLS = ["opp_tier_expert", "opp_deck_id", "opp_model_version", "is_torch_expert"]

# Model inputs. Keeping this equal to FEATURE_NAMES is what lets
# retrain_dagger.py warm-start the deployed booster instead of cold-starting.
TRAIN_COLS = FEATURE_NAMES

LABEL_COL = "label_action_idx"
WEIGHT_COL = "sample_weight"

TIER_EXPERT = "expert"
TIER_GENERAL = "general"
SELF_DECK_NAME = "self_deck"
UNKNOWN_DECK_ID = -1.0

# The three decks the previous phase was specialised on. The pipeline no longer
# locks onto them: the full 8-deck pool is the default. This set survives only
# for the `--focus-nemesis` debugging/ablation switch and for the arena report.
# Deck csv stems must match ptcg_owen/decks/*.csv names.
NEMESIS_DECKS = {"crustle", "dragapult", "froslass"}

# Default share of rollout matches played against the official expert pool; the
# remainder is self-mirror, which keeps a stable source of self-deck winning
# trajectories flowing into DAgger.
DEFAULT_EXPERT_FRAC = 0.70
DEFAULT_MIRROR_FRAC = 1.0 - DEFAULT_EXPERT_FRAC

# Per-deck sampling boost applied to the nemesis decks. 1.0 = uniform attention
# across the whole pool; the per-deck deficit term in
# rollout_worker.compute_adaptive_weights already upweights decks we lose to.
DEFAULT_NEMESIS_BOOST = 1.0

# `models/experts/<stem>_v<version>` — the stem is the deck identity and is
# what the DeckRegistry encodes. The version deliberately stays out of the
# registry key so bumping v4 -> v5 cannot renumber existing decks.
EXPERT_RECORD_RE = re.compile(r"^(?P<stem>.+)_v(?P<version>\d+)$")


def log(message: str) -> None:
    print(message, flush=True)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def atomic_copy_replace(source: Path, target: Path, backup: Path | None = None) -> None:
    if backup is not None and target.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copy2(source, temporary)
    os_replace(temporary, target)


def os_replace(source: Path, target: Path) -> None:
    import os

    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(source), str(target))


def round_dir(out_dir: Path, round_no: int) -> Path:
    return out_dir / f"round_{int(round_no):03d}"


def load_deck(path: Path) -> list[int]:
    with path.open(encoding="utf-8") as file:
        deck = [int(line.strip()) for line in file if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck {path} has {len(deck)} cards, expected 60")
    return deck


def discover_deck_pool(directory: Path) -> list[tuple[str, Path]]:
    if not directory.is_dir():
        return []
    return sorted((path.stem, path) for path in directory.glob("*.csv"))


@dataclass(frozen=True)
class ExpertSpec:
    """One official expert checkpoint and the deck it was trained with."""

    stem: str
    version: int
    model_path: Path
    deck_path: Path


def discover_expert_models(
    models_dir: Path | None = None,
    owen_root: Path | None = None,
) -> dict[str, ExpertSpec]:
    """Find the newest expert checkpoint per deck stem.

    Pairing is driven by ``record.json``'s ``deck.path`` rather than assuming
    ``decks/<stem>.csv``: that field is the authoritative deck the policy was
    trained against, and ``make_agent`` feeds it into ``encode_decision``, so a
    mismatch would silently give the opponent the wrong deck context.

    Stems whose newest record is unreadable, lacks a deck path, or has no
    ``model.pt`` next to it are skipped with a warning rather than raising —
    the rollout pool degrades to the XGB fallback for that deck.
    """
    models_dir = models_dir or EXPERT_MODELS_DIR
    owen_root = owen_root or OWEN_ROOT
    if not models_dir.is_dir():
        return {}

    best: dict[str, ExpertSpec] = {}
    for record_path in sorted(models_dir.glob("*/record.json")):
        record = read_json(record_path, default={})
        if not isinstance(record, dict) or record.get("model_type") != "expert":
            continue
        match = EXPERT_RECORD_RE.match(str(record.get("name") or record_path.parent.name))
        if match is None:
            log(f"[decks] skipping expert record with unparsable name: {record_path}")
            continue
        stem, version = match["stem"], int(match["version"])

        model_path = record_path.parent / "model.pt"
        if not model_path.exists():
            log(f"[decks] skipping {stem}_v{version}: missing {model_path.name}")
            continue

        deck_rel = (record.get("deck") or {}).get("path")
        if not deck_rel:
            log(f"[decks] skipping {stem}_v{version}: record.json has no deck.path")
            continue
        deck_path = Path(deck_rel)
        if not deck_path.is_absolute():
            deck_path = owen_root / deck_path
        if not deck_path.exists():
            log(f"[decks] skipping {stem}_v{version}: deck missing at {deck_path}")
            continue

        current = best.get(stem)
        if current is None or version > current.version:
            best[stem] = ExpertSpec(stem, version, model_path, deck_path)

    return best


class DeckRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.registry: dict[str, int] = {}
        existing = read_json(path, default={})
        if isinstance(existing, dict):
            self.registry = {str(name): int(idx) for name, idx in existing.items()}

    def id_of(self, name: str) -> int:
        name = str(name)
        if name not in self.registry:
            self.registry[name] = (max(self.registry.values(), default=0)) + 1
            write_json(self.path, self.registry)
        return self.registry[name]

    def seed_experts(self, stems: Any) -> None:
        """Pre-assign deterministic ids to the canonical expert decks.

        Ids are handed out in sorted stem order, so a fresh checkout reproduces
        exactly the encoding an existing deck_registry.json already holds, and
        ids never churn when a checkpoint version is bumped (the version is not
        part of the key).
        """
        changed = False
        for stem in sorted(str(s) for s in stems):
            key = f"expert:{stem}"
            if key not in self.registry:
                self.registry[key] = (max(self.registry.values(), default=0)) + 1
                changed = True
        if changed:
            write_json(self.path, self.registry)


def compute_sample_weight(
    final_score: float,
    n_rows: int,
    mode: str = "match_total",
    w_min: float = 0.25,
    w_max: float = 5.0,
    score_floor: float = 0.0,
) -> float | None:
    if n_rows <= 0 or final_score < score_floor:
        return None
    if mode == "flat":
        mass = 1.0
    elif mode == "match_total":
        mass = max(w_min, min(float(final_score), w_max))
    else:
        raise ValueError(f"unknown weight mode: {mode!r}")
    return mass / n_rows
