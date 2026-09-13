from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from base_data import ROOT, file_sha256


FORMAT = "learn"


@dataclass(frozen=True)
class OpponentSpec:
    model: Path | None
    deck: Path
    weight: float
    model_type: str


@dataclass(frozen=True)
class DeckSpec:
    deck: Path
    weight: float
    opponents: tuple[OpponentSpec, ...]


@dataclass(frozen=True)
class LearningData:
    source: Path
    sha256: str
    model_type: str
    model_deck: dict[str, Any] | None
    decks: tuple[DeckSpec, ...]


@dataclass(frozen=True)
class Matchup:
    learner_deck: Path
    opponent_model: Path | None
    opponent_deck: Path


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _model_path(value: str | Path) -> Path:
    path = _repository_path(value)
    return path / "model.pt" if path.is_dir() else path


def _record(model: Path) -> dict[str, Any]:
    return json.loads((model.parent / "record.json").read_text(encoding="utf-8"))


def _expert_deck(record: dict[str, Any]) -> Path:
    value = record.get("deck", {}).get("path")
    if record.get("model_type") != "expert" or not value:
        raise ValueError("expert model record has no deck")
    return _repository_path(value)


def _weight(value: Any) -> float:
    weight = float(value)
    if weight <= 0:
        raise ValueError("sampling weights must be positive")
    return weight


def _deck(value: Any, learner_deck: Path) -> Path:
    if value == "same":
        return learner_deck
    if not isinstance(value, str):
        raise ValueError("opponent deck must be a path or 'same'")
    return _repository_path(value)


def _opponent(value: dict[str, Any], learner_deck: Path) -> OpponentSpec:
    kind = value.get("type")
    weight = _weight(value.get("weight", 1))
    if kind == "self":
        return OpponentSpec(
            None,
            _deck(value.get("deck", "same"), learner_deck),
            weight,
            "self",
        )
    if kind != "model" or not value.get("model"):
        raise ValueError("opponent type must be 'self' or a model path")

    model = _model_path(value["model"])
    record = _record(model)
    model_type = record.get("model_type")
    if model_type == "expert":
        if "deck" in value:
            raise ValueError("expert opponent deck comes from its record")
        deck = _expert_deck(record)
    elif model_type == "base":
        deck = _deck(value.get("deck"), learner_deck)
    else:
        raise ValueError(f"opponent is not a base or expert model: {model}")
    return OpponentSpec(model, deck, weight, model_type)


def load_learning_data(path: Path, initial: Path) -> LearningData:
    """Load one JSON sampling strategy for an initial model."""
    source = _repository_path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("format") != FORMAT:
        raise ValueError(f"learning data format must be {FORMAT}")

    initial_record = _record(_model_path(initial))
    model_type = initial_record.get("model_type")
    if model_type not in ("base", "expert"):
        raise ValueError("initial model must be a base or expert model")
    model_deck = initial_record.get("deck") if model_type == "expert" else None
    fixed_deck = _expert_deck(initial_record) if model_deck else None

    decks = []
    for item in value.get("decks") or []:
        if item.get("path") == "record":
            if fixed_deck is None:
                raise ValueError("'record' learner deck requires an expert model")
            learner_deck = fixed_deck
        else:
            learner_deck = _repository_path(item["path"])
        if fixed_deck is not None and learner_deck.resolve() != fixed_deck.resolve():
            raise ValueError("an expert learner must use its recorded deck")
        opponents = tuple(
            _opponent(opponent, learner_deck)
            for opponent in item.get("opponents") or []
        )
        if not opponents:
            raise ValueError(f"learner deck has no opponents: {learner_deck}")
        decks.append(DeckSpec(learner_deck, _weight(item.get("weight", 1)), opponents))
    if not decks:
        raise ValueError("learning data has no decks")

    return LearningData(
        source,
        file_sha256(source),
        model_type,
        dict(model_deck) if model_deck else None,
        tuple(decks),
    )


def sample_matchups(data: LearningData, games: int, seed: int) -> list[Matchup]:
    """Sample a concrete, reproducible rollout plan."""
    rng = random.Random(seed)
    result = []
    deck_weights = [item.weight for item in data.decks]
    for _ in range(games):
        deck = rng.choices(data.decks, deck_weights)[0]
        opponent = rng.choices(
            deck.opponents,
            [item.weight for item in deck.opponents],
        )[0]
        result.append(Matchup(deck.deck, opponent.model, opponent.deck))
    return result


def learning_data_record(data: LearningData) -> dict[str, Any]:
    """Return the compact, resolved strategy stored with a model."""
    def relative(path: Path) -> str:
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    return {
        "path": relative(data.source),
        "sha256": data.sha256,
        "decks": [
            {
                "path": relative(deck.deck),
                "weight": deck.weight,
                "opponents": [
                    {
                        "type": opponent.model_type,
                        **(
                            {"model": relative(opponent.model)}
                            if opponent.model is not None
                            else {}
                        ),
                        "deck": relative(opponent.deck),
                        "weight": opponent.weight,
                    }
                    for opponent in deck.opponents
                ],
            }
            for deck in data.decks
        ],
    }
