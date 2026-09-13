import argparse
import hashlib
import json
import random
import tempfile
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
MANIFESTS_DIR = DATASETS_DIR / "manifests"
METADATA_DIR = DATASETS_DIR / "metadata"
BASE_DIR = PROCESSED_DIR / "base"
MANIFEST_PATH = MANIFESTS_DIR / "base.json"
DECKS_PATH = METADATA_DIR / "decks.json"
WORK_DIR = ROOT / "workspace" / "base_data"
SPLITS = ("train", "validation", "test")
FORMAT = "base"
PIPELINE_VERSION = 2
REMOVED_OBSERVATION_FIELDS = {
    "remainingOverageTime",
    "search_begin_input",
    "visualize",
}


class EntityKind(IntEnum):
    STATE = 1
    SELECT = 2
    PLAYER = 3
    CARD = 4
    POKEMON = 5
    ENERGY = 6
    HIDDEN = 7
    LOG = 8
    CONTEXT_CARD = 9
    EFFECT_CARD = 10
    OPTION = 11


class Owner(IntEnum):
    UNKNOWN = 1
    SELF = 2
    OPPONENT = 3


class Zone(IntEnum):
    NONE = 1
    DECK = 2
    HAND = 3
    DISCARD = 4
    ACTIVE = 5
    BENCH = 6
    PRIZE = 7
    STADIUM = 8
    ENERGY = 9
    TOOL = 10
    PRE_EVOLUTION = 11
    PLAYER = 12
    LOOKING = 13
    OWN_DECK = 14


CATEGORICAL_FIELDS = (
    "kind",
    "zone",
    "target_zone",
    "owner",
    "target_owner",
    "card_id",
    "target_card_id",
    "serial",
    "target_serial",
    "position",
    "target_position",
    "select_type",
    "select_context",
    "option_type",
    "log_type",
    "attack_id",
    "special_condition",
    "energy_type",
    "reason",
)
NUMERIC_FIELDS = (
    "turn",
    "action_count",
    "step",
    "deck_count",
    "hand_count",
    "prize_count",
    "bench_max",
    "hp",
    "max_hp",
    "min_count",
    "max_count",
    "number",
    "count",
    "remaining_damage",
    "remaining_energy",
    "value",
    "result",
)
FLAG_FIELDS = (
    "supporter_played",
    "stadium_played",
    "energy_attached",
    "retreated",
    "poisoned",
    "burned",
    "asleep",
    "paralyzed",
    "confused",
    "appear_this_turn",
    "has_basic",
    "is_recover",
    "coin_head",
    "put_damage",
)
NUMERIC_SCALES = torch.tensor(
    [100, 100, 1000, 60, 30, 6, 8, 400, 400, 10, 10, 60, 60, 40, 10, 400, 10],
    dtype=torch.float32,
)
CAT_INDEX = {name: index for index, name in enumerate(CATEGORICAL_FIELDS)}


@dataclass(frozen=True)
class Token:
    categorical: tuple[int, ...]
    numeric: tuple[float, ...]
    flags: tuple[float, ...]


@dataclass(frozen=True)
class EncodedDecision:
    state: tuple[Token, ...]
    options: tuple[Token, ...]
    minimum: int
    maximum: int
    action: tuple[int, ...] | None
    deck_key: int = 0


@dataclass(frozen=True)
class TokenBatch:
    categorical: torch.Tensor
    numeric: torch.Tensor
    flags: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device | str) -> "TokenBatch":
        """Move a token batch to a device."""
        return TokenBatch(
            self.categorical.to(device),
            self.numeric.to(device),
            self.flags.to(device),
            self.mask.to(device),
        )


@dataclass(frozen=True)
class DecisionBatch:
    state: TokenBatch
    options: TokenBatch
    minimum: torch.Tensor
    maximum: torch.Tensor
    targets: torch.Tensor | None

    def to(self, device: torch.device | str) -> "DecisionBatch":
        """Move a decision batch to a device."""
        return DecisionBatch(
            self.state.to(device),
            self.options.to(device),
            self.minimum.to(device),
            self.maximum.to(device),
            None if self.targets is None else self.targets.to(device),
        )


def ensure_directories() -> None:
    """Create the fixed dataset and workspace directories."""
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        MANIFESTS_DIR,
        METADATA_DIR,
        WORK_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def value_sha256(value: Any) -> str:
    """Hash a value using canonical JSON."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    """Calculate a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_fingerprint() -> str:
    """Return the current token schema fingerprint."""
    payload = json.dumps(
        {
            "categorical": CATEGORICAL_FIELDS,
            "numeric": NUMERIC_FIELDS,
            "flags": FLAG_FIELDS,
            "scales": NUMERIC_SCALES.tolist(),
        },
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _category(value: Any) -> int:
    """Encode a non-negative category while reserving zero for missing values."""
    return value + 1 if type(value) is int and value >= 0 else 0


def _number(value: Any) -> float:
    """Convert a numeric value or return zero."""
    return float(value) if type(value) in (int, float) else 0.0


def _flag(value: Any) -> float:
    """Convert a value to a binary flag."""
    return float(bool(value))


def _token(
    categorical: dict[str, int] | None = None,
    numeric: dict[str, float] | None = None,
    flags: dict[str, float] | None = None,
) -> Token:
    """Create a token with the fixed field order."""
    categorical = categorical or {}
    numeric = numeric or {}
    flags = flags or {}
    return Token(
        tuple(int(categorical.get(name, 0)) for name in CATEGORICAL_FIELDS),
        tuple(float(numeric.get(name, 0.0)) for name in NUMERIC_FIELDS),
        tuple(float(flags.get(name, 0.0)) for name in FLAG_FIELDS),
    )


def _relative_owner(player_index: Any, your_index: int) -> Owner:
    """Convert an absolute player index to a relative owner."""
    if type(player_index) is not int:
        return Owner.UNKNOWN
    return Owner.SELF if player_index == your_index else Owner.OPPONENT


def _owner_from_entity(entity: Any, your_index: int, default: Owner) -> Owner:
    """Read an entity owner using a fallback value."""
    if isinstance(entity, dict) and type(entity.get("playerIndex")) is int:
        return _relative_owner(entity["playerIndex"], your_index)
    return default


def _iter_entities(value: Any) -> Iterable[dict[str, Any]]:
    """Yield entity objects from a single value or list."""
    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        yield from (item for item in value if isinstance(item, dict))


def _identity(entity: Any) -> tuple[int, int]:
    """Read a card identity and serial pair."""
    if not isinstance(entity, dict):
        return 0, 0
    return _category(entity.get("id", entity.get("cardId"))), _category(entity.get("serial"))


def _card_token(
    card: dict[str, Any],
    zone: Zone,
    your_index: int,
    default_owner: Owner,
    position: int = 0,
    kind: EntityKind = EntityKind.CARD,
) -> Token:
    """Encode a visible card."""
    card_id, serial = _identity(card)
    return _token(
        {
            "kind": kind,
            "zone": zone,
            "owner": _owner_from_entity(card, your_index, default_owner),
            "card_id": card_id,
            "serial": serial,
            "position": _category(position) if position else 0,
        }
    )


def _attachment_tokens(pokemon: dict[str, Any], your_index: int, owner: Owner) -> list[Token]:
    """Encode energy, tools, and pre-evolutions attached to a Pokemon."""
    parent_id, parent_serial = _identity(pokemon)
    tokens = []
    for position, energy_type in enumerate(pokemon.get("energies") or []):
        tokens.append(
            _token(
                {
                    "kind": EntityKind.ENERGY,
                    "zone": Zone.ENERGY,
                    "owner": owner,
                    "target_card_id": parent_id,
                    "target_serial": parent_serial,
                    "position": _category(position),
                    "energy_type": _category(energy_type),
                }
            )
        )
    for field, zone in (
        ("energyCards", Zone.ENERGY),
        ("tools", Zone.TOOL),
        ("preEvolution", Zone.PRE_EVOLUTION),
    ):
        for position, card in enumerate(pokemon.get(field) or []):
            if not isinstance(card, dict):
                continue
            token = _card_token(card, zone, your_index, owner, position)
            categories = list(token.categorical)
            categories[CAT_INDEX["target_card_id"]] = parent_id
            categories[CAT_INDEX["target_serial"]] = parent_serial
            tokens.append(replace(token, categorical=tuple(categories)))
    return tokens


def _pokemon_tokens(
    pokemon: dict[str, Any],
    zone: Zone,
    your_index: int,
    default_owner: Owner,
    position: int,
) -> list[Token]:
    """Encode an in-play Pokemon and its attachments."""
    owner = _owner_from_entity(pokemon, your_index, default_owner)
    card_id, serial = _identity(pokemon)
    token = _token(
        {
            "kind": EntityKind.POKEMON,
            "zone": zone,
            "owner": owner,
            "card_id": card_id,
            "serial": serial,
            "position": _category(position),
        },
        {"hp": _number(pokemon.get("hp")), "max_hp": _number(pokemon.get("maxHp"))},
        {"appear_this_turn": _flag(pokemon.get("appearThisTurn"))},
    )
    return [token, *_attachment_tokens(pokemon, your_index, owner)]


def _player_tokens(player: dict[str, Any], player_index: int, your_index: int) -> list[Token]:
    """Encode one player's public and perspective-visible state."""
    owner = _relative_owner(player_index, your_index)
    prize = player.get("prize") or []
    tokens = [
        _token(
            {"kind": EntityKind.PLAYER, "zone": Zone.PLAYER, "owner": owner},
            {
                "deck_count": _number(player.get("deckCount")),
                "hand_count": _number(player.get("handCount")),
                "prize_count": float(len(prize)),
                "bench_max": _number(player.get("benchMax")),
            },
            {name: _flag(player.get(name)) for name in FLAG_FIELDS[4:9]},
        )
    ]
    for zone, field in ((Zone.ACTIVE, "active"), (Zone.BENCH, "bench")):
        for position, pokemon in enumerate(player.get(field) or []):
            if isinstance(pokemon, dict):
                tokens.extend(_pokemon_tokens(pokemon, zone, your_index, owner, position))
    for position, card in enumerate(player.get("discard") or []):
        if isinstance(card, dict):
            tokens.append(_card_token(card, Zone.DISCARD, your_index, owner, position))
    hand = player.get("hand")
    if isinstance(hand, list):
        for position, card in enumerate(hand):
            if isinstance(card, dict):
                tokens.append(_card_token(card, Zone.HAND, your_index, owner, position))
    elif _number(player.get("handCount")):
        tokens.append(
            _token(
                {"kind": EntityKind.HIDDEN, "zone": Zone.HAND, "owner": owner},
                {"count": _number(player.get("handCount"))},
            )
        )
    if prize:
        tokens.append(
            _token(
                {"kind": EntityKind.HIDDEN, "zone": Zone.PRIZE, "owner": owner},
                {"count": float(len(prize))},
            )
        )
    return tokens


def _first_integer(value: dict[str, Any], *keys: str) -> int | None:
    """Return the first integer stored under the requested keys."""
    return next((value[key] for key in keys if type(value.get(key)) is int), None)


def _log_token(log: dict[str, Any], position: int, your_index: int) -> Token:
    """Encode one visible battle log entry."""
    return _token(
        {
            "kind": EntityKind.LOG,
            "zone": _category(log.get("fromArea")),
            "target_zone": _category(log.get("toArea")),
            "owner": _relative_owner(log.get("playerIndex"), your_index),
            "card_id": _category(_first_integer(log, "cardId", "cardIdAfter", "cardIdActive")),
            "target_card_id": _category(
                _first_integer(log, "cardIdTarget", "cardIdBefore", "cardIdBench")
            ),
            "serial": _category(_first_integer(log, "serial", "serialAfter", "serialActive")),
            "target_serial": _category(
                _first_integer(log, "serialTarget", "serialBefore", "serialBench")
            ),
            "position": _category(position),
            "log_type": _category(log.get("type")),
            "attack_id": _category(log.get("attackId")),
            "reason": _category(log.get("reason")),
        },
        {"value": _number(log.get("value")), "result": _number(log.get("result"))},
        {
            "has_basic": _flag(log.get("hasBasicPokemon")),
            "is_recover": _flag(log.get("isRecover")),
            "coin_head": _flag(log.get("head")),
            "put_damage": _flag(log.get("putDamageCounter")),
        },
    )


def _visible_entity(
    observation: dict[str, Any],
    player_index: int,
    area: Any,
    index: Any,
) -> dict[str, Any] | None:
    """Resolve an option reference to a visible entity."""
    current = observation["current"]
    select = observation["select"]
    players = current["players"]
    player = players[player_index] if 0 <= player_index < len(players) else {}
    collections = {
        1: select.get("deck"),
        2: player.get("hand"),
        3: player.get("discard"),
        4: player.get("active"),
        5: player.get("bench"),
        6: player.get("prize"),
        7: current.get("stadium"),
        12: current.get("looking"),
    }
    if area == 11:
        return {"playerIndex": player_index}
    values = collections.get(area)
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list) or type(index) is not int or not 0 <= index < len(values):
        return None
    entity = values[index]
    return entity if isinstance(entity, dict) else None


def _option_token(
    option: dict[str, Any],
    observation: dict[str, Any],
    your_index: int,
) -> Token:
    """Encode one legal action option."""
    option_type = option.get("type")
    player_index = option.get("playerIndex")
    if type(player_index) is not int:
        player_index = your_index
    area = option.get("area")
    if area is None and option_type == 7:
        area = 2
    source_position = option.get("index")
    target_area = option.get("inPlayArea")
    target_position = option.get("inPlayIndex")
    source = _visible_entity(observation, player_index, area, source_position)
    target = _visible_entity(observation, player_index, target_area, target_position)
    energy_type = 0

    if isinstance(source, dict) and type(option.get("toolIndex")) is int:
        parent = source
        attachment_index = option["toolIndex"]
        tools = parent.get("tools") or []
        source = tools[attachment_index] if 0 <= attachment_index < len(tools) else None
        target = parent
        target_area = area
        target_position = source_position
        source_position = attachment_index
        area = 9
    elif isinstance(source, dict) and type(option.get("energyIndex")) is int:
        parent = source
        attachment_index = option["energyIndex"]
        energies = parent.get("energies") or []
        energy_cards = parent.get("energyCards") or []
        if 0 <= attachment_index < len(energies):
            energy_type = _category(energies[attachment_index])
        source = energy_cards[attachment_index] if 0 <= attachment_index < len(energy_cards) else None
        target = parent
        target_area = area
        target_position = source_position
        source_position = attachment_index
        area = 8

    source_id, source_serial = _identity(source)
    target_id, target_serial = _identity(target)
    source_owner = _owner_from_entity(
        source,
        your_index,
        _relative_owner(player_index, your_index),
    )
    target_owner = _owner_from_entity(target, your_index, Owner.UNKNOWN)
    return _token(
        {
            "kind": EntityKind.OPTION,
            "zone": _category(area),
            "target_zone": _category(target_area),
            "owner": source_owner,
            "target_owner": target_owner,
            "card_id": _category(option.get("cardId")) or source_id,
            "target_card_id": target_id,
            "serial": _category(option.get("serial")) or source_serial,
            "target_serial": target_serial,
            "position": _category(source_position),
            "target_position": _category(target_position),
            "option_type": _category(option_type),
            "attack_id": _category(option.get("attackId")),
            "special_condition": _category(option.get("specialConditionType")),
            "energy_type": energy_type,
        },
        {"number": _number(option.get("number")), "count": _number(option.get("count"))},
    )


def _validated_action(
    action: Sequence[int],
    option_count: int,
    minimum: int,
    maximum: int,
) -> tuple[int, ...]:
    """Return an action after checking the indices needed for training."""
    values = tuple(action)
    if (
        any(type(index) is not int or not 0 <= index < option_count for index in values)
        or len(values) != len(set(values))
        or not minimum <= len(values) <= maximum
    ):
        raise ValueError("invalid action")
    return values


def encode_decision(
    observation: dict[str, Any],
    own_deck: Sequence[int],
    action: Sequence[int] | None = None,
    deck_key: int = 0,
) -> EncodedDecision:
    """Encode one observation, legal option set, and optional training action."""
    current = observation["current"]
    select = observation["select"]
    your_index = current["yourIndex"]
    players = current["players"]
    options = select["option"]
    minimum = select["minCount"]
    maximum = select["maxCount"]
    if len(own_deck) != 60:
        raise ValueError("a deck must contain 60 card IDs")
    if not 0 <= minimum <= maximum <= len(options):
        raise ValueError("invalid selection range")

    state = [
        _token(
            {
                "kind": EntityKind.STATE,
                "target_owner": _relative_owner(current.get("firstPlayer"), your_index),
            },
            {
                "turn": _number(current.get("turn")),
                "action_count": _number(current.get("turnActionCount")),
                "step": _number(observation.get("step")),
                "result": _number(current.get("result")),
            },
            {name: _flag(current.get(name)) for name in FLAG_FIELDS[:4]},
        ),
        _token(
            {
                "kind": EntityKind.SELECT,
                "select_type": _category(select.get("type")),
                "select_context": _category(select.get("context")),
            },
            {
                "min_count": float(minimum),
                "max_count": float(maximum),
                "remaining_damage": _number(select.get("remainDamageCounter")),
                "remaining_energy": _number(select.get("remainEnergyCost")),
            },
        ),
    ]
    for player_index, player in enumerate(players):
        state.extend(_player_tokens(player, player_index, your_index))
    for position, card in enumerate(current.get("stadium") or []):
        if isinstance(card, dict):
            state.append(_card_token(card, Zone.STADIUM, your_index, Owner.UNKNOWN, position))
    for position, card in enumerate(current.get("looking") or []):
        if isinstance(card, dict):
            state.append(_card_token(card, Zone.LOOKING, your_index, Owner.SELF, position))
    for position, card in enumerate(select.get("deck") or []):
        if isinstance(card, dict):
            state.append(_card_token(card, Zone.DECK, your_index, Owner.SELF, position))
    for field, kind in (
        ("contextCard", EntityKind.CONTEXT_CARD),
        ("effect", EntityKind.EFFECT_CARD),
    ):
        for card in _iter_entities(select.get(field)):
            state.append(_card_token(card, Zone.NONE, your_index, Owner.UNKNOWN, kind=kind))
    for card_id, count in sorted(Counter(own_deck).items()):
        state.append(
            _token(
                {
                    "kind": EntityKind.CARD,
                    "zone": Zone.OWN_DECK,
                    "owner": Owner.SELF,
                    "card_id": _category(card_id),
                },
                {"count": float(count)},
            )
        )
    for position, log in enumerate(observation.get("logs") or []):
        if isinstance(log, dict):
            state.append(_log_token(log, position, your_index))

    encoded_action = (
        None
        if action is None
        else _validated_action(action, len(options), minimum, maximum)
    )
    return EncodedDecision(
        tuple(state),
        tuple(_option_token(option, observation, your_index) for option in options),
        minimum,
        maximum,
        encoded_action,
        deck_key,
    )


def _pack_token_batch(rows: Sequence[Sequence[Token]]) -> TokenBatch:
    """Pad token sequences into an in-memory batch."""
    batch_size = len(rows)
    width = max((len(row) for row in rows), default=0)
    categorical = torch.zeros(batch_size, width, len(CATEGORICAL_FIELDS), dtype=torch.long)
    numeric = torch.zeros(batch_size, width, len(NUMERIC_FIELDS), dtype=torch.float32)
    flags = torch.zeros(batch_size, width, len(FLAG_FIELDS), dtype=torch.float32)
    mask = torch.zeros(batch_size, width, dtype=torch.bool)
    for row_index, row in enumerate(rows):
        if not row:
            continue
        categorical[row_index, : len(row)] = torch.tensor(
            [token.categorical for token in row]
        )
        numeric[row_index, : len(row)] = (
            torch.tensor([token.numeric for token in row]) / NUMERIC_SCALES
        )
        flags[row_index, : len(row)] = torch.tensor([token.flags for token in row])
        mask[row_index, : len(row)] = True
    return TokenBatch(categorical, numeric, flags, mask)


def collate_decisions(decisions: Sequence[EncodedDecision]) -> DecisionBatch:
    """Build a padded model batch from encoded decisions."""
    if not decisions:
        raise ValueError("cannot collate an empty batch")
    labeled = [decision.action is not None for decision in decisions]
    if any(labeled) and not all(labeled):
        raise ValueError("cannot mix labeled and unlabeled decisions")
    state = _pack_token_batch([decision.state for decision in decisions])
    options = _pack_token_batch([decision.options for decision in decisions])
    minimum = torch.tensor([decision.minimum for decision in decisions], dtype=torch.long)
    maximum = torch.tensor([decision.maximum for decision in decisions], dtype=torch.long)
    targets = None
    if all(labeled):
        target_width = max(len(decision.action or ()) for decision in decisions) + 1
        stop_index = options.mask.shape[1]
        targets = torch.full((len(decisions), target_width), -100, dtype=torch.long)
        for row_index, decision in enumerate(decisions):
            action = decision.action or ()
            if action:
                targets[row_index, : len(action)] = torch.tensor(action)
            targets[row_index, len(action)] = stop_index
    return DecisionBatch(state, options, minimum, maximum, targets)


def _pack_tokens(
    decisions: Sequence[EncodedDecision],
    field: str,
) -> dict[str, torch.Tensor]:
    """Pack ragged token sequences into values and offsets."""
    sequences = [getattr(decision, field) for decision in decisions]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.int32)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0)))
    tokens = [token for sequence in sequences for token in sequence]
    if tokens:
        categorical = torch.tensor(
            [token.categorical for token in tokens],
            dtype=torch.int32,
        )
        if categorical.min() < 0 or categorical.max() >= 32768:
            raise ValueError("categorical token exceeds int16 storage")
        categorical = categorical.to(torch.int16)
        numeric = (
            torch.tensor([token.numeric for token in tokens], dtype=torch.float32)
            / NUMERIC_SCALES
        ).to(torch.float16)
        flags = torch.tensor([token.flags for token in tokens], dtype=torch.bool)
    else:
        categorical = torch.empty(
            (0, len(CATEGORICAL_FIELDS)),
            dtype=torch.int16,
        )
        numeric = torch.empty((0, len(NUMERIC_FIELDS)), dtype=torch.float16)
        flags = torch.empty((0, len(FLAG_FIELDS)), dtype=torch.bool)
    return {
        f"{field}_categorical": categorical,
        f"{field}_numeric": numeric,
        f"{field}_flags": flags,
        f"{field}_offsets": offsets,
    }


def pack_decisions(decisions: Sequence[EncodedDecision]) -> dict[str, torch.Tensor]:
    """Pack labeled decisions into one training shard."""
    if not decisions or any(decision.action is None for decision in decisions):
        raise ValueError("training shards require labeled decisions")
    action_lengths = torch.tensor(
        [len(decision.action or ()) for decision in decisions],
        dtype=torch.int32,
    )
    action_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int32), action_lengths.cumsum(0))
    )
    return {
        **_pack_tokens(decisions, "state"),
        **_pack_tokens(decisions, "options"),
        "minimum": torch.tensor(
            [decision.minimum for decision in decisions],
            dtype=torch.int16,
        ),
        "maximum": torch.tensor(
            [decision.maximum for decision in decisions],
            dtype=torch.int16,
        ),
        "actions": torch.tensor(
            [index for decision in decisions for index in decision.action or ()],
            dtype=torch.int16,
        ),
        "action_offsets": action_offsets,
        "deck_key": torch.tensor(
            [decision.deck_key for decision in decisions],
            dtype=torch.int64,
        ),
    }


def save_shard(decisions: Sequence[EncodedDecision], path: Path) -> None:
    """Save one shard through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(pack_decisions(decisions), temporary)
    temporary.replace(path)


def load_shard(path: Path) -> dict[str, torch.Tensor]:
    """Load one packed shard on CPU."""
    shard = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(shard, dict) or "minimum" not in shard:
        raise ValueError(f"invalid shard: {path}")
    return shard


def shard_size(shard: dict[str, torch.Tensor]) -> int:
    """Return the number of decisions in a shard."""
    return int(shard["minimum"].shape[0])


def _gather_tokens(
    shard: dict[str, torch.Tensor],
    field: str,
    indices: torch.Tensor,
) -> TokenBatch:
    """Gather ragged token rows into a padded batch."""
    offsets = shard[f"{field}_offsets"]
    lengths = offsets[indices + 1] - offsets[indices]
    width = int(lengths.max().item()) if len(indices) else 0
    batch_size = len(indices)
    categorical = torch.zeros(
        batch_size,
        width,
        len(CATEGORICAL_FIELDS),
        dtype=torch.long,
    )
    numeric = torch.zeros(
        batch_size,
        width,
        len(NUMERIC_FIELDS),
        dtype=torch.float32,
    )
    flags = torch.zeros(
        batch_size,
        width,
        len(FLAG_FIELDS),
        dtype=torch.float32,
    )
    mask = torch.zeros(batch_size, width, dtype=torch.bool)
    for row, index in enumerate(indices.tolist()):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        length = end - start
        categorical[row, :length] = shard[f"{field}_categorical"][start:end].long()
        numeric[row, :length] = shard[f"{field}_numeric"][start:end].float()
        flags[row, :length] = shard[f"{field}_flags"][start:end].float()
        mask[row, :length] = True
    return TokenBatch(categorical, numeric, flags, mask)


def batch_from_shard(
    shard: dict[str, torch.Tensor],
    indices: torch.Tensor,
) -> DecisionBatch:
    """Build a model batch from selected shard rows."""
    state = _gather_tokens(shard, "state", indices)
    options = _gather_tokens(shard, "options", indices)
    minimum = shard["minimum"][indices].long()
    maximum = shard["maximum"][indices].long()
    action_offsets = shard["action_offsets"]
    action_lengths = action_offsets[indices + 1] - action_offsets[indices]
    targets = torch.full(
        (len(indices), int(action_lengths.max().item()) + 1),
        -100,
        dtype=torch.long,
    )
    stop_index = options.mask.shape[1]
    for row, index in enumerate(indices.tolist()):
        start = int(action_offsets[index])
        end = int(action_offsets[index + 1])
        length = end - start
        if length:
            targets[row, :length] = shard["actions"][start:end].long()
        targets[row, length] = stop_index
    return DecisionBatch(state, options, minimum, maximum, targets)


def read_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Read the unified base dataset manifest."""
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if (
        manifest.get("format") != FORMAT
        or manifest.get("token_schema_sha256") != schema_fingerprint()
    ):
        raise ValueError("dataset format does not match base_data.py")
    return manifest


def split_shards(
    split: str,
    manifest: dict[str, Any] | None = None,
) -> list[Path]:
    """Return the shard paths for one dataset split."""
    manifest = manifest or read_manifest()
    entries = manifest["splits"][split]["shards"]
    return [
        BASE_DIR / (entry["path"] if isinstance(entry, dict) else entry)
        for entry in entries
    ]


def iter_batches(
    paths: Sequence[Path],
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> Iterator[DecisionBatch]:
    """Yield model batches from a sequence of shards."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    ordered = list(paths)
    if shuffle:
        random.Random(seed).shuffle(ordered)
    for path in ordered:
        shard = load_shard(path)
        count = shard_size(shard)
        if shuffle:
            path_seed = int(hashlib.sha256(path.name.encode()).hexdigest()[:8], 16)
            generator = torch.Generator().manual_seed(seed ^ path_seed)
            indices = torch.randperm(count, generator=generator)
        else:
            indices = torch.arange(count)
        for start in range(0, count, batch_size):
            yield batch_from_shard(shard, indices[start : start + batch_size])


def discover_archives() -> list[Path]:
    """Return all raw ZIP archives in stable order."""
    archives = sorted(RAW_DIR.rglob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"no ZIP archives found in {RAW_DIR}")
    return archives


def _shared_observation_fields(replay: dict[str, Any]) -> tuple[str, ...]:
    """Return shared observation fields that should be restored for both players."""
    specification = replay.get("specification", {}).get("observation", {})
    return tuple(
        name
        for name, field in specification.items()
        if isinstance(field, dict)
        and field.get("shared")
        and name not in REMOVED_OBSERVATION_FIELDS
    )


def _clean_observation(
    step: list[dict[str, Any]],
    player: int,
    shared_fields: Sequence[str],
) -> dict[str, Any]:
    """Remove unused fields and restore shared observation values."""
    observation = dict(step[player]["observation"])
    for field in REMOVED_OBSERVATION_FIELDS:
        observation.pop(field, None)
    shared = step[0]["observation"]
    for field in shared_fields:
        if field in shared:
            observation[field] = shared[field]
    return observation


def _encode_replay(
    replay: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[Any],
    list[EncodedDecision],
    int,
    int,
    dict[str, dict[str, Any]],
]:
    """Encode only useful decisions made by the winning player."""
    steps = replay["steps"]
    decks = [list(steps[1][player]["action"]) for player in range(2)]
    if any(len(deck) != 60 for deck in decks):
        raise ValueError("a replay deck does not contain 60 cards")

    rewards = replay["rewards"]
    environment = {
        "name": replay.get("name"),
        "version": replay.get("version"),
        "module_version": replay.get("module_version"),
        "schema_version": replay.get("schema_version"),
    }
    winner = next((player for player, reward in enumerate(rewards) if reward == 1), None)
    if winner is None:
        return environment, rewards, [], 0, 0, {}

    shared_fields = _shared_observation_fields(replay)
    deck = decks[winner]
    catalog: dict[str, dict[str, Any]] = {}
    encoded = []
    winning_decisions = 0
    forced_decisions = 0
    deck_key = 0

    for step, next_step in zip(steps, steps[1:]):
        state = step[winner]
        observation = state.get("observation")
        if not isinstance(observation, dict):
            continue
        select = observation.get("select")
        if state.get("status") != "ACTIVE" or not isinstance(select, dict):
            continue
        winning_decisions += 1
        if is_forced_selection(select):
            forced_decisions += 1
            continue
        if not deck_key:
            deck_key = _register_deck(deck, catalog)
        cleaned = _clean_observation(step, winner, shared_fields)
        action = _validated_action(
            next_step[winner]["action"],
            len(select["option"]),
            select["minCount"],
            select["maxCount"],
        )
        encoded.append(encode_decision(cleaned, deck, action, deck_key))

    return environment, rewards, encoded, winning_decisions, forced_decisions, catalog


def is_forced_selection(select: dict[str, Any]) -> bool:
    """Return whether a selection contains no meaningful choice."""
    options = select["option"]
    return (
        select["maxCount"] == 0
        or not options
        or (len(options) == 1 and select["minCount"] >= 1)
    )


def split_for_replay(replay_sha256: str) -> str:
    """Assign an episode to a stable 80/10/10 split."""
    bucket = int(replay_sha256[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def deck_fingerprint(deck: Sequence[int]) -> str:
    """Hash a deck without depending on card order."""
    return value_sha256(sorted(deck))


def _register_deck(
    deck: Sequence[int],
    catalog: dict[str, dict[str, Any]],
) -> int:
    """Add a deck to the catalog and return its integer key."""
    fingerprint = deck_fingerprint(deck)
    if fingerprint not in catalog:
        key = int(fingerprint[:15], 16) + 1
        catalog[fingerprint] = {
            "key": key,
            "sha256": fingerprint,
            "cards": [
                {"id": card_id, "count": count}
                for card_id, count in sorted(Counter(deck).items())
            ],
        }
    return catalog[fingerprint]["key"]


def _flush_buffers(
    buffers: dict[str, list[EncodedDecision]],
    output_dir: Path,
    parts: dict[str, int],
    split_data: dict[str, dict[str, Any]],
    shard_size_value: int,
    shard_prefix: str,
    force: bool,
) -> None:
    """Write complete shards and optionally flush remaining decisions."""
    for split in SPLITS:
        while len(buffers[split]) >= shard_size_value or force and buffers[split]:
            count = min(shard_size_value, len(buffers[split]))
            decisions = buffers[split][:count]
            del buffers[split][:count]
            relative = Path(split) / f"{shard_prefix}-{parts[split]:05d}.pt"
            path = output_dir / relative
            save_shard(decisions, path)
            split_data[split]["shards"].append(
                {
                    "path": relative.as_posix(),
                    "decisions": count,
                    "bytes": path.stat().st_size,
                }
            )
            parts[split] += 1


def _write_json(value: Any, path: Path) -> None:
    """Write a readable UTF-8 JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _plan_archives(
    archives: Sequence[Path],
) -> list[tuple[str, tuple[str, ...]]]:
    """Plan fast cross-archive deduplication from ZIP metadata."""
    seen: set[tuple[int, int]] = set()
    plans = []
    for path in archives:
        skipped = []
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                (
                    member
                    for member in archive.infolist()
                    if not member.is_dir() and member.filename.lower().endswith(".json")
                ),
                key=lambda member: member.filename,
            )
        for member in members:
            identity = (member.CRC, member.file_size)
            if identity in seen:
                skipped.append(member.filename)
            else:
                seen.add(identity)
        plans.append((str(path), tuple(skipped)))
    return plans


def _archive_prefix(path: Path, raw_dir: Path) -> str:
    """Return a stable unique prefix for one archive's shards."""
    relative = path.relative_to(raw_dir).as_posix()
    digest = hashlib.sha256(relative.encode()).hexdigest()[:12]
    return f"{path.stem}-{digest}"


def _process_archive(
    archive_path: str,
    raw_path: str,
    output_path: str,
    skipped_members: tuple[str, ...],
    shard_size_value: int,
) -> dict[str, Any]:
    """Process one ZIP archive and write its shards directly."""
    path = Path(archive_path)
    raw_dir = Path(raw_path)
    output_dir = Path(output_path)
    prefix = _archive_prefix(path, raw_dir)
    skipped = set(skipped_members)
    buffers: dict[str, list[EncodedDecision]] = {split: [] for split in SPLITS}
    parts = {split: 0 for split in SPLITS}
    split_data: dict[str, dict[str, Any]] = {
        split: {"episodes": 0, "decisions": 0, "shards": []}
        for split in SPLITS
    }
    deck_catalog: dict[str, dict[str, Any]] = {}
    environments: Counter[tuple[Any, ...]] = Counter()
    rejection_reasons: Counter[str] = Counter()
    stats = {
        "examined_episodes": 0,
        "accepted_episodes": 0,
        "rejected_episodes": 0,
        "duplicate_episodes": len(skipped),
        "draw_episodes": 0,
        "winning_decisions": 0,
        "forced_decisions_removed": 0,
        "learning_decisions": 0,
    }

    with zipfile.ZipFile(path) as archive:
        members = sorted(
            (
                member
                for member in archive.infolist()
                if not member.is_dir() and member.filename.lower().endswith(".json")
            ),
            key=lambda member: member.filename,
        )
        stats["examined_episodes"] = len(members)
        for member in members:
            if member.filename in skipped:
                continue
            try:
                payload = archive.read(member)
                replay_digest = hashlib.sha256(payload).hexdigest()
                (
                    environment,
                    rewards,
                    encoded,
                    winning_decisions,
                    forced_decisions,
                    episode_catalog,
                ) = _encode_replay(json.loads(payload))
                split = split_for_replay(replay_digest)
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as error:
                stats["rejected_episodes"] += 1
                rejection_reasons[type(error).__name__] += 1
                continue

            stats["accepted_episodes"] += 1
            stats["winning_decisions"] += winning_decisions
            stats["forced_decisions_removed"] += forced_decisions
            if 1 not in rewards:
                stats["draw_episodes"] += 1
            environments[
                (
                    environment["name"],
                    environment["version"],
                    environment["module_version"],
                    environment["schema_version"],
                )
            ] += 1
            deck_catalog.update(episode_catalog)
            if encoded:
                buffers[split].extend(encoded)
                split_data[split]["episodes"] += 1
                split_data[split]["decisions"] += len(encoded)
                stats["learning_decisions"] += len(encoded)
                _flush_buffers(
                    buffers,
                    output_dir,
                    parts,
                    split_data,
                    shard_size_value,
                    prefix,
                    False,
                )

    _flush_buffers(
        buffers,
        output_dir,
        parts,
        split_data,
        shard_size_value,
        prefix,
        True,
    )
    relative = path.relative_to(raw_dir).as_posix()
    source = {
        "path": relative,
        "bytes": path.stat().st_size,
        "json_files": len(members),
        "examined_episodes": stats["examined_episodes"],
        "accepted_episodes": stats["accepted_episodes"],
        "rejected_episodes": stats["rejected_episodes"],
        "duplicate_episodes": stats["duplicate_episodes"],
    }
    return {
        "source": source,
        "stats": stats,
        "splits": split_data,
        "decks": deck_catalog,
        "environments": environments,
        "rejection_reasons": rejection_reasons,
    }


def _progress(result: dict[str, Any]) -> str:
    """Return one compact archive completion message."""
    return json.dumps(
        {"archive": result["source"]["path"], **result["stats"]},
        ensure_ascii=False,
        sort_keys=True,
    )


def build_base_dataset(
    shard_size_value: int = 2048,
    workers: int = 2,
) -> dict[str, Any]:
    """Build the unified base dataset with archive-level parallelism."""
    if shard_size_value <= 0 or workers <= 0:
        raise ValueError("shard size and workers must be positive")
    ensure_directories()
    archives = discover_archives()
    if BASE_DIR.exists() or MANIFEST_PATH.exists() or DECKS_PATH.exists():
        raise FileExistsError("base dataset output already exists")

    stage_root = Path(tempfile.mkdtemp(prefix="build-", dir=WORK_DIR))
    stage_data = stage_root / "base"
    stage_manifest = stage_root / "base.json"
    stage_decks = stage_root / "decks.json"
    arguments = [
        (
            archive_path,
            str(RAW_DIR),
            str(stage_data),
            skipped,
            shard_size_value,
        )
        for archive_path, skipped in _plan_archives(archives)
    ]

    results = []
    worker_count = min(workers, len(arguments))
    if worker_count == 1:
        for values in arguments:
            result = _process_archive(*values)
            results.append(result)
            print(_progress(result), flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            max_tasks_per_child=1,
        ) as executor:
            futures = [executor.submit(_process_archive, *values) for values in arguments]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(_progress(result), flush=True)

    results.sort(key=lambda result: result["source"]["path"])
    sources = [result["source"] for result in results]
    stats = {
        key: sum(result["stats"][key] for result in results)
        for key in (
            "examined_episodes",
            "accepted_episodes",
            "rejected_episodes",
            "duplicate_episodes",
            "draw_episodes",
            "winning_decisions",
            "forced_decisions_removed",
            "learning_decisions",
        )
    }
    split_data: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        shards = [
            shard
            for result in results
            for shard in result["splits"][split]["shards"]
        ]
        split_data[split] = {
            "episodes": sum(
                result["splits"][split]["episodes"]
                for result in results
            ),
            "decisions": sum(
                result["splits"][split]["decisions"]
                for result in results
            ),
            "shards": sorted(shards, key=lambda shard: shard["path"]),
        }
    if not split_data["train"]["decisions"] or not split_data["validation"]["decisions"]:
        raise ValueError("dataset has no train or validation decisions")

    deck_catalog: dict[str, dict[str, Any]] = {}
    environments: Counter[tuple[Any, ...]] = Counter()
    rejection_reasons: Counter[str] = Counter()
    for result in results:
        deck_catalog.update(result["decks"])
        environments.update(result["environments"])
        rejection_reasons.update(result["rejection_reasons"])

    decks = {
        "format": "ptcg-decks-v1",
        "decks": sorted(deck_catalog.values(), key=lambda deck: deck["key"]),
    }
    _write_json(decks, stage_decks)
    deck_catalog_sha256 = file_sha256(stage_decks)
    identity = {
        "format": FORMAT,
        "pipeline_version": PIPELINE_VERSION,
        "token_schema_sha256": schema_fingerprint(),
        "sources": [
            {
                "path": source["path"],
                "bytes": source["bytes"],
                "json_files": source["json_files"],
            }
            for source in sources
        ],
        "selection": "winner_only_non_forced",
        "split": "sha256_80_10_10",
        "deduplication": "zip_crc32_and_uncompressed_size",
        "shard_size": shard_size_value,
    }
    manifest = {
        **identity,
        "dataset_sha256": value_sha256(identity),
        "source_root": "datasets/raw",
        "sources": sources,
        "stats": stats,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "environments": [
            {
                "name": values[0],
                "version": values[1],
                "module_version": values[2],
                "schema_version": values[3],
                "episodes": count,
            }
            for values, count in sorted(environments.items(), key=lambda item: str(item[0]))
        ],
        "deck_catalog": {
            "path": "datasets/metadata/decks.json",
            "sha256": deck_catalog_sha256,
            "decks": len(deck_catalog),
        },
        "splits": split_data,
    }
    _write_json(manifest, stage_manifest)
    stage_data.replace(BASE_DIR)
    stage_decks.replace(DECKS_PATH)
    stage_manifest.replace(MANIFEST_PATH)
    stage_root.rmdir()
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse the data build settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Build the base dataset using the repository's fixed directories."""
    args = parse_args()
    try:
        manifest = build_base_dataset(args.shard_size, args.workers)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "dataset_sha256": manifest["dataset_sha256"],
                **manifest["stats"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
