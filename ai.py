import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

Observation = Mapping[str, Any]
Request = dict[str, Any]
PokemonKey = tuple[int, int]
PokemonState = tuple[int, int]

MAIN_SELECT_TYPE = 0
MAIN_FALLBACK_SCORES = {7: 1.5, 8: 4.0, 9: 2.0, 13: 6.0, 14: 2.0}

TURN_END = 3
CHANGE = 9
EVOLVE = 12
DEVOLVE = 13
ATTACK = 15
HP_CHANGE = 16

LOG_MOVE = 6
LOG_PLAY = 10

WIN_REWARD = 1.0
PRIZE_REWARD = 0.1
OWN_TURN_PENALTY = 0.01
DAMAGE_REWARD = 0.1
ATTACK_PENALTY_DIVISOR = 1000.0

AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_INPLAY = 5
AREA_PRIZE = 6
AREA_LOOKING = 12

RAGING_BOLT_EX_ID = 63
BLOODMOON_EX_ID = 44
OGERPON_EX_ID = 96
LATIAS_EX_ID = 184
IRON_LEAVES_ID = 75
BLOODMOON_ID = 135
POKEMON_CARD_IDS = (63, 44, 96, 75, 135, 184)
AUXILIARY_POKEMON_IDS = (184,)

CYRANO_ID = 1205
CRISPIN_ID = 1198
LILLIE_ID = 1227
BOSS_ORDERS_ID = 1182
SUPPORTER_CARD_IDS = (1198, 1227, 1182, 1205)

ULTRA_BALL_ID = 1121
POKEGEAR_ID = 1122
BUG_CATCHING_SET_ID = 1094
ENERGY_SEARCH_ID = 1119
ENERGY_RETRIEVAL_ID = 1118
SWITCH_ID = 1123
UNFAIR_STAMP_ID = 1080
NIGHT_STRETCHER_ID = 1097
PRIME_CATCHER_ID = 1088
TOOL_SCRAPPER_ID = 1137
ENERGY_SWITCH_ID = 1116
KEY_ITEM_CARD_IDS = (1121, 1122, 1094, 1119, 1118, 1123, 1080, 1097, 1088, 1137, 1116)

GRASS_ENERGY_ID = 1
LIGHTNING_ENERGY_ID = 4
FIGHTING_ENERGY_ID = 6
BASIC_ENERGY_IDS = (1, 4, 6)

IMMUNE_TO_EX_IDS = (345, 533)
CRUSTLE_ID = 345
DWEBBLE_ID = 344
MEGA_KANGASKHAN_EX_ID = 756
DRAGAPULT_ID = 121
FROSLASS_ID = 104

# Recovery cards that let an opponent rebuild after a knockout. Once these are
# in their discard pile they cannot come back, so a KO made now sticks.
RECOVERY_CARD_IDS = (NIGHT_STRETCHER_ID, ENERGY_RETRIEVAL_ID)
OPP_RECOVERY_EXHAUSTED = 2

# Headline attackers across the eight environment decks we now train against.
# Seeing one in the opponent's discard pile means that line is a spent copy.
KEY_OPP_ATTACKER_IDS = (CRUSTLE_ID, DRAGAPULT_ID, FROSLASS_ID, BLOODMOON_ID, MEGA_KANGASKHAN_EX_ID)

# Self-play pipeline (arena_judge) loads this module under aliased names with env
# overrides so candidate and baseline models can coexist in one worker process.
_MODEL_PATH = Path(os.environ.get("PTCG_MODEL_PATH") or Path(__file__).resolve().parent / "xgb_model.json")
_MAPPING_PATH = Path(os.environ.get("PTCG_MAPPING_PATH") or Path(__file__).resolve().parent / "action_label_mapping.json")

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

_POKEGEAR_TIERS = {1205: 100.0, 1198: 80.0, 1227: 60.0, 1182: 40.0}
_BUG_CATCHING_TIERS = {75: 80.0, 96: 60.0, 1: 50.0}
_CYRANO_SEARCH_TIERS = {63: 90.0, 96: 75.0, 75: 60.0, 184: 45.0, 44: 30.0}
_ENERGY_SEARCH_TIERS = {4: 50.0, 6: 50.0, 1: 30.0}

_MODEL_CACHE: dict[str, Any] = {}


def transform_score(raw_score: float) -> float:
    return raw_score**2 if raw_score > 0 else raw_score


def calculate_damage_score(lost_hp: int, max_hp: int) -> float:
    if max_hp <= 0:
        return 0.0
    return DAMAGE_REWARD * min(max(lost_hp, 0), max_hp) / max_hp


def calculate_attack_penalty(attack_count: int) -> float:
    return attack_count**2 / ATTACK_PENALTY_DIVISOR


def _pokemon_states(current: Observation) -> dict[PokemonKey, PokemonState]:
    states: dict[PokemonKey, PokemonState] = {}
    for player_index, player in enumerate(current.get("players", [])):
        pokemon_in_play = [
            *(player.get("active") or []),
            *(player.get("bench") or []),
        ]
        for pokemon in pokemon_in_play:
            if pokemon is not None:
                key = (player_index, int(pokemon["serial"]))
                states[key] = (int(pokemon["hp"]), int(pokemon["maxHp"]))
    return states


def _prize_counts(current: Observation) -> tuple[int, ...]:
    return tuple(len(player.get("prize") or []) for player in current.get("players", []))


def _turn_player(current: Observation) -> int | None:
    turn = int(current.get("turn", 0))
    first_player = int(current.get("firstPlayer", -1))
    if turn <= 0 or first_player not in (0, 1):
        return None
    return first_player if turn % 2 else 1 - first_player


def _pokemon_transitions(logs: Sequence[Observation]) -> dict[PokemonKey, PokemonKey]:
    transitions: dict[PokemonKey, PokemonKey] = {}
    for log in logs:
        player_index = log.get("playerIndex")
        if player_index is None:
            continue
        log_type = log.get("type")
        if log_type == CHANGE:
            old_serial = log.get("serialBefore")
            new_serial = log.get("serialAfter")
        elif log_type == EVOLVE:
            old_serial = log.get("serialTarget")
            new_serial = log.get("serial")
        elif log_type == DEVOLVE:
            old_serial = log.get("serial")
            new_serial = log.get("serialTarget")
        else:
            continue
        if old_serial is not None and new_serial is not None:
            transitions[(int(player_index), int(old_serial))] = (int(player_index), int(new_serial))
    return transitions


def _observation_fingerprint(
    current: Observation,
    logs: Sequence[Observation],
    pokemon: Mapping[PokemonKey, PokemonState],
    prizes: tuple[int, ...],
) -> str:
    payload = {
        "turn": current.get("turn"),
        "turn_action_count": current.get("turnActionCount"),
        "result": current.get("result"),
        "logs": logs,
        "pokemon": sorted((*key, *state) for key, state in pokemon.items()),
        "prizes": prizes,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class ScoreTracker:
    def __init__(self, player_index: int) -> None:
        if player_index not in (0, 1):
            raise ValueError("player_index must be 0 or 1.")
        self.player_index = player_index
        self.reset()

    def reset(self) -> None:
        self.turn_score = 0.0
        self.raw_score = 0.0
        self.final_score: float | None = None
        self.turns_scored = 0
        self.last_turn_player: int | None = None
        self.own_turn_score = 0.0
        self.opponent_turn_score = 0.0
        self._pending_score = 0.0
        self._attacks_since_prize = 0
        self._active_turn_player: int | None = None
        self._pokemon: dict[PokemonKey, PokemonState] = {}
        self._prizes: tuple[int, ...] | None = None
        self._last_observation = ""

    def observe(self, observation: Observation) -> None:
        current = observation.get("current")
        if current is None:
            return

        logs = observation.get("logs") or []
        pokemon = _pokemon_states(current)
        prizes = _prize_counts(current)
        fingerprint = _observation_fingerprint(current, logs, pokemon, prizes)
        if fingerprint == self._last_observation:
            return
        self._last_observation = fingerprint

        current_turn_player = _turn_player(current)
        if self._prizes is not None and int(current.get("turn", 0)) > 0:
            self._collect_attack_count(logs)
            self._collect_damage(logs, pokemon)
            self._collect_prizes(prizes)

            ended_turns = [int(log["playerIndex"]) for log in logs if log.get("type") == TURN_END]
            for player_index in ended_turns:
                self._end_turn(player_index)

            if ended_turns and int(current.get("result", -1)) >= 0:
                self._active_turn_player = None
            else:
                self._active_turn_player = current_turn_player
        else:
            self._active_turn_player = current_turn_player

        self._pokemon = pokemon
        self._prizes = prizes

    def _collect_attack_count(self, logs: Sequence[Observation]) -> None:
        self._attacks_since_prize += sum(log.get("type") == ATTACK and log.get("playerIndex") == self.player_index for log in logs)

    def _collect_damage(
        self,
        logs: Sequence[Observation],
        current_pokemon: Mapping[PokemonKey, PokemonState],
    ) -> None:
        transitions = _pokemon_transitions(logs)
        reverse_transitions = {new: old for old, new in transitions.items()}
        changed = {(int(log["playerIndex"]), int(log["serial"])) for log in logs if log.get("type") == HP_CHANGE and log.get("playerIndex") is not None and log.get("serial") is not None}

        for key in changed:
            previous_key = key if key in self._pokemon else reverse_transitions.get(key)
            current_key = key if key in current_pokemon else transitions.get(key)
            previous = self._pokemon.get(previous_key) if previous_key else None
            if previous is None:
                continue

            previous_hp, previous_max_hp = previous
            current = current_pokemon.get(current_key) if current_key else None
            if current is None:
                lost_hp = previous_hp
                damage_max_hp = previous_max_hp
            else:
                current_hp, current_max_hp = current
                previous_damage = previous_max_hp - previous_hp
                current_damage = current_max_hp - current_hp
                lost_hp = current_damage - previous_damage
                damage_max_hp = current_max_hp

            score = calculate_damage_score(lost_hp, damage_max_hp)
            self._pending_score += -score if key[0] == self.player_index else score

    def _collect_prizes(self, current_prizes: tuple[int, ...]) -> None:
        if self._prizes is None:
            return

        opponent_index = 1 - self.player_index
        prizes_taken = max(self._prizes[self.player_index] - current_prizes[self.player_index], 0)
        opponent_prizes_taken = max(self._prizes[opponent_index] - current_prizes[opponent_index], 0)

        if prizes_taken:
            self._pending_score += PRIZE_REWARD * prizes_taken - calculate_attack_penalty(self._attacks_since_prize)
            self._attacks_since_prize = 0

        self._pending_score -= PRIZE_REWARD * opponent_prizes_taken

    def _end_turn(self, player_index: int) -> None:
        self.turn_score = self._pending_score
        if player_index == self.player_index:
            self.turn_score -= OWN_TURN_PENALTY

        self.raw_score += self.turn_score
        self.turns_scored += 1
        self.last_turn_player = player_index
        if player_index == self.player_index:
            self.own_turn_score = self.turn_score
        else:
            self.opponent_turn_score = self.turn_score
        self._pending_score = 0.0

    def finish(self, won: bool) -> tuple[float, float]:
        if self.final_score is None:
            if self._active_turn_player is not None:
                self._end_turn(self._active_turn_player)
                self._active_turn_player = None
            if won:
                self.raw_score += WIN_REWARD
            self.final_score = transform_score(self.raw_score)
        return self.raw_score, self.final_score

    def feedback(self) -> dict[str, float | int | None]:
        return {
            "turn_score": self.turn_score,
            "last_turn_player": self.last_turn_player,
            "own_turn_score": self.own_turn_score,
            "opponent_turn_score": self.opponent_turn_score,
            "raw_score": self.raw_score,
            "turns_scored": self.turns_scored,
        }


def validate_deck(deck: list[int]) -> None:
    if len(deck) != 60:
        raise ValueError(f"Deck must contain exactly 60 card IDs, got {len(deck)}.")
    if not all(type(card_id) is int for card_id in deck):
        raise TypeError("Every card ID must be an integer.")


def build_request(observation: Observation, score: Mapping[str, Any]) -> Request:
    select = observation.get("select")
    if select is None:
        raise RuntimeError("A battle observation does not contain a selection.")

    options = select.get("option") or []
    min_count = int(select.get("minCount", 0))
    max_count = min(int(select.get("maxCount", 0)), len(options))
    if min_count > max_count:
        raise RuntimeError("CABT returned fewer options than the required count.")

    return {
        "state": observation["current"],
        "logs": observation.get("logs") or [],
        "score": dict(score),
        "type": select.get("type"),
        "context": select.get("context"),
        "min_count": min_count,
        "max_count": max_count,
        "options": options,
    }


def select_legal_options(request: Request, scores: Sequence[float]) -> list[int]:
    options = request["options"]
    if len(scores) != len(options):
        raise ValueError("The policy must score every legal option exactly once.")

    count = request["max_count"]
    ranked_indices = sorted(range(len(options)), key=lambda i: (-scores[i], i))
    return ranked_indices[:count]


def _dict_card_id(card: Any) -> int | None:
    if card is None:
        return None
    if isinstance(card, dict):
        cid = card.get("id")
        return int(cid) if cid is not None else None
    if isinstance(card, int):
        return card
    return None


def _zone_entries(player: Any, area: int) -> list[Any]:
    if not isinstance(player, dict):
        return []
    if area == AREA_HAND:
        zone = player.get("hand")
    elif area == AREA_DISCARD:
        zone = player.get("discard")
    elif area == AREA_ACTIVE:
        zone = player.get("active")
    elif area == AREA_BENCH:
        zone = player.get("bench")
    elif area == AREA_PRIZE:
        zone = player.get("prize")
    else:
        zone = []
    return list(zone) if isinstance(zone, list) else []


def _zone_ids(player: Any, area: int) -> list[int]:
    ids: list[int] = []
    for entry in _zone_entries(player, area):
        cid = _dict_card_id(entry)
        if cid is not None:
            ids.append(cid)
    return ids


def _pokemon_at(player: Any, area: Any, index: Any) -> Any:
    if not isinstance(player, dict) or not isinstance(index, int):
        return None
    if area == AREA_ACTIVE:
        zone = _zone_entries(player, AREA_ACTIVE)
    else:
        zone = _zone_entries(player, AREA_ACTIVE) + _zone_entries(player, AREA_BENCH)
    if 0 <= index < len(zone) and isinstance(zone[index], dict):
        return zone[index]
    return None


def _get_pokemon_energies(pokemon: Any) -> list[int]:
    if not isinstance(pokemon, dict):
        return []
    cards = pokemon.get("energyCards")
    if not isinstance(cards, list):
        cards = pokemon.get("energies")
    if not isinstance(cards, list):
        return []
    res = []
    for c in cards:
        cid = _dict_card_id(c)
        if cid is not None:
            res.append(cid)
    return res


def _energy_id_at(pokemon: Any, energy_index: Any) -> int | None:
    if not isinstance(pokemon, dict) or not isinstance(energy_index, int):
        return None
    cards = pokemon.get("energyCards")
    if not isinstance(cards, list):
        cards = pokemon.get("energies")
    if not isinstance(cards, list) or not 0 <= energy_index < len(cards):
        return None
    return _dict_card_id(cards[energy_index])


def _last_played_card(logs: Sequence[Any], my_idx: int) -> int | None:
    for log in reversed(list(logs or [])):
        if isinstance(log, dict) and log.get("type") == LOG_PLAY and log.get("playerIndex") == my_idx:
            cid = log.get("cardId")
            if cid is not None:
                return int(cid)
    return None


def _our_pokemon_knocked_out(logs: Sequence[Any], my_idx: int) -> bool:
    for log in logs or []:
        if isinstance(log, dict) and log.get("type") == LOG_MOVE and log.get("playerIndex") == my_idx and log.get("toArea") == AREA_DISCARD and log.get("fromArea") in (AREA_ACTIVE, AREA_INPLAY):
            return True
    return False


class _PureXgbPredictor:
    def __init__(self, model_dict: Mapping[str, Any]) -> None:
        learner = model_dict.get("learner") or model_dict
        lmp = learner["learner_model_param"]
        self.num_class = int(lmp.get("num_class", 1))
        gbm = learner["gradient_booster"]["model"]
        self.trees = gbm["trees"]
        tree_info = gbm.get("tree_info") or [i % self.num_class for i in range(len(self.trees))]
        self.tree_info = [int(t) for t in tree_info]

        base = lmp.get("base_score")
        if isinstance(base, str):
            try:
                base = json.loads(base)
            except Exception:
                base = float(base)
        if isinstance(base, (list, tuple)):
            self.base_scores = [float(v) for v in base]
            if len(self.base_scores) < self.num_class:
                self.base_scores += [0.0] * (self.num_class - len(self.base_scores))
        else:
            self.base_scores = [float(base if base is not None else 0.5)] * self.num_class

    def _leaf_value(self, tree: Mapping[str, Any], features: Sequence[float]) -> float:
        node = 0
        conditions = tree["split_conditions"]
        left = tree["left_children"]
        right = tree["right_children"]
        split_features = tree["split_indices"]
        feature_count = len(features)
        while True:
            if left[node] < 0:
                return float(tree["base_weights"][node])
            cond = conditions[node]
            if cond is None or cond != cond:
                return float(tree["base_weights"][node])
            feature_index = int(split_features[node])
            value = features[feature_index] if feature_index < feature_count else 0.0
            node = left[node] if value < cond else right[node]

    def predict(self, features: Sequence[float]) -> list[float]:
        logits = list(self.base_scores)
        for tree_index, tree in enumerate(self.trees):
            class_index = self.tree_info[tree_index]
            logits[class_index] += self._leaf_value(tree, features)
        max_logit = max(logits)
        exp_values = [math.exp(v - max_logit) for v in logits]
        total = sum(exp_values)
        return [e / total for e in exp_values]


def _build_predictor() -> Any:
    try:
        import numpy as np
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(_MODEL_PATH))

        def predict(features: Sequence[float]) -> list[float]:
            probs = booster.inplace_predict(np.asarray([features], dtype=np.float32))[0]
            return [float(p) for p in probs]

        return predict
    except Exception:
        pass

    try:
        if _MODEL_PATH.exists():
            model_json = json.loads(_MODEL_PATH.read_text(encoding="utf-8"))
            return _PureXgbPredictor(model_json).predict
    except Exception:
        return None
    return None


def _get_predictor() -> Any:
    if "predictor" not in _MODEL_CACHE:
        _MODEL_CACHE["predictor"] = _build_predictor()
    return _MODEL_CACHE["predictor"]


def _get_class_mapping() -> dict[int, int] | None:
    if "mapping" in _MODEL_CACHE:
        return _MODEL_CACHE["mapping"]
    mapping: dict[int, int] | None = None
    if _MAPPING_PATH.exists():
        try:
            data = json.loads(_MAPPING_PATH.read_text(encoding="utf-8"))
            arr = data.get("class_to_option") if isinstance(data, dict) else data
            mapping = {int(i): int(v) for i, v in enumerate(arr)}
        except Exception:
            mapping = None
    _MODEL_CACHE["mapping"] = mapping
    return mapping


def _model_probs(features: Sequence[float]) -> list[float] | None:
    predict = _get_predictor()
    if predict is not None:
        try:
            return predict(features)
        except Exception:
            return None
    return None


_MODEL_CACHE["predictor"] = _build_predictor()


def _extract_features(
    current: Mapping[str, Any],
    my_idx: int,
    options_count: int,
) -> list[float] | None:
    players = current.get("players") or []
    if len(players) < 2:
        return None

    my_board = players[my_idx]
    opp_board = players[1 - my_idx]

    hand = _zone_entries(my_board, AREA_HAND)
    hand_grass = hand_lightning = hand_fight = 0
    for card in hand:
        cid = _dict_card_id(card)
        if cid == GRASS_ENERGY_ID:
            hand_grass += 1
        elif cid == LIGHTNING_ENERGY_ID:
            hand_lightning += 1
        elif cid == FIGHTING_ENERGY_ID:
            hand_fight += 1
    hand_size = len(hand)

    active = _pokemon_at(my_board, AREA_ACTIVE, 0)
    if active is not None:
        active_cid = _dict_card_id(active) or 0
        active_hp = int(active.get("hp", 0))
        active_max_hp = int(active.get("maxHp", 1))
        active_hp_ratio = active_hp / max(active_max_hp, 1)
        active_energy_cnt = len(_get_pokemon_energies(active))
    else:
        active_cid = 0
        active_hp_ratio = 0.0
        active_energy_cnt = 0

    bench = [p for p in _zone_entries(my_board, AREA_BENCH) if isinstance(p, dict)]
    has_latias_ex = 1 if (active_cid == LATIAS_EX_ID or any(_dict_card_id(p) == LATIAS_EX_ID for p in bench)) else 0
    bench_ogerpon_cnt = sum(1 for p in bench if _dict_card_id(p) == OGERPON_EX_ID)

    total_board_energies = 0
    for pokemon in ([active] if active is not None else []) + bench:
        total_board_energies += len(_get_pokemon_energies(pokemon))

    my_prizes = len(_zone_entries(my_board, AREA_PRIZE))
    opp_prizes = len(_zone_entries(opp_board, AREA_PRIZE))
    prize_diff = my_prizes - opp_prizes

    opp_active = _pokemon_at(opp_board, AREA_ACTIVE, 0)
    opp_active_cid = _dict_card_id(opp_active) or 0
    opp_active_immune_ex = 1 if opp_active_cid in IMMUNE_TO_EX_IDS else 0

    bench_hps = [int(p.get("maxHp", 0)) - int(p.get("damage", 0)) for p in bench]
    my_bench_lowest_hp = min(bench_hps) if bench_hps else 300.0

    opp_bench = [p for p in _zone_entries(opp_board, AREA_BENCH) if isinstance(p, dict)]
    opp_mons = ([opp_active] if opp_active is not None else []) + opp_bench
    opp_has_crustle = 0
    opp_has_dragapult = 0
    opp_has_froslass = 0
    for pokemon in opp_mons:
        cid = _dict_card_id(pokemon)
        name = str(pokemon.get("name") or "").lower()
        if cid == 345 or "crustle" in name:
            opp_has_crustle = 1
        if cid == 121 or "dragapult" in name:
            opp_has_dragapult = 1
        if cid == 104 or "froslass" in name:
            opp_has_froslass = 1

    return [
        float(int(current.get("turn", 0))),
        float(int(current.get("turnActionCount", 0))),
        float(hand_size),
        float(hand_grass),
        float(hand_lightning),
        float(hand_fight),
        float(1 if active_cid == RAGING_BOLT_EX_ID else 0),
        round(active_hp_ratio, 3),
        float(active_energy_cnt),
        float(has_latias_ex),
        float(bench_ogerpon_cnt),
        float(total_board_energies),
        float(my_prizes),
        float(opp_prizes),
        float(prize_diff),
        float(opp_active_immune_ex),
        float(options_count),
        float(my_bench_lowest_hp),
        float(opp_has_crustle),
        float(opp_has_dragapult),
        float(opp_has_froslass),
    ]


def _attach_score(
    energy_id: int | None,
    target_id: int | None,
    target_pokemon: Any = None,
) -> float:
    if target_id in AUXILIARY_POKEMON_IDS:
        return -200.0

    cur_energies = _get_pokemon_energies(target_pokemon) if target_pokemon else []
    has_lightning = LIGHTNING_ENERGY_ID in cur_energies
    has_fighting = FIGHTING_ENERGY_ID in cur_energies

    if energy_id == GRASS_ENERGY_ID:
        if target_id == IRON_LEAVES_ID:
            return 110.0
        if target_id == OGERPON_EX_ID:
            return 85.0
        if target_id == BLOODMOON_EX_ID:
            return 30.0
        return 10.0

    if energy_id == LIGHTNING_ENERGY_ID:
        if target_id == RAGING_BOLT_EX_ID:
            return 120.0 if not has_lightning else 70.0
        if target_id == BLOODMOON_ID:
            return 40.0
        if target_id == BLOODMOON_EX_ID:
            return 30.0
        return 10.0

    if energy_id == FIGHTING_ENERGY_ID:
        if target_id == RAGING_BOLT_EX_ID:
            return 120.0 if not has_fighting else 70.0
        if target_id == BLOODMOON_ID:
            return 90.0
        if target_id == BLOODMOON_EX_ID:
            return 35.0
        return 10.0

    return {
        RAGING_BOLT_EX_ID: 100.0,
        IRON_LEAVES_ID: 80.0,
        OGERPON_EX_ID: 75.0,
        BLOODMOON_ID: 60.0,
        BLOODMOON_EX_ID: 30.0,
    }.get(target_id, 5.0)


def _discard_energy_score(
    pokemon_id: int | None,
    energy_id: int | None,
    is_active_bolt: bool = False,
    bolt_energy_count: int = 0,
) -> float:
    if pokemon_id == OGERPON_EX_ID:
        return 120.0
    if pokemon_id == IRON_LEAVES_ID:
        return 80.0
    if pokemon_id in (BLOODMOON_ID, BLOODMOON_EX_ID):
        return 60.0
    if pokemon_id in AUXILIARY_POKEMON_IDS:
        return 70.0
    if pokemon_id == RAGING_BOLT_EX_ID:
        if is_active_bolt and bolt_energy_count <= 2:
            return -80.0
        return 20.0
    return 30.0


def _score_recover_select(
    current: Mapping[str, Any],
    options: Sequence[Any],
    my_idx: int,
    trigger: int | None,
) -> list[float]:
    players = current.get("players") or []
    if my_idx >= len(players):
        return [0.0] * len(options)
    discard = _zone_entries(players[my_idx], AREA_DISCARD)

    if trigger == NIGHT_STRETCHER_ID:
        hand_and_bench = _zone_ids(players[my_idx], AREA_HAND) + _zone_ids(players[my_idx], AREA_BENCH)
        has_raging_bolt = RAGING_BOLT_EX_ID in hand_and_bench

        if not has_raging_bolt:
            tiers = {RAGING_BOLT_EX_ID: 110.0, BLOODMOON_EX_ID: 90.0, BLOODMOON_ID: 90.0, IRON_LEAVES_ID: 60.0, OGERPON_EX_ID: 40.0, 1: 30.0, 4: 20.0, 6: 20.0}
        else:
            tiers = {BLOODMOON_EX_ID: 100.0, BLOODMOON_ID: 100.0, RAGING_BOLT_EX_ID: 80.0, IRON_LEAVES_ID: 60.0, 1: 50.0, OGERPON_EX_ID: 40.0, 4: 20.0, 6: 20.0}
    elif trigger == ENERGY_RETRIEVAL_ID:
        tiers = {GRASS_ENERGY_ID: 50.0, LIGHTNING_ENERGY_ID: 35.0, FIGHTING_ENERGY_ID: 35.0}
    else:
        tiers = {63: 60.0, 44: 55.0, 135: 50.0, 75: 40.0, 96: 30.0, 1: 25.0, 4: 20.0, 6: 20.0}

    scores: list[float] = []
    for option in options:
        if not isinstance(option, dict) or option.get("playerIndex") != my_idx:
            scores.append(0.0)
            continue
        index = option.get("index")
        cid = _dict_card_id(discard[index]) if isinstance(index, int) and 0 <= index < len(discard) else None
        scores.append(tiers.get(cid, 5.0))
    return scores


def _score_looking_select(
    current: Mapping[str, Any],
    options: Sequence[Any],
    my_idx: int,
    tiers: Mapping[int, float],
    default: float,
) -> list[float]:
    looking = current.get("looking") or []
    scores: list[float] = []
    for option in options:
        if not isinstance(option, dict) or option.get("playerIndex") != my_idx:
            scores.append(0.0)
            continue
        index = option.get("index")
        cid = _dict_card_id(looking[index]) if isinstance(index, int) and 0 <= index < len(looking) else None
        scores.append(tiers.get(cid, default))
    return scores


def _score_discard_cost(
    current: Mapping[str, Any],
    options: Sequence[Any],
    my_idx: int,
) -> list[float]:
    players = current.get("players") or []
    if my_idx >= len(players):
        return [0.0] * len(options)
    hand = _zone_entries(players[my_idx], AREA_HAND)
    hand_ids = [_dict_card_id(c) for c in hand]
    pokemon_counts = Counter(cid for cid in hand_ids if cid in POKEMON_CARD_IDS)
    opp = players[1 - my_idx] if len(players) > 1 - my_idx else None
    opp_has_crustle = CRUSTLE_ID in {_dict_card_id(p) or 0 for p in _inplay_roster(opp)}
    my_board_ids = _zone_ids(players[my_idx], AREA_ACTIVE) + _zone_ids(players[my_idx], AREA_BENCH)

    scores: list[float] = []
    for option in options:
        if not isinstance(option, dict) or option.get("playerIndex") != my_idx:
            scores.append(0.0)
            continue
        index = option.get("index")
        cid = hand_ids[index] if isinstance(index, int) and 0 <= index < len(hand) else None

        if opp_has_crustle and cid in (TOOL_SCRAPPER_ID, BLOODMOON_ID):
            if hand_ids.count(cid) + my_board_ids.count(cid) <= 1:
                scores.append(-300.0)
                continue

        if cid in KEY_ITEM_CARD_IDS:
            scores.append(-200.0)
        elif cid in BASIC_ENERGY_IDS:
            scores.append(100.0)
        elif cid == CYRANO_ID:
            scores.append(80.0)
        elif cid == CRISPIN_ID:
            scores.append(65.0)
        elif cid == LILLIE_ID:
            scores.append(45.0)
        elif cid == BOSS_ORDERS_ID:
            scores.append(10.0)
        elif cid in POKEMON_CARD_IDS:
            scores.append(40.0 if pokemon_counts.get(cid, 0) >= 2 else -50.0)
        else:
            scores.append(20.0)
    return scores


def _score_energy_discard(
    current: Mapping[str, Any],
    options: Sequence[Any],
    my_idx: int,
) -> list[float]:
    players = current.get("players") or []
    scores: list[float] = []
    player = players[my_idx] if my_idx < len(players) else None
    active_mon = _pokemon_at(player, AREA_ACTIVE, 0)
    active_cid = _dict_card_id(active_mon)
    active_energy_count = len(_get_pokemon_energies(active_mon))

    for option in options:
        if not isinstance(option, dict) or option.get("playerIndex") != my_idx:
            scores.append(0.0)
            continue
        area = option.get("area")
        idx = option.get("index")
        pokemon = _pokemon_at(player, area, idx)
        pokemon_id = _dict_card_id(pokemon)
        energy_id = _energy_id_at(pokemon, option.get("energyIndex"))

        is_active_bolt = (area == AREA_ACTIVE or (area == AREA_INPLAY and idx == 0)) and (pokemon_id == RAGING_BOLT_EX_ID)
        scores.append(_discard_energy_score(pokemon_id, energy_id, is_active_bolt, active_energy_count))
    return scores


def _score_attach_sub(
    current: Mapping[str, Any],
    options: Sequence[Any],
    my_idx: int,
) -> list[float]:
    players = current.get("players") or []
    if my_idx >= len(players):
        return [0.0] * len(options)
    hand = _zone_entries(players[my_idx], AREA_HAND)
    hand_energy_ids = [_dict_card_id(c) for c in hand if _dict_card_id(c) in BASIC_ENERGY_IDS]

    scores: list[float] = []
    for option in options:
        if not isinstance(option, dict) or option.get("playerIndex") != my_idx:
            scores.append(0.0)
            continue
        pokemon = _pokemon_at(players[my_idx], option.get("area"), option.get("index"))
        target_id = _dict_card_id(pokemon)
        energy_index = option.get("energyIndex")
        energy_id = hand_energy_ids[energy_index] if isinstance(energy_index, int) and 0 <= energy_index < len(hand_energy_ids) else None
        scores.append(_attach_score(energy_id, target_id, pokemon))
    return scores


def _score_switch_target(
    current: Mapping[str, Any],
    options: Sequence[Any],
    my_idx: int,
) -> list[float]:
    players = current.get("players") or []
    if my_idx >= len(players):
        return [0.0] * len(options)
    opp_active_ids = _zone_ids(players[1 - my_idx], AREA_ACTIVE) if len(players) > 1 - my_idx else []
    immune_ex = any(cid in IMMUNE_TO_EX_IDS for cid in opp_active_ids)

    scores: list[float] = []
    for option in options:
        if not isinstance(option, dict) or option.get("playerIndex") != my_idx:
            scores.append(0.0)
            continue
        pokemon = _inplay_by_area(players[my_idx], option.get("area"), option.get("index"))
        target_cid = _dict_card_id(pokemon)

        if immune_ex:
            scores.append(100.0 if target_cid == BLOODMOON_ID else 10.0)
        else:
            energies = len(_get_pokemon_energies(pokemon))
            if target_cid == RAGING_BOLT_EX_ID and energies >= 2:
                scores.append(90.0)
            elif target_cid == IRON_LEAVES_ID and energies >= 3:
                scores.append(70.0)
            else:
                scores.append(20.0)
    return scores


def _inplay_roster(player: Any) -> list[Any]:
    mons = list(player.get("active") or []) if player is not None else []
    mons += [p for p in _zone_entries(player, AREA_BENCH) if isinstance(p, dict)]
    return mons


def _bench_pokemon(player: Any, index: Any) -> Any:
    if player is None:
        return None
    bench = _zone_entries(player, AREA_BENCH)
    if isinstance(index, int) and 0 <= index < len(bench):
        return bench[index]
    return None


def _inplay_by_area(player: Any, area: Any, index: Any) -> Any:
    if area == AREA_ACTIVE:
        return _pokemon_at(player, AREA_ACTIVE, 0)
    return _bench_pokemon(player, index)


def _roster_tool_ids(roster: Sequence[Any]) -> list[int]:
    ids: list[int] = []
    for pokemon in roster:
        if not isinstance(pokemon, dict):
            continue
        for tool in (pokemon.get("tools") or []):
            if isinstance(tool, dict):
                ids.append(_dict_card_id(tool) or 0)
    return ids


def _score_drag_target(
    current: Mapping[str, Any],
    options: Sequence[Any],
    my_idx: int,
) -> list[float]:
    players = current.get("players") or []
    opp_idx = 1 - my_idx
    opp = players[opp_idx] if opp_idx < len(players) else None
    my = players[my_idx] if my_idx < len(players) else None
    opp_ids = {_dict_card_id(p) or 0 for p in _inplay_roster(opp)} if opp is not None else set()
    my_roster = _inplay_roster(my)
    my_has_bloodmoon = any(_dict_card_id(p) == BLOODMOON_ID for p in my_roster)
    opp_has_crustle = CRUSTLE_ID in opp_ids

    scores: list[float] = []
    for option in options:
        if not isinstance(option, dict) or option.get("playerIndex") != opp_idx:
            scores.append(0.0)
            continue
        pokemon = _bench_pokemon(opp, option.get("index"))
        cid = _dict_card_id(pokemon) or 0
        if cid == 0:
            scores.append(0.0)
            continue
        score = 0.0
        if cid == MEGA_KANGASKHAN_EX_ID:
            score += 100.0
        elif cid == DWEBBLE_ID:
            score += 90.0 if opp_has_crustle else 45.0
        elif cid == CRUSTLE_ID:
            score += 15.0 if my_has_bloodmoon else -70.0
        max_hp = int(pokemon.get("maxHp", 0))
        cur_hp = max_hp - int(pokemon.get("damage", 0))
        if max_hp > 0 and 0 < cur_hp <= max_hp * 0.45:
            score += 50.0
        elif cid not in (CRUSTLE_ID, DWEBBLE_ID, MEGA_KANGASKHAN_EX_ID):
            score += 25.0
        scores.append(score)
    return scores


def _score_transfer_source(
    current: Mapping[str, Any],
    options: Sequence[Any],
    my_idx: int,
) -> list[float]:
    players = current.get("players") or []
    my = players[my_idx] if my_idx < len(players) else None
    scores: list[float] = []
    for option in options:
        if not isinstance(option, dict) or option.get("playerIndex") != my_idx:
            scores.append(0.0)
            continue
        pokemon = _inplay_by_area(my, option.get("area"), option.get("index"))
        cid = _dict_card_id(pokemon) or 0
        n = len(_get_pokemon_energies(pokemon))
        if cid == OGERPON_EX_ID:
            scores.append(45.0)
        elif cid == LATIAS_EX_ID:
            scores.append(40.0)
        elif cid == IRON_LEAVES_ID:
            scores.append(35.0)
        elif cid == BLOODMOON_ID and n <= 2:
            scores.append(-60.0)
        elif cid == RAGING_BOLT_EX_ID and n <= 4:
            scores.append(-50.0)
        else:
            scores.append(10.0)
    return scores


def _score_transfer_dest(
    current: Mapping[str, Any],
    options: Sequence[Any],
    my_idx: int,
) -> list[float]:
    players = current.get("players") or []
    my = players[my_idx] if my_idx < len(players) else None
    scores: list[float] = []
    for option in options:
        if not isinstance(option, dict) or option.get("playerIndex") != my_idx:
            scores.append(0.0)
            continue
        pokemon = _inplay_by_area(my, option.get("area"), option.get("index"))
        cid = _dict_card_id(pokemon) or 0
        n = len(_get_pokemon_energies(pokemon))
        if cid == BLOODMOON_ID and n < 3:
            scores.append(85.0)
        elif cid == RAGING_BOLT_EX_ID and n < 4:
            scores.append(70.0)
        elif cid == IRON_LEAVES_ID and n < 3:
            scores.append(55.0)
        elif cid == OGERPON_EX_ID:
            scores.append(20.0)
        elif cid == LATIAS_EX_ID:
            scores.append(-40.0)
        else:
            scores.append(10.0)
    return scores


def _score_sub_select(
    request: Request,
    current: Mapping[str, Any],
    logs: Sequence[Any],
    my_idx: int,
) -> list[float]:
    options = request["options"]
    count = len(options)
    select_type = request["type"]
    context = str(request.get("context"))
    trigger = _last_played_card(logs, my_idx)

    if select_type == 1 and context == "7":
        areas = {option.get("area") for option in options if isinstance(option, dict)}
        if AREA_DISCARD in areas:
            return _score_recover_select(current, options, my_idx, trigger)
        if AREA_LOOKING in areas:
            if trigger == POKEGEAR_ID:
                return _score_looking_select(current, options, my_idx, _POKEGEAR_TIERS, 10.0)
            if trigger == BUG_CATCHING_SET_ID:
                return _score_looking_select(current, options, my_idx, _BUG_CATCHING_TIERS, 5.0)
            if trigger == CYRANO_ID:
                return _score_looking_select(current, options, my_idx, _CYRANO_SEARCH_TIERS, 20.0)
            if trigger == ENERGY_SEARCH_ID:
                return _score_looking_select(current, options, my_idx, _ENERGY_SEARCH_TIERS, 15.0)
        return [0.0] * count

    if select_type == 1 and context == "8":
        return _score_discard_cost(current, options, my_idx)

    if select_type == 1 and context == "3" and trigger in (SWITCH_ID, BOSS_ORDERS_ID, PRIME_CATCHER_ID):
        opp_idx = 1 - my_idx
        opp_indices = [i for i, o in enumerate(options) if isinstance(o, dict) and o.get("playerIndex") == opp_idx]
        my_indices = [i for i, o in enumerate(options) if isinstance(o, dict) and o.get("playerIndex") == my_idx]
        results = [0.0] * count
        if opp_indices:
            opp_scores = _score_drag_target(current, [options[i] for i in opp_indices], my_idx)
            for i, score in zip(opp_indices, opp_scores):
                results[i] = score
        if my_indices:
            my_scores = _score_switch_target(current, [options[i] for i in my_indices], my_idx)
            for i, score in zip(my_indices, my_scores):
                results[i] = score
        return results

    if select_type == 2 and context == "28" and trigger == ENERGY_SWITCH_ID:
        return _score_transfer_source(current, options, my_idx)

    if select_type == 1 and context == "21" and trigger == ENERGY_SWITCH_ID:
        return _score_transfer_dest(current, options, my_idx)

    if select_type == 2 and context == "26":
        return _score_energy_discard(current, options, my_idx)

    if select_type == 4:
        return _score_attach_sub(current, options, my_idx)

    return [0.0] * count


def _main_attach_target_pokemon(player: Any, option: Mapping[str, Any]) -> Any:
    in_play_area = option.get("inPlayArea")
    if in_play_area == AREA_ACTIVE:
        return _pokemon_at(player, AREA_ACTIVE, 0)
    elif in_play_area == AREA_BENCH:
        bench = _zone_entries(player, AREA_BENCH)
        index = option.get("inPlayIndex")
        return bench[index] if isinstance(index, int) and 0 <= index < len(bench) else None
    return None


def _score_main_select(
    request: Request,
    current: Mapping[str, Any],
    logs: Sequence[Any],
    my_idx: int,
) -> list[float]:
    options = request["options"]
    count = len(options)
    players = current.get("players") or []
    if len(players) < 2:
        return [MAIN_FALLBACK_SCORES.get(option.get("type"), 0.0) if isinstance(option, dict) else 0.0 for option in options]

    features = _extract_features(current, my_idx, count)
    probs = _model_probs(features) if features is not None else None
    if probs:
        mapping = _get_class_mapping()
        prob_by_option: dict[int, float] = {}
        if mapping:
            for class_index, option_index in mapping.items():
                if 0 <= class_index < len(probs):
                    prob_by_option[int(option_index)] = probs[class_index]
        else:
            prob_by_option = {i: p for i, p in enumerate(probs)}
        base_scores = [prob_by_option.get(i, 0.0) for i in range(count)]
    else:
        base_scores = [MAIN_FALLBACK_SCORES.get(option.get("type"), 0.0) if isinstance(option, dict) else 0.0 for option in options]

    my_board = players[my_idx]
    opp_board = players[1 - my_idx]
    my_hand = _zone_entries(my_board, AREA_HAND)
    my_hand_ids = [_dict_card_id(c) for c in my_hand]
    my_bench_ids = _zone_ids(my_board, AREA_BENCH)
    my_active_ids = _zone_ids(my_board, AREA_ACTIVE)
    opp_active_ids = _zone_ids(opp_board, AREA_ACTIVE)

    turn = int(current.get("turn", 0))
    has_latias_ex = LATIAS_EX_ID in (my_active_ids + my_bench_ids)
    has_raging_bolt_board = RAGING_BOLT_EX_ID in (my_active_ids + my_bench_ids)
    immune_ex = any(cid in IMMUNE_TO_EX_IDS for cid in opp_active_ids)
    bloodmoon_ready = (BLOODMOON_ID in my_bench_ids) or (BLOODMOON_ID in my_hand_ids)

    active_mon = _pokemon_at(my_board, AREA_ACTIVE, 0)
    bench_mons = [p for p in _zone_entries(my_board, AREA_BENCH) if isinstance(p, dict)]
    total_grass_energies = sum(_get_pokemon_energies(p).count(GRASS_ENERGY_ID) for p in ([active_mon] if active_mon else []) + bench_mons)

    discard_energies = [cid for cid in _zone_ids(my_board, AREA_DISCARD) if cid in BASIC_ENERGY_IDS]
    discard_energy_count = len(discard_energies)

    my_hand_size = len(my_hand_ids)
    opp_hand_size = len(_zone_entries(opp_board, AREA_HAND))
    my_prizes = len(_zone_entries(my_board, AREA_PRIZE))
    opp_prizes = len(_zone_entries(opp_board, AREA_PRIZE))
    knocked_out = bool(request.get("our_knocked_out_this_turn")) or _our_pokemon_knocked_out(logs, my_idx)
    stamp_trigger = (opp_hand_size - my_hand_size >= 3) and (my_prizes > opp_prizes) and knocked_out

    my_board_ids = my_active_ids + my_bench_ids
    my_mons = ([active_mon] if active_mon else []) + bench_mons
    hand_basic = sum(1 for c in my_hand_ids if c in BASIC_ENERGY_IDS)
    my_discard_ids = _zone_ids(my_board, AREA_DISCARD)
    opp_active_mon = _pokemon_at(opp_board, AREA_ACTIVE, 0)
    opp_active_cid = _dict_card_id(opp_active_mon) or 0
    opp_active_hp = (int(opp_active_mon.get("maxHp", 0)) - int(opp_active_mon.get("damage", 0))) if opp_active_mon is not None else 0
    opp_bench_mons = [p for p in _zone_entries(opp_board, AREA_BENCH) if isinstance(p, dict)]
    opp_bench_ids = [_dict_card_id(p) for p in opp_bench_mons]
    opp_roster_ids = set(opp_active_ids) | set(opp_bench_ids)
    opp_has_crustle = CRUSTLE_ID in opp_roster_ids
    opp_active_crustle = opp_active_cid == CRUSTLE_ID
    opp_bench_target = any(c in (DWEBBLE_ID, MEGA_KANGASKHAN_EX_ID) for c in opp_bench_ids)
    opp_any_tool = bool(_roster_tool_ids(_inplay_roster(opp_board)))
    ogerpon_has_spare_grass = any(
        (_dict_card_id(p) == OGERPON_EX_ID) and GRASS_ENERGY_ID in _get_pokemon_energies(p)
        for p in my_mons
    )
    crustle_bloodmoon_kod = opp_has_crustle and (BLOODMOON_ID in my_discard_ids) and knocked_out

    # Opponent key-resource stock. The previous phase only reasoned about our own
    # discard pile (my_discard_ids) plus a hardcoded Crustle check; across eight
    # decks we also need to know what the opponent has already spent, because a
    # KO only sticks if they cannot rebuild.
    opp_discard_ids = _zone_ids(opp_board, AREA_DISCARD)
    opp_recovery_spent = sum(1 for cid in opp_discard_ids if cid in RECOVERY_CARD_IDS)
    opp_recovery_gone = opp_recovery_spent >= OPP_RECOVERY_EXHAUSTED
    opp_attacker_spent = any(cid in KEY_OPP_ATTACKER_IDS for cid in opp_discard_ids)
    # Their board cannot be rebuilt, so committing to a KO race is safe.
    opp_cannot_rebuild = opp_recovery_gone or opp_attacker_spent

    scores = list(base_scores)
    for i, option in enumerate(options):
        if not isinstance(option, dict):
            continue
        option_type = option.get("type")

        if option_type == 7:
            index = option.get("index")
            cid = my_hand_ids[index] if isinstance(index, int) and 0 <= index < len(my_hand_ids) else None

            if cid == IRON_LEAVES_ID:
                if total_grass_energies < 2:
                    scores[i] -= 40.0
                else:
                    scores[i] += 80.0

            if turn <= 2 and not has_raging_bolt_board:
                if cid in (ULTRA_BALL_ID, CYRANO_ID, BUG_CATCHING_SET_ID, POKEGEAR_ID):
                    scores[i] += 120.0
                if cid in (LILLIE_ID, CRISPIN_ID):
                    scores[i] += 90.0

            if cid == ENERGY_RETRIEVAL_ID:
                if discard_energy_count >= 2:
                    scores[i] += 85.0
                else:
                    scores[i] -= 20.0

            elif cid == UNFAIR_STAMP_ID:
                scores[i] += 130.0 if stamp_trigger else -80.0

            elif cid == SWITCH_ID and has_latias_ex:
                scores[i] -= 25.0

            elif cid == BOSS_ORDERS_ID and immune_ex and not bloodmoon_ready:
                scores[i] += 150.0

            elif cid == BLOODMOON_ID and immune_ex:
                scores[i] += 85.0

            elif cid == BLOODMOON_EX_ID and opp_prizes <= 2:
                scores[i] += 60.0

            if cid == CYRANO_ID and not has_raging_bolt_board and OGERPON_EX_ID not in my_board_ids:
                scores[i] += 160.0

            if cid == IRON_LEAVES_ID and ogerpon_has_spare_grass and opp_active_cid not in IMMUNE_TO_EX_IDS and 0 < opp_active_hp <= 180:
                scores[i] += 70.0

            if cid == TOOL_SCRAPPER_ID:
                scores[i] += 180.0 if opp_any_tool else -200.0

            if cid == PRIME_CATCHER_ID:
                if opp_active_crustle:
                    scores[i] += 200.0 if opp_bench_target else 40.0
                elif opp_bench_target and opp_has_crustle:
                    scores[i] += 120.0
                elif not opp_bench_mons:
                    scores[i] -= 90.0

            if cid == BOSS_ORDERS_ID and not (immune_ex and not bloodmoon_ready) and opp_bench_target:
                scores[i] += 60.0

            if cid == NIGHT_STRETCHER_ID:
                if not any(c in (RAGING_BOLT_EX_ID, BLOODMOON_ID) or c in BASIC_ENERGY_IDS for c in my_discard_ids):
                    scores[i] -= 150.0
                elif crustle_bloodmoon_kod:
                    scores[i] += 140.0
                elif opp_cannot_rebuild:
                    # They have no way back, so our own recovery matters less and
                    # we should be spending cards on pressure instead.
                    scores[i] -= 40.0

            if cid == BOSS_ORDERS_ID and opp_cannot_rebuild and opp_bench_target:
                # Gusting a target they cannot rebuild around closes the game.
                scores[i] += 75.0

            if cid == ENERGY_SWITCH_ID:
                need_charge = any(
                    (_dict_card_id(p) == BLOODMOON_ID and len(_get_pokemon_energies(p)) < 3)
                    or (_dict_card_id(p) == RAGING_BOLT_EX_ID and len(_get_pokemon_energies(p)) < 4)
                    for p in my_mons
                )
                spare = any(
                    _dict_card_id(p) in (OGERPON_EX_ID, LATIAS_EX_ID, IRON_LEAVES_ID) and len(_get_pokemon_energies(p)) >= 1
                    for p in my_mons
                )
                scores[i] += 60.0 if (need_charge and spare) else -40.0

            if cid == ENERGY_SEARCH_ID:
                scores[i] += 60.0 if hand_basic == 0 else (-15.0 if hand_basic >= 2 else 0.0)

        elif option_type == 8:
            energy_id = None
            index = option.get("index")
            if isinstance(index, int) and 0 <= index < len(my_hand_ids):
                energy_id = my_hand_ids[index]
            target_pokemon = _main_attach_target_pokemon(my_board, option)
            target_id = _dict_card_id(target_pokemon)
            scores[i] += _attach_score(energy_id, target_id, target_pokemon)

        elif option_type == 9:
            if immune_ex and BLOODMOON_ID in my_bench_ids:
                scores[i] += 90.0
            elif has_latias_ex:
                scores[i] += 30.0

    return scores


def score_options(request: Request, observation: Observation | None = None) -> list[float]:
    observation = observation or {}
    current = observation.get("current") or request.get("state") or {}
    logs = observation.get("logs") or request.get("logs") or []
    my_idx = int(current.get("yourIndex", request.get("player_index", 0)))

    if request["type"] != MAIN_SELECT_TYPE:
        return _score_sub_select(request, current, logs, my_idx)
    return _score_main_select(request, current, logs, my_idx)


class AIAgent:
    def __init__(self, deck: list[int], player_index: int = 0) -> None:
        validate_deck(deck)
        self.deck = deck.copy()
        self.player_index = player_index
        self.score = ScoreTracker(player_index)
        self._ko_turn = -1
        self._ko_this_turn = False

    def observe(self, observation: Observation) -> None:
        self.score.observe(observation)

    def finish(self, won: bool) -> tuple[float, float]:
        return self.score.finish(won)

    def __call__(
        self,
        observation: Observation,
        _configuration: Any = None,
    ) -> list[int]:
        if observation.get("current") is None:
            self.score.reset()
            return self.deck.copy()

        self.observe(observation)

        current = observation["current"]
        my_idx = int(current.get("yourIndex", self.player_index))
        turn = int(current.get("turn", 0))
        acting_player = _turn_player(current)

        if acting_player != my_idx or turn != self._ko_turn:
            self._ko_turn = turn
            self._ko_this_turn = False
        if _our_pokemon_knocked_out(observation.get("logs") or [], my_idx):
            self._ko_this_turn = True

        request = build_request(observation, self.score.feedback())
        request["player_index"] = my_idx
        request["our_knocked_out_this_turn"] = self._ko_this_turn
        return select_legal_options(request, score_options(request, observation))


def make_ai_agent(deck: list[int], player_index: int = 0) -> AIAgent:
    return AIAgent(deck, player_index)
