import json
from typing import Any, Mapping, Sequence


Observation = Mapping[str, Any]
Request = dict[str, Any]
PokemonKey = tuple[int, int]
PokemonState = tuple[int, int]

MAIN_SELECT_TYPE = 0
MAIN_OPTION_SCORES = {8: 3.0, 13: 2.0, 14: 1.0}

TURN_END = 3
CHANGE = 9
EVOLVE = 12
DEVOLVE = 13
ATTACK = 15
HP_CHANGE = 16

WIN_REWARD = 1.0
PRIZE_REWARD = 0.1
OWN_TURN_PENALTY = 0.01
DAMAGE_REWARD = 0.1
ATTACK_PENALTY_DIVISOR = 1000.0


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
    return tuple(len(player.get("prize") or []) for player in current["players"])


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
            old_key = (int(player_index), int(old_serial))
            transitions[old_key] = (int(player_index), int(new_serial))
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
    """Accumulates one AI player's score from CABT states and event logs."""

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

            ended_turns = [
                int(log["playerIndex"])
                for log in logs
                if log.get("type") == TURN_END
            ]
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
        self._attacks_since_prize += sum(
            log.get("type") == ATTACK
            and log.get("playerIndex") == self.player_index
            for log in logs
        )

    def _collect_damage(
        self,
        logs: Sequence[Observation],
        current_pokemon: Mapping[PokemonKey, PokemonState],
    ) -> None:
        transitions = _pokemon_transitions(logs)
        reverse_transitions = {new: old for old, new in transitions.items()}
        changed = {
            (int(log["playerIndex"]), int(log["serial"]))
            for log in logs
            if log.get("type") == HP_CHANGE
            and log.get("playerIndex") is not None
            and log.get("serial") is not None
        }

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
        prizes_taken = max(
            self._prizes[self.player_index] - current_prizes[self.player_index],
            0,
        )
        opponent_prizes_taken = max(
            self._prizes[opponent_index] - current_prizes[opponent_index],
            0,
        )

        if prizes_taken:
            self._pending_score += (
                PRIZE_REWARD * prizes_taken
                - calculate_attack_penalty(self._attacks_since_prize)
            )
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


def score_options(request: Request) -> list[float]:
    """Replace this heuristic with model scores; options remain engine-defined."""
    if request["type"] != MAIN_SELECT_TYPE:
        return [0.0] * len(request["options"])
    return [
        MAIN_OPTION_SCORES.get(option.get("type"), 0.0)
        for option in request["options"]
    ]


def select_legal_options(request: Request, scores: Sequence[float]) -> list[int]:
    options = request["options"]
    if len(scores) != len(options):
        raise ValueError("The policy must score every legal option exactly once.")

    count = request["max_count"]
    ranked_indices = sorted(range(len(options)), key=lambda i: (-scores[i], i))
    return ranked_indices[:count]


class AIAgent:
    def __init__(self, deck: list[int], player_index: int = 0) -> None:
        validate_deck(deck)
        self.deck = deck.copy()
        self.score = ScoreTracker(player_index)

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
        request = build_request(observation, self.score.feedback())
        return select_legal_options(request, score_options(request))


def make_ai_agent(deck: list[int], player_index: int = 0) -> AIAgent:
    return AIAgent(deck, player_index)
