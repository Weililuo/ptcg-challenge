from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import create_environment, load_deck, model_path

from base_data import EncodedDecision, collate_decisions, encode_decision, file_sha256
from common import (
    ActorCriticPolicy,
    EntityPointerPolicy,
    load_actor_critic,
    load_checkpoint,
    load_model,
)
from learn_data import (
    LearningData,
    Matchup,
    learning_data_record,
    load_learning_data,
    sample_matchups,
)
from train import print_event, save_checkpoint, write_json


@dataclass(frozen=True)
class LearningConfig:
    iterations: int = 100
    games_per_iteration: int = 16
    rollout_workers: int = 4
    batch_size: int = 128
    update_epochs: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    reference_coefficient: float = 0.01
    prize_coefficient: float = 0.25
    gradient_clip: float = 1.0
    seed: int = 42


@dataclass
class Transition:
    decision: EncodedDecision
    log_probability: float
    value: float
    prize_potential: float
    advantage: float = 0.0
    return_value: float = 0.0


_WORKER_MODELS: dict[Path, EntityPointerPolicy] = {}


def prize_potential(observation: dict[str, Any]) -> float:
    """Return normalized prize-card advantage from the acting player's view."""
    current = observation.get("current")
    if not isinstance(current, dict):
        return 0.0
    players = current.get("players") or []
    own_index = current.get("yourIndex")
    if own_index not in (0, 1) or len(players) != 2:
        return 0.0

    def count(player: dict[str, Any]) -> int:
        prize = player.get("prize")
        return len(prize) if isinstance(prize, list) else int(player.get("prizeCount", 0))

    own = count(players[own_index])
    opponent = count(players[1 - own_index])
    return (opponent - own) / 6.0


def make_agent(
    policy: EntityPointerPolicy | ActorCriticPolicy,
    deck: list[int],
    device: torch.device,
    transitions: list[Transition] | None = None,
    sample: bool = False,
):
    """Create a rollout agent and optionally record its decisions."""
    def agent(observation: dict[str, Any], _configuration: Any = None) -> list[int]:
        if observation.get("current") is None:
            return deck.copy()
        decision = encode_decision(observation, deck)
        batch = collate_decisions([decision]).to(device)
        if isinstance(policy, ActorCriticPolicy):
            output = policy.act(batch, sample=sample)
            action = output.actions[0]
            if transitions is not None:
                transitions.append(
                    Transition(
                        replace(decision, action=tuple(action)),
                        float(output.log_probability[0].item()),
                        float(output.value[0].item()),
                        prize_potential(observation),
                    )
                )
            return action
        return policy.decode(batch)[0]

    return agent


def state_field(state: Any, name: str) -> Any:
    """Read one Kaggle state field."""
    if hasattr(state, name):
        return getattr(state, name)
    return state[name]


def finish_episode(
    transitions: list[Transition],
    outcome: float,
    final_potential: float,
    config: LearningConfig,
) -> None:
    """Add terminal and prize-difference rewards, then calculate GAE."""
    advantage = 0.0
    next_value = 0.0
    next_potential = final_potential
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        reward = config.prize_coefficient * (
            next_potential - transition.prize_potential
        )
        if index == len(transitions) - 1:
            reward += outcome
        delta = reward + config.gamma * next_value - transition.value
        advantage = (
            delta + config.gamma * config.gae_lambda * advantage
        )
        transition.advantage = advantage
        transition.return_value = advantage + transition.value
        next_value = transition.value
        next_potential = transition.prize_potential


def play_episode(
    actor: ActorCriticPolicy,
    matchup: Matchup,
    fixed_models: dict[Path, EntityPointerPolicy],
    learner_seat: int,
    device: torch.device,
    config: LearningConfig,
) -> tuple[list[Transition], float]:
    """Play one episode against a fixed opponent or the current policy."""
    transitions: list[Transition] = []
    learner_agent = make_agent(
        actor,
        load_deck(matchup.learner_deck),
        device,
        transitions,
        sample=True,
    )
    if matchup.opponent_model is None:
        opponent_agent = make_agent(
            actor,
            load_deck(matchup.opponent_deck),
            device,
            sample=True,
        )
    else:
        if matchup.opponent_model not in fixed_models:
            fixed_models[matchup.opponent_model] = load_model(
                matchup.opponent_model,
                device,
            )
        opponent_agent = make_agent(
            fixed_models[matchup.opponent_model],
            load_deck(matchup.opponent_deck),
            device,
        )

    agents = (
        [learner_agent, opponent_agent]
        if learner_seat == 0
        else [opponent_agent, learner_agent]
    )
    environment = create_environment()
    environment.run(agents)
    final_state = environment.state[learner_seat]
    outcome = state_field(final_state, "reward")
    if outcome not in (-1, 0, 1) or not transitions:
        raise RuntimeError("CABT did not produce a valid learning episode")
    final_observation = state_field(final_state, "observation")
    finish_episode(
        transitions,
        float(outcome),
        prize_potential(final_observation),
        config,
    )
    return transitions, float(outcome)


def collect_rollouts(
    actor: ActorCriticPolicy,
    matchups: list[Matchup],
    fixed_models: dict[Path, EntityPointerPolicy],
    first_game: int,
    device: torch.device,
    config: LearningConfig,
) -> tuple[list[Transition], dict[str, int]]:
    """Collect a sequential subset of one on-policy rollout set."""
    transitions = []
    outcomes = {"wins": 0, "losses": 0, "draws": 0}
    for game, matchup in enumerate(matchups):
        episode, outcome = play_episode(
            actor,
            matchup,
            fixed_models,
            (first_game + game) % 2,
            device,
            config,
        )
        transitions.extend(episode)
        outcomes[("wins" if outcome > 0 else "losses" if outcome < 0 else "draws")] += 1
    return transitions, outcomes


def rollout_worker(
    snapshot: Path,
    matchups: list[Matchup],
    first_game: int,
    seed: int,
    config: LearningConfig,
) -> tuple[list[Transition], dict[str, int]]:
    """Collect one worker's games with CPU inference."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    actor = load_actor_critic(snapshot, "cpu")
    return collect_rollouts(
        actor,
        matchups,
        _WORKER_MODELS,
        first_game,
        torch.device("cpu"),
        config,
    )


def collect_parallel(
    executor: ProcessPoolExecutor,
    snapshot: Path,
    data: LearningData,
    iteration: int,
    config: LearningConfig,
) -> tuple[list[Transition], dict[str, int]]:
    """Split one rollout evenly across persistent worker processes."""
    matchups = sample_matchups(
        data,
        config.games_per_iteration,
        config.seed + iteration,
    )
    workers = min(config.rollout_workers, config.games_per_iteration)
    quotient, remainder = divmod(config.games_per_iteration, workers)
    futures = []
    first_game = 0
    for worker in range(workers):
        games = quotient + int(worker < remainder)
        futures.append(
            executor.submit(
                rollout_worker,
                snapshot,
                matchups[first_game : first_game + games],
                first_game,
                config.seed + iteration * workers + worker,
                config,
            )
        )
        first_game += games

    transitions = []
    outcomes = {"wins": 0, "losses": 0, "draws": 0}
    for future in futures:
        worker_transitions, worker_outcomes = future.result()
        transitions.extend(worker_transitions)
        for key in outcomes:
            outcomes[key] += worker_outcomes[key]
    return transitions, outcomes


def distribution_kl(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Calculate legal-action KL along the sampled action prefixes."""
    current_log = F.log_softmax(current_logits.float(), dim=-1)
    reference_log = F.log_softmax(reference_logits.float(), dim=-1)
    current_probability = current_log.exp()
    values = (current_probability * (current_log - reference_log)).sum(dim=-1)
    return values[targets != -100].mean()


def ppo_update(
    actor: ActorCriticPolicy,
    reference: EntityPointerPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    rng: random.Random,
    device: torch.device,
    config: LearningConfig,
) -> dict[str, float]:
    """Run a small PPO update over the current on-policy decisions."""
    advantages = torch.tensor([row.advantage for row in transitions])
    advantages = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    ).clamp_min(1e-6)
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "reference_kl": 0.0,
        "clip_fraction": 0.0,
    }
    examples = 0
    indices = list(range(len(transitions)))
    actor.eval()

    for _ in range(config.update_epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), config.batch_size):
            chosen = indices[start : start + config.batch_size]
            batch = collate_decisions(
                [transitions[index].decision for index in chosen]
            ).to(device)
            old_log_probability = torch.tensor(
                [transitions[index].log_probability for index in chosen],
                device=device,
            )
            advantage = advantages[chosen].to(device)
            returns = torch.tensor(
                [transitions[index].return_value for index in chosen],
                device=device,
            )

            output = actor.evaluate_actions(batch)
            ratio = (output.log_probability - old_log_probability).exp()
            unclipped = ratio * advantage
            clipped = ratio.clamp(1.0 - config.clip, 1.0 + config.clip) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(output.value, returns)
            entropy = output.entropy.mean()
            reference_kl = torch.zeros((), device=device)
            if config.reference_coefficient:
                with torch.inference_mode():
                    reference_logits = reference.teacher_forced_logits(batch)
                if output.logits is None:
                    raise RuntimeError("action logits are unavailable")
                reference_kl = distribution_kl(
                    output.logits,
                    reference_logits,
                    batch.targets,
                )
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
                + config.reference_coefficient * reference_kl
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), config.gradient_clip)
            optimizer.step()

            count = len(chosen)
            examples += count
            metrics = {
                "loss": loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "reference_kl": reference_kl,
                "clip_fraction": ((ratio - 1.0).abs() > config.clip).float().mean(),
            }
            for name, value in metrics.items():
                totals[name] += float(value.detach().item()) * count

    return {name: value / examples for name, value in totals.items()}


def validate_config(config: LearningConfig) -> None:
    """Reject only settings that make the update invalid."""
    positive = (
        config.iterations,
        config.games_per_iteration,
        config.rollout_workers,
        config.batch_size,
        config.update_epochs,
        config.learning_rate,
        config.gradient_clip,
    )
    if min(positive) <= 0:
        raise ValueError("learning counts and rates must be positive")
    if not 0 <= config.gae_lambda <= 1 or not 0 < config.gamma <= 1:
        raise ValueError("invalid GAE settings")
    if min(
        config.clip,
        config.value_coefficient,
        config.entropy_coefficient,
        config.reference_coefficient,
        config.prize_coefficient,
        config.weight_decay,
    ) < 0:
        raise ValueError("loss coefficients cannot be negative")


def resolve_device(value: str) -> torch.device:
    """Resolve the learning device."""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return device


def learn(
    name: str,
    initial: Path,
    data: LearningData,
    config: LearningConfig | None = None,
    resume: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    """Train and publish one lightweight self-play PPO model."""
    settings = config or LearningConfig()
    validate_config(settings)
    selected_device = resolve_device(device)
    initial_path = model_path(initial)
    model_type = data.model_type
    output_dir = ROOT / "models" / ("experts" if model_type == "expert" else "base") / name
    work_dir = ROOT / "workspace" / "learn" / ("experts" if model_type == "expert" else "base") / name
    model_output = output_dir / "model.pt"
    record_output = output_dir / "record.json"
    last_output = work_dir / "last.pt"
    rollout_output = work_dir / "rollout.pt"
    initial_sha256 = file_sha256(initial_path)

    if not resume and any(path.exists() for path in (model_output, record_output, last_output)):
        raise FileExistsError(f"learning output already exists: {name}")
    if resume and not last_output.exists():
        raise FileNotFoundError(f"no learning checkpoint to resume: {last_output}")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    actor = load_actor_critic(initial_path, selected_device)
    reference = load_model(initial_path, selected_device)
    reference.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    start_iteration = 0
    total_games = 0
    total_decisions = 0
    total_outcomes = {"wins": 0, "losses": 0, "draws": 0}
    if resume:
        checkpoint = load_checkpoint(last_output)
        if checkpoint["initial_sha256"] != initial_sha256:
            raise ValueError("initial model differs from the checkpoint")
        if checkpoint["learning_config"] != asdict(settings):
            raise ValueError("learning settings differ from the checkpoint")
        if checkpoint.get("learning_data_sha256") != data.sha256:
            raise ValueError("learning data differs from the checkpoint")
        actor.policy.load_state_dict(checkpoint["model"])
        actor.value_head.load_state_dict(checkpoint["value_head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint["iteration"])
        stats = checkpoint.get("stats", {})
        total_games = int(stats.get("games", 0))
        total_decisions = int(stats.get("decisions", 0))
        for key in total_outcomes:
            total_outcomes[key] = int(stats.get(key, 0))

    seed = settings.seed + start_iteration
    rng = random.Random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    last_metrics: dict[str, float] = {}

    with ProcessPoolExecutor(max_workers=settings.rollout_workers) as executor:
        for iteration in range(start_iteration, settings.iterations):
            save_checkpoint(
                rollout_output,
                {
                    "model": actor.policy.state_dict(),
                    "value_head": actor.value_head.state_dict(),
                    "model_config": asdict(actor.policy.config),
                },
            )
            transitions, outcomes = collect_parallel(
                executor,
                rollout_output,
                data,
                iteration,
                settings,
            )
            last_metrics = ppo_update(
                actor,
                reference,
                optimizer,
                transitions,
                rng,
                selected_device,
                settings,
            )
            total_games += settings.games_per_iteration
            total_decisions += len(transitions)
            for key in total_outcomes:
                total_outcomes[key] += outcomes[key]
            save_checkpoint(
                last_output,
                {
                    "model": actor.policy.state_dict(),
                    "value_head": actor.value_head.state_dict(),
                    "model_config": asdict(actor.policy.config),
                    "optimizer": optimizer.state_dict(),
                    "learning_config": asdict(settings),
                    "learning_data_sha256": data.sha256,
                    "initial_sha256": initial_sha256,
                    "iteration": iteration + 1,
                    "stats": {
                        "games": total_games,
                        "decisions": total_decisions,
                        **total_outcomes,
                    },
                },
            )
            print_event(
                "learn",
                iteration=iteration + 1,
                games=settings.games_per_iteration,
                decisions=len(transitions),
                wins=outcomes["wins"],
                losses=outcomes["losses"],
                draws=outcomes["draws"],
                **last_metrics,
            )

    published = {
        "model": actor.policy.state_dict(),
        "value_head": actor.value_head.state_dict(),
        "model_config": asdict(actor.policy.config),
    }
    temporary = model_output.with_name(f"{model_output.name}.tmp")
    torch.save(published, temporary)
    temporary.replace(model_output)
    record = {
        "name": name,
        "model_type": model_type,
        **({"deck": data.model_deck} if data.model_deck else {}),
        "model": {
            "path": model_output.name,
            "sha256": file_sha256(model_output),
            "parameters": sum(parameter.numel() for parameter in actor.parameters()),
            "config": asdict(actor.policy.config),
        },
        "initialized_from": {
            "path": initial_path.relative_to(ROOT).as_posix(),
            "sha256": initial_sha256,
        },
        "learning": {
            "algorithm": "ppo_actor_critic",
            "config": asdict(settings),
            "data": learning_data_record(data),
            "reward": {
                "terminal": {"win": 1, "loss": -1, "draw": 0},
                "prize": "normalized_prize_difference_delta",
                "prize_coefficient": settings.prize_coefficient,
            },
            "expert_decks_are_fixed": True,
            "iterations": settings.iterations,
            "games": total_games,
            "decisions": total_decisions,
            **total_outcomes,
            "last_update": last_metrics,
            "device": (
                torch.cuda.get_device_name(selected_device)
                if selected_device.type == "cuda"
                else "cpu"
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    write_json(record_output, record)
    print_event("learn_complete", name=name, games=total_games, decisions=total_decisions)
    return record


def parse_args() -> argparse.Namespace:
    """Parse the lightweight learning command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("initial", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    """Launch self-play reinforcement learning."""
    args = parse_args()
    try:
        config = (
            LearningConfig(**json.loads(args.config.read_text(encoding="utf-8")))
            if args.config
            else LearningConfig()
        )
        data = load_learning_data(args.data, args.initial)
        learn(args.name, args.initial, data, config, args.resume, args.device)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
