from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parent
RESULT_PATH = ROOT / "result.html"
sys.path.insert(0, str(ROOT / "src"))

from base_data import collate_decisions, encode_decision, file_sha256
from common import EntityPointerPolicy, load_model


Agent = Callable[[dict[str, Any], Any], list[int]]
_MAKE: Callable[..., Any] | None = None


def repository_path(path: Path) -> Path:
    """Resolve a path relative to the repository."""
    return path if path.is_absolute() else ROOT / path


def model_path(path: Path) -> Path:
    """Resolve a model directory or model.pt path."""
    path = repository_path(path)
    return path / "model.pt" if path.is_dir() else path


def load_deck(path: Path) -> list[int]:
    """Load one 60-card deck CSV."""
    deck = [
        int(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        raise ValueError(f"deck must contain 60 card IDs: {path}")
    return deck


def resolve_device(value: str) -> torch.device:
    """Resolve the inference device."""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return device


def make_agent(
    model: EntityPointerPolicy,
    deck: list[int],
    device: torch.device,
) -> Agent:
    """Create one CABT agent from a model and deck."""
    def agent(
        observation: dict[str, Any],
        _configuration: Any = None,
    ) -> list[int]:
        """Return the deck or one model-selected action."""
        if observation.get("current") is None:
            return deck.copy()
        decision = encode_decision(observation, deck)
        batch = collate_decisions([decision]).to(device)
        return model.decode(batch)[0]

    return agent


def create_environment():
    """Create CABT without unrelated OpenSpiel import output."""
    global _MAKE
    if _MAKE is None:
        sys.stdout.flush()
        sys.stderr.flush()
        null = os.open(os.devnull, os.O_WRONLY)
        stdout = os.dup(1)
        stderr = os.dup(2)
        try:
            os.dup2(null, 1)
            os.dup2(null, 2)
            from kaggle_environments import make
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout, 1)
            os.dup2(stderr, 2)
            os.close(null)
            os.close(stdout)
            os.close(stderr)
        _MAKE = make
    return _MAKE(
        "cabt",
        configuration={"actTimeout": 30, "runTimeout": 2000},
        debug=False,
    )


def load_players(
    first_model: Path,
    first_deck: Path,
    second_model: Path,
    second_deck: Path,
    device: str,
) -> tuple[Path, Path, Path, Path, Agent, Agent]:
    """Load both models, decks, and agents once."""
    selected_device = resolve_device(device)
    torch.set_float32_matmul_precision("high")
    first_path = model_path(first_model)
    second_path = model_path(second_model)
    first_deck_path = repository_path(first_deck)
    second_deck_path = repository_path(second_deck)
    first_policy = load_model(first_path, selected_device)
    second_policy = (
        first_policy
        if second_path.resolve() == first_path.resolve()
        else load_model(second_path, selected_device)
    )
    return (
        first_path,
        second_path,
        first_deck_path,
        second_deck_path,
        make_agent(first_policy, load_deck(first_deck_path), selected_device),
        make_agent(second_policy, load_deck(second_deck_path), selected_device),
    )


def run_match(first_agent: Agent, second_agent: Agent, reverse: bool = False):
    """Run one match and return CABT plus player1's reward."""
    agents = [second_agent, first_agent] if reverse else [first_agent, second_agent]
    environment = create_environment()
    environment.run(agents)
    return environment, environment.state[int(reverse)].reward


def play(
    first_model: Path,
    first_deck: Path,
    second_model: Path,
    second_deck: Path,
    output: Path = RESULT_PATH,
    device: str = "auto",
) -> Path:
    """Run one CABT match and write its HTML replay."""
    *_, first_agent, second_agent = load_players(
        first_model,
        first_deck,
        second_model,
        second_deck,
        device,
    )
    environment, _ = run_match(first_agent, second_agent)
    output.write_text(environment.render(mode="html") or "", encoding="utf-8")
    return output


def file_info(path: Path) -> dict[str, str]:
    """Describe one repository file for a matchup record."""
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(path),
    }


def write_matchup(
    first_model: Path,
    first_deck: Path,
    second_model: Path,
    second_deck: Path,
    result: dict[str, int | float],
) -> Path:
    """Merge one batch result into player1's model record."""
    record_path = first_model.parent / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    opponent = second_model.parent.name if second_model.name == "model.pt" else second_model.stem
    key = f"{first_deck.stem}_vs_{opponent}_{second_deck.stem}"
    record.setdefault("matchups", {})[key] = {
        "player_deck": file_info(first_deck),
        "opponent_model": file_info(second_model),
        "opponent_deck": file_info(second_deck),
        "seat_order": "alternating",
        **result,
    }
    temporary = record_path.with_name(f"{record_path.name}.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(record_path)
    return record_path


def simulate(
    first_model: Path,
    first_deck: Path,
    second_model: Path,
    second_deck: Path,
    games: int,
    device: str = "auto",
) -> Path:
    """Run alternating batch matches and record player1's results."""
    (
        first_path,
        second_path,
        first_deck_path,
        second_deck_path,
        first_agent,
        second_agent,
    ) = load_players(first_model, first_deck, second_model, second_deck, device)
    counts = {"wins": 0, "losses": 0, "draws": 0}
    outcomes = {1: "wins", -1: "losses", 0: "draws"}
    for game in range(games):
        _, reward = run_match(first_agent, second_agent, bool(game % 2))
        outcome = outcomes.get(reward)
        if outcome is None:
            raise RuntimeError(f"CABT returned invalid reward: {reward}")
        counts[outcome] += 1
    result = {
        "games": games,
        **counts,
        "win_rate": counts["wins"] / games,
    }
    return write_matchup(
        first_path,
        first_deck_path,
        second_path,
        second_deck_path,
        result,
    )


def parse_args() -> argparse.Namespace:
    """Parse the two players and simulation mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("first_model", type=Path)
    parser.add_argument("first_deck", type=Path)
    parser.add_argument("second_model", type=Path)
    parser.add_argument("second_deck", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--large", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run one replay or a recorded batch simulation."""
    args = parse_args()
    try:
        if args.batch or args.large:
            output = simulate(
                args.first_model,
                args.first_deck,
                args.second_model,
                args.second_deck,
                1000 if args.large else 100,
                args.device,
            )
        else:
            output = play(
                args.first_model,
                args.first_deck,
                args.second_model,
                args.second_deck,
                device=args.device,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(output)


if __name__ == "__main__":
    main()
