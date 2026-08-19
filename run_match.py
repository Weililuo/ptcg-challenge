from dbm import error
import random
from pathlib import Path
from typing import Any, Callable, Mapping

from kaggle_environments import make
from kaggle_environments.envs.cabt import cabt as cabt_environment

from ai import AIAgent, make_ai_agent


BASE_DIR = Path(__file__).resolve().parent
DECK_PATH = BASE_DIR / "deck.csv"
RESULT_PATH = BASE_DIR / "result.html"


def load_deck(path: Path) -> list[int]:
    with path.open(encoding="utf-8") as file:
        return [int(line) for line in file if line.strip()]


def make_observed_random_agent(
    deck: list[int],
    observer: Callable[[Mapping[str, Any]], None],
) -> Callable[[Mapping[str, Any]], list[int]]:
    def agent(
    observation: Mapping[str, Any],
    _configuration: Any = None,
    ) -> list[int]:
        observer(observation)
        if observation.get("current") is None:
            return deck.copy()

        select = observation["select"]
        options = select.get("option") or []
        count = min(int(select.get("maxCount", 0)), len(options))
        if count < int(select.get("minCount", 0)):
            raise RuntimeError("CABT returned fewer options than the required count.")
        return random.sample(range(len(options)), count)

    return agent


def run_match(ai_agent: AIAgent, deck: list[int]) -> Any:
    opponent = make_observed_random_agent(deck, ai_agent.observe)
    env = make(
        "cabt",
        configuration={"actTimeout": 1, "runTimeout": 2000},
        debug=True,
    )

    original_battle_select = cabt_environment.battle_select

    def observed_battle_select(action: list[int]) -> dict[str, Any]:
        observation = original_battle_select(action)
        ai_agent.observe(observation)
        return observation

    cabt_environment.battle_select = observed_battle_select

    try:
        env.run([ai_agent, opponent])
        print(
            "match:",
            [(state.status, state.reward) for state in env.state],
        )

        error = env.steps[0][0].get("error")
        if error:
            print("error:", error)
    finally:
        cabt_environment.battle_select = original_battle_select

    return env


def main() -> None:
    deck = load_deck(DECK_PATH)
    ai_agent = make_ai_agent(deck, player_index=0)
    env = run_match(ai_agent, deck)

    result_html = env.render(mode="html") or ""
    RESULT_PATH.write_text(result_html, encoding="utf-8")

    raw_score, final_score = ai_agent.finish(env.state[0].reward == 1)
    
    feedback = ai_agent.score.feedback()
    formatted = {
        key: f"{value:.6f}" if isinstance(value, float) else value
        for key, value in feedback.items()
    }
    print("score:", formatted)

    print(
        f"Simulation finished: raw_score={raw_score:.6f}, "
        f"final_score={final_score:.6f}"
    )


if __name__ == "__main__":
    main()
