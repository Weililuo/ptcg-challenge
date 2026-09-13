import argparse
import multiprocessing as mp
import random
from pathlib import Path
import pandas as pd
from kaggle_environments import make
from kaggle_environments.envs.cabt import cabt as cabt_environment

from ai import make_ai_agent

BASE_DIR = Path(__file__).resolve().parent
DECK_PATH = BASE_DIR / "deck.csv"
OWEN_DECKS_DIR = BASE_DIR / "ptcg-ai" / "decks"


def load_deck(path: Path) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def run_single_match(match_id: int, my_deck: list[int], opp_deck: list[int]):
    agent_0 = make_ai_agent(my_deck, player_index=0)
    agent_1 = make_ai_agent(opp_deck, player_index=1)

    env = make("cabt", configuration={"actTimeout": 1, "runTimeout": 2000}, debug=False)

    orig_battle_select = cabt_environment.battle_select

    def tracked_select(action):
        obs = orig_battle_select(action)
        agent_0.observe(obs)
        agent_1.observe(obs)
        return obs

    cabt_environment.battle_select = tracked_select
    try:
        env.run([agent_0, agent_1])
        won = 1 if env.state[0].reward == 1 else 0
        raw_score, final_score = agent_0.finish(won == 1)
    finally:
        cabt_environment.battle_select = orig_battle_select

    return {
        "match_id": match_id,
        "won": won,
        "raw_score": raw_score,
        "final_score": final_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    my_deck = load_deck(DECK_PATH)

    opp_deck_pool = [my_deck]
    if OWEN_DECKS_DIR.exists():
        for csv_file in OWEN_DECKS_DIR.glob("*.csv"):
            try:
                opp_deck_pool.append(load_deck(csv_file))
            except Exception:
                pass

    print(f"Loaded {len(opp_deck_pool)} deck variants for evaluation/self-play.")
    print(f"Launching {args.matches} matches on {args.workers} CPU cores...")

    tasks = []
    for i in range(args.matches):
        chosen_opp = random.choice(opp_deck_pool)
        tasks.append((i + 1, my_deck, chosen_opp))

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        results = pool.starmap(run_single_match, tasks)

    df = pd.DataFrame(results)
    win_rate = df["won"].mean() * 100.0
    avg_final = df["final_score"].mean()

    print("=" * 40)
    print(f"Completed {len(df)} matches.")
    print(f"Overall Win Rate: {win_rate:.2f}%")
    print(f"Average Final Reward: {avg_final:.4f}")
    print("=" * 40)

    output_csv = BASE_DIR / "results.csv"
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()