from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import orjson
import pandas as pd
from tqdm import tqdm

DATASETS_DIR = Path(r"D:\Hackathon-Black Pearl\PTCG\datasets")
OUTPUT_CSV = Path(r"D:\Hackathon-Black Pearl\PTCG\train_features.csv")


def extract_features_from_observation(
    current: Dict[str, Any],
    select: Dict[str, Any],
    winner_idx: int,
) -> Optional[Dict[str, Any]]:
    """Fast extraction using direct dict lookups."""
    players = current.get("players")
    if not players or len(players) < 2:
        return None

    my_board = players[winner_idx]
    opp_board = players[1 - winner_idx]

    # 1. Hand resources
    hand: List[int] = my_board.get("hand", [])
    hand_grass_cnt = 0
    hand_lightning_cnt = 0
    hand_fight_cnt = 0
    for cid in hand:
        if cid == 1:
            hand_grass_cnt += 1
        elif cid == 4:
            hand_lightning_cnt += 1
        elif cid == 6:
            hand_fight_cnt += 1
    hand_size = len(hand)

    # 2. Active spot
    active_list = my_board.get("active")
    if active_list and active_list[0] is not None:
        active_mon = active_list[0]
        active_cid = int(active_mon.get("cardId", 0))
        active_hp = int(active_mon.get("hp", 0))
        active_max_hp = int(active_mon.get("maxHp", 1))
        active_hp_ratio = active_hp / max(active_max_hp, 1)
        active_energy_cnt = len(active_mon.get("energy", []))
    else:
        active_cid, active_hp_ratio, active_energy_cnt = 0, 0.0, 0

    is_active_raging_bolt = int(active_cid == 63)

    # 3. Bench & Synergy
    bench = [p for p in my_board.get("bench", []) if p is not None]
    has_latias_ex = int(active_cid == 184 or any(p.get("cardId") == 184 for p in bench))
    bench_ogerpon_cnt = sum(1 for p in bench if p.get("cardId") == 96)

    all_my_pokemon = (active_list if active_list else []) + bench
    total_board_energies = sum(len(p.get("energy", [])) for p in all_my_pokemon if p)

    # 4. Prize & Matchup Context
    my_prizes = len(my_board.get("prize", []))
    opp_prizes = len(opp_board.get("prize", []))
    prize_diff = my_prizes - opp_prizes

    opp_active_list = opp_board.get("active")
    opp_active_cid = int(opp_active_list[0].get("cardId", 0)) if opp_active_list and opp_active_list[0] else 0
    opp_active_immune_ex = int(opp_active_cid in (345, 533))

    return {
        "turn": int(current.get("turn", 0)),
        "turn_action_count": int(current.get("turnActionCount", 0)),
        "hand_size": hand_size,
        "hand_grass_cnt": hand_grass_cnt,
        "hand_lightning_cnt": hand_lightning_cnt,
        "hand_fight_cnt": hand_fight_cnt,
        "is_active_raging_bolt": is_active_raging_bolt,
        "active_hp_ratio": round(active_hp_ratio, 3),
        "active_energy_cnt": active_energy_cnt,
        "has_latias_ex": has_latias_ex,
        "bench_ogerpon_cnt": bench_ogerpon_cnt,
        "total_board_energies": total_board_energies,
        "my_prizes_remaining": my_prizes,
        "opp_prizes_remaining": opp_prizes,
        "prize_diff": prize_diff,
        "opp_active_immune_ex": opp_active_immune_ex,
        "options_count": len(select.get("option", [])),
    }


def parse_batch_episodes(file_paths: List[Path]) -> List[Dict[str, Any]]:
    """Processes a chunk of files per worker with fast binary IO and orjson."""
    batch_records: List[Dict[str, Any]] = []

    for file_path in file_paths:
        try:
            with file_path.open("rb") as f:
                data = orjson.loads(f.read())
        except Exception:
            continue

        rewards = data.get("rewards")
        if not rewards or len(rewards) < 2 or rewards[0] is None or rewards[1] is None or rewards[0] == rewards[1]:
            continue

        winner_idx = 0 if rewards[0] > rewards[1] else 1
        steps = data.get("steps", [])

        for step_idx in range(1, len(steps)):
            step_data = steps[step_idx]
            if len(step_data) <= winner_idx:
                continue
            winner_node = step_data[winner_idx]

            observation = winner_node.get("observation", {})
            current = observation.get("current")
            select = observation.get("select")
            action = winner_node.get("action")

            if not current or not select or action is None:
                continue
            if select.get("type") != 0:
                continue

            feat_dict = extract_features_from_observation(current, select, winner_idx)
            if not feat_dict:
                continue

            chosen_idx = action[0] if isinstance(action, list) and len(action) > 0 else action
            if not isinstance(chosen_idx, int):
                continue

            feat_dict["label_action_idx"] = chosen_idx
            batch_records.append(feat_dict)

    return batch_records


def chunkify(lst: List[Any], n_chunks: int) -> List[List[Any]]:
    """Splits a list into n evenly sized sublists."""
    k, m = divmod(len(lst), n_chunks)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


def main() -> None:
    cpu_count = os.cpu_count() or 16
    print(f"Detected {cpu_count} CPU threads.")

    all_files = list(DATASETS_DIR.rglob("*.json"))
    total_files = len(all_files)
    print(f"Discovered {total_files:,} episode replays.")

    chunks = chunkify(all_files, cpu_count * 5)

    all_rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = [executor.submit(parse_batch_episodes, chunk) for chunk in chunks]
        for f in tqdm(futures, desc="Fast parsing"):
            batch_res = f.result()
            if batch_res:
                all_rows.extend(batch_res)

    print(f"\nExtraction finished! Total training samples: {len(all_rows):,}")
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset successfully saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
