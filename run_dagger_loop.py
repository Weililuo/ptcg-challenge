from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import dagger_common as dc

REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative DAgger Self-Play Training Loop")
    parser.add_argument("--rounds", type=int, default=10, help="number of DAgger iterations to run")
    parser.add_argument("--matches", type=int, default=200, help="number of self-play games per round")
    parser.add_argument("--workers", type=int, default=6, help="workers for rollout self-play")
    parser.add_argument("--arena-games", type=int, default=80, help="games for the candidate-vs-expert + candidate-vs-champion arena")
    parser.add_argument("--arena-workers", type=int, default=6, help="workers for arena matches")
    parser.add_argument("--boost-rounds", type=int, default=50, help="boost rounds per retrain step")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for booster updates")
    parser.add_argument("--threshold", type=float, default=0.55, help="arena promotion overall win-rate threshold")
    parser.add_argument("--min-expert-winrate", type=float, default=0.30, help="minimum AGGREGATE win rate across the whole expert deck pool")
    parser.add_argument("--min-champion-winrate", type=float, default=0.50, help="minimum mirror win rate vs the deployed champion (regression guard)")
    parser.add_argument("--expert-frac", type=float, default=dc.DEFAULT_EXPERT_FRAC, help="rollout/arena share played against the official expert deck pool")
    parser.add_argument("--nemesis-boost", type=float, default=dc.DEFAULT_NEMESIS_BOOST, help="per-deck sampling boost for the nemesis decks (1.0 = uniform)")
    parser.add_argument("--owen-root", type=Path, default=dc.OWEN_ROOT, help="ptcg_owen checkout holding decks/ and models/experts/")
    parser.add_argument("--expert-models-dir", type=Path, default=None, help="expert checkpoints dir; defaults to <owen-root>/models/experts")
    parser.add_argument("--focus-nemesis", type=lambda v: v.lower() not in ("0", "false", "no"), default=False, help="DEBUG/ablation: restrict the expert pool to crustle/dragapult/froslass")
    parser.add_argument("--mirror-frac", type=float, default=dc.DEFAULT_MIRROR_FRAC, help="self-mirror share of rollout matches; defaults to 1 - expert-frac")
    parser.add_argument("--start-round", type=int, default=None, help="override the resume point (defaults to the last round in loop_log.jsonl)")
    parser.add_argument("--promote", action="store_true", help="swap deployed model on promotion")
    parser.add_argument("--viz", action="store_true", help="run visualization pipeline after each round")
    parser.add_argument("--seed", type=int, default=20260830, help="base random seed")
    return parser.parse_args()


def run_stage(cmd: list[str], stage_name: str) -> None:
    dc.log(f"[loop] stage: {stage_name} | executing: {' '.join(cmd)}")
    started = time.perf_counter()
    res = subprocess.run(cmd, cwd=REPO_ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"stage '{stage_name}' failed with code {res.returncode}")
    dc.log(f"[loop] stage '{stage_name}' finished in {time.perf_counter() - started:.1f}s")


def get_latest_round(loop_log: Path) -> int:
    if not loop_log.exists():
        return 0
    latest = 0
    with open(loop_log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                latest = max(latest, int(entry.get("round", 0)))
            except Exception:
                pass
    return latest


def read_round_outcome(round_no: int) -> dict[str, Any]:
    """Summarize one arena decision for the loop log.

    Values are coerced to plain Python types because pandas hands back numpy
    scalars, which json.dumps cannot serialize.
    """
    record: dict[str, Any] = {
        "round": int(round_no),
        "decision": None,
        "promoted": 0,
        "win_rate": None,
        "expert_win_rate": None,
        "champion_win_rate": None,
    }
    rounds_csv = dc.SELFPLAY_DIR / "arena" / "arena_rounds.csv"
    try:
        df = pd.read_csv(rounds_csv)
    except Exception as error:
        dc.log(f"[loop] WARNING: could not read {rounds_csv} ({error})")
        return record
    if df.empty or "round" not in df.columns:
        return record
    rows = df[df["round"] == round_no]
    if rows.empty:
        return record

    last = rows.iloc[-1]
    for key in ("decision", "note"):
        if key in df.columns and pd.notna(last[key]):
            record[key] = str(last[key])
    for key in ("win_rate", "expert_win_rate", "champion_win_rate"):
        if key in df.columns and pd.notna(last[key]):
            record[key] = float(last[key])
    if "promoted" in df.columns and pd.notna(last["promoted"]):
        record["promoted"] = int(last["promoted"])
    return record


def main() -> None:
    args = parse_args()
    loop_log = dc.SELFPLAY_DIR / "loop_log.jsonl"
    dc.SELFPLAY_DIR.mkdir(parents=True, exist_ok=True)

    if args.start_round is not None:
        start_round = args.start_round
    else:
        start_round = get_latest_round(loop_log) + 1
    end_round = start_round + args.rounds - 1

    dc.log(f"[loop] starting loop: rounds {start_round} -> {end_round} (total: {args.rounds})")
    dc.log(f"[loop] settings: matches={args.matches}, workers={args.workers}, arena_games={args.arena_games}, threshold={args.threshold}, min_expert_wr={args.min_expert_winrate}, min_champion_wr={args.min_champion_winrate}, expert_frac={args.expert_frac}")

    for current_round in range(start_round, end_round + 1):
        round_seed = args.seed + current_round * 1000
        dc.log(f"\n[loop] ==================== ROUND {current_round} / {end_round} ====================")

        # 1. Rollout Stage
        rollout_cmd = [
            sys.executable,
            str(REPO_ROOT / "rollout_worker.py"),
            "--round",
            str(current_round),
            "--matches",
            str(args.matches),
            "--workers",
            str(args.workers),
            "--seed",
            str(round_seed),
            "--expert-frac",
            str(args.expert_frac),
            "--mirror-frac",
            str(args.mirror_frac),
            "--nemesis-boost",
            str(args.nemesis_boost),
            "--owen-root",
            str(args.owen_root),
        ]
        if args.expert_models_dir:
            rollout_cmd += ["--expert-models-dir", str(args.expert_models_dir)]
        if args.focus_nemesis:
            rollout_cmd.append("--focus-nemesis")
        run_stage(rollout_cmd, "rollout")

        # 2. Retrain Stage
        round_dir = dc.SELFPLAY_DIR / f"round_{current_round:03d}"
        new_rows_csv = round_dir / "rollout_rows.csv"

        retrain_cmd = [
            sys.executable,
            str(REPO_ROOT / "retrain_dagger.py"),
            "--round",
            str(current_round),
            "--new-rows",
            str(new_rows_csv),
            "--boost-rounds",
            str(args.boost_rounds),
            "--lr",
            str(args.lr),
        ]
        run_stage(retrain_cmd, "retrain")

        # 3. Arena Stage (with dual gate: overall + expert min win rate)
        arena_cmd = [
            sys.executable,
            str(REPO_ROOT / "arena_judge.py"),
            "--round",
            str(current_round),
            "--games",
            str(args.arena_games),
            "--workers",
            str(args.arena_workers),
            "--threshold",
            str(args.threshold),
            "--min-expert-winrate",
            str(args.min_expert_winrate),
            "--min-champion-winrate",
            str(args.min_champion_winrate),
            "--expert-frac",
            str(args.expert_frac),
            "--owen-root",
            str(args.owen_root),
            "--seed",
            str(round_seed),
        ]
        if args.expert_models_dir:
            arena_cmd += ["--expert-models-dir", str(args.expert_models_dir)]
        if args.focus_nemesis:
            arena_cmd.append("--focus-nemesis")
        if args.promote:
            arena_cmd.append("--promote")
        run_stage(arena_cmd, "arena")

        # Record the round so the next invocation resumes here instead of
        # restarting at round 1 and overwriting earlier rounds.
        dc.append_jsonl(loop_log, {**read_round_outcome(current_round), "timestamp": datetime.now(timezone.utc).isoformat()})
        dc.log(f"[loop] round {current_round} logged -> {loop_log}")

        # 4. Visualize Stage
        if args.viz:
            viz_cmd = [
                sys.executable,
                str(REPO_ROOT / "visualize_pipeline.py"),
                "--out-dir",
                str(dc.SELFPLAY_DIR),
                "--plots-dir",
                str(dc.SELFPLAY_DIR / "plots"),
            ]
            run_stage(viz_cmd, "visualize")

    dc.log(f"[loop] all {args.rounds} rounds completed successfully.")


if __name__ == "__main__":
    main()
