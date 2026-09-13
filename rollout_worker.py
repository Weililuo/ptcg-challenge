from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from kaggle_environments import make
from kaggle_environments.envs.cabt import cabt as cabt_environment

import dagger_common as dc

_AI: Any = None

MAIN_SELECT_TYPE = 0


def _worker_init(model_path: str, mapping_path: str) -> None:
    import os

    os.environ[dc.ENV_MODEL] = model_path
    os.environ[dc.ENV_MAPPING] = mapping_path

    global _AI
    import ai

    _AI = ai
    if list(ai.FEATURE_NAMES) != dc.FEATURE_NAMES:
        raise RuntimeError("FEATURE_NAMES drifted between ai.py and dagger_common.py - " "rollout rows would not match the training schema.")


class TrajectoryRecorder:
    def __init__(self, agent: Any, side: int, match_id: int, record: bool = True) -> None:
        self.agent = agent
        self.side = side
        self.match_id = match_id
        # Expert opponents are never mined into our dataset, so their recorder
        # skips feature extraction entirely instead of computing rows to discard.
        self.record = record
        self.records: list[dict[str, Any]] = []

    def observe(self, observation: Mapping[str, Any]) -> None:
        # Torch experts are stateless per observation and have no observe hook.
        observe = getattr(self.agent, "observe", None)
        if observe is not None:
            observe(observation)

    def __call__(self, observation: Mapping[str, Any], _configuration: Any = None) -> list[int]:
        current = observation.get("current")
        if current is None:
            return self.agent(observation, _configuration)

        action = self.agent(observation, _configuration)
        if not self.record:
            return action

        select = observation.get("select")
        if not select or select.get("type") != MAIN_SELECT_TYPE or not action:
            return action

        my_idx = int(current.get("yourIndex", getattr(self.agent, "player_index", self.side)))
        features = _AI._extract_features(current, my_idx, len(select.get("option") or []))
        if features is None:
            return action
        self.records.append(
            {
                "features": [float(v) for v in features],
                "label": int(action[0]),
            }
        )
        return action


def _finish(agent: Any, won: bool) -> tuple[float, float]:
    """Score an agent at match end, tolerating agents with no scoring state.

    Both sides are scored unconditionally even when only one trajectory is
    recorded, and torch experts expose no `finish`.
    """
    finish = getattr(agent, "finish", None)
    if finish is None:
        return (0.0, 0.0)
    return finish(won)


def _match_had_errors(env: Any) -> str | None:
    try:
        for step_pair in env.steps:
            for entry in step_pair or []:
                error = (entry or {}).get("error")
                if error:
                    return str(error)
    except Exception as error:
        return f"error scan failed: {error}"
    return None


def run_single_match(task: tuple) -> dict[str, Any]:
    (
        match_id,
        my_deck,
        opp_deck,
        opp_tier,
        opp_deck_name,
        opp_deck_id,
        opp_model_path,
        opp_model_version,
        seed,
        record_sides,
        weight_cfg,
    ) = task

    started = time.perf_counter()
    result: dict[str, Any] = {
        "round": weight_cfg["round"],
        "match_id": match_id,
        "opp_tier": opp_tier,
        "opp_deck_name": opp_deck_name,
        "opp_deck_id": opp_deck_id,
        "opp_model_version": opp_model_version,
        "is_torch_expert": int(opp_model_path is not None),
        "won": 0,
        "raw_score": None,
        "final_score": None,
        "n_rows": 0,
        "rows": [],
        "error": None,
        "seed": seed,
        "duration_s": None,
    }

    try:
        random.seed(seed)

        record_self = record_sides in ("self", "both")
        record_opp = record_sides == "both"

        player_0 = TrajectoryRecorder(_AI.make_ai_agent(my_deck, player_index=0), side=0, match_id=match_id, record=record_self)
        if opp_model_path:
            # Imported here so mirror-only workers never pay the torch import.
            from expert_agent import ExpertOpponentAgent

            opp_agent = ExpertOpponentAgent(opp_model_path, opp_deck, player_index=1)
        else:
            opp_agent = _AI.make_ai_agent(opp_deck, player_index=1)
        player_1 = TrajectoryRecorder(opp_agent, side=1, match_id=match_id, record=record_opp)

        env = make("cabt", configuration={"actTimeout": 1, "runTimeout": 2000}, debug=False)
        original_battle_select = cabt_environment.battle_select

        def tracked_select(action: list[int]) -> dict[str, Any]:
            observation = original_battle_select(action)
            player_0.observe(observation)
            player_1.observe(observation)
            return observation

        cabt_environment.battle_select = tracked_select
        try:
            env.run([player_0, player_1])
        finally:
            cabt_environment.battle_select = original_battle_select

        error = _match_had_errors(env)
        if error:
            result["error"] = error
            return result

        won_0 = 1 if env.state[0].reward == 1 else 0
        won_1 = 1 if env.state[1].reward == 1 else 0
        raw_0, final_0 = _finish(player_0.agent, won_0 == 1)
        raw_1, final_1 = _finish(player_1.agent, won_1 == 1)
        result["raw_score"], result["final_score"] = raw_0, final_0
        result["won"] = won_0

        winners: list[tuple[TrajectoryRecorder, str, str, int, float]] = []
        if won_0 and record_sides in ("self", "both"):
            winners.append((player_0, opp_tier, opp_deck_name, opp_deck_id, final_0))
        if won_1 and record_sides == "both":
            winners.append((player_1, dc.TIER_GENERAL, dc.SELF_DECK_NAME, dc.UNKNOWN_DECK_ID, final_1))

        rows: list[dict[str, Any]] = []
        for recorder, tier, deck_name, deck_id, final_score in winners:
            weight = dc.compute_sample_weight(
                final_score,
                len(recorder.records),
                mode=weight_cfg["mode"],
                w_min=weight_cfg["w_min"],
                w_max=weight_cfg["w_max"],
                score_floor=weight_cfg["score_floor"],
            )
            if weight is None:
                continue
            for record in recorder.records:
                row = {name: record["features"][i] for i, name in enumerate(dc.FEATURE_NAMES)}
                row["opp_tier_expert"] = 1.0 if tier == dc.TIER_EXPERT else 0.0
                row["opp_deck_id"] = float(deck_id)
                row["opp_model_version"] = float(opp_model_version)
                row["is_torch_expert"] = float(1 if opp_model_path else 0)
                row[dc.LABEL_COL] = record["label"]
                row[dc.WEIGHT_COL] = weight
                row["round"] = weight_cfg["round"]
                row["match_id"] = match_id
                row["side"] = recorder.side
                row["opp_tier"] = tier
                row["opp_deck_name"] = deck_name
                rows.append(row)

        result["rows"] = rows
        result["n_rows"] = len(rows)
        return result
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result
    finally:
        result["duration_s"] = round(time.perf_counter() - started, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play rollout with winner-trajectory tagging")
    parser.add_argument("--round", type=int, required=True, help="DAgger iteration index")
    parser.add_argument("--matches", type=int, default=1000, help="matches to play this round")
    parser.add_argument("--workers", type=int, default=16, help="spawn worker processes")
    parser.add_argument("--chunksize", type=int, default=1, help="tasks per pool chunk")
    parser.add_argument("--seed", type=int, default=20260830, help="base seed; per-match seed = base + match_id")
    parser.add_argument("--model", type=Path, default=dc.DEPLOYED_MODEL, help="deployed model the policy rolls out with")
    parser.add_argument("--mapping", type=Path, default=dc.DEPLOYED_MAPPING, help="class->option mapping of the deployed model")
    parser.add_argument("--owen-root", type=Path, default=dc.OWEN_ROOT, help="ptcg_owen checkout holding decks/ and models/experts/")
    parser.add_argument("--expert-models-dir", type=Path, default=None, help="expert checkpoints dir; defaults to <owen-root>/models/experts")
    parser.add_argument("--expert-decks-dir", type=Path, default=None, help="fallback deck dir for stems with no expert model; defaults to <owen-root>/decks")
    parser.add_argument("--general-decks-dir", type=Path, default=None, help="extra general pool; defaults to mirror deck.csv")
    parser.add_argument("--focus-nemesis", action="store_true", help="DEBUG/ablation: restrict the expert tier to crustle/dragapult/froslass")
    parser.add_argument("--mirror-frac", type=float, default=dc.DEFAULT_MIRROR_FRAC, help="self-mirror share; defaults to 1 - expert-frac")
    parser.add_argument("--expert-frac", type=float, default=dc.DEFAULT_EXPERT_FRAC, help="fraction of matches vs the expert pool")
    parser.add_argument("--nemesis-boost", type=float, default=dc.DEFAULT_NEMESIS_BOOST, help="per-deck sampling boost for the nemesis decks (1.0 = uniform)")
    parser.add_argument("--record-sides", choices=("self", "both"), default="self", help="self: only deck.csv's winning trajectory; both: also mine the opponent's wins (rejected against torch experts)")
    parser.add_argument("--weight-mode", choices=("match_total", "flat"), default="match_total")
    parser.add_argument("--w-min", type=float, default=0.25, help="per-match sample mass lower clip")
    parser.add_argument("--w-max", type=float, default=5.0, help="per-match sample mass upper clip")
    parser.add_argument("--score-floor", type=float, default=0.0, help="drop winning matches whose final_score is below this")
    parser.add_argument("--out-dir", type=Path, default=dc.SELFPLAY_DIR)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit (no games played)")
    return parser.parse_args()


def compute_adaptive_weights(
    pool: list[tuple],
    last_results_csv: Path | None,
    nemesis_boost: float = dc.DEFAULT_NEMESIS_BOOST,
) -> list[float]:
    """Weight decks within one tier by how badly we are losing to them.

    Tier share is expressed as a match COUNT in main(), not here, so this only
    normalizes inside the pool it is given. Grouping stays on `opp_deck_name`
    (the deck stem) rather than the checkpoint version, so a version bump does
    not reset the accumulated win-rate statistics.
    """
    raw_weights: list[float] = []
    stats: dict[str, float] = {}

    if last_results_csv and last_results_csv.exists():
        try:
            df = pd.read_csv(last_results_csv)
            if not df.empty and "opp_deck_name" in df.columns and "won" in df.columns:
                stats = df.groupby("opp_deck_name")["won"].mean().to_dict()
        except Exception:
            stats = {}

    for entry in pool:
        name = entry["name"]
        wr = stats.get(name, 0.40)
        boost = nemesis_boost if name in dc.NEMESIS_DECKS else 1.00
        raw_weights.append(max(0.10, 1.05 - wr) * boost)

    total = sum(raw_weights)
    if total <= 0:
        return [1.0 / len(pool)] * len(pool)
    return [w / total for w in raw_weights]


def main() -> None:
    args = parse_args()
    out_round = dc.round_dir(args.out_dir, args.round)
    out_round.mkdir(parents=True, exist_ok=True)

    my_deck = dc.load_deck(dc.MAIN_DECK_CSV)

    owen_root = args.owen_root
    models_dir = args.expert_models_dir or (owen_root / "models" / "experts")
    decks_dir = args.expert_decks_dir or (owen_root / "decks")

    specs = dc.discover_expert_models(models_dir, owen_root)
    if args.focus_nemesis:
        dropped = sorted(set(specs) - dc.NEMESIS_DECKS)
        for stem in dropped:
            dc.log(f"[rollout] focus-nemesis: excluding expert deck '{stem}' from the training pool")
        specs = {stem: spec for stem, spec in specs.items() if stem in dc.NEMESIS_DECKS}
        if not specs:
            raise SystemExit(f"focus-nemesis: no nemesis experts {sorted(dc.NEMESIS_DECKS)} found under {models_dir}")

    registry = dc.DeckRegistry(args.out_dir / "deck_registry.json")
    registry.seed_experts(specs)

    # Expert tier: a real torch checkpoint per deck. Stems present as decks but
    # missing a checkpoint fall back to our own XGB agent so a partial
    # models/experts/ still trains rather than aborting.
    expert_pool: list[dict[str, Any]] = []
    for stem, spec in sorted(specs.items()):
        expert_pool.append(
            {
                "name": stem,
                "deck": dc.load_deck(spec.deck_path),
                "tier": dc.TIER_EXPERT,
                "deck_id": registry.id_of(f"expert:{stem}"),
                "model_path": str(spec.model_path),
                "model_version": spec.version,
            }
        )
    covered = set(specs)
    for stem, path in dc.discover_deck_pool(decks_dir):
        if stem in covered:
            continue
        if args.focus_nemesis and stem not in dc.NEMESIS_DECKS:
            continue
        dc.log(f"[rollout] WARNING: deck '{stem}' has no expert checkpoint - falling back to the XGB agent")
        expert_pool.append(
            {
                "name": stem,
                "deck": dc.load_deck(path),
                "tier": dc.TIER_EXPERT,
                "deck_id": registry.id_of(f"expert:{stem}"),
                "model_path": None,
                "model_version": 0,
            }
        )

    # Mirror tier: our own deck, piloted by our own policy on both sides.
    mirror_pool = [
        {
            "name": dc.SELF_DECK_NAME,
            "deck": my_deck,
            "tier": dc.TIER_GENERAL,
            "deck_id": registry.id_of(f"general:{dc.SELF_DECK_NAME}"),
            "model_path": None,
            "model_version": 0,
        }
    ]

    n_experts = sum(1 for e in expert_pool if e["model_path"])
    dc.log(
        f"[rollout] round={args.round} matches={args.matches} workers={args.workers} "
        f"expert_decks={len(expert_pool)} (torch={n_experts}) expert_frac={args.expert_frac} "
        f"focus_nemesis={args.focus_nemesis}"
    )
    if args.dry_run:
        return

    if args.record_sides == "both" and n_experts:
        raise SystemExit(
            "--record-sides both would mine the torch experts' own winning trajectories into our "
            "dataset (tagged as our self deck), i.e. clone the expert rather than self-play. "
            "Use --record-sides self when the expert pool is active."
        )

    weight_cfg = {
        "round": args.round,
        "mode": args.weight_mode,
        "w_min": args.w_min,
        "w_max": args.w_max,
        "score_floor": args.score_floor,
    }

    last_results_csv = dc.round_dir(args.out_dir, args.round - 1) / "match_results.csv"

    # Tier split is expressed as match counts; the adaptive deficit weighting
    # only ever reallocates *within* the expert tier, so the two never multiply.
    n_mirror = max(0, min(args.matches, round(args.matches * args.mirror_frac)))
    n_expert = args.matches - n_mirror if expert_pool else 0
    if not expert_pool:
        n_mirror = args.matches

    chosen_opponents: list[dict[str, Any]] = []
    if n_expert:
        expert_weights = compute_adaptive_weights(expert_pool, last_results_csv, args.nemesis_boost)
        chosen_opponents += random.choices(expert_pool, weights=expert_weights, k=n_expert)
    chosen_opponents += random.choices(mirror_pool, k=n_mirror)
    random.shuffle(chosen_opponents)
    dc.log(
        f"[rollout] sampled {n_expert} expert matches across {len(expert_pool)} deck(s) "
        f"and {n_mirror} self-mirror matches"
    )

    tasks = []
    for match_id, entry in enumerate(chosen_opponents, start=1):
        tasks.append(
            (
                match_id,
                my_deck,
                entry["deck"],
                entry["tier"],
                entry["name"],
                entry["deck_id"],
                entry["model_path"],
                entry["model_version"],
                args.seed + match_id,
                args.record_sides,
                weight_cfg,
            )
        )

    dc.log(f"[rollout] dispatching {len(tasks)} tasks to {args.workers} spawn workers...")
    started = time.perf_counter()

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(str(args.model), str(args.mapping)),
    ) as pool_ctx:
        results = pool_ctx.map(run_single_match, tasks, chunksize=args.chunksize)

    rows: list[dict[str, Any]] = []
    outcome_records: list[dict[str, Any]] = []
    errors = 0
    for result in results:
        outcome_records.append({key: value for key, value in result.items() if key != "rows"})
        if result["error"]:
            errors += 1
            continue
        rows.extend(result["rows"])

    outcomes = pd.DataFrame(outcome_records)
    outcomes.to_csv(out_round / "match_results.csv", index=False)

    summary: dict[str, Any] = {
        "round": args.round,
        "config": {key: str(value) for key, value in vars(args).items()},
        "matches_total": len(outcomes),
        "matches_with_error": int(errors),
        "matches_valid": int(len(outcomes) - errors),
        "win_rate": float(outcomes["won"].mean()) if len(outcomes) else None,
        "rows_kept": len(rows),
        "duration_s": round(time.perf_counter() - started, 1),
    }
    if len(outcomes):
        by_tier = outcomes.groupby("opp_tier").agg(matches=("won", "size"), wins=("won", "sum"))
        summary["by_tier"] = {tier: {"matches": int(row.matches), "wins": int(row.wins), "win_rate": round(float(row.wins / max(row.matches, 1)), 4)} for tier, row in by_tier.iterrows()}
        # Win-only recording biases the dataset toward decks we already beat.
        # Per-deck visibility is how a starved matchup gets noticed.
        by_deck = outcomes.groupby("opp_deck_name").agg(matches=("won", "size"), wins=("won", "sum"))
        summary["by_deck"] = {
            str(name): {
                "matches": int(row.matches),
                "wins": int(row.wins),
                "win_rate": round(float(row.wins / max(row.matches, 1)), 4),
            }
            for name, row in by_deck.iterrows()
        }

    if rows:
        rollout = pd.DataFrame(rows)
        column_order = (
            dc.TRAIN_COLS
            + dc.NEW_FEATURE_COLS
            + [dc.LABEL_COL, dc.WEIGHT_COL, "round", "match_id", "side", "opp_tier", "opp_deck_name", "opp_model_version", "is_torch_expert"]
        )
        rollout = rollout[[col for col in column_order if col in rollout.columns]]
        rollout.to_csv(out_round / "rollout_rows.csv", index=False)
        summary["weight_stats"] = {
            "min": float(rollout[dc.WEIGHT_COL].min()),
            "mean": float(rollout[dc.WEIGHT_COL].mean()),
            "max": float(rollout[dc.WEIGHT_COL].max()),
            "unique_labels": int(rollout[dc.LABEL_COL].nunique()),
        }
        rows_by_deck = rollout.groupby("opp_deck_name").size().to_dict()
        summary["rows_kept_by_deck"] = {str(name): int(count) for name, count in rows_by_deck.items()}
    else:
        dc.log("[rollout] WARNING: zero winner rows kept this round - " "retrain stage should be skipped and the seed rotated")

    dc.write_json(out_round / "rollout_summary.json", summary)
    dc.log(f"[rollout] done in {summary['duration_s']}s | valid={summary['matches_valid']} " f"errors={errors} win_rate={summary['win_rate']} rows_kept={summary['rows_kept']}")
    dc.log(f"[rollout] outputs -> {out_round}")


if __name__ == "__main__":
    main()
