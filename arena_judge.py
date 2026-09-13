from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from kaggle_environments import make
from kaggle_environments.envs.cabt import cabt as cabt_environment

import dagger_common as dc

_CHALLENGER: Any = None
_CHAMPION: Any = None

PROMOTE_THRESHOLD = 0.55
DEFAULT_MIN_EXPERT_WINRATE = 0.30
DEFAULT_MIN_CHAMPION_WINRATE = 0.50
DEFAULT_MIN_DECK_GAMES = 10
NEMESIS_DECKS = dc.NEMESIS_DECKS

# Two independent blocks. `expert` measures absolute capability against the
# frozen torch experts; `champion` is the regression guard, because once the
# expert seat is a torch model nothing else compares the candidate to the
# deployed model any more.
BLOCK_EXPERT = "expert"
BLOCK_CHAMPION = "champion"


def _build_opponent(deck: list[int], player_index: int, block: str, opp_model_path: str | None) -> Any:
    """Seat the expert block's torch checkpoint, or the champion for the mirror block."""
    if block == BLOCK_EXPERT and opp_model_path:
        from expert_agent import ExpertOpponentAgent

        return ExpertOpponentAgent(opp_model_path, deck, player_index)
    return _CHAMPION.make_ai_agent(deck, player_index)


def _rotated_path(path: Path, n_columns: int) -> Path:
    """Pick a non-colliding name for a schema-stale archive, e.g. arena_rounds_v16.csv."""
    candidate = path.with_name(f"{path.stem}_v{n_columns}{path.suffix}")
    suffix = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_v{n_columns}_{suffix}{path.suffix}")
        suffix += 1
    return candidate


def _append_round_record(rounds_csv: Path, round_record: dict[str, Any]) -> None:
    """Append one arena round, rotating a schema-stale file aside first.

    `round_record` gained columns after arena_rounds.csv was first written, and
    appending a wider row under the old header corrupts the file: pandas writes
    values positionally, so every later reader binds them to the wrong column
    names. Rotate the stale file rather than rewriting real history, then start
    a clean one.
    """
    rounds_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = list(round_record)
    header = not rounds_csv.exists()
    if not header:
        on_disk = list(pd.read_csv(rounds_csv, nrows=0).columns)
        if on_disk != columns:
            rotated = _rotated_path(rounds_csv, len(on_disk))
            dc.log(
                f"[arena] {rounds_csv.name} has the old {len(on_disk)}-column schema, "
                f"current is {len(columns)}; rotating -> {rotated.name} and starting a fresh file"
            )
            dc.os_replace(rounds_csv, rotated)
            header = True
    pd.DataFrame([round_record], columns=columns).to_csv(rounds_csv, mode="a", header=header, index=False)


def _load_policy_module(module_name: str, model_path: Path, mapping_path: Path) -> Any:
    os.environ[dc.ENV_MODEL] = str(model_path)
    os.environ[dc.ENV_MAPPING] = str(mapping_path)
    spec = importlib.util.spec_from_file_location(module_name, dc.AI_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build import spec for {dc.AI_PY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arena_worker_init(candidate_model: str, candidate_mapping: str, champion_model: str, champion_mapping: str) -> None:
    global _CHALLENGER, _CHAMPION
    _CHALLENGER = _load_policy_module("ai_challenger", Path(candidate_model), Path(candidate_mapping))
    _CHAMPION = _load_policy_module("ai_champion", Path(champion_model), Path(champion_mapping))


def _env_errors(env: Any) -> str | None:
    try:
        for step_pair in env.steps:
            for entry in step_pair or []:
                error = (entry or {}).get("error")
                if error:
                    return str(error)
    except Exception as error:
        return f"error scan failed: {error}"
    return None


def play_game(task: tuple) -> dict[str, Any]:
    game_id, candidate_side, p0_deck, p1_deck, opp_name, block, opp_model_path, seed = task
    started = time.perf_counter()
    record: dict[str, Any] = {
        "game_id": game_id,
        "candidate_side": candidate_side,
        "opp_deck": opp_name,
        "block": block,
        "is_expert": int(block == BLOCK_EXPERT),
        "candidate_won": 0,
        "champion_won": 0,
        "draw": 0,
        "valid": 0,
        "candidate_raw_score": None,
        "candidate_final_score": None,
        "champion_raw_score": None,
        "champion_final_score": None,
        "error": None,
        "duration_s": None,
    }
    try:
        random.seed(seed)
        if candidate_side == 0:
            p0 = _CHALLENGER.make_ai_agent(p0_deck, player_index=0)
            p1 = _build_opponent(p1_deck, 1, block, opp_model_path)
        else:
            p0 = _build_opponent(p0_deck, 0, block, opp_model_path)
            p1 = _CHALLENGER.make_ai_agent(p1_deck, player_index=1)

        env = make("cabt", configuration={"actTimeout": 1, "runTimeout": 2000}, debug=False)
        original_battle_select = cabt_environment.battle_select

        def tracked_select(action: list[int]) -> dict[str, Any]:
            observation = original_battle_select(action)
            p0.observe(observation)
            p1.observe(observation)
            return observation

        cabt_environment.battle_select = tracked_select
        try:
            env.run([p0, p1])
        finally:
            cabt_environment.battle_select = original_battle_select

        error = _env_errors(env)
        if error:
            record["error"] = error
            return record

        won_0 = 1 if env.state[0].reward == 1 else 0
        won_1 = 1 if env.state[1].reward == 1 else 0
        challenger_raw, challenger_final = p0.finish(won_0 == 1) if candidate_side == 0 else p1.finish(won_1 == 1)
        champion_raw, champion_final = p1.finish(won_1 == 1) if candidate_side == 0 else p0.finish(won_0 == 1)
        record["candidate_raw_score"], record["candidate_final_score"] = challenger_raw, challenger_final
        record["champion_raw_score"], record["champion_final_score"] = champion_raw, champion_final

        if won_0 == won_1:
            record["draw"] = 1
        else:
            record["valid"] = 1
            challenger_won = won_0 if candidate_side == 0 else won_1
            record["candidate_won"], record["champion_won"] = challenger_won, 1 - challenger_won
        return record
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    finally:
        record["duration_s"] = round(time.perf_counter() - started, 3)


def load_expert_decks(directory: Path) -> dict[str, list[int]]:
    expert_decks: dict[str, list[int]] = {}
    for name, path in dc.discover_deck_pool(directory):
        try:
            deck = dc.load_deck(path)
        except Exception:
            dc.log(f"[arena] WARNING: skipping unloadable expert deck {path} ({type(expert_decks).__name__})")
            continue
        if len(deck) != 60:
            continue
        expert_decks[name] = deck
    return expert_decks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Candidate vs champion self-play arena with promotion gate")
    parser.add_argument("--round", type=int, required=True, help="DAgger iteration index")
    parser.add_argument("--candidate", type=Path, default=dc.CANDIDATE_MODEL)
    parser.add_argument("--candidate-mapping", type=Path, default=dc.CANDIDATE_MAPPING)
    parser.add_argument("--champion", type=Path, default=dc.DEPLOYED_MODEL)
    parser.add_argument("--champion-mapping", type=Path, default=dc.DEPLOYED_MAPPING)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--owen-root", type=Path, default=dc.OWEN_ROOT, help="ptcg_owen checkout holding decks/ and models/experts/")
    parser.add_argument("--expert-models-dir", type=Path, default=None, help="expert checkpoints dir; defaults to <owen-root>/models/experts")
    parser.add_argument("--expert-decks-dir", type=Path, default=None, help="fallback deck dir; defaults to <owen-root>/decks")
    parser.add_argument("--expert-frac", type=float, default=dc.DEFAULT_EXPERT_FRAC, help="share of arena games spent in the expert (capability) block; the rest is the champion regression block")
    parser.add_argument("--threshold", type=float, default=PROMOTE_THRESHOLD, help="promotion overall win-rate gate")
    parser.add_argument("--min-expert-winrate", type=float, default=DEFAULT_MIN_EXPERT_WINRATE, help="minimum aggregate win-rate against the whole expert deck pool (not per deck)")
    parser.add_argument("--min-champion-winrate", type=float, default=DEFAULT_MIN_CHAMPION_WINRATE, help="minimum mirror win-rate against the deployed champion, i.e. the regression guard")
    parser.add_argument("--min-deck-games", type=int, default=DEFAULT_MIN_DECK_GAMES, help="report-only: below this a per-deck win rate is flagged as low sample")
    parser.add_argument("--focus-nemesis", action="store_true", help="DEBUG/ablation: restrict the expert block to crustle/dragapult/froslass")
    parser.add_argument("--min-valid-frac", type=float, default=0.8, help="round is inconclusive below this fraction of valid games")
    parser.add_argument("--promote", action="store_true", help="actually swap the deployed model on promotion")
    parser.add_argument("--archive-rejected", action="store_true", help="move a rejected candidate into the archive")
    parser.add_argument("--deck", type=Path, default=dc.MAIN_DECK_CSV)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--out-dir", type=Path, default=dc.SELFPLAY_DIR / "arena")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.candidate.exists():
        raise SystemExit(f"candidate model missing: {args.candidate} - run retrain_dagger first")
    if not args.candidate_mapping.exists():
        dc.log(f"[arena] WARNING: candidate mapping missing ({args.candidate_mapping}) - " "ai.py will assume the identity mapping")

    main_deck = dc.load_deck(args.deck)

    owen_root = args.owen_root
    models_dir = args.expert_models_dir or (owen_root / "models" / "experts")
    decks_dir = args.expert_decks_dir or (owen_root / "decks")

    specs = dc.discover_expert_models(models_dir, owen_root)
    if args.focus_nemesis:
        for stem in sorted(set(specs) - NEMESIS_DECKS):
            dc.log(f"[arena] focus-nemesis: excluding expert deck '{stem}'")
        specs = {stem: spec for stem, spec in specs.items() if stem in NEMESIS_DECKS}
        if not specs:
            raise SystemExit(f"focus-nemesis: no nemesis experts {sorted(NEMESIS_DECKS)} found under {models_dir}")

    expert_decks = load_expert_decks(decks_dir)
    for stem in sorted(expert_decks):
        if stem in specs:
            continue
        if args.focus_nemesis and stem not in NEMESIS_DECKS:
            continue
        dc.log(f"[arena] WARNING: deck '{stem}' has no expert checkpoint - the champion will pilot it")

    if not specs and not expert_decks:
        dc.log("[arena] WARNING: no expert decks found; the expert block will fall back to mirror-only")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_expert_games = max(0, min(args.games, round(args.games * args.expert_frac)))
    n_champion_games = args.games - n_expert_games

    expert_order = sorted(set(specs) | {s for s in expert_decks if not (args.focus_nemesis and s not in NEMESIS_DECKS)})
    if not expert_order:
        expert_order = ["mirror"]

    tasks = []
    game_id = 0
    # Block A: candidate vs the frozen torch experts, round-robin so every deck
    # in the pool gets comparable exposure, candidate side alternating.
    for index in range(n_expert_games):
        game_id += 1
        candidate_side = game_id % 2
        opp_name = expert_order[index % len(expert_order)]
        opp_deck = specs[opp_name].deck_path if opp_name in specs else expert_decks.get(opp_name, main_deck)
        if not isinstance(opp_deck, list):
            opp_deck = dc.load_deck(opp_deck)
        opp_model_path = str(specs[opp_name].model_path) if opp_name in specs else None
        if candidate_side == 0:
            p0_deck, p1_deck = main_deck, opp_deck
        else:
            p0_deck, p1_deck = opp_deck, main_deck
        tasks.append((game_id, candidate_side, p0_deck, p1_deck, opp_name, BLOCK_EXPERT, opp_model_path, args.seed + game_id))

    # Block B: candidate vs the deployed champion, both on our own deck. This is
    # the only remaining measurement of "did the new model regress?".
    for _ in range(n_champion_games):
        game_id += 1
        candidate_side = game_id % 2
        tasks.append((game_id, candidate_side, main_deck, main_deck, "champion_mirror", BLOCK_CHAMPION, None, args.seed + game_id))

    dc.log(
        f"[arena] round={args.round} games={args.games} workers={args.workers} " f"threshold={args.threshold} min_expert_wr={args.min_expert_winrate} min_champion_wr={args.min_champion_winrate} " f"expert_block={n_expert_games} champion_block={n_champion_games} torch_experts={len(specs)} focus_nemesis={args.focus_nemesis} promote={'YES' if args.promote else 'dry-run'}"
    )

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_arena_worker_init,
        initargs=(str(args.candidate), str(args.candidate_mapping), str(args.champion), str(args.champion_mapping)),
    ) as pool:
        results = pool.map(play_game, tasks, chunksize=1)

    games_df = pd.DataFrame(results)
    games_df["round"] = args.round
    games_df.to_csv(args.out_dir / f"round_{int(args.round):03d}_games.csv", index=False)

    valid_games = games_df[games_df["valid"] == 1]
    valid = int(valid_games["valid"].sum())
    candidate_wins = int(valid_games["candidate_won"].sum())
    champion_wins = int(valid_games["champion_won"].sum())
    draws = int(games_df["draw"].sum())
    errors = int(games_df["error"].notna().sum())
    win_rate = candidate_wins / valid if valid else None

    expert_df = valid_games[valid_games["block"] == BLOCK_EXPERT]
    valid_expert = len(expert_df)
    expert_wins = int(expert_df["candidate_won"].sum()) if valid_expert else 0
    expert_win_rate = expert_wins / valid_expert if valid_expert else None

    champion_df = valid_games[valid_games["block"] == BLOCK_CHAMPION]
    valid_champion = len(champion_df)
    champion_wins = int(champion_df["candidate_won"].sum()) if valid_champion else 0
    champion_win_rate = champion_wins / valid_champion if valid_champion else None

    # Report-only: no longer a promotion gate now that the pool covers 8 decks
    # (a per-deck floor across every matchup makes promotion near-unreachable).
    per_deck: dict[str, dict[str, int | float]] = {}
    for name, group in expert_df.groupby("opp_deck"):
        per_deck[str(name)] = {
            "games": int(len(group)),
            "wins": int(group["candidate_won"].sum()),
            "win_rate": round(float(group["candidate_won"].mean()), 6),
            "low_sample": int(len(group) < args.min_deck_games),
        }

    min_valid = int(args.games * args.min_valid_frac)
    if valid < min_valid:
        decision = "inconclusive"
        note = f"only {valid} valid games (< {min_valid} required); {errors} errored"
    elif win_rate is None or win_rate < args.threshold:
        decision = "reject"
        note = f"overall win rate {win_rate if win_rate is not None else 0.0:.1%} < {args.threshold:.0%} over {valid} valid games"
    elif expert_win_rate is not None and expert_win_rate < args.min_expert_winrate:
        decision = "reject"
        note = f"overall wr {win_rate:.1%} >= {args.threshold:.0%}, but aggregate expert wr {expert_win_rate:.1%} < {args.min_expert_winrate:.0%}"
    elif champion_win_rate is not None and champion_win_rate < args.min_champion_winrate:
        decision = "reject"
        note = f"overall wr {win_rate:.1%} and expert wr {expert_win_rate:.1%} pass, but the candidate regresses on the mirror: {champion_win_rate:.1%} < {args.min_champion_winrate:.0%} vs the champion"
    else:
        decision = "promote"
        expert_txt = f"{expert_win_rate:.1%}" if expert_win_rate is not None else "n/a"
        champion_txt = f"{champion_win_rate:.1%}" if champion_win_rate is not None else "n/a"
        note = f"overall wr {win_rate:.1%} >= {args.threshold:.0%}; expert wr {expert_txt} >= {args.min_expert_winrate:.0%}; champion mirror wr {champion_txt} >= {args.min_champion_winrate:.0%}"

    if decision == "promote" and args.promote:
        archive_dir = dc.SELFPLAY_DIR / "archive" / f"round_{int(args.round):03d}"
        dc.atomic_copy_replace(args.candidate, args.champion, backup=archive_dir / "baseline" / args.champion.name)
        dc.atomic_copy_replace(args.candidate_mapping, args.champion_mapping, backup=archive_dir / "baseline" / args.champion_mapping.name)
        dc.append_jsonl(
            args.out_dir / "promotions.jsonl",
            {
                "round": args.round,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "win_rate": round(win_rate, 6),
                "expert_win_rate": round(expert_win_rate, 6) if expert_win_rate is not None else None,
                "champion_win_rate": round(champion_win_rate, 6) if champion_win_rate is not None else None,
                "per_deck_win_rates": per_deck,
                "valid_games": valid,
                "candidate": str(args.candidate),
                "archive": str(archive_dir),
            },
        )
        dc.log(f"[arena] PROMOTED: {args.candidate.name} is now the deployed model (backup -> {archive_dir})")
    elif decision == "reject" and args.archive_rejected:
        archive_dir = dc.SELFPLAY_DIR / "archive" / f"round_{int(args.round):03d}" / "rejected"
        dc.os_replace(args.candidate, archive_dir / args.candidate.name)
        dc.os_replace(args.candidate_mapping, archive_dir / args.candidate_mapping.name)
        dc.log(f"[arena] candidate archived (rejected) -> {archive_dir}")

    round_record = {
        "round": args.round,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "games": args.games,
        "valid_games": valid,
        "candidate_wins": candidate_wins,
        "champion_wins": champion_wins,
        "draws": draws,
        "errors": errors,
        "win_rate": round(win_rate, 6) if win_rate is not None else None,
        "expert_win_rate": round(expert_win_rate, 6) if expert_win_rate is not None else None,
        "champion_win_rate": round(champion_win_rate, 6) if champion_win_rate is not None else None,
        "expert_block_games": valid_expert,
        "champion_block_games": valid_champion,
        "per_deck_win_rates": json.dumps(per_deck),
        "threshold": args.threshold,
        "min_expert_winrate": args.min_expert_winrate,
        "min_champion_winrate": args.min_champion_winrate,
        "focus_nemesis": int(args.focus_nemesis),
        "promoted": 1 if (decision == "promote" and args.promote) else 0,
        "decision": decision,
        "note": note,
    }
    _append_round_record(args.out_dir / "arena_rounds.csv", round_record)

    expert_log_str = f", expert_wr={expert_win_rate:.3f}" if expert_win_rate is not None else ""
    champion_log_str = f", champion_mirror_wr={champion_win_rate:.3f}" if champion_win_rate is not None else ""
    deck_log = ", ".join(
        f"{name}={stats['wins']}/{stats['games']}({stats['win_rate']:.0%})"
        for name, stats in sorted(per_deck.items())
    )
    dc.log(f"[arena] decision={decision.upper()} | candidate {candidate_wins}-{champion_wins} " f"(valid={valid}, draws={draws}, errors={errors}, win_rate={win_rate}{expert_log_str}{champion_log_str})")
    dc.log(f"[arena] per-deck: {deck_log if deck_log else 'n/a'}")
    dc.log(f"[arena] results -> {args.out_dir}")


if __name__ == "__main__":
    main()
