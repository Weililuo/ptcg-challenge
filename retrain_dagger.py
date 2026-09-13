from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

import dagger_common as dc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm-started incremental XGBoost fine-tuning")
    parser.add_argument("--round", type=int, required=True, help="DAgger iteration index (for logs)")
    parser.add_argument("--new-rows", type=Path, required=True, help="rollout_rows.csv from rollout_worker")
    parser.add_argument("--history", type=Path, default=dc.HISTORY_CSV, help="historical train_features.csv")
    parser.add_argument("--history-max-rows", type=int, default=1_000_000, help="read cap on the history CSV (memory)")
    parser.add_argument("--old-model", type=Path, default=dc.DEPLOYED_MODEL)
    parser.add_argument("--old-mapping", type=Path, default=dc.DEPLOYED_MAPPING)
    parser.add_argument("--out-model", type=Path, default=dc.CANDIDATE_MODEL)
    parser.add_argument("--out-mapping", type=Path, default=dc.CANDIDATE_MAPPING)
    parser.add_argument("--lr", type=float, default=0.01, help="fine-tune learning rate (original BC run used 0.08)")
    parser.add_argument("--boost-rounds", type=int, default=50, help="new trees per round (num_class trees each); 50 suits the wider 8-deck data distribution")
    parser.add_argument("--max-depth", type=int, default=7, help="must match the old booster's structural params")
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample", type=float, default=0.85)
    parser.add_argument("--history-frac", type=float, default=0.5, help="target fraction of training rows taken from history; 0 disables mixing")
    parser.add_argument("--history-weight", type=float, default=1.0, help="sample weight of history rows")
    parser.add_argument("--baseline-dir", type=Path, default=dc.SELFPLAY_DIR / "baseline_promoted", help="promoted rounds' rollout CSVs, kept as long-term baseline data")
    parser.add_argument("--min-label-count", type=int, default=2, help="drop new-row labels with fewer samples")
    parser.add_argument("--val-frac", type=float, default=0.2, help="held-out fraction of the NEW rows")
    parser.add_argument("--early-stop", type=int, default=0, help="early_stopping_rounds; 0 disables")
    parser.add_argument("--nthread", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--log-dir", type=Path, default=dc.SELFPLAY_DIR / "retrain")
    return parser.parse_args()


def booster_meta(booster: xgb.Booster) -> tuple[int, int]:
    config = json.loads(booster.save_config())
    learner = config["learner"]
    num_class = int(learner["learner_model_param"]["num_class"])
    num_trees = int(learner["gradient_booster"]["gbtree_model_param"]["num_trees"])
    return num_class, num_trees


def old_booster_feature_count(booster: xgb.Booster) -> int | None:
    feature_names = getattr(booster, "feature_names", None)
    if feature_names:
        return len(feature_names)
    num_features_attr = getattr(booster, "num_features", None)
    if num_features_attr is None:
        return None
    try:
        return int(num_features_attr() if callable(num_features_attr) else num_features_attr)
    except Exception:
        return None


def load_class_alignment(old_model: Path, old_mapping: Path) -> tuple[int, list[int], dict[int, int]]:
    booster = xgb.Booster()
    booster.load_model(str(old_model))
    num_class, num_trees = booster_meta(booster)
    dc.log(f"[retrain] old booster: num_class={num_class} n_trees={num_trees}")

    data = dc.read_json(old_mapping)
    if data is None:
        class_to_option = list(range(num_class))
        dc.log("[retrain] WARNING: no action_label_mapping.json found - assuming identity class->option mapping")
    else:
        class_to_option = [int(v) for v in data.get("class_to_option", [])]
        if len(class_to_option) != num_class:
            raise SystemExit(f"mapping has {len(class_to_option)} classes but the booster has {num_class} - " "re-train the base model or restore a matching mapping file")
    option_to_class = {option: cls for cls, option in enumerate(class_to_option)}
    return num_class, class_to_option, option_to_class


def stratified_subsample(df: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    if n_target >= len(df):
        return df
    rng = np.random.default_rng(seed)
    frac = n_target / len(df)
    pieces = []
    for label, count in df[dc.LABEL_COL].value_counts().items():
        group = df[df[dc.LABEL_COL] == label]
        keep = max(min(int(round(count * frac)), len(group)), 1)
        pieces.append(group.sample(n=keep, random_state=int(rng.integers(0, 2**31 - 1))))
    out = pd.concat(pieces)
    if len(out) > n_target:
        out = out.sample(n=n_target, random_state=seed)
    elif len(out) < n_target:
        top_up = df.drop(out.index).sample(n=n_target - len(out), random_state=seed)
        out = pd.concat([out, top_up])
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    log_path = args.log_dir / f"retrain_round_{int(args.round):03d}.json"
    stats: dict[str, Any] = {"round": args.round, "config": {key: str(value) for key, value in vars(args).items()}}

    num_class, class_to_option, option_to_class = load_class_alignment(args.old_model, args.old_mapping)
    allowed_labels = set(option_to_class)

    if not args.new_rows.exists():
        raise SystemExit(f"no rollout rows at {args.new_rows} - run rollout_worker first")
    new_df = pd.read_csv(args.new_rows)
    stats["new_rows_raw"] = len(new_df)

    new_df = new_df[new_df[dc.LABEL_COL].isin(allowed_labels)].copy()
    stats["new_rows_dropped_unrepresentable"] = stats["new_rows_raw"] - len(new_df)

    counts = new_df[dc.LABEL_COL].value_counts()
    new_df = new_df[new_df[dc.LABEL_COL].isin(counts[counts >= args.min_label_count].index)].copy()
    stats["new_rows_dropped_rare"] = stats["new_rows_raw"] - stats["new_rows_dropped_unrepresentable"] - len(new_df)

    if new_df.empty:
        raise SystemExit("all new rows were filtered out (labels outside the deployed class set) - nothing to learn")
    new_df[dc.LABEL_COL] = new_df[dc.LABEL_COL].map(option_to_class)
    stats["new_rows_used"] = len(new_df)

    history_rows: list[pd.DataFrame] = []
    if args.history_frac > 0:
        if args.history.exists():
            history_df = pd.read_csv(args.history, nrows=args.history_max_rows)
            history_df = history_df[history_df[dc.LABEL_COL].isin(allowed_labels)].copy()
            history_df[dc.LABEL_COL] = history_df[dc.LABEL_COL].map(option_to_class)
            history_df["opp_tier_expert"] = 0.0
            history_df["opp_deck_id"] = float(dc.UNKNOWN_DECK_ID)
            history_rows.append(history_df)

        if args.baseline_dir and args.baseline_dir.is_dir():
            baseline_frames = []
            for baseline_csv in sorted(args.baseline_dir.glob("*.csv")):
                try:
                    frame = pd.read_csv(baseline_csv)
                    frame = frame[frame[dc.LABEL_COL].isin(allowed_labels)].copy()
                    frame[dc.LABEL_COL] = frame[dc.LABEL_COL].map(option_to_class)
                    baseline_frames.append(frame[dc.TRAIN_COLS + [dc.LABEL_COL]])
                except Exception as error:
                    dc.log(f"[retrain] WARNING: skipping baseline {baseline_csv}: {error}")
            if baseline_frames:
                history_rows.append(pd.concat(baseline_frames, ignore_index=True))

        if history_rows:
            history_df = pd.concat(history_rows, ignore_index=True)
            n_target = int(len(new_df) * args.history_frac / max(1.0 - args.history_frac, 1e-9))
            history_df = stratified_subsample(history_df, min(n_target, args.history_max_rows), args.seed)
            history_df[dc.WEIGHT_COL] = args.history_weight
            stats["history_rows_used"] = len(history_df)
            history_rows = [history_df]

    rng = np.random.default_rng(args.seed)
    y_new = new_df[dc.LABEL_COL].to_numpy()
    try:
        train_idx, val_idx = train_test_split(np.arange(len(new_df)), test_size=args.val_frac, random_state=args.seed, stratify=y_new)
    except ValueError:
        train_idx, val_idx = train_test_split(np.arange(len(new_df)), test_size=args.val_frac, random_state=args.seed)
    new_train, new_val = new_df.iloc[train_idx].copy(), new_df.iloc[val_idx].copy()
    stats["train_new"] = len(new_train)
    stats["val_new"] = len(new_val)

    train_parts = [new_train, *history_rows]
    train_df = pd.concat(train_parts, ignore_index=True)

    new_feature_defaults = {"my_bench_lowest_hp": 300.0, "opp_has_crustle": 0.0, "opp_has_dragapult": 0.0, "opp_has_froslass": 0.0}
    for name, default in new_feature_defaults.items():
        if name in train_df.columns:
            train_df[name] = train_df[name].fillna(float(default))
        else:
            train_df[name] = float(default)

    # XGBoost accepts NaN silently, so a feature added to FEATURE_NAMES without a
    # matching entry in new_feature_defaults would train a quietly broken model
    # instead of failing. Fail loudly here instead.
    missing_cols = [name for name in dc.TRAIN_COLS if name not in train_df.columns]
    if missing_cols:
        raise SystemExit(
            f"training data is missing feature column(s): {', '.join(missing_cols)} - "
            "add a default to new_feature_defaults or rebuild the history CSV"
        )
    nan_cols = [name for name in dc.TRAIN_COLS if train_df[name].isna().any()]
    if nan_cols:
        raise SystemExit(
            f"training data has NaN in feature column(s): {', '.join(nan_cols)} - "
            "XGBoost would accept these silently; fix the rollout or add a default"
        )

    def to_dmatrix(df: pd.DataFrame, with_weight: bool) -> xgb.DMatrix:
        x = df[dc.TRAIN_COLS].to_numpy(dtype=np.float32)
        kwargs: dict[str, Any] = {"feature_names": dc.TRAIN_COLS, "label": df[dc.LABEL_COL].to_numpy()}
        if with_weight:
            kwargs["weight"] = df[dc.WEIGHT_COL].to_numpy(dtype=np.float32)
        return xgb.DMatrix(x, **kwargs)

    dtrain = to_dmatrix(train_df, with_weight=True)
    dval = to_dmatrix(new_val, with_weight=False)

    old_booster = xgb.Booster()
    old_booster.load_model(str(args.old_model))
    old_feature_count = old_booster_feature_count(old_booster)
    new_feature_count = len(dc.FEATURE_NAMES)
    warm_start = old_feature_count is not None and old_feature_count == new_feature_count
    params = {
        "objective": "multi:softprob",
        "num_class": num_class,
        "eval_metric": "mlogloss",
        "learning_rate": args.lr,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample,
        "tree_method": "hist",
        "nthread": args.nthread,
        "seed": args.seed,
    }

    if warm_start:
        boost_rounds = args.boost_rounds
        train_kwargs: dict[str, Any] = {"xgb_model": old_booster}
        dc.log(f"[retrain] warm-start boosting: +{boost_rounds} rounds @ lr={args.lr} " f"(train={len(train_df):,} rows, val={len(new_val):,})")
    else:
        boost_rounds = 100
        train_kwargs = {}
        dc.log(
            f"[retrain] feature dimension changed ({old_feature_count} -> {new_feature_count}): "
            f"cold-start training {boost_rounds} rounds @ lr={args.lr} "
            f"(train={len(train_df):,} rows, val={len(new_val):,})"
        )

    evals_result: dict[str, list[float]] = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=boost_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stop or None,
        evals_result=evals_result,
        verbose_eval=5,
        **train_kwargs,
    )

    val_probs = booster.predict(dval)
    val_y = new_val[dc.LABEL_COL].to_numpy()
    top_k = min(3, num_class)
    order = np.argsort(val_probs, axis=1)[:, ::-1]
    topk_hits = (order[:, :top_k] == val_y[:, None]).any(axis=1)
    stats["val_top1_acc"] = round(float((order[:, 0] == val_y).mean()), 6)
    stats["val_top3_acc"] = round(float(topk_hits.mean()), 6)
    stats["train_mlogloss"] = float(evals_result["train"]["mlogloss"][-1])
    stats["val_mlogloss"] = float(evals_result["val"]["mlogloss"][-1])
    stats["old_n_trees"] = booster_meta(old_booster)[1]
    stats["new_n_trees"] = booster_meta(booster)[1]
    stats["best_iteration"] = int(getattr(booster, "best_iteration", -1))
    stats["old_feature_count"] = old_feature_count
    stats["new_feature_count"] = new_feature_count
    stats["warm_start"] = bool(warm_start)
    stats["boost_rounds_used"] = boost_rounds

    booster.save_model(str(args.out_model))
    dc.write_json(args.out_mapping, {"class_to_option": [int(v) for v in class_to_option]})

    stats["duration_s"] = round(time.perf_counter() - started, 1)
    dc.write_json(log_path, stats)
    dc.log(
        f"[retrain] candidate saved -> {args.out_model} ({stats['new_n_trees']} trees, "
        f"val top1={stats['val_top1_acc']:.4f} top3={stats['val_top3_acc']:.4f} "
        f"mlogloss={stats['val_mlogloss']:.4f})"
    )
    dc.log(f"[retrain] metrics -> {log_path}")


if __name__ == "__main__":
    main()
