from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import dagger_common as dc

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES_EXPERT = "#2a78d6"
SERIES_GENERAL = "#eb6834"
SERIES_OVERALL = "#1baf7a"
SERIES_WON = "#2a78d6"
SERIES_LOST = "#eb6834"
THRESHOLD_INK = "#0d366b"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 13,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "text.color": INK_PRIMARY,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 8.0,
            "axes.axisbelow": True,
        }
    )
    sns.set_theme(style="whitegrid", rc={"grid.color": GRID})


def load_rollout_logs(out_dir: Path) -> pd.DataFrame | None:
    frames = []
    for csv_path in sorted(out_dir.glob("round_*/match_results.csv")):
        try:
            frame = pd.read_csv(csv_path)
            if not frame.empty:
                frames.append(frame)
        except Exception as error:
            warnings.warn(f"skipping unreadable {csv_path}: {error}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["round"] = df["round"].astype(int)
    return df


def load_arena_rounds(arena_dir: Path) -> pd.DataFrame | None:
    csv_path = arena_dir / "arena_rounds.csv"
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as error:
        warnings.warn(f"skipping unreadable {csv_path}: {error}")
        return None


def _placeholder(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", color=INK_MUTED, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    dc.log(f"[viz] saved -> {path}")


def plot_win_rate_trend(df: pd.DataFrame, arena: pd.DataFrame | None, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    rounds = sorted(df["round"].unique())
    overall = df.groupby("round")["won"].mean().reindex(rounds) * 100.0

    ax.plot(rounds, overall, color=SERIES_OVERALL, marker="o", label="overall")
    for tier, color, label in (
        (dc.TIER_EXPERT, SERIES_EXPERT, "vs expert pool"),
        (dc.TIER_GENERAL, SERIES_GENERAL, "vs general pool"),
    ):
        tier_df = df[df["opp_tier"] == tier]
        if tier_df.empty:
            continue
        by_round = tier_df.groupby("round")["won"].mean().reindex(rounds) * 100.0
        ax.plot(by_round.index, by_round.values, color=color, marker="o", label=label)

    if arena is not None and not arena.empty:
        threshold = arena["threshold"].dropna()
        if not threshold.empty:
            ax.axhline(threshold.iloc[-1] * 100.0, color=THRESHOLD_INK, linestyle="--", linewidth=1.5, label=f"promotion gate ({threshold.iloc[-1]:.0%})")
        promoted_rounds = arena.loc[arena["promoted"] == 1, "round"]
        if not promoted_rounds.empty:
            xs = promoted_rounds[promoted_rounds.isin(overall.index)]
            ax.scatter(xs, [overall.loc[r] for r in xs], marker="*", s=220, color=THRESHOLD_INK, zorder=5, label="promoted candidate")

    ax.set_xlabel("DAgger round")
    ax.set_ylabel("win rate (%)")
    ax.set_title("Self-play win rate trend")
    ax.set_ylim(0, 100)
    ax.margins(x=0.05)
    if len(ax.get_legend_handles_labels()[0]) > 1:
        ax.legend(loc="best")
    _save(fig, path)


def _plot_deck_radar(ax: plt.Axes, df: pd.DataFrame) -> None:
    deck_stats = df.groupby("opp_deck_name")["won"].agg(["mean", "size"])
    decks = sorted(deck_stats.index, key=lambda name: -deck_stats.loc[name, "mean"])
    if not decks:
        _placeholder(ax, "no deck-level data")
        return

    values = [deck_stats.loc[name, "mean"] * 100.0 for name in decks]
    angles = np.linspace(0.0, 2.0 * np.pi, len(decks), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    values_closed = values + values[:1]

    ax.plot(angles, values_closed, color=SERIES_EXPERT, linewidth=2.0)
    ax.fill(angles, values_closed, color=SERIES_EXPERT, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{name}\n({int(deck_stats.loc[name, 'size'])} games)" for name in decks], fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("Win rate vs each opponent deck", pad=20)
    for angle, value in zip(angles[:-1], values):
        ax.text(angle, value + 8, f"{value:.0f}%", ha="center", va="center", fontsize=8.5, color=INK_PRIMARY, fontweight="bold")


def _plot_tier_bars(ax: plt.Axes, df: pd.DataFrame) -> None:
    order = [dc.TIER_EXPERT, dc.TIER_GENERAL]
    present = [tier for tier in order if (df["opp_tier"] == tier).any()]
    if not present:
        _placeholder(ax, "no tier-level data")
        return

    colors = {"expert": SERIES_EXPERT, "general": SERIES_GENERAL, "overall": SERIES_OVERALL}
    labels = []
    values = []
    bar_colors = []
    for tier in present:
        labels.append(f"vs {tier}")
        values.append(float(df.loc[df["opp_tier"] == tier, "won"].mean() * 100.0))
        bar_colors.append(colors[tier])
    if len(present) > 1:
        labels.append("overall")
        values.append(float(df["won"].mean() * 100.0))
        bar_colors.append(colors["overall"])

    bars = ax.bar(labels, values, color=bar_colors, width=0.62)
    ax.set_ylim(0, 100)
    ax.set_ylabel("win rate (%)")
    ax.set_title("Aggregate win rate by opponent tier")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2.5, f"{value:.1f}%", ha="center", va="bottom", fontsize=10, color=INK_PRIMARY, fontweight="bold")
    ax.grid(axis="x", visible=False)


def plot_deck_performance(df: pd.DataFrame, path: Path) -> None:
    fig = plt.figure(figsize=(13, 5.6))
    ax_radar = fig.add_subplot(1, 2, 1, projection="polar", facecolor=SURFACE)
    ax_bars = fig.add_subplot(1, 2, 2)
    _plot_deck_radar(ax_radar, df)
    _plot_tier_bars(ax_bars, df)
    fig.suptitle("Opponent pool breakdown", fontsize=14, y=1.02)
    _save(fig, path)


def plot_final_score_hist(df: pd.DataFrame, path: Path) -> None:
    fig, (ax_tier, ax_outcome) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    scores = df.dropna(subset=["final_score"])
    if scores.empty:
        _placeholder(ax_tier, "no final scores recorded")
        _placeholder(ax_outcome, "no final scores recorded")
        _save(fig, path)
        return

    sns.histplot(scores, x="final_score", hue="opp_tier", bins=36, alpha=0.45, palette={"expert": SERIES_EXPERT, "general": SERIES_GENERAL}, ax=ax_tier, legend=len(scores["opp_tier"].unique()) > 1)
    for tier, color in ((dc.TIER_EXPERT, SERIES_EXPERT), (dc.TIER_GENERAL, SERIES_GENERAL)):
        tier_scores = scores.loc[scores["opp_tier"] == tier, "final_score"]
        if len(tier_scores) > 3:
            sns.kdeplot(tier_scores, color=color, linewidth=2.2, ax=ax_tier, legend=False)
    ax_tier.set_xlabel("final score (shaped reward)")
    ax_tier.set_ylabel("matches")
    ax_tier.set_title("Final score by opponent tier")
    ax_tier.grid(axis="x", visible=False)

    sns.histplot(scores, x="final_score", hue="won", bins=36, alpha=0.45, palette={1: SERIES_WON, 0: SERIES_LOST}, ax=ax_outcome, legend=True)
    ax_outcome.set_xlabel("final score (shaped reward)")
    ax_outcome.set_ylabel("matches")
    ax_outcome.set_title("Final score by match outcome")
    ax_outcome.grid(axis="x", visible=False)

    handles, _ = ax_outcome.get_legend_handles_labels()
    if handles:
        ax_outcome.legend(handles, ["won", "lost"], title=None)

    fig.suptitle("Reward shaping distribution", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play pipeline monitoring dashboard")
    parser.add_argument("--out-dir", type=Path, default=dc.SELFPLAY_DIR, help="root containing round_*/match_results.csv")
    parser.add_argument("--arena-dir", type=Path, default=dc.SELFPLAY_DIR / "arena", help="dir with arena_rounds.csv")
    parser.add_argument("--plots-dir", type=Path, default=dc.SELFPLAY_DIR / "plots", help="PNG output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()

    df = load_rollout_logs(args.out_dir)
    arena = load_arena_rounds(args.arena_dir)

    if df is None:
        dc.log(f"[viz] no match_results.csv files under {args.out_dir} - run rollout_worker first")
        return

    dc.log(f"[viz] {len(df):,} matches across {df['round'].nunique()} round(s); " f"arena rounds: {0 if arena is None else len(arena)}")
    plot_win_rate_trend(df, arena, args.plots_dir / "win_rate_trend.png")
    plot_deck_performance(df, args.plots_dir / "deck_performance.png")
    plot_final_score_hist(df, args.plots_dir / "final_score_hist.png")


if __name__ == "__main__":
    main()
