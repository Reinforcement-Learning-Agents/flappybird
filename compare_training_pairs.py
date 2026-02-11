#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PAIR_LIST = [
    ("dqn", "nfq"),
    ("dqn", "ddqn"),
]


def load_train_episode_curve(metrics_csv: Path) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: episode, value
    using only rows where type == 'train_episode'.
    """
    df = pd.read_csv(metrics_csv)

    required = {"type", "episode", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"{metrics_csv} missing required columns. Found: {list(df.columns)}")

    tr = df[df["type"] == "train_episode"][["episode", "value"]].copy()
    tr = tr.dropna(subset=["episode", "value"])
    tr["episode"] = tr["episode"].astype(int)
    tr["value"] = tr["value"].astype(float)
    tr = tr.sort_values("episode").reset_index(drop=True)

    if tr.empty:
        raise ValueError(f"No train_episode rows found in {metrics_csv}")

    return tr


def smooth_series(y: pd.Series, window: int) -> pd.Series:
    """
    Rolling mean smoothing (keeps same length).
    """
    if window <= 1:
        return y
    return y.rolling(window=window, min_periods=1).mean()


def collect_algo_runs(results_dir: Path, algo: str, seeds: list[int], smooth_w: int) -> pd.DataFrame:
    """
    Returns DF indexed by episode, columns seed0/seed1/seed2...
    We inner-join on episode so all seeds share the same x-axis.
    """
    per_seed = []
    for seed in seeds:
        run_dir = results_dir / f"{algo}_seed{seed}"
        metrics_csv = run_dir / "metrics.csv"
        if not metrics_csv.exists():
            raise FileNotFoundError(f"Missing {metrics_csv}")

        tr = load_train_episode_curve(metrics_csv)
        col = f"seed{seed}"
        tr[col] = smooth_series(tr["value"], smooth_w)
        per_seed.append(tr[["episode", col]])

    merged = per_seed[0]
    for nxt in per_seed[1:]:
        merged = merged.merge(nxt, on="episode", how="inner")

    return merged.set_index("episode").sort_index()


def stats_from_merged(merged: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=merged.index)
    out["min"] = merged.min(axis=1)
    out["max"] = merged.max(axis=1)
    out["mean"] = merged.mean(axis=1)
    return out


def plot_band(ax, stats: pd.DataFrame, label: str):
    x = stats.index.to_numpy()
    ax.plot(x, stats["mean"].to_numpy(), linewidth=2, label=label)
    ax.fill_between(x, stats["min"].to_numpy(), stats["max"].to_numpy(), alpha=0.25)


def make_pair_plot(stats_a: pd.DataFrame, stats_b: pd.DataFrame, name_a: str, name_b: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_band(ax, stats_a, label=name_a.upper())
    plot_band(ax, stats_b, label=name_b.upper())

    ax.set_title(f"{name_a.upper()} vs {name_b.upper()}")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Score (episode return)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results", help="Path to results/ directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds to aggregate")
    parser.add_argument("--smooth", type=int, default=50, help="Smoothing window in episodes (e.g., 25/50/100)")
    parser.add_argument("--outdir", type=str, default="compare_pairs_episodes", help="Output directory")
    args = parser.parse_args()

    results_dir = Path(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for a, b in PAIR_LIST:
        merged_a = collect_algo_runs(results_dir, a, args.seeds, args.smooth)
        merged_b = collect_algo_runs(results_dir, b, args.seeds, args.smooth)

        stats_a = stats_from_merged(merged_a)
        stats_b = stats_from_merged(merged_b)

        out_png = outdir / f"{a}_vs_{b}_episodes.png"
        make_pair_plot(stats_a, stats_b, a, b, out_png)
        print(f"[OK] Saved {out_png}")


if __name__ == "__main__":
    main()
