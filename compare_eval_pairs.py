#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PAIR_LIST = [
    ("nfq", "dqn"),
    ("dqn", "ddqn"),
]


def load_eval_curve(metrics_csv: Path) -> pd.DataFrame:
    """Load only eval rows: columns step, value."""
    df = pd.read_csv(metrics_csv)
    required = {"type", "step", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"{metrics_csv} missing columns. Found: {list(df.columns)}")

    eval_df = df[df["type"] == "eval"][["step", "value"]].copy()
    eval_df = eval_df.dropna(subset=["step", "value"])
    eval_df["step"] = eval_df["step"].astype(int)
    eval_df["value"] = eval_df["value"].astype(float)
    eval_df = eval_df.sort_values("step").reset_index(drop=True)

    if eval_df.empty:
        raise ValueError(f"No eval rows found in {metrics_csv}")
    return eval_df


def smooth_series(y: pd.Series, window: int) -> pd.Series:
    """Rolling mean smoothing keeping same length."""
    if window <= 1:
        return y
    return y.rolling(window=window, min_periods=1).mean()


def collect_algo_runs(results_dir: Path, algo: str, seeds: list[int], smooth_w: int) -> pd.DataFrame:
    """
    Returns DF indexed by step with one column per seed.
    Inner-join across seeds to keep a shared x-axis.
    """
    per_seed = []
    for seed in seeds:
        run_dir = results_dir / f"{algo}_seed{seed}"
        metrics_csv = run_dir / "metrics.csv"
        if not metrics_csv.exists():
            raise FileNotFoundError(f"Missing {metrics_csv}")

        eval_df = load_eval_curve(metrics_csv)
        col = f"seed{seed}"
        eval_df[col] = smooth_series(eval_df["value"], smooth_w)
        per_seed.append(eval_df[["step", col]])

    merged = per_seed[0]
    for nxt in per_seed[1:]:
        merged = merged.merge(nxt, on="step", how="inner")

    return merged.set_index("step").sort_index()


def stats_from_merged(merged: pd.DataFrame) -> pd.DataFrame:
    """Columns -> min/max/mean across seeds."""
    out = pd.DataFrame(index=merged.index)
    out["min"] = merged.min(axis=1)
    out["max"] = merged.max(axis=1)
    out["mean"] = merged.mean(axis=1)
    return out


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Area under curve using trapezoidal rule.
    Compatible with old and new NumPy versions.
    """
    if len(x) < 2:
        return float("nan")
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))



def plot_band(ax, stats: pd.DataFrame, label: str):
    x = stats.index.to_numpy()
    ax.plot(x, stats["mean"].to_numpy(), linewidth=2, label=label)
    ax.fill_between(x, stats["min"].to_numpy(), stats["max"].to_numpy(), alpha=0.25)


def summarize(stats: pd.DataFrame, algo: str, seeds: list[int], smooth_w: int, lastk: int) -> dict:
    x = stats.index.to_numpy()
    y = stats["mean"].to_numpy()
    k = min(lastk, len(y))
    return {
        "algo": algo,
        "seeds": ",".join(map(str, seeds)),
        "smooth_window": smooth_w,
        "n_eval_points": len(x),
        "final_mean_lastk": float(np.mean(y[-k:])),
        "auc_mean": auc(x, y),
    }


def make_pair_plot(stats_a: pd.DataFrame, stats_b: pd.DataFrame, name_a: str, name_b: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_band(ax, stats_a, label=name_a.upper())
    plot_band(ax, stats_b, label=name_b.upper())

    ax.set_title(f"FlappyBird: {name_a.upper()} vs {name_b.upper()} (mean ± [min,max] over seeds)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval return")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results", help="Path to results/ directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds to aggregate")
    parser.add_argument("--smooth", type=int, default=3, help="Rolling window (eval points) for smoothing")
    parser.add_argument("--outdir", type=str, default="compare_pairs_eval", help="Output directory")
    parser.add_argument("--lastk", type=int, default=10, help="Last K eval points for final mean")
    args = parser.parse_args()

    results_dir = Path(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []

    for a, b in PAIR_LIST:
        merged_a = collect_algo_runs(results_dir, a, args.seeds, args.smooth)
        merged_b = collect_algo_runs(results_dir, b, args.seeds, args.smooth)

        stats_a = stats_from_merged(merged_a)
        stats_b = stats_from_merged(merged_b)

        # Save per-algo stats for this pair (optional but handy)
        stats_a.reset_index().rename(columns={"index": "step"}).to_csv(outdir / f"{a}_mean_min_max.csv", index=False)
        stats_b.reset_index().rename(columns={"index": "step"}).to_csv(outdir / f"{b}_mean_min_max.csv", index=False)

        # Save plot
        out_png = outdir / f"{a}_vs_{b}_eval.png"
        make_pair_plot(stats_a, stats_b, a, b, out_png)

        # Summary table rows
        all_summary_rows.append(summarize(stats_a, a, args.seeds, args.smooth, args.lastk))
        all_summary_rows.append(summarize(stats_b, b, args.seeds, args.smooth, args.lastk))

        print(f"[OK] Saved {out_png}")

    summary = pd.DataFrame(all_summary_rows)
    summary.to_csv(outdir / "summary_pairs.csv", index=False)

    print(f"\n[OK] Summary saved to {outdir / 'summary_pairs.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

# esegui: python compare_pairs.py --results results --seeds 0 1 2 --smooth 3 --outdir compare_pairs_eval
