#!/usr/bin/env python3
"""Plot agent-vs-random trends across training epoch groups.

Reads `per_class_summary.csv` (from `aggregate_views.py`, run on a CSV produced
with multiple --epoch_groups) and writes three plots per (model, metric) into
the output directory:

    <model>_<metric>_per_class_grid.png  — small-multiples (one subplot per class);
                                            agent vs random lines over epoch groups
    <model>_<metric>_delta_heatmap.png   — class × epoch_group heatmap of
                                            (agent_mean - random_mean)
    <model>_<metric>_macro.png           — single panel: macro-mean across classes
                                            of agent_mean and random_mean over epochs

Usage:
    python scripts/plot_epoch_trends.py \\
        --summary results/views/summary/per_class_summary.csv \\
        --output_dir results/views/plots/
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VIEW_TYPE_NAMES = ["Expanded", "Expanded-like", "Foreshortened", "Foreshortened-like", "Remainder"]
BUCKET_COUNT_COLS = [
    "agent_n_expanded", "agent_n_expanded_like",
    "agent_n_foreshortened", "agent_n_foreshortened_like",
    "agent_n_remainder",
]


def empirical_bucket_subtitle(df, view_type_str=None):
    """Sum the agent_n_* count columns across rows and return a one-line
    description of the empirical bucket mix among the agent's selected views.

    Buckets that contributed zero selections are dropped from the line.
    """
    if not all(c in df.columns for c in BUCKET_COUNT_COLS):
        return None
    totals = [int(df[c].sum()) for c in BUCKET_COUNT_COLS]
    total = sum(totals)
    if total <= 0:
        return None
    parts = [f"{name} {n / total:.1%} (n={n})"
             for name, n in zip(VIEW_TYPE_NAMES, totals) if n > 0]
    prefix = f"Agent selections" + (f" (v{view_type_str})" if view_type_str else "") + ": "
    return prefix + ", ".join(parts)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=str, required=True,
                   help="per_class_summary.csv with epoch_start/epoch_end columns.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--models", nargs="+", default=None,
                   help="Filter to specific models (default: all in the CSV).")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="Filter to specific metrics (default: all in the CSV).")
    return p.parse_args()


def _epoch_midpoint(es, ee):
    return (es + ee) / 2.0


def _epoch_label(es, ee):
    return f"{int(es)}-{int(ee)}"


def plot_per_class_grid(df, model, metric, out_path, subtitle=None):
    """One subplot per dataset class; agent (orange) and random (gray) vs epoch."""
    classes = sorted(df["dataset"].unique())
    n = len(classes)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.2),
                             sharex=True, squeeze=False)

    for idx, cls in enumerate(classes):
        ax = axes[idx // ncols][idx % ncols]
        cls_df = df[df["dataset"] == cls].sort_values("epoch_start")
        if cls_df.empty:
            ax.set_visible(False)
            continue
        x = [_epoch_midpoint(s, e) for s, e in zip(cls_df["epoch_start"], cls_df["epoch_end"])]
        ax.plot(x, cls_df["agent_mean"], "o-", color="#d95f02", label="agent", lw=1.5, ms=4)
        ax.plot(x, cls_df["random_mean"], "s--", color="#7f7f7f", label="random", lw=1.0, ms=4)
        short = cls.split(",")[0]
        ax.set_title(short, fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=7)

    # blank any unused subplots
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    # one legend at the figure level
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    title = f"{model} / {metric}: agent vs random by class"
    if subtitle:
        title = title + "\n" + subtitle
    fig.suptitle(title, fontsize=11)
    fig.supxlabel("Training epoch midpoint", fontsize=10)
    fig.supylabel("Confidence / similarity", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(df, model, metric, out_path, subtitle=None):
    """class × epoch_group heatmap of (agent_mean − random_mean)."""
    df = df.copy()
    df["epoch_label"] = [_epoch_label(s, e) for s, e in zip(df["epoch_start"], df["epoch_end"])]
    df["epoch_mid"] = [_epoch_midpoint(s, e) for s, e in zip(df["epoch_start"], df["epoch_end"])]
    pivot = df.pivot_table(index="dataset", columns="epoch_label", values="delta_mean",
                           aggfunc="mean")
    # column order by epoch midpoint, not lexicographic
    col_order = (df[["epoch_label", "epoch_mid"]]
                 .drop_duplicates().sort_values("epoch_mid")["epoch_label"].tolist())
    pivot = pivot.reindex(columns=col_order)

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * pivot.shape[1] + 4), max(6, 0.25 * pivot.shape[0] + 1)))
    vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 1.0
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([c.split(",")[0] for c in pivot.index], fontsize=8)
    ax.set_xlabel("Epoch group")
    ax.set_ylabel("Class")
    title = f"{model} / {metric}: agent − random (positive = agent better)"
    if subtitle:
        title = title + "\n" + subtitle
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Δ (agent − random)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_macro(df, model, metric, out_path, subtitle=None):
    """Single panel: macro-mean across classes of agent_mean and random_mean."""
    df = df.copy()
    df["epoch_mid"] = [_epoch_midpoint(s, e) for s, e in zip(df["epoch_start"], df["epoch_end"])]
    macro = (df.groupby(["epoch_start", "epoch_end", "epoch_mid"])
               .agg(agent=("agent_mean", "mean"),
                    random=("random_mean", "mean"),
                    delta=("delta_mean", "mean"),
                    agent_sem=("agent_mean", "sem"),
                    random_sem=("random_mean", "sem"))
               .reset_index()
               .sort_values("epoch_mid"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(macro["epoch_mid"], macro["agent"], yerr=macro["agent_sem"],
                fmt="o-", color="#d95f02", label="agent (macro mean)", lw=1.8, capsize=3)
    ax.errorbar(macro["epoch_mid"], macro["random"], yerr=macro["random_sem"],
                fmt="s--", color="#7f7f7f", label="random (macro mean)", lw=1.4, capsize=3)
    ax.set_xlabel("Training epoch midpoint")
    ax.set_ylabel("Confidence / similarity")
    title = f"{model} / {metric}: macro mean across classes"
    if subtitle:
        title = title + "\n" + subtitle
    ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.summary)

    required = {"dataset", "model", "metric", "epoch_start", "epoch_end",
                "agent_mean", "random_mean", "delta_mean"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Summary CSV missing columns: {sorted(missing)}. "
                         "Make sure aggregate_views.py was run on a CSV that has "
                         "epoch_start/epoch_end (i.e. produced with --epoch_groups).")

    models = args.models or sorted(df["model"].unique())
    metrics = args.metrics or sorted(df["metric"].unique())

    for model in models:
        for metric in metrics:
            sub = df[(df["model"] == model) & (df["metric"] == metric)]
            if sub.empty:
                continue
            n_groups = sub[["epoch_start", "epoch_end"]].drop_duplicates().shape[0]
            if n_groups < 2:
                print(f"  Skip {model}/{metric}: only {n_groups} epoch group(s).")
                continue

            svt = None
            if "selected_view_type" in sub.columns and sub["selected_view_type"].notna().any():
                svt = str(sub["selected_view_type"].dropna().iloc[0])
            subtitle = empirical_bucket_subtitle(sub, svt)
            suffix = f"_v{svt}" if svt else ""
            base = os.path.join(args.output_dir, f"{model}_{metric}{suffix}")

            plot_per_class_grid(sub, model, metric, f"{base}_per_class_grid.png", subtitle)
            plot_delta_heatmap(sub, model, metric, f"{base}_delta_heatmap.png", subtitle)
            plot_macro(sub, model, metric, f"{base}_macro.png", subtitle)
            print(f"  Wrote 3 plots for {model}/{metric}{suffix}: {base}_*.png")


if __name__ == "__main__":
    main()
