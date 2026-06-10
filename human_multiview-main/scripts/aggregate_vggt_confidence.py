"""Overlay VGGT (or Pi3 / DUSt3R / MASt3R) confidence-on-selections curves
from multiple experiments onto two figures: pair-level and set-level.

Reads `overall_summary.csv` (produced by `aggregate_views.py`) from each
--summary path given on the CLI, concatenates them with a per-experiment
label, and saves overlay plots:

    <model>_pair_<value>_over_epochs.png       — pair-level VGGT confidence
    <model>_image_<value>_over_epochs.png      — joint set-level (VGGT/Pi3 only)
    aggregated.csv                              — concatenated raw rows

<value> defaults to `agent_mean` (the VGGT confidence on the agent's selected
views); switch to `delta_mean` to plot the agent-over-random advantage.

Usage:
    python3 aggregate_vggt_confidence.py \\
        --summary results/views/v01234/summary/no_freeze:no_freeze \\
        --summary results/views/v01234/summary/freeze_10:freeze_10 \\
        --model vggt \\
        --output_dir compare/vggt_confidence_freeze_sweep
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Columns we know live in overall_summary.csv (written by aggregate_views.py)
DEFAULT_VALUE_COL_FOR = {
    "agent_mean": "instance_pooled_agent_mean",
    "random_mean": "instance_pooled_random_mean",
    "delta_mean": "instance_pooled_delta_mean",
    "macro_agent": "macro_mean_agent",
    "macro_random": "macro_mean_random",
    "macro_delta": "macro_mean_delta",
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--summary", action="append", required=True,
                   help="Summary directory or overall_summary.csv path, "
                        "optionally PATH:LABEL. Repeatable.")
    p.add_argument("--model", default="vggt",
                   help="Filter rows to this model (vggt / pi3 / dust3r / mast3r).")
    p.add_argument("--value", default="agent_mean",
                   choices=list(DEFAULT_VALUE_COL_FOR.keys()),
                   help="Which numeric column to plot (default: agent_mean = "
                        "instance-pooled mean of VGGT confidence on the agent's selections).")
    p.add_argument("--pair_metric_name", default="pair_mean",
                   help="`metric` column value identifying pairwise confidence.")
    p.add_argument("--set_metric_name", default="image_mean",
                   help="`metric` column value identifying joint set-level confidence. "
                        "Empty string disables the set-level plot.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--title_suffix", default="")
    p.add_argument("--style", default="line",
                   choices=["line", "heatmap", "rank_stacked", "sorted_bars",
                            "both", "all"],
                   help="line | heatmap | sorted_bars (grouped bars per epoch "
                        "sorted left-to-right by value, real y-axis) | "
                        "rank_stacked (vertically stacked; y-axis is a sum) | "
                        "both | all.")
    p.add_argument("--bin_epochs", type=int, default=0,
                   help="Heatmap only: bin into N columns (0 = no binning).")
    p.add_argument("--smooth", type=int, default=1,
                   help="Line-plot rolling mean window (1 = off).")
    return p.parse_args()


def parse_summary_spec(spec):
    """Parse PATH or PATH:LABEL. PATH may be a directory containing
    overall_summary.csv, or a direct path to the CSV. Returns (csv_path, label).

    Treats the suffix after the LAST ':' as a label iff it doesn't contain a
    path separator. This way absolute paths like '/abs/path:label' parse
    correctly.
    """
    path, label = spec, None
    if ":" in spec:
        prefix, _, suffix = spec.rpartition(":")
        if prefix and suffix and "/" not in suffix:
            path, label = prefix, suffix

    if os.path.isdir(path):
        csv_path = os.path.join(path, "overall_summary.csv")
    else:
        csv_path = path

    if label is None or label == "":
        # Use the parent of the summary CSV (assumed = experiment label)
        parent = os.path.basename(os.path.normpath(os.path.dirname(csv_path)))
        label = parent if parent else os.path.basename(csv_path)
    return csv_path, label


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _epoch_mid(df):
    return (df["epoch_start"].astype(float) + df["epoch_end"].astype(float)) / 2.0


def smooth_series(s, window):
    return s.rolling(window=window, center=True, min_periods=1).mean() if window > 1 else s


def _bin_matrix(epochs, values, n_bins):
    if n_bins <= 0 or len(epochs) <= n_bins:
        return epochs, values
    epochs = np.asarray(epochs, dtype=float)
    values = np.asarray(values, dtype=float)
    edges = np.linspace(epochs.min() - 1e-6, epochs.max() + 1e-6, n_bins + 1)
    centers, bvals = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (epochs >= lo) & (epochs < hi)
        if mask.any():
            centers.append(float((lo + hi) / 2))
            bvals.append(float(np.nanmean(values[mask])))
    return np.array(centers), np.array(bvals)


def plot_line(combined, value_col, metric_name, title_word, out_path, args):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for exp_label, df_exp in combined.groupby("experiment"):
        sub = df_exp[df_exp["metric"] == metric_name].sort_values("epoch_mid")
        if sub.empty:
            continue
        y = smooth_series(sub[value_col].astype(float), args.smooth)
        ax.plot(sub["epoch_mid"].to_numpy(), y.to_numpy(),
                "o-", lw=1.5, ms=4, alpha=0.85, label=exp_label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("Training epoch midpoint")
    ax.set_ylabel(f"{args.model.upper()} {title_word} confidence ({value_col})")
    title = f"{args.model.upper()} {title_word} confidence on agent's selections"
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sorted_bars(combined, value_col, metric_name, title_word, out_path, args):
    """Grouped bars per epoch, sorted left-to-right by value (descending).

    Real y-axis (no spurious sum). Same experiment = same colour everywhere;
    a colour changing horizontal position between epoch groups = a rank flip.
    """
    rows_of_values, exp_labels, all_epochs = [], [], None
    for exp_label, df_exp in combined.groupby("experiment"):
        sub = df_exp[df_exp["metric"] == metric_name].sort_values("epoch_mid")
        if sub.empty:
            continue
        epochs_i = sub["epoch_mid"].to_numpy()
        vals_i = sub[value_col].astype(float).to_numpy()
        epochs_i, vals_i = _bin_matrix(epochs_i, vals_i, args.bin_epochs)
        if all_epochs is None:
            all_epochs = epochs_i
        elif not np.array_equal(all_epochs, epochs_i):
            all_epochs = np.union1d(all_epochs, epochs_i)
        exp_labels.append(exp_label)
        rows_of_values.append((epochs_i, vals_i))
    if not exp_labels:
        return

    matrix = np.full((len(exp_labels), len(all_epochs)), np.nan)
    for i, (epochs_i, vals_i) in enumerate(rows_of_values):
        idx = np.searchsorted(all_epochs, epochs_i)
        matrix[i, idx] = vals_i

    n_exp, n_T = matrix.shape
    cmap_base = plt.cm.tab10 if n_exp <= 10 else plt.cm.tab20
    color_for = {lbl: cmap_base(i % cmap_base.N) for i, lbl in enumerate(exp_labels)}

    group_width = 0.85
    bar_width = group_width / max(n_exp, 1)
    fig_w = max(8.0, 0.18 * n_T * n_exp + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, max(4.5, 0.3 * n_exp + 3.0)))

    for j in range(n_T):
        col = matrix[:, j]
        order = np.argsort(-np.where(np.isnan(col), -np.inf, col))
        valid = [i for i in order if not np.isnan(col[i])]
        n_valid = len(valid)
        for pos_in_group, i in enumerate(valid):
            x = j + (pos_in_group - (n_valid - 1) / 2.0) * bar_width
            ax.bar(x, col[i], width=bar_width * 0.9,
                   color=color_for[exp_labels[i]],
                   edgecolor="white", linewidth=0.3)

    n_xticks = min(n_T, 10)
    tick_idx = np.linspace(0, n_T - 1, n_xticks).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{int(all_epochs[i])}" for i in tick_idx], fontsize=8)
    ax.set_xlabel("Training epoch midpoint")
    ax.set_ylabel(value_col)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_for[lbl]) for lbl in exp_labels]
    ax.legend(legend_handles, exp_labels, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False,
              title="experiment\n(within each epoch:\nleft = highest)")
    title = f"{args.model.upper()} {title_word} confidence ({value_col})"
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    title += "\n(sorted-bars: within each epoch, bars left→right by descending value)"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rank_stacked(combined, value_col, metric_name, title_word, out_path, args):
    """Stacked bars sorted by value at each epoch (segment colour = experiment,
    segment order = rank). Same idea as the temporal-test rank_stacked plot."""
    rows_of_values, exp_labels, all_epochs = [], [], None
    for exp_label, df_exp in combined.groupby("experiment"):
        sub = df_exp[df_exp["metric"] == metric_name].sort_values("epoch_mid")
        if sub.empty:
            continue
        epochs_i = sub["epoch_mid"].to_numpy()
        vals_i = sub[value_col].astype(float).to_numpy()
        epochs_i, vals_i = _bin_matrix(epochs_i, vals_i, args.bin_epochs)
        if all_epochs is None:
            all_epochs = epochs_i
        elif not np.array_equal(all_epochs, epochs_i):
            all_epochs = np.union1d(all_epochs, epochs_i)
        exp_labels.append(exp_label)
        rows_of_values.append((epochs_i, vals_i))
    if not exp_labels:
        return

    matrix = np.full((len(exp_labels), len(all_epochs)), np.nan)
    for i, (epochs_i, vals_i) in enumerate(rows_of_values):
        idx = np.searchsorted(all_epochs, epochs_i)
        matrix[i, idx] = vals_i

    n_exp, n_T = matrix.shape
    cmap_base = plt.cm.tab10 if n_exp <= 10 else plt.cm.tab20
    color_for = {lbl: cmap_base(i % cmap_base.N) for i, lbl in enumerate(exp_labels)}

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * n_T + 2),
                                    max(4.5, 0.3 * n_exp + 3)))
    for j in range(n_T):
        col = matrix[:, j]
        order = np.argsort(-np.where(np.isnan(col), -np.inf, col))
        bottom_pos = 0.0
        bottom_neg = 0.0
        for i in order:
            v = col[i]
            if np.isnan(v):
                continue
            lbl = exp_labels[i]
            if v >= 0:
                ax.bar(j, v, bottom=bottom_pos, width=0.85,
                       color=color_for[lbl], edgecolor="white", linewidth=0.4)
                bottom_pos += v
            else:
                ax.bar(j, v, bottom=bottom_neg, width=0.85,
                       color=color_for[lbl], edgecolor="white", linewidth=0.4)
                bottom_neg += v

    n_xticks = min(n_T, 10)
    tick_idx = np.linspace(0, n_T - 1, n_xticks).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{int(all_epochs[i])}" for i in tick_idx], fontsize=8)
    ax.set_xlabel("Training epoch midpoint")
    ax.set_ylabel(f"Stacked {value_col} — sum across experiments")
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_for[lbl]) for lbl in exp_labels]
    ax.legend(legend_handles, exp_labels, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False,
              title="experiment (segment colour fixed; order = rank)")
    title = f"{args.model.upper()} {title_word} confidence ({value_col})"
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    title += "\n(rank-stacked: segment order = sorted by value at each epoch)"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(combined, value_col, metric_name, title_word, out_path, args):
    rows_of_values, exp_labels, all_epochs = [], [], None
    for exp_label, df_exp in combined.groupby("experiment"):
        sub = df_exp[df_exp["metric"] == metric_name].sort_values("epoch_mid")
        if sub.empty:
            continue
        epochs_i = sub["epoch_mid"].to_numpy()
        vals_i = sub[value_col].astype(float).to_numpy()
        epochs_i, vals_i = _bin_matrix(epochs_i, vals_i, args.bin_epochs)
        if all_epochs is None:
            all_epochs = epochs_i
        elif not np.array_equal(all_epochs, epochs_i):
            all_epochs = np.union1d(all_epochs, epochs_i)
        exp_labels.append(exp_label)
        rows_of_values.append((epochs_i, vals_i))
    if not exp_labels:
        return

    matrix = np.full((len(exp_labels), len(all_epochs)), np.nan)
    for i, (epochs_i, vals_i) in enumerate(rows_of_values):
        idx = np.searchsorted(all_epochs, epochs_i)
        matrix[i, idx] = vals_i

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return

    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(all_epochs) + 2),
                                    max(2.5, 0.4 * len(exp_labels) + 1)))
    cmap = "RdBu_r" if "delta" in value_col else "viridis"
    if "delta" in value_col:
        vmax = float(np.nanmax(np.abs(finite)))
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
    else:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                       vmin=float(np.nanmin(finite)), vmax=float(np.nanmax(finite)),
                       interpolation="nearest")
    ax.set_yticks(range(len(exp_labels)))
    ax.set_yticklabels(exp_labels, fontsize=8)
    n_xticks = min(len(all_epochs), 10)
    tick_idx = np.linspace(0, len(all_epochs) - 1, n_xticks).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{int(all_epochs[i])}" for i in tick_idx], fontsize=8)
    ax.set_xlabel("Training epoch midpoint")
    title = f"{args.model.upper()} {title_word} confidence ({value_col})"
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    value_col = DEFAULT_VALUE_COL_FOR[args.value]

    rows = []
    for spec in args.summary:
        csv_path, label = parse_summary_spec(spec)
        if not os.path.exists(csv_path):
            print(f"SKIP {label}: no overall_summary.csv at {csv_path}",
                  file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        df = df[df["model"] == args.model]
        if df.empty:
            print(f"SKIP {label}: no rows for model={args.model} in {csv_path}",
                  file=sys.stderr)
            continue
        if value_col not in df.columns:
            print(f"SKIP {label}: column {value_col!r} not in {csv_path}.\n"
                  f"  Available: {sorted(df.columns)}", file=sys.stderr)
            continue
        df = df.copy()
        df["epoch_mid"] = _epoch_mid(df)
        df["experiment"] = label
        rows.append(df)
        print(f"loaded {label}  ({len(df)} rows from {csv_path})")

    if not rows:
        sys.exit("No summaries loaded; nothing to plot.")

    combined = pd.concat(rows, ignore_index=True)
    out_csv = os.path.join(args.output_dir, "aggregated.csv")
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  "
          f"({combined['experiment'].nunique()} experiments × "
          f"{combined['metric'].nunique()} metrics)")

    do_line = args.style in ("line", "both", "all")
    do_heatmap = args.style in ("heatmap", "both", "all")
    do_rank_stacked = args.style in ("rank_stacked", "all")
    do_sorted_bars = args.style in ("sorted_bars", "all")

    # Pair-level
    if args.pair_metric_name:
        if do_line:
            out = os.path.join(args.output_dir,
                               f"{args.model}_pair_{args.value}_over_epochs.png")
            plot_line(combined, value_col, args.pair_metric_name, "pair-level", out, args)
            print(f"Saved: {out}")
        if do_heatmap:
            out = os.path.join(args.output_dir,
                               f"{args.model}_pair_{args.value}_heatmap.png")
            plot_heatmap(combined, value_col, args.pair_metric_name, "pair-level", out, args)
            print(f"Saved: {out}")
        if do_rank_stacked:
            out = os.path.join(args.output_dir,
                               f"{args.model}_pair_{args.value}_rank_stacked.png")
            plot_rank_stacked(combined, value_col, args.pair_metric_name, "pair-level", out, args)
            print(f"Saved: {out}")
        if do_sorted_bars:
            out = os.path.join(args.output_dir,
                               f"{args.model}_pair_{args.value}_sorted_bars.png")
            plot_sorted_bars(combined, value_col, args.pair_metric_name, "pair-level", out, args)
            print(f"Saved: {out}")

    # Set-level (skip if disabled or absent)
    if args.set_metric_name and (combined["metric"] == args.set_metric_name).any():
        if do_line:
            out = os.path.join(args.output_dir,
                               f"{args.model}_set_{args.value}_over_epochs.png")
            plot_line(combined, value_col, args.set_metric_name, "set-level", out, args)
            print(f"Saved: {out}")
        if do_heatmap:
            out = os.path.join(args.output_dir,
                               f"{args.model}_set_{args.value}_heatmap.png")
            plot_heatmap(combined, value_col, args.set_metric_name, "set-level", out, args)
            print(f"Saved: {out}")
        if do_rank_stacked:
            out = os.path.join(args.output_dir,
                               f"{args.model}_set_{args.value}_rank_stacked.png")
            plot_rank_stacked(combined, value_col, args.set_metric_name, "set-level", out, args)
            print(f"Saved: {out}")
        if do_sorted_bars:
            out = os.path.join(args.output_dir,
                               f"{args.model}_set_{args.value}_sorted_bars.png")
            plot_sorted_bars(combined, value_col, args.set_metric_name, "set-level", out, args)
            print(f"Saved: {out}")
    elif args.set_metric_name:
        print(f"Note: no {args.set_metric_name} rows present (model probably "
              f"isn't VGGT/Pi3); skipping set-level plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
