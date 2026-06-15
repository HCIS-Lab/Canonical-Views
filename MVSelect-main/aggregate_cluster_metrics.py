"""Overlay cluster_metrics.csv from multiple experiments.

Reads `<exp>/cluster_metrics.csv` (produced by compute_cluster_metrics.py)
for each --exp path given on the CLI and overlays the chosen metric on a
single figure per metric, in the same plot styles as
`aggregate_temporal_tests.py`:

    line          — overlaid line plots (default)
    heatmap       — experiments × epochs grid, color = metric value
    sorted_bars   — grouped bars per epoch sorted by value (real y-axis)
    rank_stacked  — vertically stacked sorted bars (y-axis is a sum)
    both          — line + heatmap
    all           — every style

Default plotted metric is `separability` (silhouette_class − silhouette_view);
override with `--metrics` (one or more of silhouette_class, silhouette_view,
silhouette_view_index, separability, silhouette_class_selected,
silhouette_class_selected_with_init, silhouette_class_all_views_mean,
silhouette_class_all_views_max).

Usage:
    python3 aggregate_cluster_metrics.py \\
        --exp meta_logs/rgb/resnet18steps5_train_ins25_..._e100:no_freeze \\
        --exp meta_logs/rgb/freeze_10_resnet18steps5_..._e100:freeze_10 \\
        --output_dir compare/cluster_metrics_freeze_sweep \\
        --metrics separability silhouette_class_selected silhouette_class silhouette_view silhouette_view_index \\
        --style heatmap
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VALID_METRICS = [
    "silhouette_class",
    "silhouette_view",
    "silhouette_view_index",
    "separability",
    "silhouette_class_selected",
    "silhouette_class_selected_with_init",
    "silhouette_class_all_views_mean",
    "silhouette_class_all_views_max",
]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--exp", action="append", required=True,
                   help="Experiment to include. PATH or PATH:LABEL. Repeatable. "
                        "PATH = folder containing cluster_metrics.csv (typically "
                        "meta_logs/<rep>/<exp>/).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--metrics", nargs="+", default=["separability"],
                   choices=VALID_METRICS,
                   help="Which metrics to plot (one figure per metric per style).")
    p.add_argument("--style", default="heatmap",
                   choices=["line", "heatmap", "sorted_bars", "rank_stacked",
                            "both", "all"],
                   help="Plot style — defaults to heatmap (best for many exps).")
    p.add_argument("--bin_epochs", type=int, default=0,
                   help="Heatmap / bars only: bin into N columns (0 = no binning).")
    p.add_argument("--every_n_epochs", type=int, default=None,
                   help="Drop rows whose epoch is not a multiple of N. "
                        "Useful when training dumped features at uneven cadence "
                        "and you want a regular grid for plotting (e.g., every 10).")
    p.add_argument("--smooth", type=int, default=1)
    p.add_argument("--title_suffix", default="")
    return p.parse_args()


def parse_exp_spec(spec):
    """PATH or PATH:LABEL → (csv_path, label).

    Treats the suffix after the LAST ':' as a label iff it doesn't contain
    a path separator. This way absolute paths like
    '/abs/path:label' are parsed correctly as ('/abs/path', 'label')
    rather than treated as part of the path.
    """
    path, label = spec, None
    if ":" in spec:
        prefix, _, suffix = spec.rpartition(":")
        # Real labels don't contain slashes; if the suffix has one, the
        # colon must be inside the path (e.g. a Windows drive or a real
        # filename containing ':'), so leave spec alone.
        if prefix and suffix and "/" not in suffix:
            path, label = prefix, suffix

    if os.path.isdir(path):
        csv_path = os.path.join(path, "cluster_metrics.csv")
    else:
        csv_path = path

    if label is None or label == "":
        parent = os.path.basename(os.path.normpath(os.path.dirname(csv_path)))
        label = parent if parent else os.path.basename(csv_path)
    return csv_path, label


# ---------- plot helpers (lifted/adapted from aggregate_temporal_tests.py) ----

def _bin_matrix(epochs, values, n_bins):
    if n_bins <= 0 or len(epochs) <= n_bins:
        return epochs, values
    epochs = np.asarray(epochs, dtype=float)
    values = np.asarray(values, dtype=float)
    edges = np.linspace(epochs.min() - 0.5, epochs.max() + 0.5, n_bins + 1)
    centers, bvals = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (epochs >= lo) & (epochs < hi)
        if mask.any():
            centers.append(float((lo + hi) / 2))
            bvals.append(float(np.nanmean(values[mask])))
    return np.array(centers), np.array(bvals)


def smooth_series(s, window):
    return s.rolling(window=window, center=True, min_periods=1).mean() if window > 1 else s


def _collect(df, metric, args):
    exp_labels, rows_of_values, all_epochs = [], [], None
    for exp_label, df_exp in df.groupby("experiment"):
        sub = df_exp.sort_values("epoch")
        if metric not in sub.columns or sub[metric].isna().all():
            continue
        epochs_i = sub["epoch"].to_numpy()
        vals_i = sub[metric].to_numpy()
        epochs_i, vals_i = _bin_matrix(epochs_i, vals_i, args.bin_epochs)
        if all_epochs is None:
            all_epochs = epochs_i
        elif not np.array_equal(all_epochs, epochs_i):
            all_epochs = np.union1d(all_epochs, epochs_i)
        exp_labels.append(exp_label)
        rows_of_values.append((epochs_i, vals_i))
    if not exp_labels:
        return None, None, None
    matrix = np.full((len(exp_labels), len(all_epochs)), np.nan)
    for i, (epochs_i, vals_i) in enumerate(rows_of_values):
        idx = np.searchsorted(all_epochs, epochs_i)
        matrix[i, idx] = vals_i
    return exp_labels, all_epochs, matrix


def plot_line(df, metric, out_path, args):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for exp_label, df_exp in df.groupby("experiment"):
        sub = df_exp.sort_values("epoch")
        if metric not in sub.columns:
            continue
        y = smooth_series(sub[metric].astype(float), args.smooth)
        ax.plot(sub["epoch"].to_numpy(), y.to_numpy(),
                "o-", lw=1.5, ms=4, alpha=0.85, label=exp_label)
        plotted += 1
    if plotted == 0:
        plt.close(fig); return
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(metric)
    if metric == "separability":
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
    title = f"Cluster quality over training: {metric}"
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(df, metric, out_path, args):
    exp_labels, all_epochs, matrix = _collect(df, metric, args)
    if matrix is None:
        return
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    if metric == "separability":
        cmap = "RdBu_r"
        vmax = float(np.nanmax(np.abs(finite)))
        vmin = -vmax
    else:
        cmap = "viridis"
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))

    fig, ax = plt.subplots(figsize=(max(8, 0.3 * len(all_epochs) + 2),
                                    max(2.5, 0.4 * len(exp_labels) + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_yticks(range(len(exp_labels)))
    ax.set_yticklabels(exp_labels, fontsize=8)
    n_xticks = min(len(all_epochs), 10)
    tick_idx = np.linspace(0, len(all_epochs) - 1, n_xticks).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{int(all_epochs[i])}" for i in tick_idx], fontsize=8)
    ax.set_xlabel("Training epoch")
    title = f"Cluster quality: {metric}"
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    # Make the true min/max explicit on the colorbar so yellow/red aren't
    # mistaken for "exactly the last labeled tick" (matplotlib's default
    # ticks land on round numbers and can hide the real endpoints).
    if metric == "separability":
        ticks = sorted({vmin, 0.0, vmax})
    else:
        ticks = [vmin, vmax]
        if vmin < 0 < vmax:
            ticks = sorted({vmin, 0.0, vmax})
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:+.3f}" if t != 0 else "0" for t in ticks])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _bar_chart(df, metric, out_path, args, *, stacked):
    """Sorted-bars (stacked=False) OR rank-stacked (stacked=True)."""
    exp_labels, all_epochs, matrix = _collect(df, metric, args)
    if matrix is None:
        return
    n_exp, n_T = matrix.shape
    cmap_base = plt.cm.tab10 if n_exp <= 10 else plt.cm.tab20
    color_for = {lbl: cmap_base(i % cmap_base.N) for i, lbl in enumerate(exp_labels)}

    if stacked:
        fig_w = max(8.0, 0.4 * n_T + 2.0)
    else:
        fig_w = max(8.0, 0.18 * n_T * n_exp + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, max(4.5, 0.3 * n_exp + 3.0)))

    group_width = 0.85
    bar_width = group_width / max(n_exp, 1)

    for j in range(n_T):
        col = matrix[:, j]
        order = np.argsort(-np.where(np.isnan(col), -np.inf, col))
        valid = [i for i in order if not np.isnan(col[i])]
        if stacked:
            bottom_pos, bottom_neg = 0.0, 0.0
            for i in valid:
                v = col[i]
                if v >= 0:
                    ax.bar(j, v, bottom=bottom_pos, width=0.85,
                           color=color_for[exp_labels[i]],
                           edgecolor="white", linewidth=0.4)
                    bottom_pos += v
                else:
                    ax.bar(j, v, bottom=bottom_neg, width=0.85,
                           color=color_for[exp_labels[i]],
                           edgecolor="white", linewidth=0.4)
                    bottom_neg += v
        else:
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
    ax.set_xlabel("Training epoch")
    if metric == "separability":
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_ylabel("Stacked " + metric if stacked else metric)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_for[lbl]) for lbl in exp_labels]
    title_style = "rank-stacked" if stacked else "sorted-bars"
    ax.legend(legend_handles, exp_labels, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False,
              title=f"experiment\n(within each epoch:\nsegments sorted by {metric})")
    ttl = f"Cluster quality: {metric} ({title_style})"
    if args.title_suffix:
        ttl += f"\n{args.title_suffix}"
    ax.set_title(ttl)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- Main ------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for spec in args.exp:
        csv_path, label = parse_exp_spec(spec)
        if not os.path.exists(csv_path):
            print(f"SKIP {label}: no cluster_metrics.csv at {csv_path}",
                  file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        df["experiment"] = label
        df["_source_csv"] = csv_path
        rows.append(df)
        print(f"loaded {label}  ({len(df)} epoch rows from {csv_path})")

    if not rows:
        sys.exit("No CSVs loaded; nothing to plot.")

    combined = pd.concat(rows, ignore_index=True)

    # Optional regular-grid filter (e.g. every 10th epoch).
    if args.every_n_epochs and args.every_n_epochs > 1:
        before = len(combined)
        combined = combined[combined["epoch"] % args.every_n_epochs == 0]
        kept_epochs = sorted(combined["epoch"].unique().tolist())
        print(f"\nFiltered to epochs % {args.every_n_epochs} == 0: "
              f"{before} → {len(combined)} rows, kept epochs {kept_epochs}")

    out_csv = os.path.join(args.output_dir, "aggregated.csv")
    combined.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  ({combined['experiment'].nunique()} experiments)")

    do_line = args.style in ("line", "both", "all")
    do_heatmap = args.style in ("heatmap", "both", "all")
    do_sorted_bars = args.style in ("sorted_bars", "all")
    do_rank_stacked = args.style in ("rank_stacked", "all")

    for metric in args.metrics:
        if do_line:
            out = os.path.join(args.output_dir, f"{metric}_line.png")
            plot_line(combined, metric, out, args)
            print(f"Saved: {out}")
        if do_heatmap:
            out = os.path.join(args.output_dir, f"{metric}_heatmap.png")
            plot_heatmap(combined, metric, out, args)
            print(f"Saved: {out}")
        if do_sorted_bars:
            out = os.path.join(args.output_dir, f"{metric}_sorted_bars.png")
            _bar_chart(combined, metric, out, args, stacked=False)
            print(f"Saved: {out}")
        if do_rank_stacked:
            out = os.path.join(args.output_dir, f"{metric}_rank_stacked.png")
            _bar_chart(combined, metric, out, args, stacked=True)
            print(f"Saved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
