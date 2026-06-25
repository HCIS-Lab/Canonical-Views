#!/usr/bin/env python3
"""Aggregate midlevel_shape_features.py outputs across experiments.

Each --exp points to a meta_logs/<dataset>/<experiment>/ folder containing:

    midlevel_features/selected_midlevel_summary.csv

The script concatenates those summaries, adds an experiment label, and plots
shared-axis heatmaps or lines for selected mid-level image descriptors.
"""

import argparse
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(),
                                                   "mvselect_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "ellipse_aspect_ratio",
    "skeleton_elongation",
    "skeleton_length_norm",
    "bilateral_symmetry",
    "medial_axis_symmetry",
    "skeleton_branch_density",
    "edge_anisotropy",
    "edge_entropy",
]

METRIC_GROUPS = {
    "axis_visibility": [
        "ellipse_orientation_deg",
        "ellipse_aspect_ratio",
        "skeleton_elongation",
        "skeleton_length_norm",
    ],
    "symmetry_part_organization": [
        "bilateral_symmetry",
        "medial_axis_symmetry",
        "skeleton_endpoint_count",
        "skeleton_branchpoint_count",
        "skeleton_branch_density",
    ],
    "edge_organization": [
        "dominant_edge_orientation_deg",
        "edge_entropy",
        "edge_anisotropy",
    ],
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--exp", action="append", required=True,
                   help="Experiment folder or folder:LABEL. Repeatable.")
    p.add_argument("--dataset", default="rgb",
                   help="Resolve bare experiment names under meta_logs/<dataset>/.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--value", default="lift",
                   choices=["selected", "lift", "baseline"],
                   help="Plot selected_<metric>, lift_<metric>, or baseline_<metric>.")
    p.add_argument("--group_value", default="selected",
                   choices=["selected", "lift", "baseline", "none"],
                   help="Value prefix for the grouped comparison curve figures. "
                        "Use 'none' to disable group figures.")
    p.add_argument("--style", default="heatmap",
                   choices=["heatmap", "line", "both"])
    p.add_argument("--bin_epochs", type=int, default=10,
                   help="Bin epochs into N columns. 0 = raw epochs.")
    p.add_argument("--title_suffix", default="")
    return p.parse_args()


def _csv_path(exp_dir):
    return os.path.join(exp_dir, "midlevel_features", "selected_midlevel_summary.csv")


def parse_exp_spec(spec, dataset):
    path, label = spec, None
    if ":" in spec:
        prefix, _, suffix = spec.rpartition(":")
        if prefix and suffix and "/" not in suffix:
            path, label = prefix, suffix

    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join("meta_logs", dataset, path))

    chosen = None
    for c in candidates:
        if os.path.exists(_csv_path(c)):
            chosen = c
            break
    if chosen is None:
        for c in candidates:
            if os.path.isdir(c):
                chosen = c
                break
    if chosen is None:
        chosen = path

    if not label:
        label = os.path.basename(os.path.normpath(chosen))
    return chosen, label, candidates


def bin_epoch_series(epoch_values, y_values, n_bins):
    if n_bins <= 0 or len(epoch_values) <= n_bins:
        labels = [str(int(e)) for e in epoch_values]
        return np.asarray(epoch_values, dtype=float), np.asarray(y_values, dtype=float), labels

    epochs = np.asarray(epoch_values, dtype=float)
    values = np.asarray(y_values, dtype=float)
    edges = np.linspace(epochs.min() - 0.5, epochs.max() + 0.5, n_bins + 1)
    centers, bvals, labels = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (epochs >= lo) & (epochs < hi)
        if not mask.any():
            continue
        centers.append(float((lo + hi) / 2.0))
        bvals.append(float(np.nanmean(values[mask])))
        labels.append(f"{int(np.ceil(lo + 0.5))}-{int(np.floor(hi - 0.5))}")
    return np.asarray(centers), np.asarray(bvals), labels


def load_all(args):
    rows = []
    for spec in args.exp:
        exp_dir, label, candidates = parse_exp_spec(spec, args.dataset)
        csv_path = _csv_path(exp_dir)
        if not os.path.exists(csv_path):
            print(f"SKIP {label}: no selected_midlevel_summary.csv found.")
            print("  Tried:")
            for c in candidates:
                print(f"    {_csv_path(c)}")
            continue
        df = pd.read_csv(csv_path)
        df["experiment"] = label
        df["experiment_dir"] = exp_dir
        rows.append(df)
        print(f"loaded {label} ({len(df)} rows from {csv_path})")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def make_matrix(df, metric, value_prefix, bin_epochs):
    col = f"{value_prefix}_{metric}"
    if col not in df.columns:
        return None, None, None

    labels, rows, common_labels = [], [], None
    common_centers = None
    for label, block in df.groupby("experiment", sort=False):
        block = block.sort_values("epoch")
        centers, values, tick_labels = bin_epoch_series(
            block["epoch"].to_numpy(),
            pd.to_numeric(block[col], errors="coerce").to_numpy(),
            bin_epochs,
        )
        if common_centers is None:
            common_centers = centers
            common_labels = tick_labels
        elif not np.array_equal(common_centers, centers):
            common_centers = np.union1d(common_centers, centers)
            common_labels = [str(int(round(e))) for e in common_centers]
        labels.append(label)
        rows.append((centers, values))

    matrix = np.full((len(labels), len(common_centers)), np.nan)
    for i, (centers, values) in enumerate(rows):
        idx = np.searchsorted(common_centers, centers)
        matrix[i, idx] = values
    return labels, common_labels, matrix


def plot_heatmap(df, metric, args):
    labels, tick_labels, matrix = make_matrix(df, metric, args.value, args.bin_epochs)
    if matrix is None:
        return
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return

    if args.value == "lift":
        vmax = float(np.nanmax(np.abs(finite)))
        vmin = -vmax
        cmap = "RdBu_r"
    else:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * matrix.shape[1] + 2),
                                    max(2.5, 0.42 * len(labels) + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Training epoch")
    title = f"Mid-level {args.value}: {metric}"
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{args.value}_{metric}")
    if args.value == "lift":
        ticks = sorted({vmin, 0.0, vmax})
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:+.3f}" if t != 0 else "0" for t in ticks])
    else:
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.3f}", f"{vmax:.3f}"])
    fig.tight_layout()
    out = os.path.join(args.output_dir, f"{args.value}_{metric}_heatmap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_line(df, metric, args):
    col = f"{args.value}_{metric}"
    if col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for label, block in df.groupby("experiment", sort=False):
        block = block.sort_values("epoch")
        centers, values, _ = bin_epoch_series(
            block["epoch"].to_numpy(),
            pd.to_numeric(block[col], errors="coerce").to_numpy(),
            args.bin_epochs,
        )
        ax.plot(centers, values, "o-", lw=1.5, ms=3, label=label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    if args.value == "lift":
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(col)
    title = f"Mid-level {args.value}: {metric}"
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = os.path.join(args.output_dir, f"{args.value}_{metric}_line.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_group_curves(df, group, metrics, value_prefix, args):
    available = [m for m in metrics if f"{value_prefix}_{m}" in df.columns]
    if not available:
        return
    fig, axes = plt.subplots(len(available), 1,
                             figsize=(9, max(3, 2.35 * len(available))),
                             sharex=True)
    if len(available) == 1:
        axes = [axes]

    plotted_any = False
    for ax, metric in zip(axes, available):
        col = f"{value_prefix}_{metric}"
        plotted = 0
        for label, block in df.groupby("experiment", sort=False):
            block = block.sort_values("epoch")
            centers, values, _ = bin_epoch_series(
                block["epoch"].to_numpy(),
                pd.to_numeric(block[col], errors="coerce").to_numpy(),
                args.bin_epochs,
            )
            if np.isfinite(values).any():
                ax.plot(centers, values, "o-", lw=1.4, ms=3, label=label)
                plotted += 1
        if value_prefix == "lift":
            ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_ylabel(metric, fontsize=8)
        ax.grid(alpha=0.3)
        if plotted:
            plotted_any = True
    if not plotted_any:
        plt.close(fig)
        return

    axes[-1].set_xlabel("Training epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)),
                   fontsize=8, bbox_to_anchor=(0.5, 0.995))
    title = f"{group.replace('_', ' ')} ({value_prefix})"
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    fig.suptitle(title, y=0.94 if handles else 0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91 if handles else 0.97))
    out = os.path.join(args.output_dir, f"{value_prefix}_{group}_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_all(args)
    if df.empty:
        print("No CSVs loaded; nothing to plot.")
        return

    out_csv = os.path.join(args.output_dir, "aggregated.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df)} rows)")

    for metric in args.metrics:
        if args.style in ("heatmap", "both"):
            plot_heatmap(df, metric, args)
        if args.style in ("line", "both"):
            plot_line(df, metric, args)

    if args.group_value != "none":
        for group, metrics in METRIC_GROUPS.items():
            plot_group_curves(df, group, metrics, args.group_value, args)

    print("Done.")


if __name__ == "__main__":
    main()
