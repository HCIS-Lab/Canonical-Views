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
    "ellipse_orientation_deg",
    "ellipse_aspect_ratio",
    "bbox_aspect_ratio",
    "skeleton_length_px",
    "skeleton_elongation",
    "skeleton_length_norm",
    "bilateral_symmetry",
    "medial_axis_symmetry",
    "skeleton_endpoint_count",
    "skeleton_branchpoint_count",
    "skeleton_branch_density",
    "dominant_edge_orientation_deg",
    "edge_anisotropy",
    "edge_entropy",
    "edge_pixel_count",
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

ORIENTATION_METRICS = {
    "ellipse_orientation_deg",
    "dominant_edge_orientation_deg",
}

PRIMARY_RAW_METRICS = [
    "ellipse_aspect_ratio",
    "bilateral_symmetry",
    "edge_entropy",
]

PRIMARY_RAW_LABELS = {
    "ellipse_aspect_ratio": "Ellipse aspect ratio",
    "bilateral_symmetry": "Bilateral symmetry",
    "edge_entropy": "Edge entropy",
}

VIEW_BUCKETS = [
    "expanded",
    "Expanded-like",
    "Foreshortened",
    "Foreshortened-like",
    "Remainder",
]

VIEW_BUCKET_LABELS = {
    "expanded": "Expanded",
    "Expanded-like": "Expanded-like",
    "Foreshortened": "Foreshortened",
    "Foreshortened-like": "Foreshortened-like",
    "Remainder": "Remainder",
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
    p.add_argument("--value", default="effect",
                   choices=["selected", "lift", "baseline", "effect"],
                   help="Plot selected_<metric>, lift_<metric>, baseline_<metric>, "
                        "or effect_<metric>.")
    p.add_argument("--group_value", default="all",
                   choices=["selected", "lift", "baseline", "effect", "both", "all", "none"],
                   help="Value prefix for the grouped comparison curve figures. "
                        "'both' writes selected and lift; 'all' writes selected, "
                        "lift, and effect. "
                        "Use 'none' to disable group figures.")
    p.add_argument("--style", default="heatmap",
                   choices=["heatmap", "line", "both"])
    p.add_argument("--bin_epochs", type=int, default=10,
                   help="Bin epochs into N columns. 0 = raw epochs.")
    p.add_argument("--title_suffix", default="")
    return p.parse_args()


def _csv_path(exp_dir):
    return os.path.join(exp_dir, "midlevel_features", "selected_midlevel_summary.csv")


def _association_csv_path(exp_dir):
    return os.path.join(
        exp_dir, "midlevel_features", "view_type_midlevel_associations.csv")


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


def _mean_values(values, metric=None, value_prefix=None):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.nan
    if metric in ORIENTATION_METRICS and value_prefix in {"selected", "baseline"}:
        theta = np.deg2rad(values)
        z = np.mean(np.exp(2j * theta))
        return float((0.5 * np.rad2deg(np.angle(z))) % 180.0)
    return float(np.nanmean(values))


def bin_epoch_series(epoch_values, y_values, n_bins, metric=None, value_prefix=None):
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
        bvals.append(_mean_values(values[mask], metric, value_prefix))
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


def load_all_view_type_associations(args):
    rows = []
    for spec in args.exp:
        exp_dir, label, _ = parse_exp_spec(spec, args.dataset)
        if label != "no_freeze":
            continue
        csv_path = _association_csv_path(exp_dir)
        if not os.path.exists(csv_path):
            print(f"SKIP {label} view-type associations: no {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["experiment"] = label
        df["experiment_dir"] = exp_dir
        rows.append(df)
        print(f"loaded {label} view-type associations "
              f"({len(df)} rows from {csv_path})")
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
            metric=metric,
            value_prefix=value_prefix,
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

    if args.value == "effect":
        vmin, vmax = -1.0, 1.0
        cmap = "RdBu_r"
    elif args.value == "lift":
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
    extend = "both" if args.value == "effect" and (
        np.nanmin(finite) < vmin or np.nanmax(finite) > vmax
    ) else "neither"
    cbar = fig.colorbar(im, ax=ax, extend=extend)
    cbar.set_label(f"{args.value}_{metric}")
    if args.value == "effect":
        cbar.set_ticks([-1.0, 0.0, 1.0])
        cbar.set_ticklabels(["-1", "0", "+1"])
    elif args.value == "lift":
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
            metric=metric,
            value_prefix=args.value,
        )
        ax.plot(centers, values, "o-", lw=1.5, ms=3, label=label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    if args.value in {"lift", "effect"}:
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
                metric=metric,
                value_prefix=value_prefix,
            )
            if np.isfinite(values).any():
                ax.plot(centers, values, "o-", lw=1.4, ms=3, label=label)
                plotted += 1
        if value_prefix in {"lift", "effect"}:
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


def _mean_correlations(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return np.nan
    clipped = np.clip(values.to_numpy(dtype=float), -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def make_long_association_matrix(df, row_specs, value_col, bin_epochs,
                                 correlation=False):
    rows = []
    common_centers = None
    common_labels = None
    kept_labels = []
    for row_label, selectors in row_specs:
        block = df
        for column, value in selectors.items():
            block = block[block[column] == value]
        if block.empty:
            continue
        aggregate = _mean_correlations if correlation else "mean"
        epoch_values = block.groupby("epoch", sort=True)[value_col].agg(
            aggregate).reset_index()
        centers, values, tick_labels = bin_epoch_series(
            epoch_values["epoch"].to_numpy(),
            pd.to_numeric(epoch_values[value_col], errors="coerce").to_numpy(),
            bin_epochs,
        )
        if common_centers is None:
            common_centers = centers
            common_labels = tick_labels
        elif not np.array_equal(common_centers, centers):
            common_centers = np.union1d(common_centers, centers)
            common_labels = [str(int(round(epoch))) for epoch in common_centers]
        kept_labels.append(row_label)
        rows.append((centers, values))
    if not rows:
        return [], [], np.empty((0, 0))
    matrix = np.full((len(rows), len(common_centers)), np.nan)
    for row_idx, (centers, values) in enumerate(rows):
        indices = np.searchsorted(common_centers, centers)
        matrix[row_idx, indices] = values
    return kept_labels, common_labels, matrix


def draw_association_comparison_heatmap(matrix, row_labels, column_labels,
                                        title, colorbar_label, out_path,
                                        vmin, vmax, cmap):
    if matrix.size == 0 or not np.isfinite(matrix).any():
        return
    label_width = min(5.0, 0.05 * max(len(str(label)) for label in row_labels))
    fig, ax = plt.subplots(
        figsize=(max(8, 0.7 * len(column_labels) + 3.0 + label_width),
                 max(3.0, 0.36 * len(row_labels) + 1.8)),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("Selection epoch")
    ax.set_title(title)
    if len(row_labels) <= 35:
        threshold = max(abs(vmin), abs(vmax)) * 0.55
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if not np.isfinite(value):
                    continue
                color = "white" if abs(value) > threshold else "black"
                ax.text(col_idx, row_idx, f"{value:+.2f}", ha="center",
                        va="center", fontsize=6, color=color)
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_cross_experiment_view_type_associations(df, args):
    if df.empty:
        return
    experiments = list(dict.fromkeys(df["experiment"].astype(str)))
    for metric in PRIMARY_RAW_METRICS:
        metric_df = df[df["metric"] == metric]
        if metric_df.empty:
            continue

        eta_df = metric_df.drop_duplicates(
            ["experiment", "run", "epoch", "metric"])
        eta_specs = [
            (experiment, {"experiment": experiment})
            for experiment in experiments
        ]
        labels, columns, matrix = make_long_association_matrix(
            eta_df, eta_specs, "eta_squared", args.bin_epochs)
        draw_association_comparison_heatmap(
            matrix,
            labels,
            columns,
            f"Exact view-type association: {PRIMARY_RAW_LABELS[metric]}",
            "Eta-squared (five view types; unsigned)",
            os.path.join(
                args.output_dir,
                f"view_type_eta_squared_{metric}_heatmap.png"),
            0.0,
            1.0,
            "viridis",
        )

        family_specs = [
            (
                f"{experiment} | {VIEW_BUCKET_LABELS[view_type]}",
                {"experiment": experiment, "view_type": view_type},
            )
            for experiment in experiments
            for view_type in VIEW_BUCKETS
        ]
        labels, columns, matrix = make_long_association_matrix(
            metric_df,
            family_specs,
            "point_biserial",
            args.bin_epochs,
            correlation=True,
        )
        draw_association_comparison_heatmap(
            matrix,
            labels,
            columns,
            f"Signed view-type association: {PRIMARY_RAW_LABELS[metric]}",
            "Point-biserial correlation (type vs all other types)",
            os.path.join(
                args.output_dir,
                f"view_type_point_biserial_{metric}_heatmap.png"),
            -1.0,
            1.0,
            "RdBu_r",
        )

        labels, columns, matrix = make_long_association_matrix(
            metric_df, family_specs, "family_mean", args.bin_epochs)
        finite = matrix[np.isfinite(matrix)]
        if not finite.size:
            continue
        vmin, vmax = float(finite.min()), float(finite.max())
        if np.isclose(vmin, vmax):
            padding = max(abs(vmin) * 0.01, 1e-6)
            vmin, vmax = vmin - padding, vmax + padding
        draw_association_comparison_heatmap(
            matrix,
            labels,
            columns,
            f"Raw selected-view {PRIMARY_RAW_LABELS[metric]} by view type",
            f"Mean {PRIMARY_RAW_LABELS[metric]} (raw units)",
            os.path.join(
                args.output_dir,
                f"view_type_raw_{metric}_heatmap.png"),
            vmin,
            vmax,
            "viridis",
        )


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
        if args.group_value == "both":
            group_values = ["selected", "lift"]
        elif args.group_value == "all":
            group_values = ["selected", "lift", "effect"]
        else:
            group_values = [args.group_value]
        for value_prefix in group_values:
            for group, metrics in METRIC_GROUPS.items():
                plot_group_curves(df, group, metrics, value_prefix, args)

    association_df = load_all_view_type_associations(args)
    if not association_df.empty:
        association_out = os.path.join(
            args.output_dir, "view_type_midlevel_associations.csv")
        association_df.to_csv(association_out, index=False)
        print(f"Saved: {association_out} ({len(association_df)} rows)")
        plot_cross_experiment_view_type_associations(association_df, args)

    print("Done.")


if __name__ == "__main__":
    main()
