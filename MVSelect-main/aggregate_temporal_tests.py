"""Overlay temporal_selection_test.py results from multiple experiments.

Reads `<exp>/temporal_test/temporal_test.csv` from each --exp path given on
the CLI, concatenates them with a per-experiment label, and saves four
overlay figures with a shared y-axis per metric:

    deviation_rotate.png         — rotation ±10° accuracy drop, one line per exp
    deviation_jitter.png         — color jitter accuracy drop, one line per exp
    deviation_rotate_jitter.png  — combined, one line per exp
    margin_over_time.png         — prediction margin over training, one line per exp
    aggregated.csv               — concatenated raw rows with `experiment` column

Each --exp can be a full meta_logs path or a bare experiment-folder name;
bare names are resolved against `meta_logs/<dataset>/`. Append `:LABEL` to
override the auto-derived (basename) legend label.

Usage:
    python3 aggregate_temporal_tests.py \\
        --exp resnet18steps3_..._e100:no_freeze \\
        --exp freeze_10_resnet18steps3_..._e100:freeze_10 \\
        --dataset rgb \\
        --output_dir compare/steps3_freeze_sweep

    # Without custom labels (basenames used as legend labels)
    python3 aggregate_temporal_tests.py \\
        --exp resnet18steps3_..._e100 \\
        --exp freeze_10_resnet18steps3_..._e100 \\
        --dataset rgb --output_dir compare/foo
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--exp", action="append", required=True,
                   help="One experiment to include. PATH or PATH:LABEL. Repeatable.")
    p.add_argument("--dataset", default="rgb",
                   help="Used to resolve bare experiment names under meta_logs/<dataset>/.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--title_suffix", default="",
                   help="Optional text appended to every plot title.")
    p.add_argument("--smooth", type=int, default=1,
                   help="Rolling-mean window over epochs (1 = no smoothing).")
    p.add_argument("--ymax_dev", type=float, default=None,
                   help="Force y-axis upper bound on the 3 deviation plots (%).")
    p.add_argument("--ymax_margin", type=float, default=None,
                   help="Force y-axis upper bound on the margin plot.")
    p.add_argument("--style", default="line",
                   choices=["line", "heatmap", "rank_stacked", "sorted_bars",
                            "both", "all"],
                   help="Plot style. 'line' = overlaid lines. 'heatmap' = "
                        "experiments × epochs grid. 'sorted_bars' = grouped "
                        "bars per epoch sorted left-to-right by value (y-axis "
                        "= actual value; same colour = same experiment, so a "
                        "colour changing position = a rank flip). 'rank_stacked' "
                        "= same idea but stacked vertically (y-axis is a sum, "
                        "less meaningful — kept for compact display). "
                        "'both' = line + heatmap. 'all' = every style.")
    p.add_argument("--bin_epochs", type=int, default=0,
                   help="Heatmap only: bin epochs into N columns (mean-aggregate "
                        "within each bin). 0 = no binning (use raw per-epoch values).")
    return p.parse_args()


def _has_csv(exp_dir):
    return os.path.exists(os.path.join(exp_dir, "temporal_test", "temporal_test.csv"))


def parse_exp_spec(spec, dataset):
    """Parse 'PATH' or 'PATH:LABEL' → (resolved_path, label, attempts).

    Resolution tries (in order): PATH as-is, then meta_logs/<dataset>/PATH.
    Prefers whichever candidate actually contains the temporal_test.csv we
    need — this matters when a stray folder of the same name exists in CWD
    but the real experiment lives under meta_logs/.

    Returns the chosen path, the legend label, and the list of paths tried
    (for diagnostic logging when nothing matches).
    """
    # Strip a trailing ':LABEL' iff the suffix has no path separator (so
    # absolute paths with a label are parsed correctly).
    path, label = spec, None
    if ":" in spec:
        prefix, _, suffix = spec.rpartition(":")
        if prefix and suffix and "/" not in suffix:
            path, label = prefix, suffix

    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join("meta_logs", dataset, path))

    # 1) any candidate with a CSV wins
    chosen = None
    for c in candidates:
        if _has_csv(c):
            chosen = c
            break
    # 2) otherwise pick the first existing directory so the error message is informative
    if chosen is None:
        for c in candidates:
            if os.path.isdir(c):
                chosen = c
                break
    if chosen is None:
        chosen = path  # last resort — will be reported as not found

    if label is None or label == "":
        label = os.path.basename(os.path.normpath(chosen))
    return chosen, label, candidates


def load_temporal_csv(exp_dir):
    csv_path = os.path.join(exp_dir, "temporal_test", "temporal_test.csv")
    if not os.path.exists(csv_path):
        return None, csv_path
    return pd.read_csv(csv_path), csv_path


def smooth_series(s, window):
    if window <= 1:
        return s
    return s.rolling(window=window, center=True, min_periods=1).mean()


# ----- Plots -----------------------------------------------------------------

def plot_deviation(df, condition, condition_label, out_path, args):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for exp_label, df_exp in df.groupby("experiment"):
        pivot = df_exp.pivot_table(index="epoch", columns="condition", values="accuracy")
        if condition not in pivot.columns or "none" not in pivot.columns:
            continue
        dev_pct = (pivot["none"] - pivot[condition]) * 100.0
        dev_pct = smooth_series(dev_pct, args.smooth)
        ax.plot(dev_pct.index, dev_pct.values, "o-", lw=1.5, ms=3,
                alpha=0.85, label=exp_label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Training epoch (selections from)")
    ax.set_ylabel("Accuracy drop (%) — clean minus perturbed")
    title = f"Manipulation robustness: {condition_label}"
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    if args.ymax_dev is not None:
        ax.set_ylim(top=args.ymax_dev)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _bin_matrix(epochs, values, n_bins):
    """Return (bin_centers, binned_values). n_bins=0 returns inputs unchanged."""
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


def plot_deviation_heatmap(df, condition, condition_label, out_path, args):
    """Heatmap: rows = experiments, columns = epochs (binned if requested),
    color = accuracy drop %. Diverging colormap centered at 0."""
    exp_labels, rows_of_values, all_epochs = [], [], None
    for exp_label, df_exp in df.groupby("experiment"):
        pivot = df_exp.pivot_table(index="epoch", columns="condition", values="accuracy")
        if condition not in pivot.columns or "none" not in pivot.columns:
            continue
        dev_pct = (pivot["none"] - pivot[condition]) * 100.0
        epochs_i = dev_pct.index.to_numpy()
        vals_i = dev_pct.to_numpy()
        epochs_i, vals_i = _bin_matrix(epochs_i, vals_i, args.bin_epochs)
        if all_epochs is None:
            all_epochs = epochs_i
        elif not np.array_equal(all_epochs, epochs_i):
            # If experiments have mismatched epoch coverage, fall back to a
            # union by reindexing onto a common axis (fill missing with NaN).
            all_epochs = np.union1d(all_epochs, epochs_i)
        exp_labels.append(exp_label)
        rows_of_values.append((epochs_i, vals_i))
    if not exp_labels:
        return

    # Align all rows to common epoch axis
    matrix = np.full((len(exp_labels), len(all_epochs)), np.nan)
    for i, (epochs_i, vals_i) in enumerate(rows_of_values):
        idx = np.searchsorted(all_epochs, epochs_i)
        matrix[i, idx] = vals_i

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    vmax = float(np.nanmax(np.abs(finite)))
    if args.ymax_dev is not None:
        vmax = args.ymax_dev

    fig, ax = plt.subplots(figsize=(max(8, 0.3 * len(all_epochs) + 2),
                                    max(2.5, 0.4 * len(exp_labels) + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(exp_labels)))
    ax.set_yticklabels(exp_labels, fontsize=8)
    n_xticks = min(len(all_epochs), 10)
    tick_idx = np.linspace(0, len(all_epochs) - 1, n_xticks).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{int(all_epochs[i])}" for i in tick_idx], fontsize=8)
    ax.set_xlabel("Training epoch (selections from)")
    title = f"Manipulation robustness: {condition_label} (% accuracy drop)"
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy drop (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_margin_heatmap(df, out_path, args):
    """Heatmap of mean prediction margin over epochs per experiment."""
    exp_labels, rows_of_values, all_epochs = [], [], None
    for exp_label, df_exp in df.groupby("experiment"):
        sub = df_exp[df_exp["condition"] == "none"].sort_values("epoch")
        if sub.empty:
            continue
        epochs_i = sub["epoch"].to_numpy()
        vals_i = sub["mean_margin"].to_numpy()
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
    vmax = float(args.ymax_margin) if args.ymax_margin is not None else float(np.nanmax(finite))
    vmin = float(np.nanmin(finite))

    fig, ax = plt.subplots(figsize=(max(8, 0.3 * len(all_epochs) + 2),
                                    max(2.5, 0.4 * len(exp_labels) + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(exp_labels)))
    ax.set_yticklabels(exp_labels, fontsize=8)
    n_xticks = min(len(all_epochs), 10)
    tick_idx = np.linspace(0, len(all_epochs) - 1, n_xticks).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{int(all_epochs[i])}" for i in tick_idx], fontsize=8)
    ax.set_xlabel("Training epoch (selections from)")
    title = "Prediction margin (top-1 − top-2 logit)"
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean margin")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _collect_matrix(df, value_extractor, args):
    """Return (exp_labels, epoch_axis, matrix[E, T]) for any series extractor.

    value_extractor: df_exp -> (epochs_array, values_array).
    """
    exp_labels, rows_of_values, all_epochs = [], [], None
    for exp_label, df_exp in df.groupby("experiment"):
        try:
            epochs_i, vals_i = value_extractor(df_exp)
        except KeyError:
            continue
        if epochs_i is None or len(epochs_i) == 0:
            continue
        epochs_i, vals_i = _bin_matrix(np.asarray(epochs_i), np.asarray(vals_i),
                                       args.bin_epochs)
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


def _rank_stacked_plot(matrix, exp_labels, epoch_axis, *,
                       title, ylabel, out_path, args):
    """Per-epoch stacked bar with segments sorted by value within each bar.

    Same color = same experiment across the chart. Segment order reshuffles
    when an experiment's value overtakes another's, so rank changes are
    visually salient.
    """
    if matrix is None or len(exp_labels) == 0:
        return
    n_exp, n_T = matrix.shape

    # Stable per-experiment colour palette (tab10/tab20 to keep colours
    # readable even with up to ~20 experiments).
    cmap_base = plt.cm.tab10 if n_exp <= 10 else plt.cm.tab20
    color_for = {lbl: cmap_base(i % cmap_base.N) for i, lbl in enumerate(exp_labels)}

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * n_T + 2),
                                    max(4.5, 0.3 * n_exp + 3)))

    for j in range(n_T):
        col = matrix[:, j]
        # Sort experiments by value desc (largest at bottom).
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
    ax.set_xticklabels([f"{int(epoch_axis[i])}" for i in tick_idx], fontsize=8)
    ax.set_xlabel("Training epoch (selections from)")
    ax.set_ylabel(ylabel)

    # Stable legend ordering (input order, not per-epoch rank).
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_for[lbl]) for lbl in exp_labels]
    ax.legend(legend_handles, exp_labels, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False,
              title="experiment (segment colour stays fixed; order = rank)")
    ttl = title
    if args.title_suffix:
        ttl += f"\n{args.title_suffix}"
    ax.set_title(ttl + "\n(rank-stacked: segment order = sorted by value at each epoch)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sorted_bars_plot(matrix, exp_labels, epoch_axis, *,
                      title, ylabel, out_path, args):
    """Grouped bars per epoch, ordered left-to-right by value (descending).

    Y-axis = actual value (no spurious sum). Same experiment = same colour
    everywhere; a colour changing horizontal position between adjacent epoch
    groups = a rank flip.
    """
    if matrix is None or len(exp_labels) == 0:
        return
    n_exp, n_T = matrix.shape

    cmap_base = plt.cm.tab10 if n_exp <= 10 else plt.cm.tab20
    color_for = {lbl: cmap_base(i % cmap_base.N) for i, lbl in enumerate(exp_labels)}

    # Width budget: leave a small gap between groups.
    group_width = 0.85
    bar_width = group_width / max(n_exp, 1)

    # Figure scales with n_T × n_exp to keep individual bars readable.
    fig_w = max(8.0, 0.18 * n_T * n_exp + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, max(4.5, 0.3 * n_exp + 3.0)))

    for j in range(n_T):
        col = matrix[:, j]
        # Sort descending; NaNs go last.
        order = np.argsort(-np.where(np.isnan(col), -np.inf, col))
        valid = [i for i in order if not np.isnan(col[i])]
        n_valid = len(valid)
        for pos_in_group, i in enumerate(valid):
            # Centre the (up to n_valid) bars around x=j.
            x = j + (pos_in_group - (n_valid - 1) / 2.0) * bar_width
            ax.bar(x, col[i], width=bar_width * 0.9,
                   color=color_for[exp_labels[i]],
                   edgecolor="white", linewidth=0.3)

    n_xticks = min(n_T, 10)
    tick_idx = np.linspace(0, n_T - 1, n_xticks).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{int(epoch_axis[i])}" for i in tick_idx], fontsize=8)
    ax.set_xlabel("Training epoch (selections from)")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_for[lbl]) for lbl in exp_labels]
    ax.legend(legend_handles, exp_labels, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False,
              title="experiment\n(within each epoch:\nleft = highest value)")
    title_text = title
    if args.title_suffix:
        title_text += f"\n{args.title_suffix}"
    ax.set_title(title_text + "\n(sorted-bars: within each epoch, "
                              "bars are placed left→right by descending value)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_deviation_sorted_bars(df, condition, condition_label, out_path, args):
    def extract(df_exp):
        pivot = df_exp.pivot_table(index="epoch", columns="condition", values="accuracy")
        if condition not in pivot.columns or "none" not in pivot.columns:
            return None, None
        dev_pct = (pivot["none"] - pivot[condition]) * 100.0
        return dev_pct.index.to_numpy(), dev_pct.to_numpy()
    exp_labels, epoch_axis, matrix = _collect_matrix(df, extract, args)
    _sorted_bars_plot(matrix, exp_labels, epoch_axis,
                      title=f"Manipulation robustness: {condition_label} (% accuracy drop)",
                      ylabel="Accuracy drop (%) — clean minus perturbed",
                      out_path=out_path, args=args)


def plot_margin_sorted_bars(df, out_path, args):
    def extract(df_exp):
        sub = df_exp[df_exp["condition"] == "none"].sort_values("epoch")
        if sub.empty:
            return None, None
        return sub["epoch"].to_numpy(), sub["mean_margin"].to_numpy()
    exp_labels, epoch_axis, matrix = _collect_matrix(df, extract, args)
    _sorted_bars_plot(matrix, exp_labels, epoch_axis,
                      title="Prediction margin (top-1 − top-2 logit)",
                      ylabel="Mean margin",
                      out_path=out_path, args=args)


def plot_deviation_rank_stacked(df, condition, condition_label, out_path, args):
    def extract(df_exp):
        pivot = df_exp.pivot_table(index="epoch", columns="condition", values="accuracy")
        if condition not in pivot.columns or "none" not in pivot.columns:
            return None, None
        dev_pct = (pivot["none"] - pivot[condition]) * 100.0
        return dev_pct.index.to_numpy(), dev_pct.to_numpy()
    exp_labels, epoch_axis, matrix = _collect_matrix(df, extract, args)
    _rank_stacked_plot(matrix, exp_labels, epoch_axis,
                       title=f"Manipulation robustness: {condition_label} (% accuracy drop)",
                       ylabel="Stacked accuracy drop (%) — sum of all experiments",
                       out_path=out_path, args=args)


def plot_margin_rank_stacked(df, out_path, args):
    def extract(df_exp):
        sub = df_exp[df_exp["condition"] == "none"].sort_values("epoch")
        if sub.empty:
            return None, None
        return sub["epoch"].to_numpy(), sub["mean_margin"].to_numpy()
    exp_labels, epoch_axis, matrix = _collect_matrix(df, extract, args)
    _rank_stacked_plot(matrix, exp_labels, epoch_axis,
                       title="Prediction margin (top-1 − top-2 logit)",
                       ylabel="Stacked mean margin — sum of all experiments",
                       out_path=out_path, args=args)


def plot_margin(df, out_path, args):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for exp_label, df_exp in df.groupby("experiment"):
        sub = df_exp[df_exp["condition"] == "none"].sort_values("epoch")
        if sub.empty:
            continue
        y = smooth_series(sub["mean_margin"], args.smooth)
        ax.plot(sub["epoch"].values, y.values, "o-", lw=1.5, ms=3,
                alpha=0.85, label=exp_label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("Training epoch (selections from)")
    ax.set_ylabel("Mean prediction margin (top-1 − top-2 logit)")
    title = "Prediction-margin stability on agent-selected views"
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    if args.ymax_margin is not None:
        ax.set_ylim(top=args.ymax_margin)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----- Main ------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for spec in args.exp:
        path, label, candidates = parse_exp_spec(spec, args.dataset)
        df, csv_path = load_temporal_csv(path)
        if df is None:
            tried = "\n    ".join(
                f"{c}/temporal_test/temporal_test.csv  "
                f"{'(exists)' if _has_csv(c) else '(missing)'}"
                for c in candidates
            )
            print(f"SKIP {label}: no temporal_test.csv found.\n"
                  f"  Tried:\n    {tried}", file=sys.stderr)
            continue
        df["experiment"] = label
        df["_source_dir"] = path
        rows.append(df)
        if "checkpoint_protocol" in df.columns:
            protocols = sorted(str(x) for x in df["checkpoint_protocol"].dropna().unique())
            suffix = f", protocol={','.join(protocols)}" if protocols else ""
        else:
            suffix = ", protocol=unknown_legacy_csv"
        print(f"loaded {label}  ({len(df)} rows from {csv_path}{suffix})")

    if not rows:
        sys.exit("No CSVs loaded; nothing to plot.")

    combined = pd.concat(rows, ignore_index=True)
    combined_path = os.path.join(args.output_dir, "aggregated.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nSaved: {combined_path}  ({len(combined)} total rows, "
          f"{combined['experiment'].nunique()} experiments)")

    do_line = args.style in ("line", "both", "all")
    do_heatmap = args.style in ("heatmap", "both", "all")
    do_rank_stacked = args.style in ("rank_stacked", "all")
    do_sorted_bars = args.style in ("sorted_bars", "all")

    for cond, cond_label in [
        ("rotate", "Rotation ±10°"),
        ("jitter", "Color jitter"),
        ("rotate_jitter", "Rotation + color jitter"),
    ]:
        if do_line:
            out = os.path.join(args.output_dir, f"deviation_{cond}.png")
            plot_deviation(combined, cond, cond_label, out, args)
            print(f"Saved: {out}")
        if do_heatmap:
            out = os.path.join(args.output_dir, f"deviation_{cond}_heatmap.png")
            plot_deviation_heatmap(combined, cond, cond_label, out, args)
            print(f"Saved: {out}")
        if do_rank_stacked:
            out = os.path.join(args.output_dir, f"deviation_{cond}_rank_stacked.png")
            plot_deviation_rank_stacked(combined, cond, cond_label, out, args)
            print(f"Saved: {out}")
        if do_sorted_bars:
            out = os.path.join(args.output_dir, f"deviation_{cond}_sorted_bars.png")
            plot_deviation_sorted_bars(combined, cond, cond_label, out, args)
            print(f"Saved: {out}")

    if do_line:
        margin_out = os.path.join(args.output_dir, "margin_over_time.png")
        plot_margin(combined, margin_out, args)
        print(f"Saved: {margin_out}")
    if do_heatmap:
        margin_out = os.path.join(args.output_dir, "margin_over_time_heatmap.png")
        plot_margin_heatmap(combined, margin_out, args)
        print(f"Saved: {margin_out}")
    if do_rank_stacked:
        margin_out = os.path.join(args.output_dir, "margin_over_time_rank_stacked.png")
        plot_margin_rank_stacked(combined, margin_out, args)
        print(f"Saved: {margin_out}")
    if do_sorted_bars:
        margin_out = os.path.join(args.output_dir, "margin_over_time_sorted_bars.png")
        plot_margin_sorted_bars(combined, margin_out, args)
        print(f"Saved: {margin_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
