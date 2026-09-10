#!/usr/bin/env python3
"""Plot active single-view bias, with an optional matched active-pair control."""

import argparse
import glob
import json
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mvselect_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VIEW_TYPES = [
    "expanded",
    "Expanded-like",
    "Foreshortened",
    "Foreshortened-like",
    "Remainder",
]
DISPLAY_NAMES = {
    "expanded": "Expanded",
    "Expanded-like": "Expanded-like",
    "Foreshortened": "Foreshortened",
    "Foreshortened-like": "Foreshortened-like",
    "Remainder": "Remainder",
    "expanded_family": "Expanded family",
    "foreshortened_family": "Foreshortened family",
    "accuracy": "Test accuracy",
}
PRIORS = {
    "expanded": 0.035,
    "Expanded-like": 0.140,
    "Foreshortened": 0.017,
    "Foreshortened-like": 0.070,
    "Remainder": 0.738,
}
PRIORS["expanded_family"] = PRIORS["expanded"] + PRIORS["Expanded-like"]
PRIORS["foreshortened_family"] = (
    PRIORS["Foreshortened"] + PRIORS["Foreshortened-like"])

CONDITIONS = ["active_pair", "active_single"]
CONDITION_LABELS = {
    "active_pair": "Active pair: initial + selected",
    "active_single": "Active single: selected only",
}
CONDITION_COLORS = {
    "active_pair": "#0072B2",
    "active_single": "#D55E00",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair_dir", default=None,
        help="Optional ordinary steps=1 experiment for a matched N=2 control.")
    parser.add_argument("--single_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--epoch_bins", type=int, default=10,
        help="Number of contiguous epoch groups in difference heatmaps.")
    args = parser.parse_args()
    if args.epoch_bins < 1:
        parser.error("--epoch_bins must be positive")
    return args


def resolve_experiment_dir(path):
    candidates = [path, os.path.join("meta_logs", "rgb", path)]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Experiment directory not found. Tried: {candidates}")


def load_condition(path, condition, expected_active):
    files = sorted(glob.glob(os.path.join(path, "*_meta.json")))
    if not files:
        raise FileNotFoundError(f"No *_meta.json files in {path}")
    runs = []
    for filename in files:
        with open(filename) as handle:
            payload = json.load(handle)
        missing = [key for key in VIEW_TYPES + ["accuracy"]
                   if key not in payload]
        if missing:
            print(f"SKIP {filename}: missing selection ratios {missing}")
            continue
        recorded_active = bool(payload.get("active_single_view", False))
        if recorded_active != expected_active:
            raise ValueError(
                f"{filename} has active_single_view={recorded_active}; "
                f"expected {expected_active} for {condition}.")
        recorded_steps = payload.get("steps")
        if recorded_steps is not None and int(recorded_steps) != 1:
            raise ValueError(
                f"{filename} has steps={recorded_steps}; this comparison "
                "requires matched one-action (steps=1) runs.")
        expected_input = (
            "selected_view_only"
            if expected_active
            else "initial_plus_selected_views"
        )
        recorded_input = payload.get("recognition_input")
        if recorded_input is not None and recorded_input != expected_input:
            raise ValueError(
                f"{filename} has recognition_input={recorded_input!r}; "
                f"expected {expected_input!r}.")
        length = min(len(payload[key]) for key in VIEW_TYPES + ["accuracy"])
        if length < 1:
            continue
        run = pd.DataFrame({
            "epoch": np.arange(1, length + 1),
            **{
                key: np.asarray(payload[key][:length], dtype=float)
                for key in VIEW_TYPES
            },
        })
        run["expanded_family"] = (
            run["expanded"] + run["Expanded-like"])
        run["foreshortened_family"] = (
            run["Foreshortened"] + run["Foreshortened-like"])
        prior = np.asarray([PRIORS[key] for key in VIEW_TYPES])
        shares = run[VIEW_TYPES].to_numpy(dtype=float)
        run["view_bias_tvd"] = 0.5 * np.abs(shares - prior).sum(axis=1)
        run["accuracy"] = np.asarray(payload["accuracy"][:length], dtype=float)
        run["condition"] = condition
        run["run"] = os.path.basename(filename)
        run["seed"] = payload.get("seed")
        run["initialization"] = payload.get("initialization", "unknown")
        runs.append(run)
    if not runs:
        raise ValueError(f"No usable metadata runs in {path}")
    print(f"Loaded {len(runs)} {condition} runs from {path}")
    return pd.concat(runs, ignore_index=True)


def mean_sem(frame, metric):
    grouped = frame.groupby(["condition", "epoch"])[metric]
    summary = grouped.agg(["mean", "count", "std"]).reset_index()
    summary["sem"] = (
        summary["std"].fillna(0.0)
        / np.sqrt(summary["count"].clip(lower=1)))
    return summary


def metric_limits(frame, metric, prior):
    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if prior is not None:
        values = np.append(values, prior)
    lower, upper = float(values.min()), float(values.max())
    span = max(upper - lower, 0.01)
    return max(0.0, lower - 0.10 * span), min(1.0, upper + 0.10 * span)


def draw_metric(axis, frame, metric, title, prior=None, y_limits=None):
    summary = mean_sem(frame, metric)
    conditions = [
        condition for condition in CONDITIONS
        if condition in set(summary["condition"])
    ]
    for condition in conditions:
        block = summary[summary["condition"] == condition]
        x = block["epoch"].to_numpy()
        y = block["mean"].to_numpy()
        sem = block["sem"].to_numpy()
        color = CONDITION_COLORS[condition]
        axis.plot(
            x, y, color=color, lw=1.8,
            label=CONDITION_LABELS[condition])
        axis.fill_between(x, y - sem, y + sem, color=color, alpha=0.16)
    if prior is not None:
        axis.axhline(
            prior, color="#555555", ls=":", lw=1.1,
            label="Availability baseline")
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Training epoch")
    axis.set_ylabel("Selection share")
    if y_limits is not None:
        axis.set_ylim(*y_limits)
    axis.grid(axis="y", alpha=0.20)


def plot_metric_grid(frame, metrics, output_path, title):
    n_columns = 2 if len(metrics) > 3 else len(metrics)
    n_rows = int(np.ceil(len(metrics) / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(6.2 * n_columns, 3.7 * n_rows),
        squeeze=False,
    )
    for axis, metric in zip(axes.flat, metrics):
        limits = metric_limits(frame, metric, PRIORS[metric])
        draw_metric(
            axis, frame, metric, DISPLAY_NAMES[metric],
            prior=PRIORS[metric], y_limits=limits)
    for axis in axes.flat[len(metrics):]:
        axis.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3,
        frameon=False, fontsize=8)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.97))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_bias_magnitude(frame, output_path):
    fig, axis = plt.subplots(figsize=(8.6, 4.8))
    limits = metric_limits(frame, "view_bias_tvd", None)
    draw_metric(
        axis,
        frame,
        "view_bias_tvd",
        "Overall view-selection bias",
        y_limits=limits,
    )
    axis.set_ylabel("Total variation distance from availability prior")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_accuracy(frame, output_path):
    fig, axis = plt.subplots(figsize=(8.6, 4.8))
    summary = mean_sem(frame, "accuracy")
    conditions = [
        condition for condition in CONDITIONS
        if condition in set(summary["condition"])
    ]
    for condition in conditions:
        block = summary[summary["condition"] == condition]
        x = block["epoch"].to_numpy()
        y = block["mean"].to_numpy()
        sem = block["sem"].to_numpy()
        color = CONDITION_COLORS[condition]
        axis.plot(x, y, color=color, lw=1.8,
                  label=CONDITION_LABELS[condition])
        axis.fill_between(x, y - sem, y + sem, color=color, alpha=0.16)
    axis.set_xlabel("Training epoch")
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title(
        "Active single-view recognition performance"
        if len(conditions) == 1 else
        "Recognition performance under the matched action budget")
    axis.grid(axis="y", alpha=0.20)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def bin_difference(frame, metrics, n_bins):
    epoch_sets = [
        set(frame.loc[frame["condition"] == condition, "epoch"])
        for condition in CONDITIONS
    ]
    common_epochs = sorted(epoch_sets[0].intersection(*epoch_sets[1:]))
    if not common_epochs:
        raise ValueError("The two conditions have no epochs in common")
    groups = [group for group in np.array_split(common_epochs, n_bins)
              if len(group)]
    rows = []
    for metric in metrics:
        summary = mean_sem(frame, metric)
        pivot = summary.pivot(
            index="epoch", columns="condition", values="mean")
        for bin_index, epochs in enumerate(groups, start=1):
            block = pivot.reindex(epochs)
            difference = (
                block["active_single"] - block["active_pair"])
            rows.append({
                "metric": metric,
                "epoch_bin": bin_index,
                "epoch_start": int(epochs[0]),
                "epoch_end": int(epochs[-1]),
                "single_minus_pair_percentage_points": (
                    100.0 * float(difference.mean())),
            })
    return pd.DataFrame(rows)


def plot_difference_heatmap(differences, metrics, output_path):
    pivot = differences.pivot(
        index="metric", columns="epoch_bin",
        values="single_minus_pair_percentage_points").reindex(metrics)
    matrix = pivot.to_numpy(dtype=float)
    limit = max(float(np.nanmax(np.abs(matrix))), 1e-6)
    bins = (
        differences.drop_duplicates("epoch_bin")
        .sort_values("epoch_bin"))
    labels = [
        f"{int(row.epoch_start)}-{int(row.epoch_end)}"
        for row in bins.itertuples()
    ]
    fig, axis = plt.subplots(
        figsize=(max(9.0, 0.75 * len(labels) + 3.0),
                 max(4.6, 0.55 * len(metrics) + 1.8)))
    image = axis.imshow(
        matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_yticks(np.arange(len(metrics)))
    axis.set_yticklabels([DISPLAY_NAMES[metric] for metric in metrics])
    axis.set_xlabel("Training epoch range")
    axis.set_title("Active single minus active pair selection share")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column, row, f"{value:+.1f}", ha="center", va="center",
                fontsize=8,
                color="white" if abs(value) > 0.55 * limit else "black")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Selection-share difference (percentage points)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    single_dir = resolve_experiment_dir(args.single_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    frames = [
        load_condition(single_dir, "active_single", expected_active=True)
    ]
    if args.pair_dir:
        pair_dir = resolve_experiment_dir(args.pair_dir)
        frames.insert(0, load_condition(
            pair_dir, "active_pair", expected_active=False))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(os.path.join(output_dir, "selection_bias_runs.csv"), index=False)

    has_pair = args.pair_dir is not None
    type_filename = (
        "active_single_vs_pair_view_types.png"
        if has_pair else "active_single_view_types.png"
    )
    family_filename = (
        "active_single_vs_pair_families.png"
        if has_pair else "active_single_families.png"
    )
    plot_metric_grid(
        frame,
        VIEW_TYPES,
        os.path.join(output_dir, type_filename),
        ("Active single-view versus active-pair selection"
         if has_pair else
         "View selection without multiview recognition pooling"),
    )
    plot_metric_grid(
        frame,
        ["expanded_family", "foreshortened_family", "Remainder"],
        os.path.join(output_dir, family_filename),
        ("Active single-view versus active-pair selection by view family"
         if has_pair else
         "Active single-view selection by view family"),
    )
    plot_bias_magnitude(
        frame, os.path.join(
            output_dir,
            ("view_bias_magnitude.png" if has_pair else
             "active_single_view_bias_magnitude.png")))
    plot_accuracy(
        frame, os.path.join(
            output_dir,
            ("recognition_accuracy.png" if has_pair else
             "active_single_recognition_accuracy.png")))

    if has_pair:
        heatmap_metrics = VIEW_TYPES + [
            "expanded_family", "foreshortened_family"]
        differences = bin_difference(frame, heatmap_metrics, args.epoch_bins)
        differences.to_csv(
            os.path.join(output_dir, "active_single_minus_pair.csv"),
            index=False)
        plot_difference_heatmap(
            differences,
            heatmap_metrics,
            os.path.join(
                output_dir,
                "active_single_minus_pair_selection_heatmap.png"),
        )
    print(f"Saved active-single comparison to {output_dir}")


if __name__ == "__main__":
    main()
