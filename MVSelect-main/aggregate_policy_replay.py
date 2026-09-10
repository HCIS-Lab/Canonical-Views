#!/usr/bin/env python3
"""Aggregate matched policy-replay recognition runs and compare features."""

import argparse
import glob
import json
import os
import re
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mvselect_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = [
    "frozen_10",
    "frozen_20",
    "frozen_30",
    "final_from_start",
    "evolving",
    "shuffled",
    "family_matched_random",
    "random_views",
    "random_warmup_10",
    "random_warmup_20",
    "random_warmup_30",
    "joint_no_freeze_reference",
]
DISPLAY_NAMES = {
    "frozen_10": "Epoch-10 selections throughout",
    "frozen_20": "Epoch-20 selections throughout",
    "frozen_30": "Epoch-30 selections throughout",
    "final_from_start": "Final policy from start",
    "evolving": "Naturally evolving",
    "shuffled": "Temporally shuffled",
    "family_matched_random": "Family-matched replacement views",
    "random_views": "Random views (fresh recognizer)",
    "random_warmup_10": "Random first 10, then evolving",
    "random_warmup_20": "Random first 20, then evolving",
    "random_warmup_30": "Random first 30, then evolving",
    "joint_no_freeze_reference": "Joint no-freeze (external reference)",
}
CONDITION_ALIASES = {
    "frozen_policy_e10": "frozen_10",
    "frozen_policy_e20": "frozen_20",
    "frozen_policy_e30": "frozen_30",
    "policy_e10": "frozen_10",
    "policy_e20": "frozen_20",
    "policy_e30": "frozen_30",
}
COLORS = {
    "frozen_10": "#56B4E9",
    "frozen_20": "#0072B2",
    "frozen_30": "#332288",
    "final_from_start": "#D55E00",
    "evolving": "#009E73",
    "shuffled": "#CC79A7",
    "family_matched_random": "#777777",
    "random_views": "#A6761D",
    "random_warmup_10": "#88CCEE",
    "random_warmup_20": "#44AA99",
    "random_warmup_30": "#117733",
    "joint_no_freeze_reference": "#000000",
}
LINE_STYLES = {
    "random_warmup_10": "--",
    "random_warmup_20": "-.",
    "random_warmup_30": ":",
}
CURVE_GROUPS = {
    "policy_history": [
        "frozen_10", "frozen_20", "frozen_30", "final_from_start",
        "evolving", "shuffled", "joint_no_freeze_reference",
    ],
    "random_warmup": [
        "evolving", "random_views", "random_warmup_10",
        "random_warmup_20", "random_warmup_30",
        "joint_no_freeze_reference",
    ],
    "identity_controls": [
        "evolving", "family_matched_random", "random_views",
        "joint_no_freeze_reference",
    ],
}
METRIC_SPECS = [
    ("accuracy_pct", "Held-out accuracy (%)", "accuracy_over_training", ".1f"),
    ("prediction_margin", "Mean prediction margin", "margin_over_training", ".3f"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--required_conditions", nargs="*", default=None,
        help=(
            "Condition IDs required for every replay seed. Missing enabled "
            "controls stop aggregation instead of disappearing from plots."
        ),
    )
    return parser.parse_args()


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    sem = (
        float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) > 1 else 0.0
    )
    return mean, sem


def ordered_conditions(values):
    values = list(dict.fromkeys(values))
    return [value for value in CONDITION_ORDER if value in values] + [
        value for value in values if value not in CONDITION_ORDER]


def canonical_condition(value):
    """Map legacy display-like condition IDs to stable internal IDs."""
    value = str(value)
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return CONDITION_ALIASES.get(normalized, value)


def linear_cka(features_a, features_b):
    x = np.asarray(features_a, dtype=np.float64)
    y = np.asarray(features_b, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = (
        np.linalg.norm(x.T @ x, ord="fro")
        * np.linalg.norm(y.T @ y, ord="fro")
    )
    return float(numerator / denominator) if denominator > 0 else float("nan")


def load_feature_file(path):
    data = np.load(path)
    labels = np.asarray(data["labels"]).astype(int)
    instance_ids = np.asarray(data["instance_ids"]).astype(str)
    keys = np.asarray([
        f"{label}|{instance_id}"
        for label, instance_id in zip(labels, instance_ids)
    ])
    order = np.argsort(keys)
    return {
        "keys": keys[order],
        "features": np.asarray(data["features"])[order],
        "logits": np.asarray(data["logits"])[order],
        "labels": labels[order],
    }


def aligned_arrays(left, right):
    common, left_idx, right_idx = np.intersect1d(
        left["keys"], right["keys"], return_indices=True)
    if len(common) == 0:
        raise ValueError("Feature files have no common held-out instances")
    return left_idx, right_idx


def metric_limits(metrics, metric):
    values = pd.to_numeric(metrics[metric], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    lower, upper = float(values.min()), float(values.max())
    padding = max(0.03 * (upper - lower), 0.02 if upper == lower else 0.0)
    return lower - padding, upper + padding


def plot_curve_set(metrics, conditions, metric, ylabel, path, y_limits):
    available = set(metrics["condition"])
    conditions = [
        condition for condition in conditions if condition in available]
    if len(conditions) < 2:
        return
    fig, ax = plt.subplots(
        figsize=(max(9.2, 0.72 * len(conditions)), 5.4))
    for condition in conditions:
        subset = metrics[metrics["condition"] == condition]
        grouped = subset.groupby("epoch")[metric]
        epochs, means, sems = [], [], []
        for epoch, values in grouped:
            mean, sem = mean_sem(values)
            epochs.append(epoch)
            means.append(mean)
            sems.append(sem)
        means = np.asarray(means)
        sems = np.asarray(sems)
        color = COLORS.get(condition)
        is_reference = condition == "joint_no_freeze_reference"
        ax.plot(epochs, means, marker="o", ms=3, color=color,
                linestyle=(
                    "--" if is_reference
                    else LINE_STYLES.get(condition, "-")),
                linewidth=2.2 if is_reference else 1.5,
                label=DISPLAY_NAMES.get(condition, condition))
        ax.fill_between(epochs, means - sems, means + sems,
                        color=color, alpha=0.16)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.legend(fontsize=8, ncol=min(3, max(1, len(conditions))))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_curves(metrics, conditions, output_dir):
    for metric, ylabel, stem, _ in METRIC_SPECS:
        y_limits = metric_limits(metrics, metric)
        plot_curve_set(
            metrics, conditions, metric, ylabel,
            os.path.join(output_dir, f"{stem}.png"), y_limits)
        for group_name, group_conditions in CURVE_GROUPS.items():
            plot_curve_set(
                metrics,
                group_conditions,
                metric,
                ylabel,
                os.path.join(
                    output_dir, f"{stem}_{group_name}_curves.png"),
                y_limits,
            )


def annotate_heatmap(ax, matrix, value_format, center=None):
    finite = matrix[np.isfinite(matrix)]
    if not len(finite):
        return
    midpoint = (
        float(center) if center is not None
        else float((finite.min() + finite.max()) / 2.0)
    )
    max_deviation = (
        float(np.max(np.abs(finite - midpoint)))
        if center is not None else None
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isfinite(value):
                continue
            label = format(value, value_format)
            if center is not None and value > 0:
                label = "+" + label
            if center is None:
                text_color = "white" if value < midpoint else "black"
            else:
                text_color = (
                    "white"
                    if max_deviation and abs(value - midpoint) > 0.55 * max_deviation
                    else "black"
                )
            ax.text(
                column, row, label,
                ha="center", va="center", fontsize=7,
                color=text_color,
            )


def plot_training_heatmaps(metrics, conditions, output_dir):
    for metric, ylabel, stem, value_format in METRIC_SPECS:
        pivot = metrics.pivot_table(
            index="condition", columns="epoch", values=metric,
            aggfunc="mean").reindex(conditions)
        epochs = list(pivot.columns)
        matrix = pivot.to_numpy(dtype=float)
        width = max(9.5, 0.72 * len(epochs) + 4.0)
        height = max(5.8, 0.48 * len(conditions) + 2.0)

        fig, ax = plt.subplots(figsize=(width, height))
        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(len(epochs)))
        ax.set_xticklabels([str(int(epoch)) for epoch in epochs])
        ax.set_yticks(np.arange(len(conditions)))
        ax.set_yticklabels([
            DISPLAY_NAMES.get(condition, condition)
            for condition in conditions
        ])
        ax.set_xlabel("Training epoch")
        ax.set_title(f"{ylabel} (mean across seeds)")
        annotate_heatmap(ax, matrix, value_format)
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(ylabel)
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"{stem}_heatmap.png"), dpi=220)
        plt.close(fig)

        if "evolving" not in pivot.index:
            continue
        evolving = pivot.loc["evolving"].to_numpy(dtype=float)
        delta = matrix - evolving[np.newaxis, :]
        finite = delta[np.isfinite(delta)]
        limit = max(
            float(np.max(np.abs(finite))) if len(finite) else 0.0, 1e-9)
        fig, ax = plt.subplots(figsize=(width, height))
        image = ax.imshow(
            delta, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(np.arange(len(epochs)))
        ax.set_xticklabels([str(int(epoch)) for epoch in epochs])
        ax.set_yticks(np.arange(len(conditions)))
        ax.set_yticklabels([
            DISPLAY_NAMES.get(condition, condition)
            for condition in conditions
        ])
        ax.set_xlabel("Training epoch")
        ax.set_title(f"{ylabel}: difference from naturally evolving replay")
        annotate_heatmap(ax, delta, value_format, center=0.0)
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(f"Difference in {ylabel.lower()}")
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                output_dir, f"{stem}_delta_vs_evolving_heatmap.png"),
            dpi=220,
        )
        plt.close(fig)


def plot_run_curve(metrics, condition, run_dir):
    """Refresh a completed run's curve using the current display labels."""
    evaluated = metrics[metrics["phase"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].plot(evaluated["epoch"], evaluated["accuracy_pct"], marker="o")
    axes[0].set(xlabel="Recognition epoch", ylabel="Held-out accuracy (%)")
    axes[1].plot(
        evaluated["epoch"], evaluated["prediction_margin"], marker="o")
    axes[1].set(
        xlabel="Recognition epoch", ylabel="Mean prediction margin")
    fig.suptitle(DISPLAY_NAMES.get(condition, condition))
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "training_curve.png"), dpi=180)
    plt.close(fig)


def plot_final_bars(summaries, conditions, output_dir):
    specifications = [
        ("final_accuracy_pct", "Held-out accuracy (%)", "final_accuracy.png"),
        ("final_class_silhouette", "Class silhouette (cosine)",
         "final_class_silhouette.png"),
        ("final_vs_initial_cka", "Linear CKA: final vs shared initialization",
         "final_vs_initial_cka.png"),
    ]
    for metric, ylabel, filename in specifications:
        metric_conditions = []
        means, sems = [], []
        for condition in conditions:
            values = summaries.loc[summaries["condition"] == condition, metric]
            mean, sem = mean_sem(values.dropna())
            if not np.isfinite(mean):
                continue
            metric_conditions.append(condition)
            means.append(mean)
            sems.append(sem)
        x = np.arange(len(metric_conditions))
        fig, ax = plt.subplots(
            figsize=(max(9.2, 0.82 * len(metric_conditions)), 5.0))
        ax.bar(x, means, yerr=sems,
               color=[COLORS.get(condition, "#777777")
                      for condition in metric_conditions],
               capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [DISPLAY_NAMES.get(condition, condition)
             for condition in metric_conditions],
            rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        if (metric == "final_vs_initial_cka"
                and "joint_no_freeze_reference" in conditions
                and "joint_no_freeze_reference" not in metric_conditions):
            ax.set_title(
                "Joint reference omitted: original epoch-0 checkpoint unavailable",
                fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=200)
        plt.close(fig)


def plot_matrix(matrix, conditions, title, colorbar_label, path,
                vmin=None, vmax=None, cmap="viridis"):
    side = max(7.4, 0.62 * len(conditions) + 2.8)
    fig, ax = plt.subplots(figsize=(side, side - 0.7))
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    labels = [DISPLAY_NAMES.get(condition, condition) for condition in conditions]
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for row in range(len(conditions)):
        for column in range(len(conditions)):
            value = matrix[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.3f}", ha="center", va="center",
                        color="white" if value < np.nanmean(matrix) else "black",
                        fontsize=8)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    input_root = os.path.abspath(args.input_root)
    output_dir = os.path.abspath(args.output_dir or input_root)
    os.makedirs(output_dir, exist_ok=True)
    summary_paths = sorted(glob.glob(
        os.path.join(input_root, "**", "run_summary.json"), recursive=True))
    if not summary_paths:
        raise SystemExit(f"No run_summary.json files under {input_root}")

    summaries = []
    metric_frames = []
    feature_sets = {}
    for summary_path in summary_paths:
        run_dir = os.path.dirname(summary_path)
        with open(summary_path) as handle:
            summary = json.load(handle)
        condition = canonical_condition(summary["condition"])
        summary["condition"] = condition
        seed = int(summary["seed"])
        final_data = load_feature_file(
            os.path.join(run_dir, "final_test_features.npz"))
        initial_path = os.path.join(run_dir, "initial_test_features.npz")
        if os.path.exists(initial_path):
            initial_data = load_feature_file(initial_path)
            left_idx, right_idx = aligned_arrays(final_data, initial_data)
            summary["final_vs_initial_cka"] = linear_cka(
                final_data["features"][left_idx],
                initial_data["features"][right_idx],
            )
        else:
            summary["final_vs_initial_cka"] = float("nan")
        summary.setdefault("comparison_role", "matched_policy_replay")
        summary["run_dir"] = run_dir
        summaries.append(summary)
        frame = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
        frame["condition"] = condition
        plot_run_curve(frame, condition, run_dir)
        metric_frames.append(frame[frame["phase"] == "test"])
        feature_sets[(seed, condition)] = final_data

    summaries_df = pd.DataFrame(summaries)
    metrics_df = pd.concat(metric_frames, ignore_index=True)
    conditions = ordered_conditions(summaries_df["condition"].tolist())
    replay_seeds = sorted(summaries_df.loc[
        summaries_df["comparison_role"] == "matched_policy_replay", "seed"
    ].unique())
    required_conditions = [
        canonical_condition(condition)
        for condition in (args.required_conditions or [])
    ]
    missing_required = []
    for seed in replay_seeds:
        available = set(summaries_df.loc[
            (summaries_df["comparison_role"] == "matched_policy_replay")
            & (summaries_df["seed"] == seed),
            "condition",
        ])
        for condition in required_conditions:
            if condition not in available:
                missing_required.append((int(seed), condition))
    if missing_required:
        details = "\n".join(
            f"  seed {seed}: {condition}" for seed, condition in missing_required)
        raise RuntimeError(
            "Required replay conditions are missing; comparison figures were "
            f"not generated:\n{details}"
        )
    print("Loaded conditions: " + ", ".join(
        DISPLAY_NAMES.get(condition, condition) for condition in conditions))
    summaries_df.to_csv(
        os.path.join(output_dir, "aggregated_run_summaries.csv"), index=False)
    metrics_df.to_csv(
        os.path.join(output_dir, "aggregated_training_metrics.csv"), index=False)

    audit_rows = []
    controlled = summaries_df[
        summaries_df["comparison_role"] == "matched_policy_replay"]
    expected_conditions = set(controlled["condition"])
    for seed, subset in controlled.groupby("seed"):
        seed_conditions = set(subset["condition"])
        audit_rows.append({
            "seed": seed,
            "n_conditions": len(subset),
            "n_initialization_hashes": subset["initialization_sha256"].nunique(),
            "n_update_counts": subset["num_optimizer_updates"].nunique(),
            "n_evaluation_input_hashes": subset[
                "evaluation_input_sha256"].nunique(),
            "same_initialization": subset["initialization_sha256"].nunique() == 1,
            "same_update_count": subset["num_optimizer_updates"].nunique() == 1,
            "same_evaluation_inputs": subset[
                "evaluation_input_sha256"].nunique() == 1,
            "all_conditions_present": seed_conditions == expected_conditions,
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(os.path.join(output_dir, "control_audit.csv"), index=False)
    if not audit["same_initialization"].all():
        raise RuntimeError("Control failure: conditions did not share initialization")
    if not audit["same_update_count"].all():
        raise RuntimeError("Control failure: conditions had different update counts")
    if not audit["same_evaluation_inputs"].all():
        raise RuntimeError("Control failure: conditions used different test inputs")
    if not audit["all_conditions_present"].all():
        raise RuntimeError("Control failure: some seeds are missing conditions")

    references = summaries_df[
        summaries_df["comparison_role"] != "matched_policy_replay"]
    reference_audit = pd.DataFrame([{
        "condition": row.condition,
        "comparison_role": row.comparison_role,
        "seed": row.seed,
        "same_evaluation_inputs_as_replay": (
            row.evaluation_input_sha256
            in set(controlled["evaluation_input_sha256"])
        ),
        "included_in_matched_control_audit": False,
        "reason": (
            "Original joint classifier-selector training did not share the "
            "replay initialization or optimizer-update protocol."
        ),
    } for row in references.itertuples()])
    reference_audit.to_csv(
        os.path.join(output_dir, "external_reference_audit.csv"), index=False)
    if len(reference_audit) and not reference_audit[
            "same_evaluation_inputs_as_replay"].all():
        raise RuntimeError(
            "External reference did not use the replay evaluation inputs")

    plot_curves(metrics_df, conditions, output_dir)
    plot_training_heatmaps(metrics_df, conditions, output_dir)

    role_by_condition = {
        condition: block["comparison_role"].iloc[0]
        for condition, block in summaries_df.groupby("condition", sort=False)
    }

    def comparison_feature_pairs(left_condition, right_condition):
        left_runs = [
            (seed, data) for (seed, condition), data in feature_sets.items()
            if condition == left_condition
        ]
        right_runs = [
            (seed, data) for (seed, condition), data in feature_sets.items()
            if condition == right_condition
        ]
        left_external = (
            role_by_condition[left_condition] != "matched_policy_replay")
        right_external = (
            role_by_condition[right_condition] != "matched_policy_replay")
        if left_condition == right_condition:
            return [(data, data) for _, data in left_runs]
        if left_external or right_external:
            # An external run has no seed matched to the replay seeds. Compare
            # it with every run of the other condition instead of pretending
            # its display identifier is a matched training seed.
            return [
                (left_data, right_data)
                for _, left_data in left_runs
                for _, right_data in right_runs
            ]
        right_by_seed = dict(right_runs)
        return [
            (left_data, right_by_seed[seed])
            for seed, left_data in left_runs if seed in right_by_seed
        ]

    cka_matrix = np.full((len(conditions), len(conditions)), np.nan)
    disagreement_matrix = np.full_like(cka_matrix, np.nan)
    pair_rows = []
    for row, left_condition in enumerate(conditions):
        for column, right_condition in enumerate(conditions):
            cka_values, disagreement_values = [], []
            for left, right in comparison_feature_pairs(
                    left_condition, right_condition):
                left_idx, right_idx = aligned_arrays(left, right)
                cka_values.append(linear_cka(
                    left["features"][left_idx], right["features"][right_idx]))
                left_pred = left["logits"][left_idx].argmax(axis=1)
                right_pred = right["logits"][right_idx].argmax(axis=1)
                disagreement_values.append(float(np.mean(left_pred != right_pred)))
            if cka_values:
                cka_matrix[row, column] = np.mean(cka_values)
                disagreement_matrix[row, column] = np.mean(disagreement_values)
                pair_rows.append({
                    "condition_a": left_condition,
                    "condition_b": right_condition,
                    "linear_cka": np.mean(cka_values),
                    "prediction_disagreement": np.mean(disagreement_values),
                    "n_comparisons": len(cka_values),
                })
    pd.DataFrame(pair_rows).to_csv(
        os.path.join(output_dir, "pairwise_representation_comparison.csv"),
        index=False)
    plot_matrix(
        cka_matrix,
        conditions,
        "Final representation similarity on identical held-out inputs",
        "Linear CKA",
        os.path.join(output_dir, "final_representation_cka.png"),
        vmin=max(0.0, float(np.nanmin(cka_matrix))),
        vmax=1.0,
    )
    plot_matrix(
        100.0 * disagreement_matrix,
        conditions,
        "Prediction disagreement on identical held-out inputs",
        "Disagreement (%)",
        os.path.join(output_dir, "final_prediction_disagreement.png"),
        vmin=0.0,
        vmax=max(1.0, float(np.nanmax(100.0 * disagreement_matrix))),
        cmap="magma",
    )
    plot_final_bars(summaries_df, conditions, output_dir)
    print(f"Aggregated {len(summaries_df)} runs into {output_dir}")


if __name__ == "__main__":
    main()
