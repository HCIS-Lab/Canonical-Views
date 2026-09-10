#!/usr/bin/env python3
"""Compare mid-level cue importance for single-view and five-view inputs.

The no-freeze experiment's final classifier is evaluated in three regimes:

    single       every candidate view independently (N=1)
    random5      deterministic random five-view sets (N=5)
    selected5    exact agent-selected sets at one epoch (N=5, initial omitted)

Raw logits are not compared across input sizes. Instead, prediction utility and
cue enrichment are centered against an object- and set-size-matched baseline:

    N=1: value - mean(value over all single views of the object)
    N=5: value - mean(value over random five-view sets of the object)

The analysis reports object-balanced Spearman correlations, grouped
cross-validated R-squared, and the incremental R-squared from adding exact view
family composition after ellipse aspect ratio, bilateral symmetry, and edge
entropy.
"""

import argparse
import hashlib
import math
import os
import random
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(
    tempfile.gettempdir(), "mvselect_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from scipy import stats

from midlevel_shape_features import PRIMARY_RAW_LABELS
from temporal_selection_test import locate_final_checkpoint, resolve_selection_dir
from view_contribution_analysis import (
    FEATURES,
    OBJECTIVES,
    VIEW_FAMILIES,
    build_dataset_view_index,
    checkpoint_for_selection_run,
    choose_initial_cams,
    create_model,
    exact_view_metadata,
    extract_fixed_feature_tensor,
    forward_aggregated_masks,
    load_epoch_feature_dump,
    load_selection_runs,
    load_model_state,
    load_torch_state,
    make_feature_index,
    objective_values,
    prepare_feature_cache,
)
from midlevel_shape_features import load_modelnet40_classnames
from src.models.architectures import ARCHITECTURE_CHOICES


DEFAULT_EXPERIMENT = (
    "resnet18steps5_train_ins25_lr0.0005base1.0other1.0"
    "select_wd0.0001select0.0001_e100"
)
REGIME_LABELS = {
    "single": "All single views (N=1)",
    "random5": "Random sets (N=5)",
    "selected5": "Agent-selected sets (N=5)",
}
SUMMARY_TYPES = ["mean", "max", "std"]
FAMILY_COLUMN_NAMES = {
    family: f"prop_{family.lower().replace('-', '_')}"
    for family in VIEW_FAMILIES
}
FAMILY_COLORS = {
    "Expanded": "#009E73",
    "Expanded-like": "#0072B2",
    "Foreshortened": "#D55E00",
    "Foreshortened-like": "#CC79A7",
    "Remainder": "#666666",
}
RANK_CORRELATION_LABEL = "Object-balanced rank correlation (rho)"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--selection_dir",
        default=None,
        help="No-freeze experiment folder containing *_selection.json.",
    )
    parser.add_argument("--selection_epoch", type=int, default=100)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default="rgb")
    parser.add_argument("--arch", default="auto", choices=ARCHITECTURE_CHOICES)
    parser.add_argument("--aggregation", default="max", choices=["mean", "max"])
    parser.add_argument(
        "--data_root",
        default=(
            "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/"
            "modelnet_32_60_1_23"
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache_csv", default=None)
    parser.add_argument("--force_recompute_cache", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--non_roll", action="store_true")
    parser.add_argument("--non_like", action="store_true")
    parser.add_argument("--test_per_cls_instances", type=int, default=5)
    parser.add_argument("--n_views", type=int, default=5)
    parser.add_argument("--random_sets_per_instance", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_initial_cams", type=int, default=10)
    parser.add_argument("--initial_cams", nargs="+", type=int, default=None)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--aggregate_batch_size", type=int, default=512)
    parser.add_argument("--limit_instances", type=int, default=None)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Regenerate figures from CSVs already present in --output_dir.",
    )
    parser.add_argument(
        "--family_means_only",
        action="store_true",
        help="Compute raw all-candidate view-family cue means and exit.",
    )
    args = parser.parse_args()
    if args.selection_dir is None:
        default_arch = "resnet18" if args.arch == "auto" else args.arch
        default_experiment = DEFAULT_EXPERIMENT.replace(
            "resnet18", default_arch, 1)
        args.selection_dir = os.path.join(
            "meta_logs", args.dataset, default_experiment)
    if args.n_views != 5:
        parser.error("--n_views must be exactly 5 for this N=1 versus N=5 analysis")
    if args.random_sets_per_instance < 2:
        parser.error("--random_sets_per_instance must be at least 2")
    if args.cv_folds < 2:
        parser.error("--cv_folds must be at least 2")
    return args


def enforce_no_freeze(selection_dir):
    name = os.path.basename(os.path.normpath(selection_dir))
    if name.startswith("freeze_") or "selview_" in name:
        raise ValueError(
            "This analysis is intentionally restricted to no_freeze; got "
            f"{name}")


def build_metadata_tensors(feature_index, view_records, instance_info, num_views):
    num_instances = len(instance_info)
    metric_values = {
        metric: np.full((num_instances, num_views), np.nan, dtype=float)
        for metric in FEATURES
    }
    family_indices = np.full((num_instances, num_views), -1, dtype=int)
    family_to_index = {family: index for index, family in enumerate(VIEW_FAMILIES)}
    view_filenames = np.empty((num_instances, num_views), dtype=object)

    for instance_idx in range(num_instances):
        for view_idx in range(num_views):
            record = view_records[(instance_idx, view_idx)]
            metadata = exact_view_metadata(feature_index, record)
            family_indices[instance_idx, view_idx] = family_to_index[
                metadata["view_family"]]
            view_filenames[instance_idx, view_idx] = record["filename"]
            for metric in FEATURES:
                metric_values[metric][instance_idx, view_idx] = metadata[metric]
    return metric_values, family_indices, view_filenames


def build_view_family_feature_means(view_df, limit_instances=None):
    """Summarize raw cues by exact view family with equal object weighting."""
    frame = view_df.copy()
    frame["view_family"] = frame["view_type"].replace(
        {"expanded": "Expanded"})
    object_keys = frame[["class_idx", "instance_id"]].drop_duplicates()
    object_keys = object_keys.sort_values(
        ["class_idx", "instance_id"], kind="stable")
    if limit_instances is not None:
        object_keys = object_keys.iloc[:limit_instances]
    frame = frame.merge(object_keys, on=["class_idx", "instance_id"], how="inner")
    rows = []
    for family in VIEW_FAMILIES:
        family_block = frame[frame["view_family"] == family]
        for metric in FEATURES:
            numeric = family_block.assign(
                _value=pd.to_numeric(family_block[metric], errors="coerce")
            ).dropna(subset=["_value"])
            valid_object_means = numeric.groupby(
                ["class_idx", "instance_id"], sort=False)["_value"].mean()
            pooled_values = numeric["_value"]
            rows.append({
                "view_family": family,
                "metric": metric,
                "object_macro_mean": (
                    float(valid_object_means.mean())
                    if len(valid_object_means) else np.nan),
                "object_macro_sem": sem(valid_object_means),
                "pooled_view_mean": (
                    float(pooled_values.mean())
                    if len(pooled_values) else np.nan),
                "n_objects": int(len(valid_object_means)),
                "n_views": int(len(pooled_values)),
            })
    return pd.DataFrame(rows)


def evaluate_masks(model, feature_tensor, instance_indices, masks, targets,
                   device, batch_size):
    logits = forward_aggregated_masks(
        model, feature_tensor, instance_indices, masks, device, batch_size)
    values = objective_values(logits, targets)
    predictions = logits.argmax(dim=1).numpy().astype(int)
    return {
        "correct": (predictions == targets).astype(int),
        "prediction": predictions,
        **{
            objective: tensor.detach().cpu().numpy().astype(float)
            for objective, tensor in values.items()
        },
    }


def summarize_mask_features(instance_indices, masks, metric_values,
                            family_indices):
    summaries = {}
    for metric, values in metric_values.items():
        trial_values = values[instance_indices]
        masked = np.where(masks, trial_values, np.nan)
        with np.errstate(invalid="ignore"):
            summaries[f"{metric}_mean"] = np.nanmean(masked, axis=1)
            summaries[f"{metric}_max"] = np.nanmax(masked, axis=1)
            summaries[f"{metric}_std"] = np.nanstd(masked, axis=1)

    selected_families = family_indices[instance_indices]
    denominator = np.maximum(masks.sum(axis=1), 1)
    for family_idx, family in enumerate(VIEW_FAMILIES):
        summaries[FAMILY_COLUMN_NAMES[family]] = (
            ((selected_families == family_idx) & masks).sum(axis=1)
            / denominator
        )
    return summaries


def make_trial_frame(run_name, checkpoint_path, regime, model, feature_tensor,
                     instance_indices, masks, instance_info, metric_values,
                     family_indices, device, batch_size, sample_indices,
                     initial_cams=None, single_view_indices=None):
    targets = np.asarray([
        instance_info[int(instance_idx)]["class_idx"]
        for instance_idx in instance_indices
    ], dtype=int)
    evaluated = evaluate_masks(
        model, feature_tensor, instance_indices, masks, targets,
        device, batch_size)
    rows = {
        "run": np.repeat(run_name, len(instance_indices)),
        "checkpoint_path": np.repeat(checkpoint_path, len(instance_indices)),
        "regime": np.repeat(regime, len(instance_indices)),
        "set_size": masks.sum(axis=1).astype(int),
        "sample_index": np.asarray(sample_indices, dtype=int),
        "instance_index": np.asarray(instance_indices, dtype=int),
        "class_idx": targets,
        "class_name": np.asarray([
            instance_info[int(index)]["class_name"] for index in instance_indices
        ]),
        "instance_id": np.asarray([
            instance_info[int(index)]["instance_id"] for index in instance_indices
        ]),
        **evaluated,
        **summarize_mask_features(
            instance_indices, masks, metric_values, family_indices),
    }
    if initial_cams is not None:
        rows["initial_cam"] = np.asarray(initial_cams, dtype=int)
    else:
        rows["initial_cam"] = np.full(len(instance_indices), -1, dtype=int)
    if single_view_indices is not None:
        rows["single_view_index"] = np.asarray(single_view_indices, dtype=int)
        selected_family_indices = family_indices[
            instance_indices, single_view_indices]
        rows["single_view_family"] = np.asarray([
            VIEW_FAMILIES[index] for index in selected_family_indices
        ])
    else:
        rows["single_view_index"] = np.full(len(instance_indices), -1, dtype=int)
        rows["single_view_family"] = np.repeat("", len(instance_indices))
    return pd.DataFrame(rows)


def build_single_trials(num_instances, num_views, active_instances):
    instance_indices = np.repeat(active_instances, num_views)
    view_indices = np.tile(np.arange(num_views, dtype=int), len(active_instances))
    masks = np.zeros((len(instance_indices), num_views), dtype=bool)
    masks[np.arange(len(instance_indices)), view_indices] = True
    return instance_indices, masks, view_indices


def build_random_trials(active_instances, num_views, set_size, samples, seed,
                        run_name):
    instance_indices = np.repeat(active_instances, samples)
    masks = np.zeros((len(instance_indices), num_views), dtype=bool)
    sample_indices = np.tile(np.arange(samples, dtype=int), len(active_instances))
    for row_idx, (instance_idx, sample_idx) in enumerate(zip(
            instance_indices, sample_indices)):
        local_seed = f"{seed}|{run_name}|{int(instance_idx)}|{int(sample_idx)}"
        chosen = random.Random(local_seed).sample(range(num_views), set_size)
        masks[row_idx, chosen] = True
    return instance_indices, masks, sample_indices


def build_selected_trials(args, dump, active_instance_set, num_views):
    chosen_initial_cams = choose_initial_cams(args, num_views)
    keep = np.flatnonzero(np.isin(dump["initial_cams"], chosen_initial_cams))
    instance_indices = []
    masks = []
    initial_cams = []
    sample_indices = []
    skipped_size = 0
    for row_idx in keep:
        instance_idx = int(dump["instance_indices"][row_idx])
        if instance_idx not in active_instance_set:
            continue
        mask = dump["selected_masks"][row_idx].copy()
        initial_cam = int(dump["initial_cams"][row_idx])
        mask[initial_cam] = False
        if int(mask.sum()) != args.n_views:
            skipped_size += 1
            continue
        instance_indices.append(instance_idx)
        masks.append(mask)
        initial_cams.append(initial_cam)
        sample_indices.append(int(row_idx))
    if not masks:
        return (np.asarray([], dtype=int),
                np.empty((0, num_views), dtype=bool),
                np.asarray([], dtype=int), np.asarray([], dtype=int), skipped_size)
    return (
        np.asarray(instance_indices, dtype=int),
        np.stack(masks),
        np.asarray(sample_indices, dtype=int),
        np.asarray(initial_cams, dtype=int),
        skipped_size,
    )


def add_matched_normalization(trials):
    keys = ["run", "class_idx", "instance_id"]
    value_columns = list(OBJECTIVES) + [
        f"{metric}_{summary}"
        for metric in FEATURES
        for summary in SUMMARY_TYPES
    ]
    single_baseline = trials[trials["regime"] == "single"].groupby(
        keys, sort=False)[value_columns].mean().add_prefix("single_baseline_")
    random_baseline = trials[trials["regime"] == "random5"].groupby(
        keys, sort=False)[value_columns].mean().add_prefix("random5_baseline_")
    normalized = trials.merge(single_baseline, on=keys, how="left")
    normalized = normalized.merge(random_baseline, on=keys, how="left")
    is_single = normalized["regime"] == "single"
    for column in value_columns:
        baseline = np.where(
            is_single,
            normalized[f"single_baseline_{column}"],
            normalized[f"random5_baseline_{column}"],
        )
        normalized[f"{column}_baseline"] = baseline
        normalized[f"{column}_enrichment"] = normalized[column] - baseline
    normalized["normalization_reference"] = np.where(
        is_single, "object_mean_single", "object_mean_random5")
    return normalized.drop(columns=[
        column for column in normalized.columns
        if column.startswith("single_baseline_")
        or column.startswith("random5_baseline_")
    ])


def safe_spearman(x, y):
    frame = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan, len(frame)
    return float(stats.spearmanr(frame["x"], frame["y"])[0]), len(frame)


def object_balanced_spearman(block, x_column, y_column):
    correlation_rows = []
    sample_count = 0
    for (run_name, class_idx, instance_id), object_block in block.groupby(
            ["run", "class_idx", "instance_id"], sort=False):
        correlation, count = safe_spearman(
            object_block[x_column], object_block[y_column])
        sample_count += count
        if np.isfinite(correlation):
            correlation_rows.append({
                "run": run_name,
                "class_idx": class_idx,
                "instance_id": instance_id,
                "z": float(np.arctanh(np.clip(
                    correlation, -0.999999, 0.999999))),
            })
    if not correlation_rows:
        return np.nan, np.nan, np.nan, 0, sample_count
    per_object = pd.DataFrame(correlation_rows).groupby(
        ["class_idx", "instance_id"], sort=False)["z"].mean()
    z_values = per_object.to_numpy(dtype=float)
    mean_z = float(z_values.mean())
    sem_z = (
        float(z_values.std(ddof=1) / math.sqrt(len(z_values)))
        if len(z_values) > 1 else 0.0)
    return (
        float(np.tanh(mean_z)),
        float(np.tanh(mean_z - 1.96 * sem_z)),
        float(np.tanh(mean_z + 1.96 * sem_z)),
        len(z_values),
        sample_count,
    )


def correlation_rows_for_block(block, scope, regime, feature_summaries):
    rows = []
    for objective in OBJECTIVES:
        y_column = f"{objective}_enrichment"
        for metric in FEATURES:
            for summary in feature_summaries:
                x_column = f"{metric}_{summary}_enrichment"
                correlation, low, high, n_objects, n = object_balanced_spearman(
                    block, x_column, y_column)
                rows.append({
                    "scope": scope,
                    "regime": regime,
                    "objective": objective,
                    "metric": metric,
                    "summary": summary,
                    "correlation": correlation,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_objects": n_objects,
                    "n": n,
                })
    return rows


def build_correlation_table(trials):
    rows = []
    for regime in REGIME_LABELS:
        block = trials[trials["regime"] == regime]
        summaries = ["mean"] if regime == "single" else SUMMARY_TYPES
        rows.extend(correlation_rows_for_block(
            block, "regime", regime, summaries))
    single = trials[trials["regime"] == "single"]
    for family in VIEW_FAMILIES:
        block = single[single["single_view_family"] == family]
        rows.extend(correlation_rows_for_block(
            block, "single_family", family, ["mean"]))
    return pd.DataFrame(rows)


def sem(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def performance_table(trials):
    groups = [("single_all", trials[trials["regime"] == "single"])]
    single = trials[trials["regime"] == "single"]
    groups.extend((f"single_{family}", single[
        single["single_view_family"] == family]) for family in VIEW_FAMILIES)
    groups.extend((regime, trials[trials["regime"] == regime])
                  for regime in ["random5", "selected5"])
    rows = []
    for label, block in groups:
        run_object_means = block.groupby(
            ["run", "class_idx", "instance_id"], sort=False
        )[["correct", *OBJECTIVES]].mean()
        object_means = run_object_means.groupby(
            ["class_idx", "instance_id"], sort=False).mean()
        rows.append({
            "group": label,
            "n_trials": int(len(block)),
            "n_objects": int(len(object_means)),
            "accuracy": float(object_means["correct"].mean()),
            "accuracy_sem": sem(object_means["correct"]),
            **{
                f"mean_{objective}": float(object_means[objective].mean())
                for objective in OBJECTIVES
            },
        })
    return pd.DataFrame(rows)


def stable_fold(class_idx, instance_id, folds):
    digest = hashlib.sha1(f"{class_idx}|{instance_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % folds


def prepare_regression_frame(block, feature_columns, target_column):
    columns = ["class_idx", "instance_id", target_column, *feature_columns]
    return block[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()


def grouped_cv_r2(block, feature_columns, target_column, folds):
    frame = prepare_regression_frame(block, feature_columns, target_column)
    if len(frame) < 20:
        return np.nan, len(frame), 0
    frame["fold"] = [
        stable_fold(class_idx, instance_id, folds)
        for class_idx, instance_id in zip(frame["class_idx"], frame["instance_id"])
    ]
    predictions = np.full(len(frame), np.nan, dtype=float)
    y = frame[target_column].to_numpy(dtype=float)
    x = frame[feature_columns].to_numpy(dtype=float)
    used_folds = 0
    for fold in sorted(frame["fold"].unique()):
        test = frame["fold"].to_numpy() == fold
        train = ~test
        if train.sum() <= len(feature_columns) + 1 or test.sum() == 0:
            continue
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale[scale < 1e-12] = 1.0
        train_x = np.column_stack([
            np.ones(train.sum()), (x[train] - mean) / scale])
        test_x = np.column_stack([
            np.ones(test.sum()), (x[test] - mean) / scale])
        coefficients = np.linalg.lstsq(train_x, y[train], rcond=None)[0]
        predictions[test] = test_x @ coefficients
        used_folds += 1
    valid = np.isfinite(predictions)
    if valid.sum() < 3:
        return np.nan, int(valid.sum()), used_folds
    ss_residual = float(np.square(y[valid] - predictions[valid]).sum())
    ss_total = float(np.square(y[valid] - y[valid].mean()).sum())
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 1e-12 else np.nan
    return float(r_squared), int(valid.sum()), used_folds


def standardized_coefficients(block, feature_columns, target_column):
    frame = prepare_regression_frame(block, feature_columns, target_column)
    if len(frame) <= len(feature_columns) + 1:
        return {column: np.nan for column in feature_columns}
    x = frame[feature_columns].to_numpy(dtype=float)
    y = frame[target_column].to_numpy(dtype=float)
    x_mean, x_scale = x.mean(axis=0), x.std(axis=0)
    y_mean, y_scale = y.mean(), y.std()
    x_scale[x_scale < 1e-12] = 1.0
    if y_scale < 1e-12:
        return {column: np.nan for column in feature_columns}
    design = np.column_stack([np.ones(len(x)), (x - x_mean) / x_scale])
    coefficients = np.linalg.lstsq(
        design, (y - y_mean) / y_scale, rcond=None)[0][1:]
    return dict(zip(feature_columns, coefficients.astype(float)))


def regression_tables(trials, folds):
    cue_columns = [f"{metric}_mean_enrichment" for metric in FEATURES]
    family_columns = [
        FAMILY_COLUMN_NAMES[family]
        for family in VIEW_FAMILIES[:-1]
    ]
    model_columns = {
        "cues_only": cue_columns,
        "family_only": family_columns,
        "cues_plus_family": cue_columns + family_columns,
    }
    score_rows = []
    coefficient_rows = []
    for regime in REGIME_LABELS:
        block = trials[trials["regime"] == regime]
        for objective in OBJECTIVES:
            target = f"{objective}_enrichment"
            regime_scores = {}
            for model_name, columns in model_columns.items():
                r_squared, n, used_folds = grouped_cv_r2(
                    block, columns, target, folds)
                regime_scores[model_name] = r_squared
                score_rows.append({
                    "regime": regime,
                    "objective": objective,
                    "model": model_name,
                    "cv_r_squared": r_squared,
                    "n": n,
                    "folds": used_folds,
                })
            score_rows.append({
                "regime": regime,
                "objective": objective,
                "model": "incremental_family_after_cues",
                "cv_r_squared": (
                    regime_scores["cues_plus_family"]
                    - regime_scores["cues_only"]
                    if np.isfinite(regime_scores["cues_plus_family"])
                    and np.isfinite(regime_scores["cues_only"])
                    else np.nan),
                "n": len(block),
                "folds": folds,
            })
            coefficients = standardized_coefficients(
                block, model_columns["cues_plus_family"], target)
            for column, coefficient in coefficients.items():
                coefficient_rows.append({
                    "regime": regime,
                    "objective": objective,
                    "predictor": column,
                    "standardized_coefficient": coefficient,
                })
    return pd.DataFrame(score_rows), pd.DataFrame(coefficient_rows)


def draw_heatmap(matrix, row_labels, column_labels, title, colorbar_label,
                 out_path, vmin, vmax, cmap):
    fig, ax = plt.subplots(
        figsize=(max(9.5, 1.45 * len(column_labels) + 5.0),
                 max(3.0, 0.55 * len(row_labels) + 1.8)),
        constrained_layout=True,
    )
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    threshold = max(abs(vmin), abs(vmax)) * 0.55
    for row_idx in range(matrix.shape[0]):
        for column_idx in range(matrix.shape[1]):
            value = matrix[row_idx, column_idx]
            if not np.isfinite(value):
                continue
            color = "white" if abs(value) > threshold else "black"
            ax.text(column_idx, row_idx, f"{value:+.2f}", ha="center",
                    va="center", fontsize=8, color=color)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_correlations(correlations, output_dir):
    direct = correlations[
        (correlations["scope"] == "regime")
        & (correlations["summary"] == "mean")
    ]
    regimes = list(REGIME_LABELS)
    for objective in OBJECTIVES:
        block = direct[direct["objective"] == objective]
        matrix = block.pivot(
            index="metric", columns="regime", values="correlation"
        ).reindex(index=FEATURES, columns=regimes).to_numpy(dtype=float)
        draw_heatmap(
            matrix,
            [PRIMARY_RAW_LABELS[metric] for metric in FEATURES],
            [REGIME_LABELS[regime] for regime in regimes],
            f"Within-object mid-level association: {OBJECTIVES[objective]}",
            RANK_CORRELATION_LABEL,
            os.path.join(output_dir, f"n1_n5_correlation_{objective}.png"),
            -1.0,
            1.0,
            "RdBu_r",
        )

        n5 = correlations[
            (correlations["scope"] == "regime")
            & (correlations["objective"] == objective)
            & (correlations["regime"].isin(["random5", "selected5"]))
        ]
        rows = [(metric, summary) for metric in FEATURES for summary in SUMMARY_TYPES]
        pivot = n5.pivot_table(
            index=["metric", "summary"], columns="regime",
            values="correlation", aggfunc="first")
        matrix = pivot.reindex(
            index=pd.MultiIndex.from_tuples(rows, names=["metric", "summary"]),
            columns=["random5", "selected5"],
        ).to_numpy(dtype=float)
        draw_heatmap(
            matrix,
            [f"{PRIMARY_RAW_LABELS[metric]} | {summary}" for metric, summary in rows],
            [REGIME_LABELS["random5"], REGIME_LABELS["selected5"]],
            f"Five-view cue summaries: {OBJECTIVES[objective]}",
            RANK_CORRELATION_LABEL,
            os.path.join(output_dir, f"n5_summary_correlation_{objective}.png"),
            -1.0,
            1.0,
            "RdBu_r",
        )

        family = correlations[
            (correlations["scope"] == "single_family")
            & (correlations["objective"] == objective)
        ]
        matrix = family.pivot(
            index="metric", columns="regime", values="correlation"
        ).reindex(index=FEATURES, columns=VIEW_FAMILIES).to_numpy(dtype=float)
        draw_heatmap(
            matrix,
            [PRIMARY_RAW_LABELS[metric] for metric in FEATURES],
            VIEW_FAMILIES,
            f"Single-view within-family association: {OBJECTIVES[objective]}",
            RANK_CORRELATION_LABEL,
            os.path.join(
                output_dir, f"single_family_correlation_{objective}.png"),
            -1.0,
            1.0,
            "RdBu_r",
        )


def plot_performance(performance, output_dir):
    order = [
        "single_Expanded",
        "single_Expanded-like",
        "single_Foreshortened",
        "single_Foreshortened-like",
        "single_Remainder",
        "random5",
        "selected5",
    ]
    labels = [
        "Expanded\nN=1",
        "Expanded-like\nN=1",
        "Foreshortened\nN=1",
        "Foreshortened-like\nN=1",
        "Remainder\nN=1",
        "Random\nN=5",
        "Selected\nN=5",
    ]
    block = performance.set_index("group").reindex(order)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(x, block["accuracy"], yerr=block["accuracy_sem"], capsize=3,
           color=["#0072B2"] * 2 + ["#D55E00"] * 2 + ["#777777", "#999999", "#009E73"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Final-classifier performance by input regime")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(block["accuracy"]):
        if np.isfinite(value):
            ax.text(idx, value + 0.025, f"{value * 100:.1f}%",
                    ha="center", fontsize=8)
    out_path = os.path.join(output_dir, "accuracy_by_input_regime.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_view_family_feature_means(summary, output_dir):
    fig, axes = plt.subplots(
        1, len(FEATURES), figsize=(15, 4.8), constrained_layout=True)
    if len(FEATURES) == 1:
        axes = [axes]
    x = np.arange(len(VIEW_FAMILIES))
    colors = [FAMILY_COLORS[family] for family in VIEW_FAMILIES]
    for axis, metric in zip(axes, FEATURES):
        block = summary[summary["metric"] == metric].set_index(
            "view_family").reindex(VIEW_FAMILIES)
        bars = axis.bar(
            x,
            block["object_macro_mean"],
            yerr=block["object_macro_sem"],
            capsize=3,
            color=colors,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(VIEW_FAMILIES, rotation=25, ha="right")
        axis.set_ylabel(f"Mean {PRIMARY_RAW_LABELS[metric]} (raw units)")
        axis.set_title(PRIMARY_RAW_LABELS[metric])
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.suptitle(
        "Raw mid-level feature means by exact view family (all candidate views)")
    out_path = os.path.join(
        output_dir, "all_candidate_view_family_midlevel_means.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_regression_scores(scores, output_dir):
    models = ["cues_only", "family_only", "cues_plus_family",
              "incremental_family_after_cues"]
    model_labels = {
        "cues_only": "Three cues",
        "family_only": "View family",
        "cues_plus_family": "Three cues + family",
        "incremental_family_after_cues": "Family gain after cues (delta R2)",
    }
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    fig, axes = plt.subplots(1, len(OBJECTIVES), figsize=(12, 4.8),
                             sharey=True, constrained_layout=True)
    if len(OBJECTIVES) == 1:
        axes = [axes]
    x = np.arange(len(REGIME_LABELS))
    width = 0.19
    for axis, objective in zip(axes, OBJECTIVES):
        block = scores[scores["objective"] == objective]
        for model_idx, (model_name, color) in enumerate(zip(models, colors)):
            values = block[block["model"] == model_name].set_index(
                "regime").reindex(REGIME_LABELS)["cv_r_squared"]
            axis.bar(
                x + (model_idx - 1.5) * width,
                values,
                width,
                label=model_labels[model_name],
                color=color,
            )
        axis.axhline(0, color="black", lw=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(
            ["N=1", "Random N=5", "Selected N=5"], rotation=15)
        axis.set_title(f"Predicting {OBJECTIVES[objective].lower()} enrichment")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Held-out R2 (or delta R2 for family gain)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Prediction of classifier-output changes on unseen objects")
    out_path = os.path.join(output_dir, "cue_and_family_cross_validated_r2.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def regenerate_plots(output_dir):
    required = {
        "correlations": "midlevel_correlations.csv",
        "performance": "performance_by_regime.csv",
        "scores": "cross_validated_r2.csv",
    }
    paths = {
        name: os.path.join(output_dir, filename)
        for name, filename in required.items()
    }
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "--plot_only requires existing analysis CSVs; missing: "
            + ", ".join(missing))
    plot_correlations(pd.read_csv(paths["correlations"]), output_dir)
    plot_performance(pd.read_csv(paths["performance"]), output_dir)
    plot_regression_scores(pd.read_csv(paths["scores"]), output_dir)
    family_summary_path = os.path.join(
        output_dir, "all_candidate_view_family_midlevel_means.csv")
    if os.path.exists(family_summary_path):
        plot_view_family_feature_means(
            pd.read_csv(family_summary_path), output_dir)


def main():
    args = parse_args()
    resolved = resolve_selection_dir(args.selection_dir, args.dataset)
    if resolved is None:
        raise SystemExit(f"No *_selection.json found for {args.selection_dir}")
    args.selection_dir = resolved
    enforce_no_freeze(args.selection_dir)
    args.output_dir = args.output_dir or os.path.join(
        args.selection_dir, "single_multiview_midlevel")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.plot_only:
        regenerate_plots(args.output_dir)
        print("Done (plot only).")
        return

    view_df = prepare_feature_cache(args)
    missing = [metric for metric in FEATURES if metric not in view_df.columns]
    if missing:
        raise KeyError(f"Mid-level cache is missing {missing}")
    family_summary = build_view_family_feature_means(
        view_df, args.limit_instances)
    family_summary_path = os.path.join(
        args.output_dir, "all_candidate_view_family_midlevel_means.csv")
    family_summary.to_csv(family_summary_path, index=False)
    print(f"Saved: {family_summary_path}")
    plot_view_family_feature_means(family_summary, args.output_dir)
    if args.family_means_only:
        print("Done (family means only).")
        return

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Selection dir: {args.selection_dir}")
    print(f"Epoch:         {args.selection_epoch}")
    print(f"Device:        {device}")
    print(f"Output:        {args.output_dir}")

    runs = load_selection_runs(args.selection_dir)
    runs = [(name, content) for name, content in runs
            if args.selection_epoch in content]
    if not runs:
        raise SystemExit(
            f"No selection run contains epoch {args.selection_epoch}")

    feature_index = make_feature_index(view_df)
    classnames = load_modelnet40_classnames()

    fallback_checkpoint = locate_final_checkpoint(args)
    model = create_model(fallback_checkpoint, args, device)
    view_records, instance_lookup, instance_info = build_dataset_view_index(
        args, classnames, model.num_cam)
    active_instances = np.arange(len(instance_info), dtype=int)
    if args.limit_instances is not None:
        active_instances = active_instances[:args.limit_instances]
    active_instance_set = set(active_instances.tolist())
    metric_values, family_indices, _ = build_metadata_tensors(
        feature_index, view_records, instance_info, model.num_cam)
    if any(np.isnan(values[active_instances]).any()
           for values in metric_values.values()):
        raise ValueError("Some active views are missing mid-level descriptors")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)),
    ])
    feature_cache = {}
    trial_frames = []
    for run_name, _ in runs:
        checkpoint_path, checkpoint_match = checkpoint_for_selection_run(
            args, run_name, fallback_checkpoint)
        print(f"Run {run_name}: {checkpoint_path} ({checkpoint_match})")
        state = load_torch_state(checkpoint_path)
        load_model_state(model, state, checkpoint_path)
        if checkpoint_path not in feature_cache:
            feature_cache[checkpoint_path] = extract_fixed_feature_tensor(
                model, view_records, instance_info, transform, device,
                args.feature_batch_size)
        feature_tensor = feature_cache[checkpoint_path]

        single_instances, single_masks, single_indices = build_single_trials(
            len(instance_info), model.num_cam, active_instances)
        trial_frames.append(make_trial_frame(
            run_name, checkpoint_path, "single", model, feature_tensor,
            single_instances, single_masks, instance_info, metric_values,
            family_indices, device, args.aggregate_batch_size,
            sample_indices=single_indices,
            single_view_indices=single_indices,
        ))

        random_instances, random_masks, random_indices = build_random_trials(
            active_instances, model.num_cam, args.n_views,
            args.random_sets_per_instance, args.random_seed, run_name)
        trial_frames.append(make_trial_frame(
            run_name, checkpoint_path, "random5", model, feature_tensor,
            random_instances, random_masks, instance_info, metric_values,
            family_indices, device, args.aggregate_batch_size,
            sample_indices=random_indices,
        ))

        run_dir = os.path.join(
            args.selection_dir, run_name[:-len("_selection.json")])
        feature_path = os.path.join(
            run_dir, f"feature_{args.selection_epoch}.npz")
        if not os.path.exists(feature_path):
            print(f"  SKIP selected5: missing {feature_path}")
            continue
        dump = load_epoch_feature_dump(
            feature_path, model, instance_lookup, instance_info,
            need_feature_tensor=False)
        (selected_instances, selected_masks, selected_indices,
         initial_cams, skipped_size) = build_selected_trials(
            args, dump, active_instance_set, model.num_cam)
        print(f"  selected5 trials={len(selected_instances)}, "
              f"skipped wrong-size={skipped_size}")
        if len(selected_instances):
            trial_frames.append(make_trial_frame(
                run_name, checkpoint_path, "selected5", model, feature_tensor,
                selected_instances, selected_masks, instance_info,
                metric_values, family_indices, device,
                args.aggregate_batch_size,
                sample_indices=selected_indices,
                initial_cams=initial_cams,
            ))

    trials = pd.concat(trial_frames, ignore_index=True)
    if "selected5" not in set(trials["regime"]):
        raise SystemExit("No selected N=5 trials were produced")
    trials = add_matched_normalization(trials)
    trial_path = os.path.join(args.output_dir, "input_regime_trials.csv")
    trials.to_csv(trial_path, index=False)
    print(f"Saved: {trial_path} ({len(trials)} rows)")

    correlations = build_correlation_table(trials)
    correlation_path = os.path.join(args.output_dir, "midlevel_correlations.csv")
    correlations.to_csv(correlation_path, index=False)
    print(f"Saved: {correlation_path}")

    performance = performance_table(trials)
    performance_path = os.path.join(args.output_dir, "performance_by_regime.csv")
    performance.to_csv(performance_path, index=False)
    print(f"Saved: {performance_path}")

    scores, coefficients = regression_tables(trials, args.cv_folds)
    score_path = os.path.join(args.output_dir, "cross_validated_r2.csv")
    coefficient_path = os.path.join(
        args.output_dir, "standardized_coefficients.csv")
    scores.to_csv(score_path, index=False)
    coefficients.to_csv(coefficient_path, index=False)
    print(f"Saved: {score_path}")
    print(f"Saved: {coefficient_path}")

    plot_correlations(correlations, args.output_dir)
    plot_performance(performance, args.output_dir)
    plot_regression_scores(scores, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
