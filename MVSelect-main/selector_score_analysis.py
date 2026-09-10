#!/usr/bin/env python3
"""Relate every valid candidate view's mid-level cues to selector score.

For each exact test rollout stored in feature_<epoch>.npz, this script loads
the matching model_e<epoch>.pth checkpoint and replays the selector one step at
a time. At every step it records the raw DQN action value for every currently
valid candidate view and computes within-decision Spearman correlations between
those scores and image-derived shape descriptors.

Timestamped runs are grouped only when the complete experiment prefix matches.
For example, ``resnet18steps5_..._e100`` is never mixed with
``resnet18steps5_selview_remainder_..._e100``.

The selector is DQN-style: its output is an action-value score, not a policy
probability. The chosen-cue percentile is also reported because it directly
asks where the greedily chosen view falls among the valid candidates for a
given cue (0.5 = candidate median, 1.0 = candidate maximum).
"""

import argparse
import datetime
import math
import os
import re
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mvselect_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from midlevel_shape_features import (
    CACHE_VERSION,
    DEFAULT_DATA_ROOT,
    build_view_feature_cache,
    load_modelnet40_classnames,
)
from src.models.mvselect import setup_args
from src.models.architectures import ARCHITECTURE_CHOICES
from view_contribution_analysis import (
    build_dataset_view_index,
    create_model,
    load_epoch_feature_dump,
    load_model_state,
    load_torch_state,
)


DEFAULT_EXPERIMENT = (
    "resnet18steps5_train_ins25_lr0.0005base1.0other1.0"
    "select_wd0.0001select0.0001_e100"
)

DEFAULT_METRICS = [
    "ellipse_aspect_ratio",
    "bilateral_symmetry",
    "medial_axis_symmetry",
    "edge_entropy",
]

METRIC_LABELS = {
    "ellipse_aspect_ratio": "Ellipse Aspect Ratio",
    "bilateral_symmetry": "Bilateral Symmetry",
    "medial_axis_symmetry": "Medial Axis Symmetry",
    "edge_entropy": "Edge Entropy",
}

PLOT_GROUPS = {
    "axis_visibility": ["ellipse_aspect_ratio"],
    "symmetry": ["bilateral_symmetry", "medial_axis_symmetry"],
    "edge_organization": ["edge_entropy"],
}

RUN_RE = re.compile(
    r"^(?P<experiment>.+)_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")
CHECKPOINT_RE = re.compile(r"model_e(?P<epoch>\d+)\.pth$")
FEATURE_RE = re.compile(r"feature_(?P<epoch>\d+)\.npz$")


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    timestamp: str
    log_dir: str
    meta_run_name: str
    meta_run_dir: str


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help=(
            "Exact untimestamped experiment name. Repeat this flag to analyze "
            "multiple settings separately. Default: the steps5 no-freeze run."
        ),
    )
    parser.add_argument("--dataset", default="rgb")
    parser.add_argument("--logs_root", default="logs")
    parser.add_argument("--meta_root", default="meta_logs")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache_csv", default=None)
    parser.add_argument("--force_recompute_cache", action="store_true")
    parser.add_argument("--arch", default="auto", choices=ARCHITECTURE_CHOICES)
    parser.add_argument("--aggregation", default="max", choices=["mean", "max"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--epoch_stride", type=int, default=10)
    parser.add_argument("--epochs", nargs="+", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--run_limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None,
                        help="Infer from experiment name when omitted.")
    parser.add_argument("--num_initial_cams", type=int, default=10)
    parser.add_argument("--initial_cams", nargs="+", type=int, default=None)
    parser.add_argument("--limit_trials", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_per_cls_instances", type=int, default=5)
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Subset of the four supported primary metrics.")
    parser.add_argument("--output_root", default=None,
                        help="Default: meta_logs/<dataset>/<experiment>/selector_score.")
    parser.add_argument("--no_save_raw", action="store_true",
                        help="Do not save reconstructed [rollout,step,view] arrays.")
    parser.add_argument("--allow_replay_mismatch", action="store_true",
                        help="Keep results even if replay differs from selected_mask.")
    parser.set_defaults(non_roll=True)
    parser.add_argument("--non_roll", dest="non_roll", action="store_true")
    parser.add_argument("--roll", dest="non_roll", action="store_false")
    parser.add_argument("--non_like", action="store_true")
    args = parser.parse_args()
    if args.experiment is None:
        default_arch = "resnet18" if args.arch == "auto" else args.arch
        args.experiment = [
            DEFAULT_EXPERIMENT.replace("resnet18", default_arch, 1)]
    args.metrics = args.metrics or list(DEFAULT_METRICS)
    unknown = sorted(set(args.metrics).difference(DEFAULT_METRICS))
    if unknown:
        parser.error(
            "Unsupported --metrics: " + ", ".join(unknown)
            + ". Supported metrics: " + ", ".join(DEFAULT_METRICS)
        )
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.num_initial_cams < 0:
        parser.error("--num_initial_cams must be non-negative")
    return args


def timestamp_from_meta_name(name):
    try:
        value = datetime.datetime.fromisoformat(name)
    except ValueError:
        return None
    return value.strftime("%Y-%m-%d_%H-%M-%S")


def discover_runs(logs_dir, meta_exp_dir, experiment, run_limit=None):
    """Find timestamped log runs with an exact experiment-prefix match."""
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError(f"Logs directory does not exist: {logs_dir}")
    if not os.path.isdir(meta_exp_dir):
        raise FileNotFoundError(
            f"Experiment metadata directory does not exist: {meta_exp_dir}")

    meta_by_timestamp = {}
    for name in sorted(os.listdir(meta_exp_dir)):
        path = os.path.join(meta_exp_dir, name)
        if not os.path.isdir(path):
            continue
        timestamp = timestamp_from_meta_name(name)
        if timestamp is not None:
            meta_by_timestamp[timestamp] = (name, path)

    runs = []
    for name in sorted(os.listdir(logs_dir)):
        log_dir = os.path.join(logs_dir, name)
        if not os.path.isdir(log_dir):
            continue
        match = RUN_RE.fullmatch(name)
        if match is None or match.group("experiment") != experiment:
            continue
        timestamp = match.group("timestamp")
        meta_match = meta_by_timestamp.get(timestamp)
        if meta_match is None:
            print(f"SKIP {name}: no timestamp-matched metadata feature directory")
            continue
        meta_run_name, meta_run_dir = meta_match
        runs.append(RunSpec(
            experiment=experiment,
            timestamp=timestamp,
            log_dir=log_dir,
            meta_run_name=meta_run_name,
            meta_run_dir=meta_run_dir,
        ))
    if run_limit is not None:
        runs = runs[:run_limit]
    return runs


def indexed_files(directory, pattern):
    indexed = {}
    for name in os.listdir(directory):
        match = pattern.fullmatch(name)
        if match is not None:
            indexed[int(match.group("epoch"))] = os.path.join(directory, name)
    return indexed


def run_epoch_files(run):
    checkpoints = indexed_files(run.log_dir, CHECKPOINT_RE)
    features = indexed_files(run.meta_run_dir, FEATURE_RE)
    # model.pth is the final epoch even if model_e<final>.pth was not retained.
    final_path = os.path.join(run.log_dir, "model.pth")
    if os.path.exists(final_path) and features:
        final_epoch = max(features)
        checkpoints.setdefault(final_epoch, final_path)
    return checkpoints, features


def select_epochs(available, explicit, stride, max_epochs):
    available = sorted(available)
    if explicit is not None:
        requested = set(explicit)
        selected = [epoch for epoch in available if epoch in requested]
    elif stride > 0:
        selected = [epoch for epoch in available if epoch % stride == 0]
    else:
        selected = available
    if max_epochs is not None:
        selected = selected[:max_epochs]
    return selected


def infer_steps(experiment, explicit_steps):
    if explicit_steps is not None:
        return explicit_steps
    match = re.search(r"(?:^|_)steps(?P<steps>\d+)_", experiment)
    if match is None:
        match = re.search(r"steps(?P<steps>\d+)_", experiment)
    if match is None:
        raise ValueError(
            f"Cannot infer selector steps from '{experiment}'; pass --steps")
    steps = int(match.group("steps"))
    if steps < 1:
        raise ValueError("Selector-score analysis requires steps > 0")
    return steps


def infer_selector_limit(experiment):
    limits = [
        "foreshortened_family_remainder",
        "foreshortened_family",
        "expanded_family",
        "remainder",
    ]
    for limit in limits:
        if f"_selview_{limit}_" in f"_{experiment}_":
            return limit
    return "all"


def prepare_cache(args):
    if args.cache_csv is None:
        args.cache_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cache",
            f"midlevel_features_{CACHE_VERSION}_{args.split}.csv",
        )
    cache_args = SimpleNamespace(
        cache_csv=args.cache_csv,
        force_recompute=args.force_recompute_cache,
        data_root=args.data_root,
        split=args.split,
        limit_images=None,
    )
    return build_view_feature_cache(cache_args)


def build_metric_arrays(view_df, view_records, num_instances, num_views, metrics):
    cache = {
        (int(row.class_idx), os.path.basename(str(row.filename))): row
        for row in view_df.itertuples(index=False)
    }
    values = np.full(
        (num_instances, num_views, len(metrics)), np.nan, dtype=np.float32)
    families = np.full((num_instances, num_views), "Unknown", dtype=object)
    missing = 0
    for (instance_idx, view_idx), record in view_records.items():
        row = cache.get((int(record["class_idx"]), record["filename"]))
        if row is None:
            missing += 1
            continue
        families[instance_idx, view_idx] = str(row.view_type)
        for metric_idx, metric in enumerate(metrics):
            values[instance_idx, view_idx, metric_idx] = float(
                getattr(row, metric))
    if missing:
        raise RuntimeError(
            f"Mid-level cache is missing {missing} dataset instance/view rows")
    return values, families


def selector_keep_mask(families, limit):
    if limit == "all":
        return np.ones(families.shape, dtype=bool)
    expanded = families == "expanded"
    expanded_like = families == "Expanded-like"
    foreshortened = families == "Foreshortened"
    foreshortened_like = families == "Foreshortened-like"
    if limit == "expanded_family":
        return expanded | expanded_like
    if limit == "foreshortened_family":
        return foreshortened | foreshortened_like
    if limit == "foreshortened_family_remainder":
        return ~(expanded | expanded_like)
    if limit == "remainder":
        return ~(expanded | expanded_like | foreshortened | foreshortened_like)
    raise ValueError(f"Unknown selector limit: {limit}")


def choose_rollout_rows(initial_cams, args, num_views):
    if args.initial_cams is not None:
        selected_cams = sorted(set(args.initial_cams))
    elif 0 < args.num_initial_cams < num_views:
        selected_cams = sorted(set(np.linspace(
            0, num_views - 1, args.num_initial_cams, dtype=int).tolist()))
    else:
        selected_cams = list(range(num_views))
    invalid = [cam for cam in selected_cams if not 0 <= cam < num_views]
    if invalid:
        raise ValueError(f"Initial cameras outside [0,{num_views - 1}]: {invalid}")
    rows = np.flatnonzero(np.isin(initial_cams, selected_cams))
    if args.limit_trials is not None:
        rows = rows[:args.limit_trials]
    return rows, selected_cams


def safe_spearman(scores, cue_values):
    valid = np.isfinite(scores) & np.isfinite(cue_values)
    scores = scores[valid]
    cue_values = cue_values[valid]
    if (len(scores) < 3 or np.unique(scores).size < 2
            or np.unique(cue_values).size < 2):
        return np.nan, int(len(scores))
    result = stats.spearmanr(scores, cue_values)
    return float(result[0]), int(len(scores))


def chosen_percentile(cue_values, chosen_position):
    finite = np.isfinite(cue_values)
    if not finite[chosen_position] or finite.sum() < 2:
        return np.nan
    values = cue_values[finite]
    ranks = stats.rankdata(values, method="average")
    finite_positions = np.flatnonzero(finite)
    rank_position = int(np.flatnonzero(finite_positions == chosen_position)[0])
    return float((ranks[rank_position] - 1.0) / (len(values) - 1.0))


@torch.inference_mode()
def replay_epoch(args, model, feature_tensor, dump, metric_values, families,
                 keep_matrix, run, epoch, checkpoint_path, feature_path,
                 steps, output_dir):
    num_views = feature_tensor.shape[1]
    rollout_rows, initial_cams = choose_rollout_rows(
        dump["initial_cams"], args, num_views)
    if not len(rollout_rows):
        raise RuntimeError("No rollouts remain after initial-camera filtering")

    raw_scores = np.full(
        (len(rollout_rows), steps, num_views), np.nan, dtype=np.float32)
    raw_candidates = np.zeros(
        (len(rollout_rows), steps, num_views), dtype=bool)
    raw_actions = np.full((len(rollout_rows), steps), -1, dtype=np.int16)
    decision_rows = []

    for start in tqdm(
            range(0, len(rollout_rows), args.batch_size),
            desc=f"{run.timestamp} epoch {epoch}", leave=False):
        chunk_global_rows = rollout_rows[start:start + args.batch_size]
        chunk_instances = dump["instance_indices"][chunk_global_rows]
        chunk_initial = dump["initial_cams"][chunk_global_rows]
        batch_features = feature_tensor[chunk_instances].to(args.device)
        batch_keep = torch.as_tensor(
            keep_matrix[chunk_instances], dtype=torch.bool, device=args.device)
        current = torch.zeros(
            (len(chunk_global_rows), num_views), dtype=torch.bool,
            device=args.device)
        current.scatter_(1, torch.as_tensor(
            chunk_initial[:, None], dtype=torch.long, device=args.device), True)

        for step_idx in range(steps):
            _, _, candidate_mask = setup_args(
                batch_features, current, batch_keep)
            _, (_, action_values, action, _) = model.select_module(
                batch_features, current, batch_keep, eps_thres=-1.0)
            action_indices = action.long().argmax(dim=1)

            scores_np = action_values.detach().cpu().numpy().astype(np.float32)
            candidates_np = candidate_mask.detach().cpu().numpy().astype(bool)
            actions_np = action_indices.detach().cpu().numpy().astype(int)
            raw_scores[start:start + len(chunk_global_rows), step_idx] = scores_np
            raw_candidates[start:start + len(chunk_global_rows), step_idx] = candidates_np
            raw_actions[start:start + len(chunk_global_rows), step_idx] = actions_np

            for local_idx, global_row in enumerate(chunk_global_rows):
                instance_idx = int(chunk_instances[local_idx])
                candidate_indices = np.flatnonzero(candidates_np[local_idx])
                chosen_view = int(actions_np[local_idx])
                chosen_position = int(np.flatnonzero(
                    candidate_indices == chosen_view)[0])
                candidate_scores = scores_np[local_idx, candidate_indices]
                sorted_scores = np.sort(candidate_scores)
                score_margin = (
                    float(sorted_scores[-1] - sorted_scores[-2])
                    if len(sorted_scores) > 1 else np.nan)
                row = {
                    "experiment": run.experiment,
                    "run": os.path.basename(run.log_dir),
                    "timestamp": run.timestamp,
                    "epoch": int(epoch),
                    "step": int(step_idx + 1),
                    "rollout_index": int(global_row),
                    "instance_index": instance_idx,
                    "class_idx": int(dump["selected_classes"][global_row]),
                    "initial_cam": int(chunk_initial[local_idx]),
                    "n_candidates": int(len(candidate_indices)),
                    "chosen_view": chosen_view,
                    "chosen_view_family": str(families[instance_idx, chosen_view]),
                    "chosen_score": float(scores_np[local_idx, chosen_view]),
                    "score_margin": score_margin,
                    "checkpoint_path": checkpoint_path,
                    "feature_path": feature_path,
                }
                candidate_metrics = metric_values[
                    instance_idx, candidate_indices]
                for metric_idx, metric in enumerate(args.metrics):
                    cue = candidate_metrics[:, metric_idx]
                    rho, n_valid = safe_spearman(candidate_scores, cue)
                    row[f"rho_{metric}"] = rho
                    row[f"n_{metric}"] = n_valid
                    row[f"chosen_percentile_{metric}"] = chosen_percentile(
                        cue, chosen_position)
                decision_rows.append(row)
            current |= action.bool()

    # Reconstruct final masks from the stored raw actions without retaining all
    # GPU chunks. This also validates the exact checkpoint/feature pairing.
    replay_np = np.zeros((len(rollout_rows), num_views), dtype=bool)
    replay_np[np.arange(len(rollout_rows)), dump["initial_cams"][rollout_rows]] = True
    for step_idx in range(steps):
        actions = raw_actions[:, step_idx].astype(int)
        replay_np[np.arange(len(rollout_rows)), actions] = True
    expected = dump["selected_masks"][rollout_rows]
    replay_match = np.all(replay_np == expected, axis=1)
    match_rate = float(replay_match.mean())
    if match_rate < 1.0 and not args.allow_replay_mismatch:
        mismatches = int((~replay_match).sum())
        raise RuntimeError(
            f"Replay mismatch for {run.timestamp} epoch {epoch}: "
            f"{mismatches}/{len(replay_match)} rollouts differ from selected_mask. "
            "This usually means the checkpoint, feature dump, selector limit, or "
            "dataset options do not match. Pass --allow_replay_mismatch only for "
            "diagnosis."
        )

    decisions = pd.DataFrame(decision_rows)
    replay_by_rollout = dict(zip(rollout_rows.tolist(), replay_match.tolist()))
    decisions["replay_match"] = decisions["rollout_index"].map(
        replay_by_rollout).astype(bool)
    if not args.no_save_raw:
        raw_dir = os.path.join(output_dir, "raw", run.timestamp)
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, f"selector_scores_e{epoch}.npz")
        np.savez_compressed(
            raw_path,
            scores=raw_scores,
            candidate_mask=raw_candidates,
            actions=raw_actions,
            rollout_index=rollout_rows.astype(np.int32),
            instance_index=dump["instance_indices"][rollout_rows].astype(np.int32),
            initial_cam=dump["initial_cams"][rollout_rows].astype(np.int16),
            replay_match=replay_match,
            checkpoint_path=np.asarray(checkpoint_path),
            feature_path=np.asarray(feature_path),
        )
        print(f"Saved: {raw_path}")
    return decisions, match_rate, len(rollout_rows), initial_cams


def summarize_decisions(decisions, metrics):
    run_rows = []
    group_keys = ["experiment", "run", "epoch"]
    for group_values, epoch_block in decisions.groupby(group_keys, sort=True):
        experiment, run, epoch = group_values
        scopes = [(0, epoch_block)]
        scopes.extend(
            (int(step), block)
            for step, block in epoch_block.groupby("step", sort=True))
        for step, block in scopes:
            for metric in metrics:
                rho_values = pd.to_numeric(
                    block[f"rho_{metric}"], errors="coerce").dropna()
                percentile_values = pd.to_numeric(
                    block[f"chosen_percentile_{metric}"],
                    errors="coerce").dropna()
                run_rows.append({
                    "experiment": experiment,
                    "run": run,
                    "epoch": int(epoch),
                    "step": int(step),
                    "metric": metric,
                    "mean_rho": float(rho_values.mean()) if len(rho_values) else np.nan,
                    "mean_chosen_percentile": (
                        float(percentile_values.mean())
                        if len(percentile_values) else np.nan),
                    "n_decisions_rho": int(len(rho_values)),
                    "n_decisions_percentile": int(len(percentile_values)),
                })
    run_summary = pd.DataFrame(run_rows)

    summary_rows = []
    for keys, block in run_summary.groupby(
            ["experiment", "epoch", "step", "metric"], sort=True):
        experiment, epoch, step, metric = keys
        rho = block["mean_rho"].dropna().to_numpy(dtype=float)
        percentile = block["mean_chosen_percentile"].dropna().to_numpy(dtype=float)

        def mean_sem(values):
            if not len(values):
                return np.nan, np.nan
            mean = float(np.mean(values))
            sem = (
                float(np.std(values, ddof=1) / math.sqrt(len(values)))
                if len(values) > 1 else 0.0)
            return mean, sem

        mean_rho, sem_rho = mean_sem(rho)
        mean_percentile, sem_percentile = mean_sem(percentile)
        summary_rows.append({
            "experiment": experiment,
            "epoch": int(epoch),
            "step": int(step),
            "metric": metric,
            "mean_rho": mean_rho,
            "sem_rho_across_runs": sem_rho,
            "mean_chosen_percentile": mean_percentile,
            "sem_chosen_percentile_across_runs": sem_percentile,
            "n_runs_rho": int(len(rho)),
            "n_runs_percentile": int(len(percentile)),
            "n_decisions_rho": int(block["n_decisions_rho"].sum()),
            "n_decisions_percentile": int(
                block["n_decisions_percentile"].sum()),
        })
    return run_summary, pd.DataFrame(summary_rows)


def draw_heatmap(matrix, rows, columns, title, colorbar_label, output_path,
                 vmin, vmax, cmap):
    fig_width = max(7.5, 0.72 * len(columns) + 2.6)
    fig_height = max(3.2, 0.5 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Checkpoint epoch")
    ax.set_title(title)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:+.2f}", ha="center",
                        va="center", fontsize=7,
                        color="white" if abs(value - (vmin + vmax) / 2)
                        > 0.3 * (vmax - vmin) else "black")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_summary(summary, metrics, output_dir):
    overall = summary[summary["step"] == 0]
    epochs = sorted(overall["epoch"].unique())
    plot_specs = [
        (
            "mean_rho",
            "Mean within-decision selector-score association",
            "Mean within-decision rank correlation (rho)",
            "selector_score_correlation",
            -1.0,
            1.0,
            "RdBu_r",
        ),
        (
            "mean_chosen_percentile",
            "Cue percentile of the greedily chosen view",
            "Chosen cue percentile (0.5 = candidate median)",
            "selector_choice_percentile",
            0.0,
            1.0,
            "RdBu_r",
        ),
    ]
    groups = {
        "all": metrics,
        **{
            group: [metric for metric in group_metrics if metric in metrics]
            for group, group_metrics in PLOT_GROUPS.items()
        },
    }
    for value_col, title, colorbar, prefix, vmin, vmax, cmap in plot_specs:
        for group, group_metrics in groups.items():
            if not group_metrics:
                continue
            pivot = overall.pivot(index="metric", columns="epoch", values=value_col)
            matrix = pivot.reindex(
                index=group_metrics, columns=epochs).to_numpy(dtype=float)
            suffix = "" if group == "all" else f"_{group}"
            group_title = "" if group == "all" else (
                f" - {group.replace('_', ' ').title()}")
            draw_heatmap(
                matrix,
                [METRIC_LABELS.get(metric, metric) for metric in group_metrics],
                epochs,
                title + group_title,
                colorbar,
                os.path.join(output_dir, f"{prefix}{suffix}_heatmap.png"),
                vmin,
                vmax,
                cmap,
            )


def analyze_experiment(args, experiment, view_df):
    logs_dir = os.path.abspath(os.path.join(args.logs_root, args.dataset))
    meta_exp_dir = os.path.abspath(os.path.join(
        args.meta_root, args.dataset, experiment))
    output_dir = (
        os.path.join(os.path.abspath(args.output_root), experiment)
        if args.output_root else os.path.join(meta_exp_dir, "selector_score"))
    os.makedirs(output_dir, exist_ok=True)
    for obsolete in [
        "selector_score_correlation_symmetry_part_organization_heatmap.png",
        "selector_choice_percentile_symmetry_part_organization_heatmap.png",
    ]:
        obsolete_path = os.path.join(output_dir, obsolete)
        if os.path.exists(obsolete_path):
            os.remove(obsolete_path)

    runs = discover_runs(
        logs_dir, meta_exp_dir, experiment, run_limit=args.run_limit)
    if not runs:
        raise RuntimeError(
            f"No timestamp-matched runs found for exact experiment '{experiment}'")
    steps = infer_steps(experiment, args.steps)
    selector_limit = infer_selector_limit(experiment)
    print("=" * 72)
    print(f"Experiment:      {experiment}")
    print(f"Selector limit:  {selector_limit}")
    print(f"Matched runs:    {len(runs)}")
    for run in runs:
        print(f"  - {os.path.basename(run.log_dir)}")
    print(f"Output:          {output_dir}")
    print("=" * 72)

    manifest_rows = []
    all_decisions = []
    shared = None
    for run in runs:
        checkpoints, features = run_epoch_files(run)
        available = sorted(set(checkpoints).intersection(features))
        epochs = select_epochs(
            available, args.epochs, args.epoch_stride, args.max_epochs)
        if not epochs:
            print(f"SKIP {run.timestamp}: no requested checkpoint/feature pairs")
            continue

        model = create_model(checkpoints[epochs[0]], args, args.device)
        if shared is None:
            classnames = load_modelnet40_classnames()
            view_records, instance_lookup, instance_info = build_dataset_view_index(
                args, classnames, model.num_cam)
            metric_values, families = build_metric_arrays(
                view_df, view_records, len(instance_info), model.num_cam,
                args.metrics)
            keep_matrix = selector_keep_mask(families, selector_limit)
            shared = (
                view_records, instance_lookup, instance_info,
                metric_values, families, keep_matrix,
            )
        (view_records, instance_lookup, instance_info,
         metric_values, families, keep_matrix) = shared

        for epoch in epochs:
            checkpoint_path = checkpoints[epoch]
            feature_path = features[epoch]
            load_model_state(
                model, load_torch_state(checkpoint_path), checkpoint_path)
            dump = load_epoch_feature_dump(
                feature_path, model, instance_lookup, instance_info,
                need_feature_tensor=True)
            decisions, match_rate, n_rollouts, initial_cams = replay_epoch(
                args, model, dump["feature_tensor"], dump, metric_values,
                families, keep_matrix, run, epoch, checkpoint_path,
                feature_path, steps, output_dir)
            all_decisions.append(decisions)
            manifest_rows.append({
                "experiment": experiment,
                "run": os.path.basename(run.log_dir),
                "timestamp": run.timestamp,
                "epoch": int(epoch),
                "checkpoint_path": checkpoint_path,
                "feature_path": feature_path,
                "selector_view_limit": selector_limit,
                "n_rollouts": int(n_rollouts),
                "initial_cams": ",".join(map(str, initial_cams)),
                "replay_match_rate": match_rate,
            })
            print(
                f"loaded {run.timestamp} epoch {epoch}: "
                f"{n_rollouts} rollouts, replay match {match_rate:.1%}")

    if not all_decisions:
        raise RuntimeError(f"No selector decisions produced for {experiment}")
    decisions = pd.concat(all_decisions, ignore_index=True)
    decision_path = os.path.join(output_dir, "selector_score_decisions.csv")
    decisions.to_csv(decision_path, index=False)
    print(f"Saved: {decision_path} ({len(decisions)} decision rows)")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(output_dir, "run_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved: {manifest_path}")

    run_summary, summary = summarize_decisions(decisions, args.metrics)
    run_summary_path = os.path.join(output_dir, "selector_score_run_summary.csv")
    summary_path = os.path.join(output_dir, "selector_score_epoch_summary.csv")
    run_summary.to_csv(run_summary_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {run_summary_path}")
    print(f"Saved: {summary_path}")
    plot_summary(summary, args.metrics, output_dir)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for selector checkpoint replay")
    args.device = torch.device(f"cuda:{args.gpu_id}")
    view_df = prepare_cache(args)
    for experiment in args.experiment:
        analyze_experiment(args, experiment, view_df)
    print("Done.")


if __name__ == "__main__":
    main()
