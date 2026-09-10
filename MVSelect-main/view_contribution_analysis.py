#!/usr/bin/env python3
"""Relate selected-view shape cues to leave-one-view-out prediction value.

For every exact rollout mask stored with an experiment's feature_<epoch>.npz
files, this script evaluates the recognition model on the agent-selected set S,
on S without each selected view v_i, and after replacing v_i with an unselected
view u. It records three contribution scores for both interventions:

    delta_correct_logit = z_y(S) - z_y(S without v_i)
    delta_prediction_margin = margin(S) - margin(S without v_i)
    delta_replace_M = M(S) - M((S without v_i) union {u})

Positive values mean removing the view weakened that objective. The per-view
scores are joined to ellipse aspect ratio, bilateral symmetry, edge entropy,
and the exact five-way view family from the shared mid-level feature cache.
Numeric relationships use Spearman correlation by default. Replacement reports
both the selected cue and selected-minus-replacement cue difference. View
family is categorical, so it is summarized with per-family means and
eta-squared rather than assigning arbitrary numeric codes to the families.

The legacy *_selection.json identifies runs and epochs but is not used to
reconstruct S: it flattens actions across initial-camera rollouts. Exact sets
come from selected_mask and selected_init_cam in the feature dump.
"""

import argparse
import datetime
import glob
import json
import math
import os
import random
import tempfile
from collections import OrderedDict
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(),
                                                   "mvselect_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from scipy import stats
from tqdm import tqdm

from midlevel_shape_features import (
    CACHE_VERSION,
    DEFAULT_DATA_ROOT,
    build_view_feature_cache,
    classify_view_by_filename,
    load_modelnet40_classnames,
)
from src.models.mvcnn import MVCNN
from src.models.architectures import ARCHITECTURE_CHOICES, resolve_architecture
from src.models.mvselect import aggregate_feat
from temporal_selection_test import locate_final_checkpoint, resolve_selection_dir


FEATURES = [
    "ellipse_aspect_ratio",
    "bilateral_symmetry",
    "edge_entropy",
]

FEATURE_LABELS = {
    "ellipse_aspect_ratio": "Ellipse aspect ratio",
    "bilateral_symmetry": "Bilateral symmetry",
    "edge_entropy": "Edge entropy",
}

OBJECTIVES = OrderedDict([
    ("correct_logit", "Correct-class logit"),
    ("prediction_margin", "Top-1 - top-2 margin"),
])

OBJECTIVE_UNIT_LABELS = {
    "correct_logit": "Mean M(full) - M(intervention) (raw logit units)",
    "prediction_margin": (
        "Mean M(full) - M(intervention) (raw logit-margin units)"),
}

OBSOLETE_NEGATIVE_LOSS_FIGURES = [
    "correlation_negative_loss_heatmap.png",
    "view_family_negative_loss_heatmap.png",
    "replacement_correlation_negative_loss_heatmap.png",
    "replacement_view_family_negative_loss_heatmap.png",
    "replacement_family_pair_negative_loss_heatmap.png",
]

VIEW_FAMILIES = [
    "Expanded",
    "Expanded-like",
    "Foreshortened",
    "Foreshortened-like",
    "Remainder",
]

VIEW_FAMILY_NORMALIZATION = {
    "expanded": "Expanded",
    "Expanded": "Expanded",
    "Expanded-like": "Expanded-like",
    "Foreshortened": "Foreshortened",
    "Foreshortened-like": "Foreshortened-like",
    "Remainder": "Remainder",
}


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--selection_dir", required=True,
                   help="Experiment folder containing *_selection.json.")
    p.add_argument("--checkpoint", default=None,
                   help="Final model.pth. Auto-located from the experiment name if omitted.")
    p.add_argument("--dataset", default="rgb")
    p.add_argument("--arch", default="auto", choices=ARCHITECTURE_CHOICES,
                   help="Backbone architecture. auto infers it from the "
                        "experiment name and checkpoint.")
    p.add_argument("--aggregation", default="max", choices=["mean", "max"])
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--non_roll", action="store_true",
                   help="Use the 114-view zero-roll dataset layout.")
    p.add_argument("--non_like", action="store_true")
    p.add_argument("--test_per_cls_instances", type=int, default=5,
                   help="Must match the test_set used when features were saved.")
    p.add_argument("--cache_csv", default=None,
                   help="Shared per-view mid-level cache. Default: "
                        f"cache/midlevel_features_{CACHE_VERSION}_<split>.csv")
    p.add_argument("--force_recompute_cache", action="store_true")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--epoch_stride", type=int, default=10,
                   help="Evaluate epochs divisible by N. 0 evaluates every epoch.")
    p.add_argument("--epochs", nargs="+", type=int, default=None,
                   help="Explicit epoch list; overrides --epoch_stride.")
    p.add_argument("--max_epochs", type=int, default=None,
                   help="Smoke-test cap after epoch filtering.")
    p.add_argument("--limit_trials", type=int, default=None,
                   help="Smoke-test cap on selected object sets per epoch and run.")
    p.add_argument("--initial_cams", nargs="+", type=int, default=None,
                   help="Only analyze these initial camera indices.")
    p.add_argument("--num_initial_cams", type=int, default=0,
                   help="Use N evenly spaced initial cameras. 0 uses all cameras; "
                        "ignored when --initial_cams is given.")
    p.add_argument("--context", default="agent_only",
                   choices=["agent_only", "with_initial"],
                   help="Prediction context for each agent-view deletion. "
                        "agent_only uses only agent actions; with_initial retains "
                        "the rollout's initial input view.")
    p.add_argument("--feature_batch_size", type=int, default=64,
                   help="Images per batch when extracting fixed-final features.")
    p.add_argument("--aggregate_batch_size", type=int, default=256,
                   help="Intervention masks per classifier batch.")
    p.add_argument("--correlation", default="spearman",
                   choices=["spearman", "pearson"])
    p.add_argument("--replacement_samples", type=int, default=5,
                   help="Number of deterministic unselected replacement views "
                        "sampled per rollout. The same sampled views are used "
                        "for every selected-view replacement in that rollout.")
    p.add_argument("--replacement_seed", type=int, default=42)
    p.add_argument("--per_epoch_checkpoint", action="store_true",
                   help="Use model_e<E>.pth for selections at epoch E. Without this "
                        "flag, the final classifier is held fixed across epochs.")
    p.add_argument("--output_dir", default=None,
                   help="Default: <selection_dir>/view_contribution.")
    args = p.parse_args()
    if args.replacement_samples < 1:
        p.error("--replacement_samples must be at least 1")
    return args


def load_selection_runs(selection_dir):
    runs = []
    for filename in sorted(os.listdir(selection_dir)):
        if not filename.endswith("_selection.json"):
            continue
        path = os.path.join(selection_dir, filename)
        with open(path, "r") as handle:
            raw = json.load(handle)
        epochs = {int(epoch): content for epoch, content in raw.items()}
        runs.append((filename, epochs))
    return runs


def checkpoint_for_selection_run(args, run_name, fallback_checkpoint):
    """Match main.py's timestamped selection file to its model directory."""
    if args.checkpoint:
        return fallback_checkpoint, "explicit"

    suffix = "_selection.json"
    timestamp_text = run_name[:-len(suffix)] if run_name.endswith(suffix) else ""
    try:
        run_time = datetime.datetime.fromisoformat(timestamp_text)
    except ValueError:
        return fallback_checkpoint, "fallback_latest"

    experiment = os.path.basename(os.path.normpath(args.selection_dir))
    timestamp = run_time.strftime("%Y-%m-%d_%H-%M-%S")
    candidate = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "logs",
        args.dataset,
        f"{experiment}_{timestamp}",
        "model.pth",
    )
    if os.path.exists(candidate):
        return candidate, "selection_timestamp"
    return fallback_checkpoint, "fallback_latest"


def select_epochs(runs, explicit_epochs, stride, max_epochs):
    available = sorted({epoch for _, run in runs for epoch in run})
    if explicit_epochs is not None:
        wanted = set(explicit_epochs)
        chosen = [epoch for epoch in available if epoch in wanted]
    elif stride > 0:
        chosen = [epoch for epoch in available if epoch % stride == 0]
    else:
        chosen = available
    if max_epochs is not None:
        chosen = chosen[:max_epochs]
    return chosen


def parse_render_filename(filename):
    attrs = filename.split(".p", 1)[0].split("_")
    instance_id = attrs[0]
    camera_number = int(attrs[1])
    elevation = float(attrs[2][1:])
    azimuth = float(attrs[3][1:])
    roll = float(attrs[4][1:])
    pose_key = f"{elevation}_{azimuth}_{roll}"
    is_expanded_like = "like" in filename and "short" not in filename
    is_foreshortened_like = "like" in filename and "short" in filename
    return (instance_id, camera_number, roll, pose_key,
            is_expanded_like, is_foreshortened_like)


def build_dataset_view_index(args, classnames, num_views):
    """Reconstruct ModelNet test instance/view ordering without loading images."""
    pose_table = {}
    class_seen_ids = {class_name: [] for class_name in classnames}
    instance_lookup = {}
    instance_info = {}
    view_records = {}

    for class_idx, class_name in enumerate(classnames):
        pattern = os.path.join(args.data_root, class_name, args.split, "*.png")
        for image_path in sorted(glob.glob(pattern)):
            filename = os.path.basename(image_path)
            (instance_id, camera_number, roll, pose_key,
             is_expanded_like, is_foreshortened_like) = parse_render_filename(filename)
            if args.non_roll and roll != 0.0:
                continue
            if args.non_like and (is_expanded_like or is_foreshortened_like):
                continue
            if pose_key not in pose_table:
                pose_table[pose_key] = len(pose_table)
            view_idx = pose_table[pose_key]
            if view_idx >= num_views:
                continue
            if not args.non_roll and camera_number > num_views:
                continue
            if (len(class_seen_ids[class_name]) >= args.test_per_cls_instances
                    and instance_id not in class_seen_ids[class_name]):
                break

            instance_key = (class_idx, instance_id)
            if instance_key not in instance_lookup:
                instance_idx = len(instance_lookup)
                instance_lookup[instance_key] = instance_idx
                class_seen_ids[class_name].append(instance_id)
                instance_info[instance_idx] = {
                    "class_idx": class_idx,
                    "class_name": class_name,
                    "instance_id": instance_id,
                }
            instance_idx = instance_lookup[instance_key]
            view_records[(instance_idx, view_idx)] = {
                **instance_info[instance_idx],
                "view_idx": view_idx,
                "filename": filename,
                "image_path": image_path,
            }

    expected = len(instance_lookup) * num_views
    if len(view_records) != expected:
        raise RuntimeError(
            "Could not reconstruct a complete test-set view grid: "
            f"found {len(view_records)} instance/view records, expected {expected}. "
            "Check --non_roll, --non_like, and --test_per_cls_instances.")
    return view_records, instance_lookup, instance_info


def load_torch_state(path):
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {path}")
    state = {}
    for key, value in loaded.items():
        clean_key = key[7:] if key.startswith("module.") else key
        state[clean_key] = value
    return state


def infer_model_dimensions(state, fallback_num_cam):
    classifier_weight = state.get("classifier.weight")
    if classifier_weight is None:
        raise KeyError("Checkpoint has no classifier.weight")
    num_class = int(classifier_weight.shape[0])
    cam_embedding = state.get("select_module.cam_emb")
    num_cam = int(cam_embedding.shape[0]) if cam_embedding is not None else fallback_num_cam
    return num_cam, num_class


def load_model_state(model, state, path):
    expected = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in expected and tuple(value.shape) == tuple(expected[key].shape)
    }
    required = ["classifier.weight", "classifier.bias"]
    missing_required = [key for key in required if key not in compatible]
    if missing_required:
        raise RuntimeError(
            f"Checkpoint {path} is incompatible; missing {missing_required}")
    model.load_state_dict(compatible, strict=False)
    model.eval()


def create_model(checkpoint_path, args, device):
    state = load_torch_state(checkpoint_path)
    experiment_hint = getattr(args, "selection_dir", None)
    if not experiment_hint:
        experiments = getattr(args, "experiment", None)
        experiment_hint = experiments[0] if experiments else checkpoint_path
    arch = resolve_architecture(args.arch, experiment_hint, state)
    fallback_num_cam = 114
    num_cam, num_class = infer_model_dimensions(state, fallback_num_cam)
    dataset_stub = SimpleNamespace(num_cam=num_cam, num_class=num_class)
    model = MVCNN(dataset_stub, arch, args.aggregation, args.dataset).to(device)
    load_model_state(model, state, checkpoint_path)
    return model


def make_feature_index(view_df):
    index = {}
    for row in view_df.itertuples(index=False):
        index[(int(row.class_idx), os.path.basename(row.filename))] = row
    return index


def prepare_feature_cache(args):
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


def objective_values(logits, targets):
    if np.isscalar(targets):
        target_tensor = torch.full(
            (logits.shape[0],), int(targets), dtype=torch.long,
            device=logits.device)
    else:
        target_tensor = torch.as_tensor(
            targets, dtype=torch.long, device=logits.device).reshape(-1)
    correct_logit = logits.gather(1, target_tensor[:, None]).squeeze(1)
    top_two = torch.topk(logits, k=2, dim=1).values
    prediction_margin = top_two[:, 0] - top_two[:, 1]
    return {
        "correct_logit": correct_logit,
        "prediction_margin": prediction_margin,
    }


@torch.inference_mode()
def extract_fixed_feature_tensor(model, view_records, instance_info, transform,
                                 device, batch_size):
    """Extract all per-view features once for a fixed final checkpoint."""
    num_instances = len(instance_info)
    num_views = model.num_cam
    feature_dim = int(model.classifier.in_features)
    tensor = torch.empty(
        num_instances, num_views, feature_dim, 1, 1, dtype=torch.float32)
    entries = sorted(view_records.items())

    for start in tqdm(range(0, len(entries), batch_size),
                      desc="fixed-checkpoint view features", leave=False):
        batch_entries = entries[start:start + batch_size]
        images = []
        for _, record in batch_entries:
            with Image.open(record["image_path"]) as image:
                images.append(transform(image.convert("RGB")))
        image_tensor = torch.stack(images).unsqueeze(1).to(device)
        features, _ = model.get_feat(image_tensor, None, 1)
        features = features[:, 0].detach().cpu().float()
        for batch_idx, ((instance_idx, view_idx), _) in enumerate(batch_entries):
            tensor[instance_idx, view_idx] = features[batch_idx]
    return tensor


def infer_instance_feature_tensor(features, view_index, view_class, feature_dim):
    """Reconstruct [instance, view, feature] from view-major feature dumps."""
    features = np.asarray(features).squeeze().reshape(-1, feature_dim)
    view_index = np.asarray(view_index).reshape(-1).astype(int)
    view_class = np.asarray(view_class).reshape(-1).astype(int)
    if len(features) != len(view_index):
        raise ValueError("features and view_index have different lengths")

    position = 0
    feature_blocks = []
    class_blocks = []
    num_views = int(view_index.max()) + 1
    while position < len(view_index):
        if view_index[position] != 0:
            raise ValueError("Feature dump does not start a batch at view index 0")
        next_position = position
        while (next_position < len(view_index)
               and view_index[next_position] == 0):
            next_position += 1
        batch_size = next_position - position
        block_end = position + batch_size * num_views
        expected = np.repeat(np.arange(num_views), batch_size)
        if (block_end > len(view_index)
                or not np.array_equal(view_index[position:block_end], expected)):
            raise ValueError("Could not reconstruct view-major feature batches")
        block = features[position:block_end].reshape(
            num_views, batch_size, feature_dim).transpose(1, 0, 2)
        feature_blocks.append(block)
        class_blocks.append(view_class[position:position + batch_size])
        position = block_end

    return (torch.from_numpy(np.concatenate(feature_blocks)).float()
            .unsqueeze(-1).unsqueeze(-1),
            np.concatenate(class_blocks).astype(int))


def infer_legacy_rollout_instances(initial_cams, selected_classes, num_views):
    """Recover object indices from trainer.test's legacy batch-major order."""
    initial_cams = np.asarray(initial_cams).reshape(-1).astype(int)
    selected_classes = np.asarray(selected_classes).reshape(-1).astype(int)
    instance_indices = np.full(len(initial_cams), -1, dtype=int)
    position = 0
    instance_offset = 0
    while position < len(initial_cams):
        if initial_cams[position] != 0:
            raise ValueError("Legacy selected-mask block does not start at init cam 0")
        next_position = position
        while (next_position < len(initial_cams)
               and initial_cams[next_position] == 0):
            next_position += 1
        batch_size = next_position - position
        block_end = position + batch_size * num_views
        expected = np.repeat(np.arange(num_views), batch_size)
        if (block_end > len(initial_cams)
                or not np.array_equal(initial_cams[position:block_end], expected)):
            raise ValueError("Could not reconstruct legacy selected-mask batches")
        base = instance_offset + np.arange(batch_size)
        instance_indices[position:block_end] = np.tile(base, num_views)
        base_classes = selected_classes[position:position + batch_size]
        if not np.array_equal(
                selected_classes[position:block_end],
                np.tile(base_classes, num_views)):
            raise ValueError("Selected classes do not match rollout batch ordering")
        position = block_end
        instance_offset += batch_size
    return instance_indices


def load_epoch_feature_dump(path, model, instance_lookup, instance_info,
                            need_feature_tensor):
    with np.load(path) as npz:
        required = {"selected_mask", "selected_init_cam", "selected_class"}
        missing = required.difference(npz.files)
        if missing:
            raise KeyError(f"{path} is missing exact rollout arrays: {sorted(missing)}")
        selected_masks = np.asarray(npz["selected_mask"]).astype(bool)
        initial_cams = np.asarray(npz["selected_init_cam"]).reshape(-1).astype(int)
        selected_classes = np.asarray(npz["selected_class"]).reshape(-1).astype(int)
        if "selected_instance" in npz.files:
            instance_names = np.asarray(npz["selected_instance"]).reshape(-1).astype(str)
            instance_indices = np.asarray([
                instance_lookup[(int(class_idx), instance_name)]
                for class_idx, instance_name in zip(selected_classes, instance_names)
            ], dtype=int)
        else:
            instance_indices = infer_legacy_rollout_instances(
                initial_cams, selected_classes, selected_masks.shape[1])

        feature_tensor = None
        feature_classes = None
        if need_feature_tensor:
            feature_tensor, feature_classes = infer_instance_feature_tensor(
                npz["features"], npz["view_index"], npz["view_class"],
                int(model.classifier.in_features))

    if not (len(selected_masks) == len(initial_cams)
            == len(selected_classes) == len(instance_indices)):
        raise ValueError(f"Selected rollout arrays have inconsistent lengths: {path}")
    for row_idx, instance_idx in enumerate(instance_indices):
        if int(instance_idx) not in instance_info:
            raise ValueError(f"Unknown instance index {instance_idx} in {path}")
        expected_class = int(instance_info[int(instance_idx)]["class_idx"])
        if int(selected_classes[row_idx]) != expected_class:
            raise ValueError(
                f"Rollout class/instance mismatch at row {row_idx} in {path}")
    if feature_classes is not None:
        expected = np.asarray([
            instance_info[idx]["class_idx"] for idx in range(len(instance_info))
        ], dtype=int)
        if not np.array_equal(feature_classes, expected):
            raise ValueError(f"Saved feature instance order does not match dataset: {path}")
    return {
        "selected_masks": selected_masks,
        "initial_cams": initial_cams,
        "selected_classes": selected_classes,
        "instance_indices": instance_indices,
        "feature_tensor": feature_tensor,
    }


def choose_initial_cams(args, num_views):
    if args.initial_cams is not None:
        cameras = sorted(set(args.initial_cams))
    elif 0 < args.num_initial_cams < num_views:
        cameras = sorted(set(np.linspace(
            0, num_views - 1, args.num_initial_cams, dtype=int).tolist()))
    else:
        cameras = list(range(num_views))
    invalid = [camera for camera in cameras if not 0 <= camera < num_views]
    if invalid:
        raise ValueError(f"Initial camera indices outside [0, {num_views - 1}]: {invalid}")
    return cameras


@torch.inference_mode()
def forward_aggregated_masks(model, feature_tensor, instance_indices, masks,
                             device, batch_size):
    logits = []
    for start in range(0, len(instance_indices), batch_size):
        end = start + batch_size
        batch_instances = torch.as_tensor(
            instance_indices[start:end], dtype=torch.long)
        batch_features = feature_tensor[batch_instances].to(device)
        batch_masks = torch.as_tensor(
            masks[start:end], dtype=torch.bool, device=device)
        aggregated = aggregate_feat(
            batch_features, batch_masks, model.aggregation)
        logits.append(model.get_output(aggregated).detach().cpu())
    return torch.cat(logits, dim=0)


def exact_view_metadata(feature_index, view_record):
    record = feature_index.get(
        (int(view_record["class_idx"]), view_record["filename"]))
    if record is None:
        family = classify_view_by_filename(view_record["filename"])
        return {
            "view_index": int(view_record["view_idx"]),
            "view_family": VIEW_FAMILY_NORMALIZATION.get(family, family),
            **{feature: np.nan for feature in FEATURES},
        }
    family = VIEW_FAMILY_NORMALIZATION.get(
        str(record.view_type), str(record.view_type))
    return {
        "view_index": int(view_record["view_idx"]),
        "view_family": family,
        **{feature: float(getattr(record, feature)) for feature in FEATURES},
    }


def sample_unselected_views(selected_mask, sample_count, seed):
    """Sample deterministically from views outside the complete saved mask."""
    replacement_pool = np.flatnonzero(~np.asarray(selected_mask, dtype=bool))
    if not len(replacement_pool):
        return np.asarray([], dtype=int)
    count = min(int(sample_count), len(replacement_pool))
    return np.asarray(
        random.Random(seed).sample(replacement_pool.tolist(), count),
        dtype=int,
    )


def evaluate_epoch_rollouts(args, model, feature_tensor, dump, feature_index,
                            view_records, run_name, epoch, active_checkpoint,
                            checkpoint_match, device):
    leave_one_out_rows = []
    replacement_rows = []
    initial_cameras = choose_initial_cams(args, dump["selected_masks"].shape[1])
    rollout_rows = np.flatnonzero(np.isin(dump["initial_cams"], initial_cameras))
    if args.limit_trials is not None:
        rollout_rows = rollout_rows[:args.limit_trials]

    rollout_chunk_size = max(
        1, args.aggregate_batch_size // (args.replacement_samples + 2))
    skipped_too_small = 0
    skipped_no_replacement = 0
    for chunk_start in range(0, len(rollout_rows), rollout_chunk_size):
        chunk_rows = rollout_rows[chunk_start:chunk_start + rollout_chunk_size]
        eval_masks = []
        eval_instances = []
        eval_targets = []
        specs = []
        for rollout_row in chunk_rows:
            selected_mask = dump["selected_masks"][rollout_row].copy()
            initial_cam = int(dump["initial_cams"][rollout_row])
            instance_idx = int(dump["instance_indices"][rollout_row])
            target = int(dump["selected_classes"][rollout_row])
            agent_mask = selected_mask.copy()
            agent_mask[initial_cam] = False
            agent_views = np.flatnonzero(agent_mask)
            context_mask = selected_mask if args.context == "with_initial" else agent_mask
            if len(agent_views) == 0 or (
                    args.context == "agent_only" and len(agent_views) < 2):
                skipped_too_small += 1
                continue

            seed = (
                f"{args.replacement_seed}|{run_name}|{epoch}|"
                f"{int(rollout_row)}|{instance_idx}|{initial_cam}")
            replacement_views = sample_unselected_views(
                selected_mask, args.replacement_samples, seed)
            if not len(replacement_views):
                skipped_no_replacement += 1

            full_idx = len(eval_masks)
            eval_masks.append(context_mask.copy())
            eval_instances.append(instance_idx)
            eval_targets.append(target)
            leave_one_out_indices = []
            for view_idx in agent_views:
                without_mask = context_mask.copy()
                without_mask[view_idx] = False
                leave_one_out_indices.append(len(eval_masks))
                eval_masks.append(without_mask)
                eval_instances.append(instance_idx)
                eval_targets.append(target)

            replacement_specs = []
            for selected_index, view_idx in enumerate(agent_views):
                for replacement_sample, replacement_view_idx in enumerate(
                        replacement_views):
                    replaced_mask = context_mask.copy()
                    replaced_mask[view_idx] = False
                    replaced_mask[replacement_view_idx] = True
                    replacement_specs.append({
                        "selected_index": int(selected_index),
                        "view_idx": int(view_idx),
                        "replacement_sample": int(replacement_sample),
                        "replacement_view_idx": int(replacement_view_idx),
                        "eval_idx": len(eval_masks),
                    })
                    eval_masks.append(replaced_mask)
                    eval_instances.append(instance_idx)
                    eval_targets.append(target)
            specs.append({
                "rollout_row": int(rollout_row),
                "instance_idx": instance_idx,
                "target": target,
                "initial_cam": initial_cam,
                "agent_views": agent_views,
                "context_size": int(context_mask.sum()),
                "full_idx": full_idx,
                "leave_one_out_indices": leave_one_out_indices,
                "replacement_specs": replacement_specs,
            })

        if not specs:
            continue
        logits = forward_aggregated_masks(
            model,
            feature_tensor,
            np.asarray(eval_instances, dtype=int),
            np.stack(eval_masks),
            device,
            args.aggregate_batch_size,
        )
        values = objective_values(logits, np.asarray(eval_targets, dtype=int))
        predictions = logits.argmax(dim=1).numpy().astype(int)

        for spec in specs:
            full_idx = spec["full_idx"]
            target = spec["target"]
            instance_record = view_records[(spec["instance_idx"], 0)]
            common = {
                "run": run_name,
                "epoch": int(epoch),
                "checkpoint_protocol": (
                    "classifier_at_epoch_t" if args.per_epoch_checkpoint
                    else "final_classifier_fixed"),
                "checkpoint_path": active_checkpoint,
                "checkpoint_match": checkpoint_match,
                "feature_source": (
                    "saved_epoch_features" if args.per_epoch_checkpoint
                    else "recomputed_final_features"),
                "context": args.context,
                "rollout_index": spec["rollout_row"],
                "initial_cam": spec["initial_cam"],
                "class_idx": int(instance_record["class_idx"]),
                "class_name": instance_record["class_name"],
                "instance_id": instance_record["instance_id"],
                "agent_set_size": len(spec["agent_views"]),
                "context_size": spec["context_size"],
                "target": target,
                "full_prediction": int(predictions[full_idx]),
                "full_correct": int(predictions[full_idx] == target),
            }

            for selected_index, (view_idx, without_idx) in enumerate(zip(
                    spec["agent_views"], spec["leave_one_out_indices"])):
                view_record = view_records[(spec["instance_idx"], int(view_idx))]
                row = {
                    **common,
                    "loo_index": int(selected_index),
                    "filename": view_record["filename"],
                    "without_prediction": int(predictions[without_idx]),
                    "without_correct": int(predictions[without_idx] == target),
                    **exact_view_metadata(feature_index, view_record),
                }
                for objective in OBJECTIVES:
                    full_value = float(values[objective][full_idx].item())
                    without_value = float(values[objective][without_idx].item())
                    row[f"full_{objective}"] = full_value
                    row[f"without_{objective}"] = without_value
                    row[f"delta_{objective}"] = full_value - without_value
                leave_one_out_rows.append(row)

            for replacement_spec in spec["replacement_specs"]:
                selected_view_idx = replacement_spec["view_idx"]
                replacement_view_idx = replacement_spec["replacement_view_idx"]
                replacement_eval_idx = replacement_spec["eval_idx"]
                selected_record = view_records[
                    (spec["instance_idx"], selected_view_idx)]
                replacement_record = view_records[
                    (spec["instance_idx"], replacement_view_idx)]
                selected_metadata = exact_view_metadata(
                    feature_index, selected_record)
                replacement_metadata = exact_view_metadata(
                    feature_index, replacement_record)
                row = {
                    **common,
                    "selected_index": replacement_spec["selected_index"],
                    "filename": selected_record["filename"],
                    **selected_metadata,
                    "replacement_sample": replacement_spec["replacement_sample"],
                    "replacement_seed": args.replacement_seed,
                    "replacement_filename": replacement_record["filename"],
                    "replacement_view_index": replacement_metadata["view_index"],
                    "replacement_view_family": replacement_metadata["view_family"],
                    "replacement_prediction": int(
                        predictions[replacement_eval_idx]),
                    "replacement_correct": int(
                        predictions[replacement_eval_idx] == target),
                }
                for feature in FEATURES:
                    replacement_value = replacement_metadata[feature]
                    row[f"replacement_{feature}"] = replacement_value
                    row[f"selected_minus_replacement_{feature}"] = (
                        selected_metadata[feature] - replacement_value)
                for objective in OBJECTIVES:
                    full_value = float(values[objective][full_idx].item())
                    replacement_value = float(
                        values[objective][replacement_eval_idx].item())
                    row[f"full_{objective}"] = full_value
                    row[f"replacement_{objective}"] = replacement_value
                    row[f"delta_replace_{objective}"] = (
                        full_value - replacement_value)
                replacement_rows.append(row)
    return (leave_one_out_rows, replacement_rows, skipped_too_small,
            skipped_no_replacement)


def evaluate_exact_experiment(args, model, feature_index, view_records,
                              instance_lookup, instance_info, runs, epochs,
                              transform, device, final_checkpoint):
    leave_one_out_rows = []
    replacement_rows = []
    skipped = {
        "missing_feature_dump": 0,
        "missing_checkpoint": 0,
        "invalid_feature_dump": 0,
        "too_few_agent_views": 0,
        "no_unselected_replacement": 0,
    }
    fixed_feature_cache = {}

    for run_name, run in runs:
        run_checkpoint, run_match = checkpoint_for_selection_run(
            args, run_name, final_checkpoint)
        run_dir = os.path.join(
            args.selection_dir, run_name[:-len("_selection.json")])
        print(f"Run {run_name}: checkpoint={run_checkpoint} ({run_match})")

        fixed_features = None
        if not args.per_epoch_checkpoint:
            state = load_torch_state(run_checkpoint)
            load_model_state(model, state, run_checkpoint)
            if run_checkpoint not in fixed_feature_cache:
                fixed_feature_cache[run_checkpoint] = extract_fixed_feature_tensor(
                    model, view_records, instance_info, transform, device,
                    args.feature_batch_size)
            fixed_features = fixed_feature_cache[run_checkpoint]

        run_epochs = [epoch for epoch in epochs if epoch in run]
        for epoch in tqdm(run_epochs, desc=run_name, leave=False):
            feature_path = os.path.join(run_dir, f"feature_{epoch}.npz")
            if not os.path.exists(feature_path):
                skipped["missing_feature_dump"] += 1
                continue

            active_checkpoint = run_checkpoint
            if args.per_epoch_checkpoint:
                active_checkpoint = os.path.join(
                    os.path.dirname(run_checkpoint), f"model_e{epoch}.pth")
                if not os.path.exists(active_checkpoint):
                    skipped["missing_checkpoint"] += 1
                    continue
                state = load_torch_state(active_checkpoint)
                load_model_state(model, state, active_checkpoint)

            try:
                dump = load_epoch_feature_dump(
                    feature_path, model, instance_lookup, instance_info,
                    need_feature_tensor=args.per_epoch_checkpoint)
            except (KeyError, ValueError) as error:
                print(f"  Skip epoch {epoch}: {error}")
                skipped["invalid_feature_dump"] += 1
                continue
            feature_tensor = (
                dump["feature_tensor"] if args.per_epoch_checkpoint
                else fixed_features)
            (epoch_leave_one_out_rows, epoch_replacement_rows, too_small,
             no_replacement) = evaluate_epoch_rollouts(
                args, model, feature_tensor, dump, feature_index, view_records,
                run_name, epoch, active_checkpoint, run_match, device)
            leave_one_out_rows.extend(epoch_leave_one_out_rows)
            replacement_rows.extend(epoch_replacement_rows)
            skipped["too_few_agent_views"] += too_small
            skipped["no_unselected_replacement"] += no_replacement
    return (pd.DataFrame(leave_one_out_rows), pd.DataFrame(replacement_rows),
            skipped)


def compute_correlation(x, y, method):
    pair = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf], np.nan).dropna()
    n = len(pair)
    if n < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan, np.nan, n
    if method == "spearman":
        result = stats.spearmanr(pair["x"], pair["y"])
    else:
        result = stats.pearsonr(pair["x"], pair["y"])
    # Indexing supports both older scipy named tuples and newer result objects.
    return float(result[0]), float(result[1]), n


def analysis_metadata(block):
    return {
        "context": ",".join(sorted(block["context"].astype(str).unique())),
        "checkpoint_protocol": ",".join(sorted(
            block["checkpoint_protocol"].astype(str).unique())),
        "n_initial_cams": int(block["initial_cam"].nunique()),
        "initial_cams": ",".join(
            str(value) for value in sorted(block["initial_cam"].unique())),
    }


def correlation_table(contributions, method):
    rows = []
    scopes = [("overall", None, contributions)]
    scopes.extend(("epoch", int(epoch), block)
                  for epoch, block in contributions.groupby("epoch", sort=True))
    for scope, epoch, block in scopes:
        for objective in OBJECTIVES:
            delta_col = f"delta_{objective}"
            for feature in FEATURES:
                correlation, p_value, n = compute_correlation(
                    block[feature], block[delta_col], method)
                rows.append({
                    "scope": scope,
                    "epoch": epoch,
                    **analysis_metadata(block),
                    "objective": objective,
                    "feature": feature,
                    "method": method,
                    "correlation": correlation,
                    "p_value": p_value,
                    "n": n,
                })
    return pd.DataFrame(rows)


def replacement_selected_view_means(replacements):
    """Average sampled replacement effects once per selected rollout view."""
    group_columns = [
        "run",
        "epoch",
        "checkpoint_protocol",
        "checkpoint_path",
        "checkpoint_match",
        "feature_source",
        "context",
        "rollout_index",
        "initial_cam",
        "class_idx",
        "class_name",
        "instance_id",
        "agent_set_size",
        "context_size",
        "selected_index",
        "filename",
        "view_index",
        "view_family",
        "target",
        "full_prediction",
        "full_correct",
        "replacement_seed",
        *FEATURES,
    ]
    aggregations = {
        "replacement_sample": "nunique",
        "replacement_view_family": lambda values: ",".join(
            sorted(set(values.astype(str)))),
        "replacement_correct": "mean",
    }
    for objective in OBJECTIVES:
        aggregations[f"full_{objective}"] = "first"
        aggregations[f"replacement_{objective}"] = "mean"
        aggregations[f"delta_replace_{objective}"] = "mean"
    summary = replacements.groupby(
        group_columns, dropna=False, sort=False
    ).agg(aggregations).reset_index()
    return summary.rename(columns={
        "replacement_sample": "n_replacement_samples",
        "replacement_view_family": "sampled_replacement_families",
        "replacement_correct": "mean_replacement_correct",
    })


def replacement_correlation_table(replacements, selected_view_means, method):
    """Correlate replacement delta with selected cues and cue differences."""
    rows = []
    scopes = [("overall", None)]
    scopes.extend(("epoch", int(epoch))
                  for epoch in sorted(replacements["epoch"].unique()))
    for scope, epoch in scopes:
        pair_block = (
            replacements if epoch is None
            else replacements[replacements["epoch"] == epoch])
        selected_block = (
            selected_view_means if epoch is None
            else selected_view_means[selected_view_means["epoch"] == epoch])
        for objective in OBJECTIVES:
            delta_col = f"delta_replace_{objective}"
            for feature in FEATURES:
                correlation, p_value, n = compute_correlation(
                    selected_block[feature], selected_block[delta_col], method)
                rows.append({
                    "scope": scope,
                    "epoch": epoch,
                    **analysis_metadata(selected_block),
                    "objective": objective,
                    "cue_basis": "selected_view",
                    "feature": feature,
                    "feature_column": feature,
                    "analysis_unit": "selected_view_mean_over_replacements",
                    "method": method,
                    "correlation": correlation,
                    "p_value": p_value,
                    "n": n,
                })
                difference_col = f"selected_minus_replacement_{feature}"
                correlation, p_value, n = compute_correlation(
                    pair_block[difference_col], pair_block[delta_col], method)
                rows.append({
                    "scope": scope,
                    "epoch": epoch,
                    **analysis_metadata(pair_block),
                    "objective": objective,
                    "cue_basis": "selected_minus_replacement",
                    "feature": feature,
                    "feature_column": difference_col,
                    "analysis_unit": "selected_replacement_pair",
                    "method": method,
                    "correlation": correlation,
                    "p_value": p_value,
                    "n": n,
                })
    return pd.DataFrame(rows)


def eta_squared(values, groups):
    frame = pd.DataFrame({"value": values, "group": groups}).replace(
        [np.inf, -np.inf], np.nan).dropna()
    grouped = [block["value"].to_numpy(dtype=float)
               for _, block in frame.groupby("group") if len(block)]
    if len(frame) < 3 or len(grouped) < 2:
        return np.nan, np.nan, np.nan, len(frame), len(grouped)
    grand_mean = float(frame["value"].mean())
    ss_total = float(np.square(frame["value"] - grand_mean).sum())
    ss_between = sum(
        len(block) * (float(np.mean(block)) - grand_mean) ** 2
        for block in grouped)
    eta2 = float(ss_between / ss_total) if ss_total > 0 else np.nan
    try:
        anova = stats.f_oneway(*grouped)
        f_stat, p_value = float(anova.statistic), float(anova.pvalue)
    except ValueError:
        f_stat, p_value = np.nan, np.nan
    return eta2, f_stat, p_value, len(frame), len(grouped)


def family_tables(contributions, delta_prefix="delta_"):
    stats_rows = []
    association_rows = []
    scopes = [("overall", None, contributions)]
    scopes.extend(("epoch", int(epoch), block)
                  for epoch, block in contributions.groupby("epoch", sort=True))

    for scope, epoch, block in scopes:
        for objective in OBJECTIVES:
            delta_col = f"{delta_prefix}{objective}"
            eta2, f_stat, p_value, n, n_families = eta_squared(
                block[delta_col], block["view_family"])
            association_rows.append({
                "scope": scope,
                "epoch": epoch,
                **analysis_metadata(block),
                "objective": objective,
                "eta_squared": eta2,
                "anova_f": f_stat,
                "p_value": p_value,
                "n": n,
                "n_families": n_families,
            })
            for family in VIEW_FAMILIES:
                values = pd.to_numeric(
                    block.loc[block["view_family"] == family, delta_col],
                    errors="coerce").dropna().to_numpy(dtype=float)
                count = len(values)
                mean = float(np.mean(values)) if count else np.nan
                std = float(np.std(values, ddof=1)) if count > 1 else np.nan
                sem = std / math.sqrt(count) if count > 1 else np.nan
                stats_rows.append({
                    "scope": scope,
                    "epoch": epoch,
                    **analysis_metadata(block),
                    "objective": objective,
                    "view_family": family,
                    "n": count,
                    "mean_delta": mean,
                    "std_delta": std,
                    "sem_delta": sem,
                    "ci95_low": mean - 1.96 * sem if count > 1 else np.nan,
                    "ci95_high": mean + 1.96 * sem if count > 1 else np.nan,
                })
    return pd.DataFrame(stats_rows), pd.DataFrame(association_rows)


def replacement_family_pair_table(replacements):
    """Summarize selected-family to replacement-family interventions."""
    rows = []
    scopes = [("overall", None, replacements)]
    scopes.extend(("epoch", int(epoch), block)
                  for epoch, block in replacements.groupby("epoch", sort=True))
    for scope, epoch, block in scopes:
        for objective in OBJECTIVES:
            delta_col = f"delta_replace_{objective}"
            for selected_family in VIEW_FAMILIES:
                for replacement_family in VIEW_FAMILIES:
                    match = (
                        (block["view_family"] == selected_family)
                        & (block["replacement_view_family"]
                           == replacement_family))
                    values = pd.to_numeric(
                        block.loc[match, delta_col], errors="coerce"
                    ).dropna().to_numpy(dtype=float)
                    count = len(values)
                    mean = float(np.mean(values)) if count else np.nan
                    std = (
                        float(np.std(values, ddof=1)) if count > 1 else np.nan)
                    sem = std / math.sqrt(count) if count > 1 else np.nan
                    rows.append({
                        "scope": scope,
                        "epoch": epoch,
                        **analysis_metadata(block),
                        "objective": objective,
                        "selected_view_family": selected_family,
                        "replacement_view_family": replacement_family,
                        "n": count,
                        "mean_delta": mean,
                        "std_delta": std,
                        "sem_delta": sem,
                        "ci95_low": (
                            mean - 1.96 * sem if count > 1 else np.nan),
                        "ci95_high": (
                            mean + 1.96 * sem if count > 1 else np.nan),
                    })
    return pd.DataFrame(rows)


def draw_heatmap(matrix, row_labels, col_labels, title, colorbar_label,
                 out_path, vmin, vmax, cmap, annotate=True,
                 x_label="Selection epoch"):
    fig_width = max(7.5, 0.72 * len(col_labels) + 2.5)
    fig_height = max(2.8, 0.55 * len(row_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    if annotate:
        threshold = max(abs(vmin), abs(vmax)) * 0.55 if vmax != vmin else 0
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if not np.isfinite(value):
                    continue
                color = "white" if abs(value) > threshold else "black"
                ax.text(col_idx, row_idx, f"{value:+.2f}", ha="center",
                        va="center", fontsize=7, color=color)
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def correlation_axis_label(method):
    if str(method).lower() == "spearman":
        return "Spearman rank correlation (unitless)"
    if str(method).lower() == "pearson":
        return "Pearson correlation (unitless)"
    return f"{str(method).title()} correlation (unitless)"


def plot_correlations(correlations, output_dir):
    epoch_rows = correlations[correlations["scope"] == "epoch"]
    epochs = sorted(epoch_rows["epoch"].dropna().astype(int).unique())
    for objective, label in OBJECTIVES.items():
        block = epoch_rows[epoch_rows["objective"] == objective]
        pivot = block.pivot(index="feature", columns="epoch", values="correlation")
        matrix = pivot.reindex(index=FEATURES, columns=epochs).to_numpy(dtype=float)
        draw_heatmap(
            matrix,
            [FEATURE_LABELS[feature] for feature in FEATURES],
            epochs,
            f"Selected-view cue vs leave-one-out contribution: {label}",
            correlation_axis_label(block["method"].iloc[0]),
            os.path.join(output_dir, f"correlation_{objective}_heatmap.png"),
            -1.0,
            1.0,
            "RdBu_r",
        )

    overall = correlations[correlations["scope"] == "overall"]
    pivot = overall.pivot(index="objective", columns="feature", values="correlation")
    matrix = pivot.reindex(index=list(OBJECTIVES), columns=FEATURES).to_numpy(dtype=float)
    draw_heatmap(
        matrix,
        [OBJECTIVES[name] for name in OBJECTIVES],
        [FEATURE_LABELS[feature] for feature in FEATURES],
        "Overall selected-view cue/contribution relationship",
        correlation_axis_label(overall["method"].iloc[0]),
        os.path.join(output_dir, "overall_correlation_heatmap.png"),
        -1.0,
        1.0,
        "RdBu_r",
        x_label="Mid-level cue",
    )


def plot_replacement_correlations(correlations, output_dir):
    epoch_rows = correlations[correlations["scope"] == "epoch"]
    epochs = sorted(epoch_rows["epoch"].dropna().astype(int).unique())
    row_order = [
        ("selected_minus_replacement", feature)
        for feature in FEATURES
    ]
    row_labels = [FEATURE_LABELS[feature] for _, feature in row_order]
    for objective, label in OBJECTIVES.items():
        block = epoch_rows[epoch_rows["objective"] == objective]
        pivot = block.pivot(
            index=["cue_basis", "feature"], columns="epoch",
            values="correlation")
        matrix = pivot.reindex(
            index=pd.MultiIndex.from_tuples(
                row_order, names=["cue_basis", "feature"]),
            columns=epochs,
        ).to_numpy(dtype=float)
        draw_heatmap(
            matrix,
            row_labels,
            epochs,
            f"Selected - replacement cue difference vs prediction change: {label}",
            correlation_axis_label(block["method"].iloc[0]),
            os.path.join(
                output_dir,
                f"replacement_correlation_{objective}_heatmap.png"),
            -1.0,
            1.0,
            "RdBu_r",
        )

    overall = correlations[correlations["scope"] == "overall"]
    pivot = overall.pivot(
        index="objective", columns=["cue_basis", "feature"],
        values="correlation")
    matrix = pivot.reindex(
        index=list(OBJECTIVES),
        columns=pd.MultiIndex.from_tuples(
            row_order, names=["cue_basis", "feature"]),
    ).to_numpy(dtype=float)
    draw_heatmap(
        matrix,
        [OBJECTIVES[name] for name in OBJECTIVES],
        row_labels,
        "Overall selected - replacement cue difference vs prediction change",
        correlation_axis_label(overall["method"].iloc[0]),
        os.path.join(output_dir, "replacement_overall_correlation_heatmap.png"),
        -1.0,
        1.0,
        "RdBu_r",
        x_label="Mid-level cue",
    )


def plot_family_results(family_stats, family_association, output_dir,
                        filename_prefix="view_family",
                        intervention_label="leave-one-view-out"):
    epoch_stats = family_stats[family_stats["scope"] == "epoch"]
    epochs = sorted(epoch_stats["epoch"].dropna().astype(int).unique())
    for objective, label in OBJECTIVES.items():
        block = epoch_stats[epoch_stats["objective"] == objective]
        pivot = block.pivot(index="view_family", columns="epoch", values="mean_delta")
        matrix = pivot.reindex(index=VIEW_FAMILIES, columns=epochs).to_numpy(dtype=float)
        finite = matrix[np.isfinite(matrix)]
        scale = float(np.max(np.abs(finite))) if finite.size else 1.0
        scale = max(scale, 1e-8)
        draw_heatmap(
            matrix,
            VIEW_FAMILIES,
            epochs,
            f"Mean {intervention_label} contribution by exact view family: {label}",
            OBJECTIVE_UNIT_LABELS[objective],
            os.path.join(
                output_dir, f"{filename_prefix}_{objective}_heatmap.png"),
            -scale,
            scale,
            "RdBu_r",
        )

    epoch_assoc = family_association[family_association["scope"] == "epoch"]
    pivot = epoch_assoc.pivot(index="objective", columns="epoch", values="eta_squared")
    matrix = pivot.reindex(index=list(OBJECTIVES), columns=epochs).to_numpy(dtype=float)
    draw_heatmap(
        matrix,
        [OBJECTIVES[name] for name in OBJECTIVES],
        epochs,
        f"Exact view-family association with {intervention_label} contribution",
        "Eta-squared",
        os.path.join(output_dir, f"{filename_prefix}_eta_squared_heatmap.png"),
        0.0,
        1.0,
        "viridis",
    )


def plot_replacement_family_pairs(pair_stats, output_dir):
    epoch_rows = pair_stats[pair_stats["scope"] == "epoch"]
    epochs = sorted(epoch_rows["epoch"].dropna().astype(int).unique())
    pair_order = [
        (selected_family, replacement_family)
        for selected_family in VIEW_FAMILIES
        for replacement_family in VIEW_FAMILIES
    ]
    pair_labels = [
        f"{selected_family} -> {replacement_family}"
        for selected_family, replacement_family in pair_order
    ]
    pair_index = pd.MultiIndex.from_tuples(
        pair_order,
        names=["selected_view_family", "replacement_view_family"],
    )
    for objective, label in OBJECTIVES.items():
        block = epoch_rows[epoch_rows["objective"] == objective]
        pivot = block.pivot(
            index=["selected_view_family", "replacement_view_family"],
            columns="epoch", values="mean_delta")
        matrix = pivot.reindex(
            index=pair_index, columns=epochs).to_numpy(dtype=float)
        finite = matrix[np.isfinite(matrix)]
        scale = float(np.max(np.abs(finite))) if finite.size else 1.0
        scale = max(scale, 1e-8)
        draw_heatmap(
            matrix,
            pair_labels,
            epochs,
            f"Mean selected-to-unselected replacement contribution: {label}",
            OBJECTIVE_UNIT_LABELS[objective],
            os.path.join(
                output_dir,
                f"replacement_family_pair_{objective}_heatmap.png"),
            -scale,
            scale,
            "RdBu_r",
            annotate=False,
        )


def main():
    args = parse_args()
    resolved = resolve_selection_dir(args.selection_dir, args.dataset)
    if resolved is None:
        raise SystemExit(
            f"No *_selection.json found in {args.selection_dir} or "
            f"meta_logs/{args.dataset}/{args.selection_dir}")
    args.selection_dir = resolved
    args.output_dir = args.output_dir or os.path.join(
        args.selection_dir, "view_contribution")
    os.makedirs(args.output_dir, exist_ok=True)
    for filename in OBSOLETE_NEGATIVE_LOSS_FIGURES:
        path = os.path.join(args.output_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed obsolete figure: {path}")

    runs = load_selection_runs(args.selection_dir)
    if not runs:
        raise SystemExit("No selection runs loaded.")
    epochs = select_epochs(
        runs, args.epochs, args.epoch_stride, args.max_epochs)
    if not epochs:
        raise SystemExit(
            "No epochs remain after filtering. Use --epoch_stride 0 or --epochs E ...")

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Selection dir: {args.selection_dir}")
    print(f"Output:        {args.output_dir}")
    print(f"Device:        {device}")
    print(f"Epochs:        {epochs}")
    print(f"Context:       {args.context}")
    print(f"Replacements:  {args.replacement_samples} unselected views per rollout "
          f"(seed={args.replacement_seed})")

    classnames = load_modelnet40_classnames()
    view_df = prepare_feature_cache(args)
    missing_features = [feature for feature in FEATURES if feature not in view_df]
    if missing_features:
        raise KeyError(
            f"Mid-level cache is missing {missing_features}. Recompute with "
            "--force_recompute_cache.")
    feature_index = make_feature_index(view_df)

    final_checkpoint = locate_final_checkpoint(args)
    model = create_model(final_checkpoint, args, device)
    view_records, instance_lookup, instance_info = build_dataset_view_index(
        args, classnames, model.num_cam)
    selected_initial_cams = choose_initial_cams(args, model.num_cam)
    print(f"Initial cams:  {selected_initial_cams} "
          f"({len(selected_initial_cams)}/{model.num_cam})")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)),
    ])

    contributions, replacements, skipped = evaluate_exact_experiment(
        args, model, feature_index, view_records, instance_lookup,
        instance_info, runs, epochs, transform, device, final_checkpoint)
    if contributions.empty:
        raise SystemExit(f"No contribution rows produced. Skipped: {skipped}")
    if replacements.empty:
        raise SystemExit(f"No replacement rows produced. Skipped: {skipped}")

    contribution_path = os.path.join(
        args.output_dir, "leave_one_view_out_contributions.csv")
    contributions.to_csv(contribution_path, index=False)
    print(f"Saved: {contribution_path} ({len(contributions)} view rows)")

    correlations = correlation_table(contributions, args.correlation)
    correlation_path = os.path.join(
        args.output_dir, "contribution_correlations.csv")
    correlations.to_csv(correlation_path, index=False)
    print(f"Saved: {correlation_path}")

    family_stats, family_association = family_tables(contributions)
    family_stats_path = os.path.join(
        args.output_dir, "view_family_contributions.csv")
    family_association_path = os.path.join(
        args.output_dir, "view_family_association.csv")
    family_stats.to_csv(family_stats_path, index=False)
    family_association.to_csv(family_association_path, index=False)
    print(f"Saved: {family_stats_path}")
    print(f"Saved: {family_association_path}")

    replacement_path = os.path.join(
        args.output_dir, "selected_to_unselected_replacements.csv")
    replacements.to_csv(replacement_path, index=False)
    print(f"Saved: {replacement_path} ({len(replacements)} replacement rows)")

    replacement_means = replacement_selected_view_means(replacements)
    replacement_means_path = os.path.join(
        args.output_dir, "replacement_selected_view_means.csv")
    replacement_means.to_csv(replacement_means_path, index=False)
    print(f"Saved: {replacement_means_path} "
          f"({len(replacement_means)} selected-view rows)")

    replacement_correlations = replacement_correlation_table(
        replacements, replacement_means, args.correlation)
    replacement_correlation_path = os.path.join(
        args.output_dir, "replacement_correlations.csv")
    replacement_correlations.to_csv(
        replacement_correlation_path, index=False)
    print(f"Saved: {replacement_correlation_path}")

    replacement_family_stats, replacement_family_association = family_tables(
        replacement_means, delta_prefix="delta_replace_")
    replacement_family_stats_path = os.path.join(
        args.output_dir, "replacement_view_family_contributions.csv")
    replacement_family_association_path = os.path.join(
        args.output_dir, "replacement_view_family_association.csv")
    replacement_family_stats.to_csv(
        replacement_family_stats_path, index=False)
    replacement_family_association.to_csv(
        replacement_family_association_path, index=False)
    print(f"Saved: {replacement_family_stats_path}")
    print(f"Saved: {replacement_family_association_path}")

    replacement_family_pairs = replacement_family_pair_table(replacements)
    replacement_family_pair_path = os.path.join(
        args.output_dir, "replacement_family_pair_contributions.csv")
    replacement_family_pairs.to_csv(
        replacement_family_pair_path, index=False)
    print(f"Saved: {replacement_family_pair_path}")

    plot_correlations(correlations, args.output_dir)
    plot_family_results(family_stats, family_association, args.output_dir)
    plot_replacement_correlations(replacement_correlations, args.output_dir)
    plot_family_results(
        replacement_family_stats,
        replacement_family_association,
        args.output_dir,
        filename_prefix="replacement_view_family",
        intervention_label="selected-to-unselected replacement",
    )
    plot_replacement_family_pairs(replacement_family_pairs, args.output_dir)

    missing_midlevel = int(contributions[FEATURES].isna().any(axis=1).sum())
    missing_replacement_midlevel = int(replacements[
        FEATURES + [f"replacement_{feature}" for feature in FEATURES]
    ].isna().any(axis=1).sum())
    print(f"Skipped: {skipped}")
    print(f"Leave-one-out rows missing descriptors: {missing_midlevel}")
    print("Replacement rows missing selected/replacement descriptors: "
          f"{missing_replacement_midlevel}")
    print("Done.")


if __name__ == "__main__":
    main()
