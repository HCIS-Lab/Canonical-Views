#!/usr/bin/env python3
"""Compare model uncertainty on RL-selected vs random view subsets.

For each ModelNet instance covered by the MVSelect agent, this script builds
TWO view sets — the agent's top-K selected views and a random K-view sample
from the same instance — and runs both through each model in the same pass.

The output CSV has one row per instance with paired `agent_*` and `random_*`
columns suitable for downstream paired tests (e.g. Wilcoxon).

Metrics per set:
    - Pairwise (works for all multi-view models + DINOv2): mean/median/min/max
      and margin of pairwise confidence (or pairwise cosine similarity for
      DINOv2). Also reports the oddity-style image-score margin.
    - Joint (VGGT and Pi3 only): all N images through one forward pass; mean /
      median / min / max / margin of per-image confidence.

Usage:
    python scripts/run_evaluation_views.py \
        --models vggt dinov2 \
        --gpu_id 0 \
        --data_root /path/to/modelnet_32_60_1_23 \
        --selection_dir /path/to/505_incremental_reward/<run>/ \
        --selected_view_type 01234 \
        --start_epoch 80 --end_epoch 100 \
        --num_cam 4 \
        --split test \
        --per_cls_instances 5 \
        --output_dir results_views/
"""

import argparse
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from human_multiview.config import RESULTS_DIR
from human_multiview.data import load_modelnet_selected, open_trial_images
from human_multiview.models import get_model
from human_multiview.models.base import clear_gpu_memory
from human_multiview.evaluate import (
    evaluate_set_pairwise_confidence,
    evaluate_set_joint_confidence,
    evaluate_set_dinov2_similarity,
)


JOINT_CAPABLE = {"vggt", "pi3"}


def parse_args():
    p = argparse.ArgumentParser(description="Agent vs random view-set uncertainty")
    p.add_argument("--models", nargs="+",
                   default=["vggt", "dinov2"],
                   help="Models to evaluate.")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--data_root", type=str,
                   default="/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23",
                   help="Root of modelnet_32_60_1_23 (per-class subfolders).")
    p.add_argument("--selection_dir", type=str,
                   default="/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/MVSelect-main/meta_logs/rgb/resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100",
                   help="Folder containing the agent's *_selection.json files.")
    p.add_argument("--selected_view_type", type=str, default="01234",
                   help="Digits 0-4 selecting view-type buckets from VIEW_TYPE_LIST.")
    p.add_argument("--epoch_groups", type=str, default="80-100",
                   help='Comma-separated inclusive epoch ranges, e.g. '
                        '"1-10,11-20,21-30,31-40,41-50,51-60,61-70,71-80,81-90,91-100". '
                        'Each group is evaluated as a separate set of trials and rows '
                        'are tagged with epoch_start/epoch_end.')
    p.add_argument("--num_cam", type=int, default=4)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--per_cls_instances", type=int, default=5,
                   help="Cap on instances per class. 0 = no cap.")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="If set, evaluate only the first N trials (smoke test).")
    p.add_argument("--shard", type=str, default="0/1",
                   help='"i/N": this worker handles every N-th trial starting at i. '
                        'Use to fan a single eval out over N GPUs in parallel.')
    return p.parse_args()


def parse_shard(s):
    """'i/N' -> (i, N) with 0 <= i < N."""
    i_str, n_str = s.split("/")
    i, n = int(i_str), int(n_str)
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"Invalid --shard '{s}'; need 0 <= i < N and N >= 1.")
    return i, n


def _prefix(prefix, results):
    """Prefix the flat result-dict keys for CSV columns."""
    return {f"{prefix}_{k}": v for k, v in results.items()}


def parse_epoch_groups(s):
    """'1-10,11-20' -> [(1,10),(11,20)]."""
    groups = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-")
        groups.append((int(a), int(b)))
    return groups


_BUCKET_TO_COL = {
    "expanded":           "agent_n_expanded",
    "Expanded-like":      "agent_n_expanded_like",
    "Foreshortened":      "agent_n_foreshortened",
    "Foreshortened-like": "agent_n_foreshortened_like",
    "Remainder":          "agent_n_remainder",
}


def _base_row(trial, model_name):
    row = {
        "trial": trial["trial"],
        "dataset": trial["dataset"],
        "class_idx": trial["class_idx"],
        "instance_id": trial["instance_id"],
        "n_cam": trial["n_cam"],
        "epoch_start": trial["epoch_start"],
        "epoch_end": trial["epoch_end"],
        "selected_view_type": trial["selected_view_type"],
        "model": model_name,
    }
    bucket_counts = trial.get("agent_bucket_counts", {})
    for bucket, col in _BUCKET_TO_COL.items():
        row[col] = int(bucket_counts.get(bucket, 0))
    return row


def run_confidence_model(model_name, device, trials, limit):
    print(f"\n{'='*60}\nEvaluating {model_name.upper()} (pairwise + joint if supported)\n{'='*60}")
    ModelClass = get_model(model_name)
    model = ModelClass()
    model.load(device)
    can_joint = model_name in JOINT_CAPABLE

    rows = []
    iter_trials = trials if limit is None else trials[:limit]
    for trial in tqdm(iter_trials, desc=model_name):
        try:
            agent_imgs = open_trial_images(trial["agent_images_paths"])
            random_imgs = open_trial_images(trial["random_images_paths"])

            agent_pair = evaluate_set_pairwise_confidence(model, agent_imgs)
            random_pair = evaluate_set_pairwise_confidence(model, random_imgs)

            row = _base_row(trial, model_name)
            row.update(_prefix(f"{model_name}_agent", agent_pair))
            row.update(_prefix(f"{model_name}_random", random_pair))

            if can_joint:
                agent_joint = evaluate_set_joint_confidence(model, agent_imgs)
                random_joint = evaluate_set_joint_confidence(model, random_imgs)
                row.update(_prefix(f"{model_name}_agent", agent_joint))
                row.update(_prefix(f"{model_name}_random", random_joint))

            rows.append(row)
        except Exception as e:
            print(f"  Error on {trial['trial']}: {e}")
            continue

    model.unload()
    clear_gpu_memory()
    return pd.DataFrame(rows)


def run_dinov2(device, trials, limit):
    print(f"\n{'='*60}\nEvaluating DINOv2 (pairwise similarity)\n{'='*60}")
    ModelClass = get_model("dinov2")
    model = ModelClass()
    model.load(device)

    rows = []
    iter_trials = trials if limit is None else trials[:limit]
    for trial in tqdm(iter_trials, desc="dinov2"):
        try:
            agent_imgs = open_trial_images(trial["agent_images_paths"])
            random_imgs = open_trial_images(trial["random_images_paths"])

            agent_sim = evaluate_set_dinov2_similarity(model, agent_imgs)
            random_sim = evaluate_set_dinov2_similarity(model, random_imgs)

            row = _base_row(trial, "dinov2")
            row.update(_prefix("dinov2_agent", agent_sim))
            row.update(_prefix("dinov2_random", random_sim))
            rows.append(row)
        except Exception as e:
            print(f"  Error on {trial['trial']}: {e}")
            continue

    model.unload()
    clear_gpu_memory()
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if "cuda" in device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = "cuda"

    output_dir = args.output_dir or str(RESULTS_DIR / "views" / f"v{args.selected_view_type}")
    os.makedirs(output_dir, exist_ok=True)

    epoch_groups = parse_epoch_groups(args.epoch_groups)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Models: {args.models}")
    print(f"Selection dir: {args.selection_dir}")
    print(f"Epoch groups: {epoch_groups}")

    trials = []
    for start, end in epoch_groups:
        group_trials = load_modelnet_selected(
            data_root=args.data_root,
            selection_dir=args.selection_dir,
            selected_view_type=args.selected_view_type,
            start_epoch=start,
            end_epoch=end,
            num_cam=args.num_cam,
            split=args.split,
            per_cls_instances=args.per_cls_instances,
            random_seed=args.random_seed,
        )
        for t in group_trials:
            t["epoch_start"] = start
            t["epoch_end"] = end
            t["selected_view_type"] = args.selected_view_type
        print(f"  epochs {start}-{end}: {len(group_trials)} trials")
        trials.extend(group_trials)

    print(f"Built {len(trials)} total trials across {len(epoch_groups)} epoch group(s).")
    if len(trials) == 0:
        print("No trials built — check data_root, selection_dir, and split.")
        return

    shard_i, shard_n = parse_shard(args.shard)
    if shard_n > 1:
        before = len(trials)
        trials = trials[shard_i::shard_n]
        print(f"Shard {shard_i}/{shard_n}: keeping {len(trials)} of {before} trials "
              f"(stride partition).")
    shard_suffix = f"_shard{shard_i}of{shard_n}" if shard_n > 1 else ""

    for model_name in args.models:
        if model_name == "dinov2":
            df = run_dinov2(device, trials, args.limit)
        else:
            df = run_confidence_model(model_name, device, trials, args.limit)
        path = os.path.join(output_dir, f"{model_name}_views{shard_suffix}.csv")
        df.to_csv(path, index=False)
        print(f"Saved: {path} ({len(df)} rows)")

    print(f"\n{'='*60}\nDone.\n{'='*60}")


if __name__ == "__main__":
    main()
