#!/usr/bin/env python3
"""Pair-confidence control test.

For each ModelNet instance, sample three categories of pair inputs to VGGT
(or any pairwise-confidence model) and compare the resulting pair confidence:

    identical            — the same view paired with itself
    random               — two distinct random views drawn uniformly from the
                           instance's full view pool
    bucket pair (b1,b2)  — one view from bucket b1 + one view from bucket b2,
                           where b1,b2 ∈ {Expanded, Expanded-like,
                           Foreshortened, Foreshortened-like, Remainder}
                           (5 same-bucket + 10 cross-bucket = 15 conditions)

Pair confidence is computed the same way as elsewhere in the pipeline:
    VGGT(img_a, img_b) -> depth_conf shape (1, 2, H, W)
    conf_a  = mean of depth_conf[0, 0] over img_a's object mask
    conf_b  = mean of depth_conf[0, 1] over img_b's object mask
    pair    = (conf_a + conf_b) / 2

Outputs:
    <model>_bars.png       — 17 bars: identical, 5 same-bucket, 10 cross-bucket,
                              random. Color-coded by category.
    <model>_heatmap.png    — 5×5 bucket × bucket pair-confidence heatmap.
    <model>_pair_control.csv

Usage:
    python scripts/pair_confidence_control_test.py --models vggt
    GPUS=4 ./scripts/run_pair_confidence_control.sh
"""

import argparse
import os
import random
import sys
from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from human_multiview.config import RESULTS_DIR
from human_multiview.data import build_view_type_index
from human_multiview.evaluate import extract_confidence_with_metric, get_object_mask
from human_multiview.models import get_model
from human_multiview.models.base import clear_gpu_memory


VIEW_TYPES = ["expanded", "Expanded-like", "Foreshortened", "Foreshortened-like", "Remainder"]
VIEW_SHORT = {
    "expanded":           "E",
    "Expanded-like":      "E-l",
    "Foreshortened":      "F",
    "Foreshortened-like": "F-l",
    "Remainder":          "R",
}
# Color groups for the bar chart
COLOR_IDENTICAL = "#2ca02c"   # dark green
COLOR_SAME      = "#a6d854"   # light green
COLOR_CROSS     = "#fdae61"   # orange
COLOR_RANDOM    = "#7f7f7f"   # grey


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["vggt"],
                   help="Subset of {vggt, pi3, dust3r, mast3r}.")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--data_root", type=str,
                   default="/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--per_cls_instances", type=int, default=25)
    p.add_argument("--n_runs", type=int, default=5,
                   help="Random repeats per (instance, condition).")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", type=str, default="0/1",
                   help='"i/N" stride partition over instances.')
    p.add_argument("--from_csv", nargs="+", default=None,
                   help="If set, skip eval; concat these CSVs and plot.")
    return p.parse_args()


def parse_shard(s):
    i_str, n_str = s.split("/")
    i, n = int(i_str), int(n_str)
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"Invalid --shard '{s}'")
    return i, n


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_pair(model, img_a, img_b, mask_a, mask_b):
    """Pair confidence: average of per-view mean-over-mask confidence."""
    conf_maps = model.extract_pairwise_confidence(img_a, img_b)
    conf_a = extract_confidence_with_metric(conf_maps[:, 0:1], mask_a, "mean")
    conf_b = extract_confidence_with_metric(conf_maps[:, 1:2], mask_b, "mean")
    return float((conf_a + conf_b) / 2.0)


def all_bucket_pair_names():
    """Returns ordered list of all unordered bucket-pair condition names."""
    return [f"bucket_{VIEW_SHORT[b1]}_{VIEW_SHORT[b2]}"
            for (b1, b2) in combinations_with_replacement(VIEW_TYPES, 2)]


def run_one_model(model_name, device, index, n_runs, random_seed, limit):
    print(f"\n{'='*60}\nProbing {model_name.upper()}\n{'='*60}")
    ModelClass = get_model(model_name)
    model = ModelClass()
    model.load(device)

    instances = index if limit is None else index[:limit]
    bucket_pairs = list(combinations_with_replacement(VIEW_TYPES, 2))

    rows = []
    for entry in tqdm(instances, desc=model_name):
        cls_name = entry["dataset"]
        ins_id = entry["instance_id"]
        all_views = sorted({v for vs in entry["by_bucket"].values() for v in vs})
        if len(all_views) < 2:
            continue

        # Per-instance image/mask cache — same files appear across conditions.
        image_cache, mask_cache = {}, {}
        def load(path):
            if path not in image_cache:
                im = Image.open(path).convert("RGB")
                image_cache[path] = im
                mask_cache[path] = get_object_mask(im)
            return image_cache[path], mask_cache[path]

        for run_idx in range(n_runs):
            seed_base = f"{random_seed}|{cls_name}|{ins_id}|{run_idx}"
            run_rng = random.Random(seed_base)

            # --- Identical: same view paired with itself ---
            v = run_rng.choice(all_views)
            try:
                img, mask = load(v)
                conf = evaluate_pair(model, img, img, mask, mask)
                rows.append(dict(model=model_name, dataset=cls_name,
                                 class_idx=entry["class_idx"], instance_id=ins_id,
                                 run=run_idx, condition="identical",
                                 bucket_a=None, bucket_b=None,
                                 path_a=v, path_b=v, pair_confidence=conf))
            except Exception as e:
                print(f"  identical err {v}: {e}")

            # --- Random: two distinct random views from the full pool ---
            try:
                v1, v2 = run_rng.sample(all_views, 2)
                img1, mask1 = load(v1)
                img2, mask2 = load(v2)
                conf = evaluate_pair(model, img1, img2, mask1, mask2)
                rows.append(dict(model=model_name, dataset=cls_name,
                                 class_idx=entry["class_idx"], instance_id=ins_id,
                                 run=run_idx, condition="random",
                                 bucket_a=None, bucket_b=None,
                                 path_a=v1, path_b=v2, pair_confidence=conf))
            except Exception as e:
                print(f"  random err: {e}")

            # --- Bucket pairs (15: 5 same-bucket + 10 cross-bucket) ---
            for (b1, b2) in bucket_pairs:
                pool_a = entry["by_bucket"].get(b1, [])
                pool_b = entry["by_bucket"].get(b2, [])
                if not pool_a or not pool_b:
                    continue
                sub_rng = random.Random(f"{seed_base}|{b1}|{b2}")
                v1 = sub_rng.choice(pool_a)
                if b1 == b2:
                    pool2 = [x for x in pool_b if x != v1]
                    if not pool2:
                        continue
                    v2 = sub_rng.choice(pool2)
                else:
                    v2 = sub_rng.choice(pool_b)
                try:
                    img1, mask1 = load(v1)
                    img2, mask2 = load(v2)
                    conf = evaluate_pair(model, img1, img2, mask1, mask2)
                    rows.append(dict(model=model_name, dataset=cls_name,
                                     class_idx=entry["class_idx"], instance_id=ins_id,
                                     run=run_idx,
                                     condition=f"bucket_{VIEW_SHORT[b1]}_{VIEW_SHORT[b2]}",
                                     bucket_a=b1, bucket_b=b2,
                                     path_a=v1, path_b=v2, pair_confidence=conf))
                except Exception as e:
                    print(f"  {b1}+{b2} err: {e}")

    model.unload()
    clear_gpu_memory()
    return pd.DataFrame(rows)


# ============================================================================
# Plots
# ============================================================================

def plot_bars(df, model_name, out_path):
    """17 bars: identical | 5 same-bucket | 10 cross-bucket | random."""
    bucket_pairs = list(combinations_with_replacement(VIEW_TYPES, 2))
    same = [(b1, b2) for (b1, b2) in bucket_pairs if b1 == b2]
    cross = [(b1, b2) for (b1, b2) in bucket_pairs if b1 != b2]

    order = (
        [("identical", "Identical", COLOR_IDENTICAL)]
        + [(f"bucket_{VIEW_SHORT[b1]}_{VIEW_SHORT[b2]}",
            f"{VIEW_SHORT[b1]}+{VIEW_SHORT[b2]}", COLOR_SAME)
           for (b1, b2) in same]
        + [(f"bucket_{VIEW_SHORT[b1]}_{VIEW_SHORT[b2]}",
            f"{VIEW_SHORT[b1]}+{VIEW_SHORT[b2]}", COLOR_CROSS)
           for (b1, b2) in cross]
        + [("random", "Random", COLOR_RANDOM)]
    )

    means, stds, labels, colors = [], [], [], []
    for cond, label, color in order:
        vals = df[df["condition"] == cond]["pair_confidence"].values
        means.append(float(np.mean(vals)) if len(vals) else float("nan"))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        labels.append(label)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(labels)), 5.0))
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.4)
    ax.set_ylabel(f"{model_name} pair confidence (mean ± std)")
    ax.set_title(f"{model_name.upper()} pair confidence by composition")
    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * (max(means) if means else 1.0),
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        Patch(facecolor=COLOR_IDENTICAL, label="Identical (same view × 2)"),
        Patch(facecolor=COLOR_SAME, label="Same-bucket pair"),
        Patch(facecolor=COLOR_CROSS, label="Cross-bucket pair"),
        Patch(facecolor=COLOR_RANDOM, label="Random pair (any 2 views)"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bucket_heatmap(df, model_name, out_path):
    """5×5 symmetric heatmap of bucket × bucket pair confidence (mean)."""
    n = len(VIEW_TYPES)
    mat = np.full((n, n), np.nan)
    for i, b1 in enumerate(VIEW_TYPES):
        for j, b2 in enumerate(VIEW_TYPES):
            # Canonical unordered key.
            ba, bb = sorted([b1, b2], key=lambda x: VIEW_TYPES.index(x))
            cond = f"bucket_{VIEW_SHORT[ba]}_{VIEW_SHORT[bb]}"
            vals = df[df["condition"] == cond]["pair_confidence"].values
            if len(vals):
                mat[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    finite = mat[np.isfinite(mat)]
    if finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
    else:
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_xticklabels(VIEW_TYPES, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(VIEW_TYPES, fontsize=9)
    ax.set_title(f"{model_name.upper()} pair confidence by bucket pair (mean)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean pair confidence")
    threshold = (vmin + vmax) / 2.0
    for i in range(n):
        for j in range(n):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] < threshold else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_all(df, model_name, output_dir):
    csv_path = os.path.join(output_dir, f"{model_name}_pair_control.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} rows)")
    plot_bars(df, model_name, os.path.join(output_dir, f"{model_name}_bars.png"))
    plot_bucket_heatmap(df, model_name, os.path.join(output_dir, f"{model_name}_heatmap.png"))
    print(f"Saved 2 plots for {model_name}.")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    output_dir = args.output_dir or str(RESULTS_DIR / "pair_control")
    os.makedirs(output_dir, exist_ok=True)

    # Plot-only path
    if args.from_csv:
        print(f"Plot-only mode: concatenating {len(args.from_csv)} CSV(s)...")
        all_df = pd.concat([pd.read_csv(p) for p in args.from_csv], ignore_index=True)
        dedup = [c for c in ("model", "instance_id", "run", "condition", "path_a", "path_b")
                 if c in all_df.columns]
        if dedup:
            before = len(all_df)
            all_df = all_df.drop_duplicates(subset=dedup, keep="first")
            if before != len(all_df):
                print(f"  Dropped {before - len(all_df)} duplicate rows.")
        for model_name, df in all_df.groupby("model"):
            print(f"\nPlotting {model_name} ({len(df)} rows)")
            _plot_all(df.reset_index(drop=True), model_name, output_dir)
        print("\nDone.")
        return

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if "cuda" in device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = "cuda"

    print(f"Device: {device}, Output: {output_dir}, Models: {args.models}")
    print(f"Indexing views from {args.data_root} (split={args.split}, "
          f"per_cls_instances={args.per_cls_instances})...")
    index = build_view_type_index(
        data_root=args.data_root,
        split=args.split,
        per_cls_instances=args.per_cls_instances,
    )
    print(f"Built {len(index)} instances across "
          f"{len(set(e['dataset'] for e in index))} classes.")
    if len(index) == 0:
        print("No instances available.")
        return

    shard_i, shard_n = parse_shard(args.shard)
    if shard_n > 1:
        before = len(index)
        index = index[shard_i::shard_n]
        print(f"Shard {shard_i}/{shard_n}: keeping {len(index)} of {before} instances.")
    shard_suffix = f"_shard{shard_i}of{shard_n}" if shard_n > 1 else ""

    for model_name in args.models:
        df = run_one_model(model_name, device, index, args.n_runs,
                           args.random_seed, args.limit)
        if df.empty:
            print(f"  {model_name}: no rows produced.")
            continue
        csv_path = os.path.join(output_dir,
                                f"{model_name}_pair_control{shard_suffix}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path} ({len(df)} rows)")

        # Skip plots in sharded mode — done in a separate --from_csv pass.
        if shard_n > 1:
            continue
        _plot_all(df, model_name, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
