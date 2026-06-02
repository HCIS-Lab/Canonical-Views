#!/usr/bin/env python3
"""Probe view-type informativeness via VGGT (or Pi3) single-view confidence.

For each of the 5 view-type buckets — Expanded, Expanded-like, Foreshortened,
Foreshortened-like, Remainder — and for each ModelNet instance, sample ONE
random view from that bucket, run it through the model in single-view mode,
and read the mean confidence (depth_conf, or sigmoid(conf) for Pi3) over the
object mask. Repeat N times per (instance, bucket) for an error bar.

Interpretation of single-view confidence:
    With N=1 input image, the model has no other view to cross-attend to, so
    the confidence is effectively a *monocular depth precision*: "how
    confidently can the model recover 3D structure from this image alone?"
    Geometrically ambiguous views (foreshortened) are expected to score
    lower than informative views (expanded).

Outputs (in <output_dir>):
    bar_overall.png        — 5 bars, mean ± std confidence per view type.
    per_class_heatmap.png  — view_type × class heatmap of mean confidence.
    per_class_grid.png     — small-multiples, one subplot per class.
    results.csv            — per-(model, run, instance, view_type) raw rows.

Usage:
    python scripts/view_type_single_view_confidence.py --models vggt --gpu_id 0
    python scripts/view_type_single_view_confidence.py --models vggt pi3 --n_runs 10
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from human_multiview.config import RESULTS_DIR
from human_multiview.data import build_view_type_index
from human_multiview.evaluate import extract_confidence_with_metric, get_object_mask
from human_multiview.models import get_model
from human_multiview.models.base import clear_gpu_memory


VIEW_TYPES = ["expanded", "Expanded-like", "Foreshortened", "Foreshortened-like", "Remainder"]
VIEW_DISPLAY = {
    "expanded":           "Expanded",
    "Expanded-like":      "Expanded-like",
    "Foreshortened":      "Foreshortened",
    "Foreshortened-like": "Foreshortened-like",
    "Remainder":          "Remainder",
}
PALETTE = ["#2ca02c", "#a6d854", "#d62728", "#fdae61", "#7f7f7f"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["vggt"],
                   help='Subset of {vggt, pi3}. Other models do not expose '
                        'single-view confidence cleanly.')
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--data_root", type=str,
                   default="/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--per_cls_instances", type=int, default=25)
    p.add_argument("--n_runs", type=int, default=1,
                   help="Random samples per (instance, view_type).")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="If set, evaluate only the first N instances (smoke test).")
    p.add_argument("--shard", type=str, default="0/1",
                   help='"i/N": this worker handles every N-th instance starting at i. '
                        'Use to fan-out the probe over N GPUs.')
    p.add_argument("--from_csv", nargs="+", default=None,
                   help="If set, skip eval; concat these CSVs and plot. "
                        "Use after parallel shards finish.")
    return p.parse_args()


def parse_shard(s):
    """'i/N' -> (i, N) with 0 <= i < N."""
    i_str, n_str = s.split("/")
    i, n = int(i_str), int(n_str)
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"Invalid --shard '{s}'; need 0 <= i < N and N >= 1.")
    return i, n


def evaluate_single_view(model, image):
    """Return one scalar: mean confidence over the object mask."""
    conf = model.extract_single_view_confidence(image)  # (1, 1, H, W)
    mask = get_object_mask(image)
    return float(extract_confidence_with_metric(conf[:, 0:1], mask, "mean"))


def run_one_model(model_name, device, index, n_runs, random_seed, limit):
    print(f"\n{'='*60}\nProbing {model_name.upper()} with single-view confidence\n{'='*60}")
    ModelClass = get_model(model_name)
    model = ModelClass()
    model.load(device)
    if not hasattr(model, "extract_single_view_confidence"):
        model.unload()
        raise SystemExit(
            f"{model_name} has no extract_single_view_confidence(); "
            "only vggt and pi3 are supported.")

    rng = random.Random(random_seed)
    instances = index if limit is None else index[:limit]
    rows = []
    for entry in tqdm(instances, desc=model_name):
        cls_name = entry["dataset"]
        ins_id = entry["instance_id"]
        for vt in VIEW_TYPES:
            available = entry["by_bucket"].get(vt, [])
            if not available:
                continue
            for run_idx in range(n_runs):
                # Deterministic seeding per (instance, view_type, run) so
                # the same path is taken across reruns. random.Random in
                # py3.11+ doesn't accept tuples; join into a string.
                local_rng = random.Random(f"{random_seed}|{cls_name}|{ins_id}|{vt}|{run_idx}")
                path = local_rng.choice(available)
                try:
                    image = Image.open(path).convert("RGB")
                    conf = evaluate_single_view(model, image)
                    rows.append({
                        "model": model_name,
                        "dataset": cls_name,
                        "class_idx": entry["class_idx"],
                        "instance_id": ins_id,
                        "view_type": vt,
                        "view_type_display": VIEW_DISPLAY[vt],
                        "run": run_idx,
                        "image_path": path,
                        "confidence_mean_masked": conf,
                    })
                except Exception as e:
                    print(f"  Error on {path}: {e}")
                    continue

    model.unload()
    clear_gpu_memory()
    return pd.DataFrame(rows)


def plot_bar_overall(df, model_name, n_runs, out_path):
    """Per-view-type bar chart, mean ± std across all (instance, run) pairs."""
    means, stds = [], []
    for vt in VIEW_TYPES:
        vals = df[df["view_type"] == vt]["confidence_mean_masked"].values
        means.append(float(np.mean(vals)) if len(vals) else float("nan"))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    labels = [VIEW_DISPLAY[v] for v in VIEW_TYPES]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=PALETTE,
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel(f"{model_name} mean confidence (over object mask)")
    ax.set_title(f"{model_name.upper()} single-view confidence by view type "
                 f"(mean ± std over {n_runs} runs × instances)")
    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.01 * max(means)),
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_heatmap(df, model_name, out_path):
    """view_type × class heatmap, mean confidence per cell."""
    classes = sorted(df["dataset"].unique())
    mat = np.full((len(VIEW_TYPES), len(classes)), np.nan)
    for i, vt in enumerate(VIEW_TYPES):
        for j, cls in enumerate(classes):
            vals = df[(df["view_type"] == vt) & (df["dataset"] == cls)]["confidence_mean_masked"].values
            if len(vals):
                mat[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(classes) + 2), 3.5))
    vmin = float(np.nanmin(mat)) if np.isfinite(np.nanmin(mat)) else 0.0
    vmax = float(np.nanmax(mat)) if np.isfinite(np.nanmax(mat)) else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(VIEW_TYPES)))
    ax.set_yticklabels([VIEW_DISPLAY[v] for v in VIEW_TYPES], fontsize=9)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.split(",")[0] for c in classes], rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Class")
    ax.set_title(f"{model_name.upper()} single-view confidence (mean per cell)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean confidence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_grid(df, model_name, n_runs, out_path):
    """Small multiples: one subplot per class, 5 bars per subplot."""
    classes = sorted(df["dataset"].unique())
    ncols = 4
    nrows = int(np.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 1.8),
                             sharey=True, squeeze=False)
    for idx, cls in enumerate(classes):
        ax = axes[idx // ncols][idx % ncols]
        means, stds = [], []
        for vt in VIEW_TYPES:
            vals = df[(df["view_type"] == vt) & (df["dataset"] == cls)]["confidence_mean_masked"].values
            means.append(float(np.mean(vals)) if len(vals) else 0.0)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ax.bar(range(len(VIEW_TYPES)), means, yerr=stds, capsize=2,
               color=PALETTE, edgecolor="black", linewidth=0.3)
        ax.set_xticks(range(len(VIEW_TYPES)))
        ax.set_xticklabels(["E", "E-l", "F", "F-l", "R"], fontsize=7)
        ax.set_title(cls.split(",")[0], fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.2)
    for k in range(len(classes), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.supylabel(f"{model_name} mean confidence", fontsize=10)
    fig.suptitle(f"{model_name.upper()} single-view confidence per class "
                 f"(E=Expanded, E-l=Expanded-like, F=Foreshortened, F-l=Foreshortened-like, R=Remainder)",
                 fontsize=10)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_all(df, model_name, n_runs, output_dir):
    csv_path = os.path.join(output_dir, f"{model_name}_single_view_confidence.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} rows)")
    plot_bar_overall(df, model_name, n_runs,
                     os.path.join(output_dir, f"{model_name}_bar_overall.png"))
    plot_per_class_heatmap(df, model_name,
                           os.path.join(output_dir, f"{model_name}_per_class_heatmap.png"))
    plot_per_class_grid(df, model_name, n_runs,
                        os.path.join(output_dir, f"{model_name}_per_class_grid.png"))
    print(f"Saved 3 plots for {model_name}.")


def main():
    args = parse_args()

    output_dir = args.output_dir or str(RESULTS_DIR / "single_view")
    os.makedirs(output_dir, exist_ok=True)

    # --- Plot-only mode: skip eval, concat shard CSVs, plot ---
    if args.from_csv:
        print(f"Plot-only mode: concatenating {len(args.from_csv)} CSV(s)...")
        all_df = pd.concat([pd.read_csv(p) for p in args.from_csv], ignore_index=True)
        # Deduplicate in case the same shard is passed twice
        dedup_cols = [c for c in ("model", "instance_id", "view_type", "run", "image_path")
                      if c in all_df.columns]
        if dedup_cols:
            before = len(all_df)
            all_df = all_df.drop_duplicates(subset=dedup_cols, keep="first")
            if before != len(all_df):
                print(f"  Dropped {before - len(all_df)} duplicate rows.")
        for model_name, df in all_df.groupby("model"):
            print(f"\n{'='*60}\nPlotting {model_name.upper()} ({len(df)} rows)\n{'='*60}")
            _plot_all(df.reset_index(drop=True), model_name, args.n_runs, output_dir)
        print("\nDone.")
        return

    # --- Normal eval path ---
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if "cuda" in device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = "cuda"

    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Models: {args.models}  n_runs={args.n_runs}")

    print(f"Indexing views from {args.data_root} (split={args.split}, "
          f"per_cls_instances={args.per_cls_instances})...")
    index = build_view_type_index(
        data_root=args.data_root,
        split=args.split,
        per_cls_instances=args.per_cls_instances,
    )
    print(f"Built index of {len(index)} instances across "
          f"{len(set(e['dataset'] for e in index))} classes.")

    if len(index) == 0:
        print("No instances available — check data_root.")
        return

    shard_i, shard_n = parse_shard(args.shard)
    if shard_n > 1:
        before = len(index)
        index = index[shard_i::shard_n]
        print(f"Shard {shard_i}/{shard_n}: keeping {len(index)} of {before} instances "
              f"(stride partition).")
    shard_suffix = f"_shard{shard_i}of{shard_n}" if shard_n > 1 else ""

    for model_name in args.models:
        df = run_one_model(model_name, device, index, args.n_runs,
                           args.random_seed, args.limit)
        if df.empty:
            print(f"  {model_name}: no rows produced.")
            continue

        csv_path = os.path.join(output_dir,
                                f"{model_name}_single_view_confidence{shard_suffix}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path} ({len(df)} rows)")

        # Skip plotting in sharded mode — plots are produced in a separate
        # --from_csv pass once all shards finish.
        if shard_n > 1:
            continue

        plot_bar_overall(df, model_name, args.n_runs,
                         os.path.join(output_dir, f"{model_name}_bar_overall.png"))
        plot_per_class_heatmap(df, model_name,
                               os.path.join(output_dir, f"{model_name}_per_class_heatmap.png"))
        plot_per_class_grid(df, model_name, args.n_runs,
                            os.path.join(output_dir, f"{model_name}_per_class_grid.png"))
        print(f"Saved 3 plots for {model_name}.")

    print("\nDone.")


if __name__ == "__main__":
    main()
