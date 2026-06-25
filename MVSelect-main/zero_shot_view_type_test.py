#!/usr/bin/env python3
"""Single-view view-type test of an MVCNN classifier.

Takes an MVCNN checkpoint and tests it with ONE random view per instance,
restricted to each of the 5 view-type buckets:

    Expanded, Expanded-like, Foreshortened, Foreshortened-like, Remainder

For each bucket, runs N_RUNS (default 5) random-sample passes and reports
mean ± std accuracy across runs. The checkpoint is auto-located via the same
mechanism main.py uses for stage-2 training: it reads
`logs/<dataset>/<arch>_performance.txt` and loads the path from the 2nd line.

Outputs (under <output_dir>, default `logs/<dataset>/zero_shot_view_test/`):
    bar_overall.png        — 5 bars (one per view type), mean ± std.
    per_class_heatmap.png  — view_type × class accuracy heatmap.
    per_class_grid.png     — small-multiples, one subplot per class.
    results.csv            — raw per-run accuracies for downstream analysis.

Usage:
    python zero_shot_view_type_test.py --dataset rgb --non_roll --num_train_instances 5
    python zero_shot_view_type_test.py --dataset rgb --non_roll --n_runs 10
"""

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets import ModelNet40, RGB_Depth_Edge_Dataset
from src.models.mvcnn import MVCNN
from src.trainer_mvcnn import ClassifierTrainer


VIEW_TYPES = [
    # (display name, internal name passed to restricted_view_test)
    ("Expanded",           "longest"),
    ("Expanded-like",      "long_like"),
    ("Foreshortened",      "short"),
    ("Foreshortened-like", "short_like"),
    ("Remainder",          "remainder"),
]

# Short class names for plotting (matches src/datasets/downstream_dataset.py order).
CLASS_NAMES = [
    "airplane", "ashcan", "basket", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "bus", "cabinet", "camera", "car", "chair", "clock",
    "display", "faucet", "guitar", "helmet", "knife", "lamp", "laptop",
    "loudspeaker", "motorcycle", "mug", "pistol", "table", "telephone",
    "tower", "train", "vessel", "washer",
]


def parse_args():
    p = argparse.ArgumentParser()
    # --- Args that mirror main.py for dataset/model construction ---
    p.add_argument("--dataset", type=str, default="rgb",
                   choices=["rgb", "depth", "edge", "rgb_depth", "rgb_edge",
                            "depth_edge", "rgb_depth_edge"])
    p.add_argument("--arch", type=str, default="resnet18")
    p.add_argument("--aggregation", type=str, default="max")
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_train_instances", type=int, default=5,
                   help="Mirrors main.py; used to build train_set for MVCNN init.")
    p.add_argument("--non_roll", action="store_true")
    p.add_argument("--non_like", action="store_true")
    p.add_argument("--down", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze_epoch", type=int, default=100,
                   help="If different from 100, looks up "
                        "`<arch>_freeze<N>_performance.txt` (matching main.py's "
                        "naming when --freeze_epoch is set), and suffixes the "
                        "default output_dir with `_freeze<N>`. Has no effect if "
                        "--checkpoint is provided.")

    # --- Test-specific args ---
    p.add_argument("--n_runs", type=int, default=5,
                   help="Random-sample repeats per view type.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Defaults to logs/<dataset>/zero_shot_view_test/.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override the auto-located checkpoint with a direct path.")
    p.add_argument("--gpu_id", type=int, default=0)
    return p.parse_args()


def build_fpath(dataset):
    if dataset == "rgb":
        return os.path.expanduser(
            "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23")
    if dataset == "edge":
        return os.path.expanduser(
            "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_edge_1_23_10.0")
    if dataset == "depth":
        return os.path.expanduser(
            "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/depth_modelnet_32_60_1_23")
    # Composite datasets: dict of representations
    fp = {}
    if "rgb" in dataset:
        fp["rgb"] = os.path.expanduser(
            "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23")
    if "depth" in dataset:
        fp["depth"] = os.path.expanduser(
            "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/depth_modelnet_32_60_1_23")
    if "edge" in dataset:
        fp["edge"] = os.path.expanduser(
            "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_edge_1_23_10.0")
    return fp


def locate_mvcnn_checkpoint(dataset, arch, freeze_epoch=100):
    """Mirror of main.py's checkpoint lookup. Reads
    logs/<dataset>/<arch>[_freeze<N>]_performance.txt and returns the path
    on its 2nd line."""
    suffix = f"_freeze{freeze_epoch}" if freeze_epoch != 100 else ""
    perf_path = f"logs/{dataset}/{arch}{suffix}_performance.txt"
    if not os.path.exists(perf_path):
        raise FileNotFoundError(
            f"Cannot find {perf_path}. Pass --checkpoint <path/to/model.pth> "
            f"or run the matching training first so the performance file is written.")
    with open(perf_path, "r") as f:
        result_str = f.read()
    print(result_str)
    load_dir = result_str.split("\n")[1].replace("# ", "")
    return os.path.join(load_dir, "model.pth")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # --- Output dir ---
    freeze_tag = f"_freeze{args.freeze_epoch}" if args.freeze_epoch != 100 else ""
    output_dir = args.output_dir or f"logs/{args.dataset}/zero_shot_view_test{freeze_tag}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # --- num_cam (mirror main.py) ---
    if args.non_roll:
        num_cam = 192 if args.non_like else 114
    else:
        num_cam = 4096

    # --- Datasets ---
    fpath = build_fpath(args.dataset)
    composite = args.dataset in ("rgb_depth", "rgb_edge", "depth_edge", "rgb_depth_edge")
    DatasetClass = RGB_Depth_Edge_Dataset if composite else ModelNet40

    train_set = DatasetClass(fpath, args.dataset, num_cam, split="train",
                             per_cls_instances=args.num_train_instances,
                             non_roll=args.non_roll, non_like=args.non_like)
    test_set = DatasetClass(fpath, args.dataset, num_cam, split="test",
                            per_cls_instances=5,
                            non_roll=args.non_roll, non_like=args.non_like)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # --- Model + checkpoint ---
    model = MVCNN(train_set, args.arch, args.aggregation, args.dataset).cuda()
    ckpt_path = args.checkpoint or locate_mvcnn_checkpoint(
        args.dataset, args.arch, args.freeze_epoch)
    print(f"Loading checkpoint: {ckpt_path}")
    pretrained = torch.load(ckpt_path, map_location="cuda")
    model_dict = model.state_dict()
    pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    model.eval()

    # --- Trainer wrapper (just to call restricted_view_test) ---
    args.steps = 0
    args.task = "mvcnn"
    args.epochs = 0
    args.base_lr_ratio = 1.0
    args.other_lr_ratio = 1.0
    trainer = ClassifierTrainer(model, output_dir, args)

    # --- Run the test for each view type ---
    print(f"\nRunning {args.n_runs} random-sample runs per view type "
          f"(single-view inference)...\n")
    per_view_runs = {}            # label -> array shape [n_runs]
    per_view_per_class = {}       # label -> array shape [n_runs, num_classes]
    for label, internal in VIEW_TYPES:
        _, _, _, (prec_per_run, per_class_per_run, _, _) = \
            trainer.restricted_view_test(test_loader, num_views=1,
                                         view_type=internal, n_runs=args.n_runs)
        per_view_runs[label] = np.asarray(prec_per_run, dtype=float)
        per_view_per_class[label] = np.asarray(per_class_per_run, dtype=float)
        print(f"  {label:20s}  mean={per_view_runs[label].mean():.2f}%  "
              f"std={per_view_runs[label].std(ddof=1) if args.n_runs > 1 else 0.0:.2f}%")

    # --- Save raw CSV ---
    rows = []
    for label in [lab for lab, _ in VIEW_TYPES]:
        for run_idx, acc in enumerate(per_view_runs[label]):
            row = {"view_type": label, "run": run_idx, "overall_acc_pct": float(acc)}
            for cls_idx, cls_name in enumerate(CLASS_NAMES[:per_view_per_class[label].shape[1]]):
                row[f"class_{cls_idx}_{cls_name}"] = float(per_view_per_class[label][run_idx, cls_idx])
            rows.append(row)
    csv_path = os.path.join(output_dir, "results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ========================
    # Plot 1: bar chart with error bars (overall accuracy per view type)
    # ========================
    labels = [lab for lab, _ in VIEW_TYPES]
    means = [per_view_runs[lab].mean() for lab in labels]
    stds = [per_view_runs[lab].std(ddof=1) if args.n_runs > 1 else 0.0 for lab in labels]
    palette = ["#2ca02c", "#a6d854", "#d62728", "#fdae61", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=palette,
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"MVCNN single-view accuracy "
                 f"(mean ± std over {args.n_runs} runs)")
    ax.set_ylim(0.0, max(100.0, max(m + s for m, s in zip(means, stds)) + 5))
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    bar_path = os.path.join(output_dir, "bar_overall.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {bar_path}")

    # ========================
    # Plot 2: per-class heatmap (view_type × class, mean across runs)
    # ========================
    n_classes = per_view_per_class[labels[0]].shape[1]
    mat = np.stack([per_view_per_class[lab].mean(axis=0) for lab in labels])  # [V, C]
    fig, ax = plt.subplots(figsize=(max(8, 0.25 * n_classes + 2), 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(CLASS_NAMES[:n_classes], rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Class")
    ax.set_title(f"MVCNN single-view accuracy by class × view type "
                 f"(mean over {args.n_runs} runs)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)")
    fig.tight_layout()
    heatmap_path = os.path.join(output_dir, "per_class_heatmap.png")
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {heatmap_path}")

    # ========================
    # Plot 3: per-class small-multiples grid (one subplot per class)
    # ========================
    ncols = 4
    nrows = int(np.ceil(n_classes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 1.8),
                             sharey=True, squeeze=False)
    for cls_idx in range(n_classes):
        ax = axes[cls_idx // ncols][cls_idx % ncols]
        cls_means = [per_view_per_class[lab][:, cls_idx].mean() for lab in labels]
        cls_stds = [per_view_per_class[lab][:, cls_idx].std(ddof=1)
                    if args.n_runs > 1 else 0.0 for lab in labels]
        ax.bar(range(len(labels)), cls_means, yerr=cls_stds, capsize=2,
               color=palette, edgecolor="black", linewidth=0.3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(["E", "E-l", "F", "F-l", "R"], fontsize=7)
        ax.set_title(CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls{cls_idx}",
                     fontsize=8)
        ax.set_ylim(0.0, 100.0)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.2)
    for k in range(n_classes, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.supylabel("Accuracy (%)", fontsize=10)
    fig.suptitle(f"MVCNN single-view accuracy per class "
                 f"(E=Expanded, E-l=Expanded-like, F=Foreshortened, F-l=Foreshortened-like, R=Remainder)",
                 fontsize=10)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    grid_path = os.path.join(output_dir, "per_class_grid.png")
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {grid_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
