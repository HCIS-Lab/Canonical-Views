"""True zero-shot single-view classification probe via CLIP.

Parallel to `zero_shot_view_type_test.py`, but uses CLIP (which has never
seen ModelNet) instead of a supervised MVCNN classifier. This makes the probe
a *genuinely* zero-shot reading of "how
class-discriminable is each view type to a vision model with no
ModelNet-specific training".

For each of the 5 view-type buckets — Expanded / Expanded-like /
Foreshortened / Foreshortened-like / Remainder — samples one random view
per instance, encodes it with CLIP's image tower, and compares against
text-tower embeddings of per-class prompts. Reports top-1 AND top-5
accuracy (CLIP zero-shot top-1 on 32-class ModelNet is usually moderate;
top-5 is the more forgiving signal).

Outputs (default `logs/clip_zero_shot_view_test/`):
    bar_top1.png                — 5 bars, top-1 accuracy ± std per view type.
    bar_top5.png                — 5 bars, top-5 accuracy ± std per view type.
    per_class_top1_heatmap.png  — view_type × class top-1 accuracy.
    per_class_top5_heatmap.png  — view_type × class top-5 accuracy.
    results.csv                 — per-(run, instance, view_type) raw rows.

Usage:
    python clip_zero_shot_view_type.py
    python clip_zero_shot_view_type.py --clip_model openai/clip-vit-large-patch14
    python clip_zero_shot_view_type.py --no_prompt_ensemble  # use just "a photo of a X"
"""

import argparse
import os
import random
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# Class-name lists. Three parallel lists, all indexed by class_idx 0..31:
#   CLIP_CLASS_NAMES    — clean single-word labels used in CLIP text prompts
#                         (chosen to match how CLIP would describe each thing).
#   PLOT_CLASS_NAMES    — short labels shown on plot axes (match MVCNN probe).
#   MODELNET_DIR_NAMES  — the actual on-disk folder names (with WordNet
#                         synonym lists) for modelnet_32_60_1_23/.
# ---------------------------------------------------------------------------

CLIP_CLASS_NAMES = [
    "airplane", "trash can", "basket", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "bus", "cabinet", "camera", "car", "chair", "clock",
    "display", "faucet", "guitar", "helmet", "knife", "lamp", "laptop",
    "speaker", "motorcycle", "mug", "pistol", "table", "telephone",
    "tower", "train", "boat", "washing machine",
]
PLOT_CLASS_NAMES = [
    "airplane", "ashcan", "basket", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "bus", "cabinet", "camera", "car", "chair", "clock",
    "display", "faucet", "guitar", "helmet", "knife", "lamp", "laptop",
    "loudspeaker", "motorcycle", "mug", "pistol", "table", "telephone",
    "tower", "train", "vessel", "washer",
]
MODELNET_DIR_NAMES = [
    "airplane,aeroplane,plane",
    "ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin",
    "basket,handbasket",
    "bathtub,bathing tub,bath,tub",
    "bed", "bench", "bookshelf", "bottle", "bowl",
    "bus,autobus,coach,charabanc,double-decker,jitney,motorbus,motorcoach,omnibus,passenger vehi",
    "cabinet",
    "camera,photographic camera",
    "car,auto,automobile,machine,motorcar",
    "chair", "clock",
    "display,video display",
    "faucet,spigot",
    "guitar", "helmet", "knife", "lamp",
    "laptop,laptop computer",
    "loudspeaker,speaker,speaker unit,loudspeaker system,speaker system",
    "motorcycle,bike",
    "mug",
    "pistol,handgun,side arm,shooting iron",
    "table",
    "telephone,phone,telephone set",
    "tower",
    "train,railroad train",
    "vessel,watercraft",
    "washer,automatic washer,washing machine",
]
assert len(CLIP_CLASS_NAMES) == len(PLOT_CLASS_NAMES) == len(MODELNET_DIR_NAMES) == 32

VIEW_TYPES = ["expanded", "Expanded-like", "Foreshortened", "Foreshortened-like", "Remainder"]
VIEW_DISPLAY = {
    "expanded":           "Expanded",
    "Expanded-like":      "Expanded-like",
    "Foreshortened":      "Foreshortened",
    "Foreshortened-like": "Foreshortened-like",
    "Remainder":          "Remainder",
}
PALETTE = ["#2ca02c", "#a6d854", "#d62728", "#fdae61", "#7f7f7f"]

# Prompt templates. Default ensemble = 5 templates suited to 3D renders.
PROMPT_TEMPLATES_SIMPLE = ["a photo of a {}."]
PROMPT_TEMPLATES_3D = [
    "a photo of a {}.",
    "a 3D rendering of a {}.",
    "a grayscale 3D model of a {}.",
    "a CAD model of a {}.",
    "a digital 3D model of a {}.",
]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--data_root", type=str,
                   default="/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23",
                   help="Root of modelnet_32_60_1_23 (per-class subfolders).")
    p.add_argument("--split", default="test")
    p.add_argument("--per_cls_instances", type=int, default=25)
    p.add_argument("--n_runs", type=int, default=5,
                   help="Random samples per (instance, view_type).")
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32",
                   choices=["openai/clip-vit-base-patch32",
                            "openai/clip-vit-base-patch16",
                            "openai/clip-vit-large-patch14"],
                   help="HuggingFace CLIP variant.")
    p.add_argument("--no_prompt_ensemble", action="store_true",
                   help="Use single template 'a photo of a X' instead of "
                        "the 5-template 3D-aware ensemble (which is default).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--output_dir", default=None,
                   help="Defaults to logs/clip_zero_shot_view_test/.")
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke-test cap: evaluate only the first N instances.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# View-type bucket assignment from filename
# (matches the rules in human_multiview-main and MVSelect's modelnet40.py)
# ---------------------------------------------------------------------------

def classify_view_by_filename(fname):
    is_long = 'planar' in fname and 'short' not in fname
    is_short = 'short' in fname and 'like' not in fname
    is_long_like = 'like' in fname and 'short' not in fname
    is_short_like = 'like' in fname and 'short' in fname
    if is_short:
        return "Foreshortened"
    if is_long:
        return "expanded"
    if is_short_like:
        return "Foreshortened-like"
    if is_long_like:
        return "Expanded-like"
    return "Remainder"


def build_view_type_index(data_root, split, per_cls_instances):
    """Returns list of dicts: {class_idx, class_dir, instance_id, by_bucket}.

    `by_bucket` maps bucket_name → list of absolute image paths.
    """
    index = []
    for cls_idx, cls_dir in enumerate(MODELNET_DIR_NAMES):
        split_dir = os.path.join(data_root, cls_dir, split)
        if not os.path.isdir(split_dir):
            continue
        by_instance = {}
        for fname in sorted(os.listdir(split_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            ins_id = fname.split("_")[0]
            bucket = classify_view_by_filename(fname)
            by_instance.setdefault(ins_id, {}).setdefault(bucket, []).append(
                os.path.join(split_dir, fname))
        ins_ids = sorted(by_instance.keys())
        if per_cls_instances:
            ins_ids = ins_ids[:per_cls_instances]
        for ins_id in ins_ids:
            index.append({
                "class_idx": cls_idx,
                "class_dir": cls_dir,
                "instance_id": ins_id,
                "by_bucket": by_instance[ins_id],
            })
    return index


# ---------------------------------------------------------------------------
# CLIP setup + encoding
# ---------------------------------------------------------------------------

def precompute_text_features(clip_model, clip_processor, device, ensemble):
    """Returns [n_classes, dim] L2-normalised text embeddings.

    With ensemble=True, averages across all 3D-aware prompt templates per class.
    """
    templates = PROMPT_TEMPLATES_3D if ensemble else PROMPT_TEMPLATES_SIMPLE
    rows = []
    for cls_name in CLIP_CLASS_NAMES:
        prompts = [t.format(cls_name) for t in templates]
        inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.mean(dim=0)
            feats = feats / feats.norm()
        rows.append(feats)
    return torch.stack(rows, dim=0)


def encode_images_batched(clip_model, clip_processor, pil_images, device):
    """Encode a batch of PIL images, return L2-normalised [B, dim]."""
    inputs = clip_processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_bar(df, metric_col, metric_label, n_runs, out_path):
    means, stds = [], []
    for vt in VIEW_TYPES:
        vals = df[df["view_type"] == vt][metric_col].astype(float).values
        means.append(float(np.mean(vals)) * 100.0 if len(vals) else float("nan"))
        stds.append(float(np.std(vals, ddof=1)) * 100.0 if len(vals) > 1 else 0.0)
    labels = [VIEW_DISPLAY[v] for v in VIEW_TYPES]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=PALETTE,
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel(f"{metric_label} accuracy (%)")
    ax.set_title(f"CLIP zero-shot single-view {metric_label} accuracy by view type\n"
                 f"(mean ± std over {n_runs} runs × instances)")
    finite_means = [m for m in means if not np.isnan(m)]
    ymax = max(100.0, max((m + s) for m, s in zip(means, stds) if not np.isnan(m)) + 5
               if finite_means else 100.0)
    ax.set_ylim(0.0, ymax)
    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{m:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_heatmap(df, metric_col, metric_label, n_runs, out_path):
    n_classes = len(PLOT_CLASS_NAMES)
    mat = np.full((len(VIEW_TYPES), n_classes), np.nan)
    for i, vt in enumerate(VIEW_TYPES):
        for j, cls_name in enumerate(PLOT_CLASS_NAMES):
            vals = df[(df["view_type"] == vt) & (df["class_name"] == cls_name)][metric_col].astype(float).values
            if len(vals):
                mat[i, j] = float(np.mean(vals)) * 100.0
    fig, ax = plt.subplots(figsize=(max(8, 0.25 * n_classes + 2), 3.8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
    ax.set_yticks(range(len(VIEW_TYPES)))
    ax.set_yticklabels([VIEW_DISPLAY[v] for v in VIEW_TYPES], fontsize=9)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(PLOT_CLASS_NAMES, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Class")
    ax.set_title(f"CLIP zero-shot single-view {metric_label} accuracy "
                 f"(mean over {n_runs} runs)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{metric_label} accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = args.output_dir or "logs/clip_zero_shot_view_test"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")

    # CLIP. use_safetensors=True avoids the pytorch_model.bin path that
    # newer transformers refuses to load on torch <2.6 (CVE-2025-32434).
    print(f"Loading CLIP: {args.clip_model}")
    try:
        clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(device).eval()
    except (TypeError, ValueError, OSError) as e:
        # Older transformers don't accept use_safetensors=True. Fall back.
        print(f"  safetensors path failed ({e}); falling back to default load.")
        clip_model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    ensemble = not args.no_prompt_ensemble
    print(f"Computing text embeddings (prompt_ensemble={ensemble})...")
    text_features = precompute_text_features(clip_model, clip_processor, device, ensemble)
    print(f"  text_features: tuple{tuple(text_features.shape)}")

    # View-type index
    print(f"Indexing views from {args.data_root}...")
    index = build_view_type_index(args.data_root, args.split, args.per_cls_instances)
    print(f"  {len(index)} instances across "
          f"{len(set(e['class_idx'] for e in index))} classes.")
    if not index:
        sys.exit("No instances found — check --data_root and --split.")
    if args.limit:
        index = index[:args.limit]
        print(f"  --limit: keeping first {len(index)} instances.")

    # Build the full list of (instance, view_type, run) → image_path tuples.
    eval_items = []
    for entry in index:
        for vt in VIEW_TYPES:
            available = entry["by_bucket"].get(vt, [])
            if not available:
                continue
            for run_idx in range(args.n_runs):
                local_rng = random.Random(
                    f"{args.random_seed}|{entry['class_dir']}|{entry['instance_id']}|{vt}|{run_idx}")
                path = local_rng.choice(available)
                eval_items.append({
                    "class_idx": entry["class_idx"],
                    "class_dir": entry["class_dir"],
                    "instance_id": entry["instance_id"],
                    "view_type": vt,
                    "run": run_idx,
                    "image_path": path,
                })
    print(f"Total CLIP forwards: {len(eval_items)}")

    # Batched CLIP inference.
    rows = []
    for i in tqdm(range(0, len(eval_items), args.batch_size), desc="CLIP batches"):
        batch = eval_items[i:i + args.batch_size]
        try:
            pil_images = [Image.open(it["image_path"]).convert("RGB") for it in batch]
        except Exception as e:
            print(f"  Load error in batch {i}: {e}")
            continue
        img_features = encode_images_batched(clip_model, clip_processor, pil_images, device)
        similarities = img_features @ text_features.T  # [B, n_classes]
        top5 = similarities.topk(5, dim=1)
        top5_indices = top5.indices.cpu().numpy()
        for j, item in enumerate(batch):
            cls_idx = item["class_idx"]
            row_top5 = top5_indices[j].tolist()
            top1_idx = int(row_top5[0])
            rows.append({
                "class_idx": cls_idx,
                "class_dir": item["class_dir"],
                "class_name": PLOT_CLASS_NAMES[cls_idx],
                "instance_id": item["instance_id"],
                "view_type": item["view_type"],
                "view_type_display": VIEW_DISPLAY[item["view_type"]],
                "run": item["run"],
                "image_path": item["image_path"],
                "top1_pred_class_idx": top1_idx,
                "top1_pred_class_name": CLIP_CLASS_NAMES[top1_idx],
                "top5_pred_class_idxs": ",".join(map(str, row_top5)),
                "top1_correct": bool(top1_idx == cls_idx),
                "top5_correct": bool(cls_idx in row_top5),
            })

    if not rows:
        sys.exit("No results produced.")
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(df)} rows)")

    overall_top1 = df["top1_correct"].mean() * 100.0
    overall_top5 = df["top5_correct"].mean() * 100.0
    print(f"Overall top-1: {overall_top1:.2f}%   top-5: {overall_top5:.2f}%")

    # Plots
    plot_bar(df, "top1_correct", "Top-1", args.n_runs,
             os.path.join(output_dir, "bar_top1.png"))
    plot_bar(df, "top5_correct", "Top-5", args.n_runs,
             os.path.join(output_dir, "bar_top5.png"))
    plot_per_class_heatmap(df, "top1_correct", "Top-1", args.n_runs,
                           os.path.join(output_dir, "per_class_top1_heatmap.png"))
    plot_per_class_heatmap(df, "top5_correct", "Top-5", args.n_runs,
                           os.path.join(output_dir, "per_class_top5_heatmap.png"))
    print(f"Saved 4 plots in {output_dir}")


if __name__ == "__main__":
    main()
