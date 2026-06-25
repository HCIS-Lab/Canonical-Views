#!/usr/bin/env python3
"""Track mid-level 2D shape features of selected views.

The script computes image-derived descriptors from the already-rendered
ModelNet PNGs, then joins those descriptors to MVSelect `*_selection.json`
files to track which mid-level structures the selector uses over training.

No ShapeNet mesh access is required for these measurements. They are computed
from silhouettes / edges visible in the rendered images.
"""

import argparse
import ast
import json
import math
import os
import tempfile
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(),
                                                   "mvselect_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


DEFAULT_DATA_ROOT = "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23"

VIEW_BUCKETS = [
    "expanded",
    "Expanded-like",
    "Foreshortened",
    "Foreshortened-like",
    "Remainder",
]

METRIC_GROUPS = {
    "axis_visibility": [
        "ellipse_orientation_deg",
        "ellipse_aspect_ratio",
        "skeleton_elongation",
        "skeleton_length_norm",
    ],
    "symmetry_part_organization": [
        "bilateral_symmetry",
        "medial_axis_symmetry",
        "skeleton_endpoint_count",
        "skeleton_branchpoint_count",
        "skeleton_branch_density",
    ],
    "edge_organization": [
        "edge_anisotropy",
        "edge_entropy",
        "dominant_edge_orientation_deg",
    ],
}

PLOT_METRICS = [
    "ellipse_aspect_ratio",
    "skeleton_elongation",
    "bilateral_symmetry",
    "medial_axis_symmetry",
    "skeleton_branch_density",
    "edge_anisotropy",
    "edge_entropy",
]

ORIENTATION_METRICS = {
    "ellipse_orientation_deg",
    "dominant_edge_orientation_deg",
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--selection_dir", default=None,
                   help="meta_logs/<dataset>/<experiment>/ folder with *_selection.json. "
                        "If omitted, only per-view feature cache is computed.")
    p.add_argument("--split", default="test")
    p.add_argument("--cache_csv", default=None,
                   help="Per-view feature cache. Default: "
                        "cache/midlevel_features_<split>.csv")
    p.add_argument("--output_dir", default=None,
                   help="Default: <selection_dir>/midlevel_features, or "
                        "logs/midlevel_features if no selection_dir.")
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--limit_images", type=int, default=None,
                   help="Smoke-test cap on number of rendered images.")
    p.add_argument("--bin_epochs", type=int, default=10,
                   help="Number of epoch bins for heatmaps/plots. 0 = no binning.")
    return p.parse_args()


def classify_view_by_filename(fname):
    is_long = "planar" in fname and "short" not in fname
    is_short = "short" in fname and "like" not in fname
    is_long_like = "like" in fname and "short" not in fname
    is_short_like = "like" in fname and "short" in fname
    if is_long:
        return "expanded"
    if is_short:
        return "Foreshortened"
    if is_long_like:
        return "Expanded-like"
    if is_short_like:
        return "Foreshortened-like"
    return "Remainder"


def load_modelnet40_classnames():
    """Read ModelNet40.classnames without importing src.datasets.

    Importing src.datasets triggers optional dataset modules (e.g. Wildtrack),
    which can require cv2 even though this script only needs the class order.
    Keeping the order identical to modelnet40.py is important because
    selection.json stores numeric class indices.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "src", "datasets", "modelnet40.py")
    with open(path, "r") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModelNet40":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    names = [
                        target.id for target in stmt.targets
                        if isinstance(target, ast.Name)
                    ]
                    if "classnames" in names:
                        return ast.literal_eval(stmt.value)
    raise RuntimeError(f"Could not find ModelNet40.classnames in {path}")


def parse_instance_and_view(fname):
    parts = os.path.basename(fname).split(".p")[0].split("_")
    instance_id = parts[0]
    view_index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
    return instance_id, view_index


def largest_component(mask):
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == counts.argmax()


def image_mask_and_gray(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    mask = gray < 0.97
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = largest_component(mask)
    return gray, mask.astype(bool)


def safe_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def crop_to_bbox(mask):
    bbox = safe_bbox(mask)
    if bbox is None:
        return mask
    y0, y1, x0, x1 = bbox
    return mask[y0:y1, x0:x1]


def mirror_iou(mask):
    crop = crop_to_bbox(mask).astype(bool)
    if crop.size == 0 or crop.sum() == 0:
        return np.nan
    lr = np.fliplr(crop)
    ud = np.flipud(crop)

    def iou(a, b):
        union = np.logical_or(a, b).sum()
        if union == 0:
            return np.nan
        return float(np.logical_and(a, b).sum() / union)

    return float(np.nanmax([iou(crop, lr), iou(crop, ud)]))


def ellipse_features(mask):
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return {
            "ellipse_orientation_deg": np.nan,
            "ellipse_aspect_ratio": np.nan,
            "mask_area": int(mask.sum()),
            "bbox_aspect_ratio": np.nan,
        }
    coords = np.stack([xs.astype(float), ys.astype(float)], axis=1)
    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    major = max(vals[0], 1e-8)
    minor = max(vals[1], 1e-8)
    angle = math.degrees(math.atan2(vec[1], vec[0])) % 180.0
    bbox = safe_bbox(mask)
    if bbox is None:
        bbox_aspect = np.nan
    else:
        y0, y1, x0, x1 = bbox
        h, w = max(y1 - y0, 1), max(x1 - x0, 1)
        bbox_aspect = max(w / h, h / w)
    return {
        "ellipse_orientation_deg": float(angle),
        "ellipse_aspect_ratio": float(math.sqrt(major / minor)),
        "mask_area": int(mask.sum()),
        "bbox_aspect_ratio": float(bbox_aspect),
    }


def zhang_suen_skeleton(mask, max_iter=200):
    """Vectorized Zhang-Suen thinning for binary masks."""
    img = mask.astype(np.uint8).copy()
    if img.sum() == 0:
        return img.astype(bool)

    def neighbors(x):
        p2 = np.roll(x, -1, axis=0)
        p3 = np.roll(np.roll(x, -1, axis=0), 1, axis=1)
        p4 = np.roll(x, 1, axis=1)
        p5 = np.roll(np.roll(x, 1, axis=0), 1, axis=1)
        p6 = np.roll(x, 1, axis=0)
        p7 = np.roll(np.roll(x, 1, axis=0), -1, axis=1)
        p8 = np.roll(x, -1, axis=1)
        p9 = np.roll(np.roll(x, -1, axis=0), -1, axis=1)
        return p2, p3, p4, p5, p6, p7, p8, p9

    # Border pixels are unreliable under roll-neighborhoods.
    img[[0, -1], :] = 0
    img[:, [0, -1]] = 0

    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = neighbors(img)
            n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            transitions = sum((seq[i] == 0) & (seq[i + 1] == 1)
                              for i in range(8))
            if step == 0:
                c3 = (p2 * p4 * p6) == 0
                c4 = (p4 * p6 * p8) == 0
            else:
                c3 = (p2 * p4 * p8) == 0
                c4 = (p2 * p6 * p8) == 0
            remove = (
                (img == 1) &
                (n >= 2) & (n <= 6) &
                (transitions == 1) &
                c3 & c4
            )
            remove[[0, -1], :] = False
            remove[:, [0, -1]] = False
            if remove.any():
                img[remove] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def skeleton_features(mask):
    skel = zhang_suen_skeleton(mask)
    skel_len = int(skel.sum())
    area = max(int(mask.sum()), 1)
    if skel_len == 0:
        return {
            "skeleton_length_px": 0,
            "skeleton_length_norm": 0.0,
            "skeleton_elongation": 0.0,
            "skeleton_endpoint_count": 0,
            "skeleton_branchpoint_count": 0,
            "skeleton_branch_density": 0.0,
            "medial_axis_symmetry": np.nan,
        }, skel

    kernel = np.ones((3, 3), dtype=int)
    neighbor_count = ndimage.convolve(skel.astype(int), kernel, mode="constant") - skel
    endpoints = int(np.logical_and(skel, neighbor_count == 1).sum())
    branchpoints = int(np.logical_and(skel, neighbor_count >= 3).sum())
    bbox = safe_bbox(mask)
    if bbox is None:
        major_bbox = math.sqrt(area)
    else:
        y0, y1, x0, x1 = bbox
        major_bbox = max(y1 - y0, x1 - x0, 1)
    return {
        "skeleton_length_px": skel_len,
        "skeleton_length_norm": float(skel_len / math.sqrt(area)),
        "skeleton_elongation": float(skel_len / major_bbox),
        "skeleton_endpoint_count": endpoints,
        "skeleton_branchpoint_count": branchpoints,
        "skeleton_branch_density": float(branchpoints / max(skel_len, 1)),
        "medial_axis_symmetry": mirror_iou(skel),
    }, skel


def edge_features(gray, mask):
    dx = ndimage.sobel(gray, axis=1)
    dy = ndimage.sobel(gray, axis=0)
    mag = np.hypot(dx, dy)
    vals = mag[mask]
    if vals.size == 0:
        edge_mask = mag > np.percentile(mag, 90)
    else:
        threshold = max(float(np.percentile(vals, 75)), 1e-6)
        edge_mask = mask & (mag >= threshold)
    weights = mag[edge_mask]
    if weights.size == 0 or weights.sum() <= 0:
        return {
            "dominant_edge_orientation_deg": np.nan,
            "edge_entropy": np.nan,
            "edge_anisotropy": np.nan,
            "edge_pixel_count": 0,
        }
    theta = np.mod(np.arctan2(dy[edge_mask], dx[edge_mask]), np.pi)
    hist, _ = np.histogram(theta, bins=18, range=(0, np.pi), weights=weights)
    p = hist / max(hist.sum(), 1e-12)
    entropy = -np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(len(hist))
    resultant = np.abs(np.sum(weights * np.exp(2j * theta))) / max(weights.sum(), 1e-12)
    dominant = 0.5 * np.angle(np.sum(weights * np.exp(2j * theta)))
    dominant_deg = math.degrees(dominant) % 180.0
    return {
        "dominant_edge_orientation_deg": float(dominant_deg),
        "edge_entropy": float(entropy),
        "edge_anisotropy": float(resultant),
        "edge_pixel_count": int(weights.size),
    }


def mean_metric(series, metric):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.nan
    if metric in ORIENTATION_METRICS:
        # Axial circular mean: 0 deg and 180 deg describe the same orientation.
        theta = np.deg2rad(values)
        z = np.mean(np.exp(2j * theta))
        return float((0.5 * np.rad2deg(np.angle(z))) % 180.0)
    return float(np.mean(values))


def metric_delta(selected, baseline, metric):
    if pd.isna(selected) or pd.isna(baseline):
        return np.nan
    if metric in ORIENTATION_METRICS:
        # Shortest signed axial angular difference in [-90, 90).
        return float(((selected - baseline + 90.0) % 180.0) - 90.0)
    return float(selected - baseline)


def compute_one(path):
    gray, mask = image_mask_and_gray(path)
    out = {}
    out.update(ellipse_features(mask))
    out["bilateral_symmetry"] = mirror_iou(mask)
    skel_metrics, _ = skeleton_features(mask)
    out.update(skel_metrics)
    out.update(edge_features(gray, mask))
    return out


def build_view_feature_cache(args):
    cache_dir = os.path.dirname(args.cache_csv)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(args.cache_csv) and not args.force_recompute:
        print(f"Using cached per-view features: {args.cache_csv}")
        return pd.read_csv(args.cache_csv)

    rows = []
    classnames = load_modelnet40_classnames()
    for cls_idx, cls_name in enumerate(classnames):
        split_dir = os.path.join(args.data_root, cls_name, args.split)
        if not os.path.isdir(split_dir):
            continue
        files = sorted(
            os.path.join(split_dir, fn)
            for fn in os.listdir(split_dir)
            if fn.lower().endswith(".png")
        )
        for path in files:
            if args.limit_images is not None and len(rows) >= args.limit_images:
                break
            fname = os.path.basename(path)
            instance_id, view_index = parse_instance_and_view(fname)
            metrics = compute_one(path)
            rows.append({
                "class_idx": cls_idx,
                "class_name": cls_name,
                "instance_id": instance_id,
                "view_index": view_index,
                "filename": fname,
                "view_type": classify_view_by_filename(fname),
                "image_path": path,
                **metrics,
            })
        if args.limit_images is not None and len(rows) >= args.limit_images:
            break
    df = pd.DataFrame(rows)
    df.to_csv(args.cache_csv, index=False)
    print(f"Saved per-view feature cache: {args.cache_csv} ({len(df)} rows)")
    return df


def load_selection_jsons(selection_dir):
    runs = []
    for fn in sorted(os.listdir(selection_dir)):
        if not fn.endswith("_selection.json"):
            continue
        path = os.path.join(selection_dir, fn)
        with open(path, "r") as f:
            runs.append((fn, json.load(f)))
    return runs


def aggregate_selected_features(view_df, selection_dir):
    idx = {
        (int(r.class_idx), r.filename): r
        for r in view_df.itertuples(index=False)
    }
    by_instance = defaultdict(list)
    for r in view_df.itertuples(index=False):
        by_instance[(int(r.class_idx), str(r.instance_id))].append(r)
    metric_cols = [
        c for c in view_df.columns
        if c not in {
            "class_idx", "class_name", "instance_id", "view_index",
            "filename", "view_type", "image_path",
        }
    ]
    global_baseline = {
        metric: mean_metric(view_df[metric], metric)
        for metric in metric_cols
    }

    rows = []
    for run_name, whole in load_selection_jsons(selection_dir):
        for epoch_str, epoch_sel in whole.items():
            values = []
            selected_counts = defaultdict(int)
            selected_instances = set()
            missing = 0
            for cls_str, view_dict in epoch_sel.items():
                cls_idx = int(cls_str)
                for bucket, filenames in view_dict.items():
                    for fname in filenames:
                        key = (cls_idx, os.path.basename(fname))
                        rec = idx.get(key)
                        if rec is None:
                            missing += 1
                            continue
                        values.append(rec)
                        selected_counts[getattr(rec, "view_type")] += 1
                        selected_instances.add((int(getattr(rec, "class_idx")),
                                                str(getattr(rec, "instance_id"))))
            if not values:
                continue
            block = pd.DataFrame([v._asdict() for v in values])
            baseline_values = []
            for key in selected_instances:
                baseline_values.extend(by_instance.get(key, []))
            if baseline_values:
                baseline_block = pd.DataFrame([v._asdict() for v in baseline_values])
            else:
                baseline_block = view_df
            row = {
                "run": run_name,
                "epoch": int(epoch_str),
                "n_selected": int(len(block)),
                "n_baseline_views": int(len(baseline_block)),
                "n_missing": int(missing),
            }
            total = max(len(block), 1)
            for bucket in VIEW_BUCKETS:
                row[f"prop_{bucket}"] = selected_counts[bucket] / total
            for metric in metric_cols:
                row[f"selected_{metric}"] = mean_metric(block[metric], metric)
                matched_baseline = mean_metric(baseline_block[metric], metric)
                if pd.isna(matched_baseline):
                    matched_baseline = global_baseline.get(metric, np.nan)
                row[f"baseline_{metric}"] = float(matched_baseline)
                row[f"lift_{metric}"] = metric_delta(
                    row[f"selected_{metric}"],
                    row[f"baseline_{metric}"],
                    metric,
                )
            rows.append(row)

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, raw
    summary = raw.groupby("epoch", as_index=False).mean(numeric_only=True)
    return raw, summary


def bin_epoch_df(df, n_bins):
    if n_bins <= 0 or df.empty or df["epoch"].nunique() <= n_bins:
        return df.copy()
    epochs = df["epoch"].to_numpy(dtype=float)
    edges = np.linspace(epochs.min() - 0.5, epochs.max() + 0.5, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        block = df[(df["epoch"] >= lo) & (df["epoch"] < hi)]
        if block.empty:
            continue
        row = block.mean(numeric_only=True).to_dict()
        row["epoch"] = (lo + hi) / 2.0
        row["epoch_bin"] = f"{int(math.ceil(lo + 0.5))}-{int(math.floor(hi - 0.5))}"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_lift_heatmap(summary, out_path, bin_epochs):
    binned = bin_epoch_df(summary, bin_epochs)
    metrics = [m for m in PLOT_METRICS if f"lift_{m}" in binned.columns]
    if not metrics:
        return
    matrix = np.vstack([binned[f"lift_{m}"].to_numpy(dtype=float) for m in metrics])
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    vmax = float(np.nanmax(np.abs(finite)))
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * matrix.shape[1] + 2),
                                    max(4, 0.45 * len(metrics) + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=8)
    labels = binned["epoch_bin"].tolist() if "epoch_bin" in binned.columns else [
        str(int(e)) for e in binned["epoch"]
    ]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Training epoch")
    ax.set_title("Selected-view mid-level feature lift vs all candidate views")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Selected mean - all-view baseline")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_group_curve_figures(summary, output_dir, bin_epochs, value_prefix="selected"):
    binned = bin_epoch_df(summary, bin_epochs)
    for group, metrics in METRIC_GROUPS.items():
        available = [m for m in metrics if f"{value_prefix}_{m}" in binned.columns]
        if not available:
            continue
        fig, axes = plt.subplots(len(available), 1,
                                 figsize=(9, max(3, 2.2 * len(available))),
                                 sharex=True)
        if len(available) == 1:
            axes = [axes]
        for ax, metric in zip(axes, available):
            col = f"{value_prefix}_{metric}"
            if col not in binned.columns:
                continue
            ax.plot(binned["epoch"], binned[col], marker="o", ms=3, lw=1.5,
                    color="#1f77b4")
            if value_prefix == "lift":
                ax.axhline(0, color="gray", ls="--", alpha=0.5)
            ax.set_ylabel(metric, fontsize=8)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Training epoch")
        fig.suptitle(f"{group.replace('_', ' ')} ({value_prefix})", y=0.995)
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"midlevel_{group}_{value_prefix}_curves.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    if args.cache_csv is None:
        args.cache_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cache",
            f"midlevel_features_{args.split}.csv",
        )
    if args.output_dir is None:
        if args.selection_dir:
            args.output_dir = os.path.join(args.selection_dir, "midlevel_features")
        else:
            args.output_dir = os.path.join("logs", "midlevel_features")
    os.makedirs(args.output_dir, exist_ok=True)

    view_df = build_view_feature_cache(args)
    per_view_out = os.path.join(args.output_dir, "per_view_midlevel_features.csv")
    view_df.to_csv(per_view_out, index=False)
    print(f"Saved: {per_view_out}")

    if not args.selection_dir:
        return
    raw, summary = aggregate_selected_features(view_df, args.selection_dir)
    raw_path = os.path.join(args.output_dir, "selected_midlevel_by_run_epoch.csv")
    summary_path = os.path.join(args.output_dir, "selected_midlevel_summary.csv")
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")
    if not summary.empty:
        plot_lift_heatmap(summary,
                          os.path.join(args.output_dir, "midlevel_lift_heatmap.png"),
                          args.bin_epochs)
        plot_group_curve_figures(summary, args.output_dir, args.bin_epochs,
                                 value_prefix="selected")
        print(f"Saved plots under: {args.output_dir}")


if __name__ == "__main__":
    main()
