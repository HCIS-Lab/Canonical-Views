#!/usr/bin/env python3
"""Visualize the image operations behind the mid-level shape descriptors.

For each test-set instance, this script samples a fixed number of rendered
views and writes a contact sheet with:

    1. the original rendered view;
    2. the object mask with Zhang-Suen skeleton overlay;
    3. Sobel gradient magnitude;
    4. Sobel gradient orientation, weighted by gradient magnitude.

The visualizations are diagnostics only; metric computation lives in
midlevel_shape_features.py.
"""

import argparse
import csv
import os
import tempfile
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(),
                                                   "mvselect_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PIL import Image
from scipy import ndimage

from midlevel_shape_features import (
    DEFAULT_DATA_ROOT,
    compute_one,
    classify_view_by_filename,
    image_mask_and_gray,
    load_modelnet40_classnames,
    parse_instance_and_view,
    zhang_suen_skeleton,
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--output_dir", default=None,
                   help="Default: logs/midlevel_visual_samples/<split>_v<views_per_instance>")
    p.add_argument("--views_per_instance", type=int, default=10)
    p.add_argument("--sampling", default="even",
                   choices=["even", "first", "random"],
                   help="How to choose views from each instance's full view set.")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--max_instances", type=int, default=None,
                   help="Optional cap for quick smoke tests.")
    p.add_argument("--classes", nargs="+", default=None,
                   help="Optional class-name subset, e.g. --classes chair table.")
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args()


def collect_instances(data_root, split, class_subset=None):
    classnames = load_modelnet40_classnames()
    wanted = set(class_subset) if class_subset else None
    instances = defaultdict(list)

    for cls_idx, cls_name in enumerate(classnames):
        if wanted is not None and cls_name not in wanted:
            continue
        split_dir = os.path.join(data_root, cls_name, split)
        if not os.path.isdir(split_dir):
            continue
        for fn in sorted(os.listdir(split_dir)):
            if not fn.lower().endswith(".png"):
                continue
            instance_id, view_index = parse_instance_and_view(fn)
            instances[(cls_idx, cls_name, instance_id)].append({
                "path": os.path.join(split_dir, fn),
                "filename": fn,
                "view_index": view_index,
                "view_type": classify_view_by_filename(fn),
            })

    for key in list(instances.keys()):
        instances[key] = sorted(instances[key], key=lambda r: (r["view_index"], r["filename"]))
    return instances


def sample_views(records, n, sampling, rng):
    if n <= 0 or len(records) <= n:
        return records
    if sampling == "first":
        return records[:n]
    if sampling == "random":
        idx = sorted(rng.choice(len(records), size=n, replace=False).tolist())
        return [records[i] for i in idx]

    # Evenly cover the view-index range. This is the most useful default for
    # debugging since adjacent camera indices are often visually redundant.
    idx = np.linspace(0, len(records) - 1, n)
    idx = sorted({int(round(i)) for i in idx})
    while len(idx) < n:
        for candidate in range(len(records)):
            if candidate not in idx:
                idx.append(candidate)
                break
    idx = sorted(idx[:n])
    return [records[i] for i in idx]


def mask_boundary(mask):
    dil = ndimage.binary_dilation(mask, iterations=1)
    ero = ndimage.binary_erosion(mask, iterations=1)
    return np.logical_xor(dil, ero)


def normalize_by_mask(values, mask, percentile=99.0):
    vals = values[mask]
    if vals.size == 0:
        scale = float(np.nanpercentile(values, percentile))
    else:
        scale = float(np.nanpercentile(vals, percentile))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.nanmax(values))
    if not np.isfinite(scale) or scale <= 1e-8:
        return np.zeros_like(values, dtype=float)
    return np.clip(values / scale, 0.0, 1.0)


def skeleton_overlay(path):
    gray, mask = image_mask_and_gray(path)
    skel = zhang_suen_skeleton(mask)
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0

    # Lighten the background so the extracted structures are readable.
    out = rgb.copy()
    out[~mask] = 0.92
    out[mask] = 0.72 * out[mask] + 0.28

    boundary = mask_boundary(mask)
    out[boundary] = np.array([0.1, 0.35, 1.0])
    out[skel] = np.array([1.0, 0.05, 0.05])

    # Thicken skeleton pixels for visibility without changing the metric code.
    thick = ndimage.binary_dilation(skel, iterations=1)
    out[thick] = np.array([1.0, 0.05, 0.05])
    return np.clip(out, 0, 1), gray, mask


def sobel_panels(gray, mask):
    dx = ndimage.sobel(gray, axis=1)
    dy = ndimage.sobel(gray, axis=0)
    mag = np.hypot(dx, dy)
    mag_norm = normalize_by_mask(mag, mask)

    mag_rgb = plt.get_cmap("inferno")(mag_norm)[..., :3]
    mag_rgb[~mask] = 0.92

    theta = np.mod(np.arctan2(dy, dx), np.pi)
    hsv = np.zeros((*theta.shape, 3), dtype=float)
    hsv[..., 0] = theta / np.pi
    hsv[..., 1] = 1.0
    hsv[..., 2] = mag_norm
    orient_rgb = colors.hsv_to_rgb(hsv)
    orient_rgb[~mask] = 0.92

    return np.clip(mag_rgb, 0, 1), np.clip(orient_rgb, 0, 1)


def format_metric(value):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.3f}"


def render_instance_sheet(key, records, out_path, dpi):
    cls_idx, cls_name, instance_id = key
    n_rows = len(records)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13.2, max(2.5, 2.35 * n_rows)))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "render",
        "mask + Zhang-Suen skeleton",
        "Sobel magnitude",
        "Sobel orientation",
    ]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10)

    for row, rec in enumerate(records):
        rgb = np.asarray(Image.open(rec["path"]).convert("RGB")).astype(np.float32) / 255.0
        skel_rgb, gray, mask = skeleton_overlay(rec["path"])
        mag_rgb, orient_rgb = sobel_panels(gray, mask)
        metrics = compute_one(rec["path"])
        rec["ellipse_aspect_ratio"] = metrics.get("ellipse_aspect_ratio", np.nan)
        rec["bilateral_symmetry"] = metrics.get("bilateral_symmetry", np.nan)
        rec["edge_entropy"] = metrics.get("edge_entropy", np.nan)
        panels = [rgb, skel_rgb, mag_rgb, orient_rgb]

        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                label = (
                    f"v{rec['view_index']:03d}  {rec['view_type']}\n"
                    f"ellipse AR={format_metric(rec['ellipse_aspect_ratio'])}\n"
                    f"bilat sym={format_metric(rec['bilateral_symmetry'])}\n"
                    f"edge H={format_metric(rec['edge_entropy'])}"
                )
                ax.set_ylabel(label, rotation=0, labelpad=78, fontsize=7,
                              va="center", ha="right")

    fig.suptitle(f"{cls_name} / instance {instance_id}", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.08, 0, 1, 0.985))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "logs", "midlevel_visual_samples",
            f"{args.split}_v{args.views_per_instance}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    instances = collect_instances(args.data_root, args.split, args.classes)
    keys = sorted(instances.keys(), key=lambda k: (k[1], k[2]))
    if args.max_instances is not None:
        keys = keys[:args.max_instances]

    rng = np.random.default_rng(args.random_seed)
    index_rows = []
    for i, key in enumerate(keys, 1):
        selected = sample_views(instances[key], args.views_per_instance,
                                args.sampling, rng)
        cls_idx, cls_name, instance_id = key
        out_path = os.path.join(args.output_dir, cls_name,
                                f"{instance_id}_midlevel_extractions.png")
        render_instance_sheet(key, selected, out_path, args.dpi)
        for rec in selected:
            index_rows.append({
                "class_idx": cls_idx,
                "class_name": cls_name,
                "instance_id": instance_id,
                "view_index": rec["view_index"],
                "view_type": rec["view_type"],
                "ellipse_aspect_ratio": rec.get("ellipse_aspect_ratio", np.nan),
                "bilateral_symmetry": rec.get("bilateral_symmetry", np.nan),
                "edge_entropy": rec.get("edge_entropy", np.nan),
                "filename": rec["filename"],
                "image_path": rec["path"],
                "sheet_path": out_path,
            })
        print(f"[{i}/{len(keys)}] saved {out_path}")

    index_path = os.path.join(args.output_dir, "visualization_index.csv")
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class_idx", "class_name", "instance_id", "view_index",
            "view_type", "ellipse_aspect_ratio", "bilateral_symmetry",
            "edge_entropy", "filename", "image_path", "sheet_path",
        ])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Saved index: {index_path} ({len(index_rows)} sampled views)")


if __name__ == "__main__":
    main()
