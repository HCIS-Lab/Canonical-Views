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
CACHE_VERSION = "v2"

VIEW_BUCKETS = [
    "expanded",
    "Expanded-like",
    "Foreshortened",
    "Foreshortened-like",
    "Remainder",
]

VIEW_BUCKET_LABELS = {
    "expanded": "Expanded",
    "Expanded-like": "Expanded-like",
    "Foreshortened": "Foreshortened",
    "Foreshortened-like": "Foreshortened-like",
    "Remainder": "Remainder",
}

METRIC_GROUPS = {
    "axis_visibility": [
        "ellipse_orientation_deg",
        "ellipse_aspect_ratio",
        "bbox_aspect_ratio",
        "skeleton_length_px",
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
        "edge_pixel_count",
    ],
}

PLOT_METRICS = sorted({m for metrics in METRIC_GROUPS.values() for m in metrics})

ORIENTATION_METRICS = {
    "ellipse_orientation_deg",
    "dominant_edge_orientation_deg",
}

PRIMARY_RAW_METRICS = [
    "ellipse_aspect_ratio",
    "bilateral_symmetry",
    "edge_entropy",
]

PRIMARY_RAW_LABELS = {
    "ellipse_aspect_ratio": "Ellipse aspect ratio",
    "bilateral_symmetry": "Bilateral symmetry",
    "edge_entropy": "Edge entropy",
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
                        f"cache/midlevel_features_{CACHE_VERSION}_<split>.csv")
    p.add_argument("--output_dir", default=None,
                   help="Default: <selection_dir>/midlevel_features, or "
                        "logs/midlevel_features if no selection_dir.")
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--limit_images", type=int, default=None,
                   help="Smoke-test cap on number of rendered images.")
    p.add_argument("--bin_epochs", type=int, default=10,
                   help="Number of epoch bins for heatmaps/plots. 0 = no binning.")
    p.add_argument("--skip_view_type_association", action="store_true",
                   help="Skip the exact-view-type association analysis. The sweep "
                        "wrapper uses this for every experiment except no_freeze.")
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
        minor_bbox = math.sqrt(area)
    else:
        y0, y1, x0, x1 = bbox
        h, w = max(y1 - y0, 1), max(x1 - x0, 1)
        minor_bbox = max(min(h, w), 1)
    return {
        "skeleton_length_px": skel_len,
        "skeleton_length_norm": float(skel_len / math.sqrt(area)),
        "skeleton_elongation": float(skel_len / minor_bbox),
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


def std_metric(series, metric):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.nan
    if metric in ORIENTATION_METRICS:
        theta = np.deg2rad(values)
        r = np.abs(np.mean(np.exp(2j * theta)))
        r = min(max(float(r), 1e-12), 1.0)
        return float(np.rad2deg(0.5 * np.sqrt(-2.0 * np.log(r))))
    return float(np.std(values, ddof=0))


def metric_delta(selected, baseline, metric):
    if pd.isna(selected) or pd.isna(baseline):
        return np.nan
    if metric in ORIENTATION_METRICS:
        # Shortest signed axial angular difference in [-90, 90).
        return float(((selected - baseline + 90.0) % 180.0) - 90.0)
    return float(selected - baseline)


def metric_effect(selected, baseline, baseline_std, metric):
    delta = metric_delta(selected, baseline, metric)
    if pd.isna(delta) or pd.isna(baseline_std) or baseline_std <= 1e-12:
        return np.nan
    return float(delta / baseline_std)


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


def aggregate_selected_features(view_df, selection_dir,
                                compute_view_type_association=True):
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
    association_rows = []
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
            if compute_view_type_association:
                association_rows.extend(view_type_midlevel_association_rows(
                    block, run_name, int(epoch_str)))
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
                baseline_std = std_metric(baseline_block[metric], metric)
                row[f"baseline_{metric}"] = float(matched_baseline)
                row[f"baseline_std_{metric}"] = float(baseline_std) if not pd.isna(baseline_std) else np.nan
                row[f"lift_{metric}"] = metric_delta(
                    row[f"selected_{metric}"],
                    row[f"baseline_{metric}"],
                    metric,
                )
                row[f"effect_{metric}"] = metric_effect(
                    row[f"selected_{metric}"],
                    row[f"baseline_{metric}"],
                    row[f"baseline_std_{metric}"],
                    metric,
                )
            rows.append(row)

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, raw, pd.DataFrame(association_rows)
    summary = summarize_by_epoch(raw)
    return raw, summary, pd.DataFrame(association_rows)


def metric_name_from_column(col):
    for prefix in ("selected_", "baseline_std_", "baseline_", "lift_", "effect_"):
        if col.startswith(prefix):
            return col[len(prefix):], prefix.rstrip("_")
    return None, None


def summarize_by_epoch(df):
    rows = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for epoch, block in df.groupby("epoch", sort=True):
        row = {"epoch": int(epoch)}
        for col in numeric_cols:
            if col == "epoch":
                continue
            metric, prefix = metric_name_from_column(col)
            if metric in ORIENTATION_METRICS and prefix in {"selected", "baseline"}:
                row[col] = mean_metric(block[col], metric)
            else:
                row[col] = pd.to_numeric(block[col], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_ratio_eta_squared(values, groups):
    """Association strength for a categorical family and continuous metric."""
    frame = pd.DataFrame({"value": values, "group": groups}).replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["group"].nunique() < 2:
        return np.nan
    grand_mean = float(frame["value"].mean())
    ss_total = float(np.square(frame["value"] - grand_mean).sum())
    if ss_total <= 1e-12:
        return np.nan
    ss_between = 0.0
    for _, block in frame.groupby("group", sort=False):
        ss_between += len(block) * (float(block["value"].mean()) - grand_mean) ** 2
    return float(ss_between / ss_total)


def point_biserial_correlation(values, membership):
    """Pearson correlation between a continuous metric and a binary family flag."""
    frame = pd.DataFrame({"value": values, "member": membership}).replace(
        [np.inf, -np.inf], np.nan).dropna()
    if (len(frame) < 3 or frame["member"].nunique() < 2
            or frame["value"].nunique() < 2):
        return np.nan
    return float(np.corrcoef(
        frame["member"].astype(float), frame["value"].astype(float))[0, 1])


def view_type_midlevel_association_rows(block, run_name, epoch):
    """Compute family/metric associations for one selected-view event block."""
    rows = []
    for metric in PRIMARY_RAW_METRICS:
        if metric not in block.columns:
            continue
        metric_block = block[["view_type", metric]].copy()
        metric_block[metric] = pd.to_numeric(
            metric_block[metric], errors="coerce")
        metric_block = metric_block.replace(
            [np.inf, -np.inf], np.nan).dropna()
        eta_squared = correlation_ratio_eta_squared(
            metric_block[metric], metric_block["view_type"])
        for view_type in VIEW_BUCKETS:
            membership = metric_block["view_type"] == view_type
            family_values = metric_block.loc[membership, metric]
            other_values = metric_block.loc[~membership, metric]
            family_mean = (
                float(family_values.mean()) if len(family_values) else np.nan)
            other_mean = (
                float(other_values.mean()) if len(other_values) else np.nan)
            rows.append({
                "run": run_name,
                "epoch": int(epoch),
                "metric": metric,
                "view_type": view_type,
                "view_type_label": VIEW_BUCKET_LABELS[view_type],
                "n": int(len(metric_block)),
                "n_view_types": int(metric_block["view_type"].nunique()),
                "n_family": int(membership.sum()),
                "n_other": int((~membership).sum()),
                "eta_squared": eta_squared,
                "point_biserial": point_biserial_correlation(
                    metric_block[metric], membership),
                "family_mean": family_mean,
                "other_mean": other_mean,
                "family_minus_other_mean": (
                    family_mean - other_mean
                    if np.isfinite(family_mean) and np.isfinite(other_mean)
                    else np.nan),
            })
    return rows


def compute_view_type_midlevel_associations(selected_views):
    """Compute selection-weighted associations from a row-level event table."""
    rows = []
    if selected_views.empty:
        return pd.DataFrame(rows)
    for (run_name, epoch), block in selected_views.groupby(
            ["run", "epoch"], sort=True):
        rows.extend(view_type_midlevel_association_rows(
            block, run_name, int(epoch)))
    return pd.DataFrame(rows)


def add_long_epoch_bins(df, n_bins):
    """Add stable integer bin ids and labels to a long-form epoch table."""
    out = df.copy()
    epochs = np.sort(out["epoch"].dropna().unique().astype(float))
    if not len(epochs):
        out["epoch_bin"] = pd.Series(dtype=int)
        out["epoch_label"] = pd.Series(dtype=str)
        return out
    if n_bins <= 0 or len(epochs) <= n_bins:
        mapping = {epoch: idx for idx, epoch in enumerate(epochs)}
        labels = {idx: str(int(epoch)) for epoch, idx in mapping.items()}
        out["epoch_bin"] = out["epoch"].map(mapping).astype(int)
        out["epoch_label"] = out["epoch_bin"].map(labels)
        return out

    edges = np.linspace(epochs.min() - 0.5, epochs.max() + 0.5, n_bins + 1)
    out["epoch_bin"] = np.digitize(
        out["epoch"].to_numpy(dtype=float), edges[1:-1], right=False)
    labels = {
        idx: (
            f"{int(math.ceil(edges[idx] + 0.5))}-"
            f"{int(math.floor(edges[idx + 1] - 0.5))}"
        )
        for idx in range(n_bins)
    }
    out["epoch_label"] = out["epoch_bin"].map(labels)
    return out


def mean_correlations(values):
    """Average correlations on Fisher's z scale."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return np.nan
    clipped = np.clip(values.to_numpy(dtype=float), -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def draw_view_type_heatmap(matrix, row_labels, column_labels, title,
                           colorbar_label, out_path, vmin, vmax, cmap):
    label_width = min(4.0, 0.045 * max(len(str(label)) for label in row_labels))
    fig, ax = plt.subplots(
        figsize=(max(8, 0.7 * len(column_labels) + 2.8 + label_width),
                 max(3.0, 0.48 * len(row_labels) + 1.8)),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Selection epoch")
    ax.set_title(title)
    threshold = max(abs(vmin), abs(vmax)) * 0.55
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
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_view_type_midlevel_associations(associations, output_dir, n_bins):
    if associations.empty:
        return
    binned = add_long_epoch_bins(associations, n_bins)
    bin_order = sorted(binned["epoch_bin"].unique())
    bin_labels = [
        str(binned.loc[binned["epoch_bin"] == idx, "epoch_label"].iloc[0])
        for idx in bin_order
    ]

    eta_source = binned.drop_duplicates(["run", "epoch", "metric"])
    eta = eta_source.groupby(
        ["metric", "epoch_bin"], sort=False)["eta_squared"].mean().unstack()
    eta_matrix = eta.reindex(
        index=PRIMARY_RAW_METRICS, columns=bin_order).to_numpy(dtype=float)
    draw_view_type_heatmap(
        eta_matrix,
        [PRIMARY_RAW_LABELS[metric] for metric in PRIMARY_RAW_METRICS],
        bin_labels,
        "Exact view type association with selected-view mid-level features",
        "Eta-squared (five view types; unsigned)",
        os.path.join(output_dir, "view_type_midlevel_eta_squared_heatmap.png"),
        0.0,
        1.0,
        "viridis",
    )

    point = binned.groupby(
        ["metric", "view_type", "epoch_bin"], sort=False
    )["point_biserial"].agg(mean_correlations).unstack()
    combined_rows = [
        (metric, view_type)
        for metric in PRIMARY_RAW_METRICS
        for view_type in VIEW_BUCKETS
    ]
    combined_matrix = point.reindex(
        index=pd.MultiIndex.from_tuples(
            combined_rows, names=["metric", "view_type"]),
        columns=bin_order,
    ).to_numpy(dtype=float)
    combined_labels = [
        f"{VIEW_BUCKET_LABELS[view_type]} | {PRIMARY_RAW_LABELS[metric]}"
        for metric, view_type in combined_rows
    ]
    draw_view_type_heatmap(
        combined_matrix,
        combined_labels,
        bin_labels,
        "Signed view-type association with selected-view mid-level features",
        "Point-biserial correlation (type vs all other types)",
        os.path.join(output_dir, "view_type_midlevel_point_biserial_heatmap.png"),
        -1.0,
        1.0,
        "RdBu_r",
    )

    means = binned.groupby(
        ["metric", "view_type", "epoch_bin"], sort=False
    )["family_mean"].mean().unstack()
    for metric in PRIMARY_RAW_METRICS:
        metric_point = point.loc[metric].reindex(
            index=VIEW_BUCKETS, columns=bin_order).to_numpy(dtype=float)
        draw_view_type_heatmap(
            metric_point,
            [VIEW_BUCKET_LABELS[view_type] for view_type in VIEW_BUCKETS],
            bin_labels,
            f"View type vs {PRIMARY_RAW_LABELS[metric]}",
            "Point-biserial correlation (type vs all other types)",
            os.path.join(
                output_dir,
                f"view_type_point_biserial_{metric}_heatmap.png"),
            -1.0,
            1.0,
            "RdBu_r",
        )

        metric_means = means.loc[metric].reindex(
            index=VIEW_BUCKETS, columns=bin_order).to_numpy(dtype=float)
        finite = metric_means[np.isfinite(metric_means)]
        if not finite.size:
            continue
        vmin, vmax = float(finite.min()), float(finite.max())
        if math.isclose(vmin, vmax):
            padding = max(abs(vmin) * 0.01, 1e-6)
            vmin, vmax = vmin - padding, vmax + padding
        draw_view_type_heatmap(
            metric_means,
            [VIEW_BUCKET_LABELS[view_type] for view_type in VIEW_BUCKETS],
            bin_labels,
            f"Raw selected-view {PRIMARY_RAW_LABELS[metric]} by view type",
            f"Mean {PRIMARY_RAW_LABELS[metric]} (raw units)",
            os.path.join(
                output_dir,
                f"view_type_raw_{metric}_heatmap.png"),
            vmin,
            vmax,
            "viridis",
        )


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
        row = {}
        for col in block.select_dtypes(include=[np.number]).columns:
            metric, prefix = metric_name_from_column(col)
            if metric in ORIENTATION_METRICS and prefix in {"selected", "baseline"}:
                row[col] = mean_metric(block[col], metric)
            else:
                row[col] = pd.to_numeric(block[col], errors="coerce").mean()
        row["epoch"] = (lo + hi) / 2.0
        row["epoch_bin"] = f"{int(math.ceil(lo + 0.5))}-{int(math.floor(hi - 0.5))}"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_value_heatmap(summary, out_path, bin_epochs, value_prefix):
    binned = bin_epoch_df(summary, bin_epochs)
    metrics = [m for m in PLOT_METRICS if f"{value_prefix}_{m}" in binned.columns]
    if not metrics:
        return
    matrix = np.vstack([binned[f"{value_prefix}_{m}"].to_numpy(dtype=float) for m in metrics])
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    if value_prefix == "effect":
        vmin, vmax = -1.0, 1.0
        cmap = "RdBu_r"
    elif value_prefix == "lift":
        vmax = float(np.nanmax(np.abs(finite)))
        vmin = -vmax
        cmap = "RdBu_r"
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        cmap = "viridis"
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * matrix.shape[1] + 2),
                                    max(4, 0.45 * len(metrics) + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=8)
    labels = binned["epoch_bin"].tolist() if "epoch_bin" in binned.columns else [
        str(int(e)) for e in binned["epoch"]
    ]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Training epoch")
    ax.set_title(f"Selected-view mid-level feature {value_prefix}")
    extend = "both" if value_prefix == "effect" and (
        np.nanmin(finite) < vmin or np.nanmax(finite) > vmax
    ) else "neither"
    cbar = fig.colorbar(im, ax=ax, extend=extend)
    if value_prefix == "effect":
        cbar.set_label("(selected mean - baseline mean) / baseline std")
        cbar.set_ticks([-1.0, 0.0, 1.0])
        cbar.set_ticklabels(["-1", "0", "+1"])
    elif value_prefix == "lift":
        cbar.set_label("Selected mean - all-view baseline")
        ticks = sorted({vmin, 0.0, vmax})
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:+.3f}" if t != 0 else "0" for t in ticks])
    else:
        cbar.set_label(value_prefix)
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.3f}", f"{vmax:.3f}"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _binned_run_values(raw, selected_col, baseline_col, n_bins):
    """Average within run/bin first, preserving runs as the SEM unit."""
    block = raw[["run", "epoch", selected_col, baseline_col]].copy()
    block[selected_col] = pd.to_numeric(block[selected_col], errors="coerce")
    block[baseline_col] = pd.to_numeric(block[baseline_col], errors="coerce")
    unique_epochs = np.sort(block["epoch"].dropna().unique().astype(float))
    if n_bins > 0 and len(unique_epochs) > n_bins:
        edges = np.linspace(
            unique_epochs.min() - 0.5,
            unique_epochs.max() + 0.5,
            n_bins + 1,
        )
        block["bin"] = np.digitize(
            block["epoch"], edges[1:-1], right=False)
        per_run = block.groupby(["run", "bin"], sort=True).agg(
            epoch=("epoch", "mean"),
            selected=(selected_col, "mean"),
            baseline=(baseline_col, "mean"),
        ).reset_index()
        labels = {}
        for bin_idx in sorted(per_run["bin"].unique()):
            lo, hi = edges[int(bin_idx)], edges[int(bin_idx) + 1]
            labels[int(bin_idx)] = (
                f"{int(math.ceil(lo + 0.5))}-"
                f"{int(math.floor(hi - 0.5))}"
            )
    else:
        per_run = block.rename(columns={
            selected_col: "selected",
            baseline_col: "baseline",
        })
        per_run["bin"] = per_run["epoch"].astype(int)
        labels = {int(epoch): str(int(epoch)) for epoch in unique_epochs}

    rows = []
    for bin_idx, values in per_run.groupby("bin", sort=True):
        selected = values["selected"].dropna().to_numpy(dtype=float)
        baseline = values["baseline"].dropna().to_numpy(dtype=float)

        def mean_sem(array):
            if not len(array):
                return np.nan, np.nan
            sem = (
                float(np.std(array, ddof=1) / math.sqrt(len(array)))
                if len(array) > 1 else 0.0
            )
            return float(np.mean(array)), sem

        selected_mean, selected_sem = mean_sem(selected)
        baseline_mean, baseline_sem = mean_sem(baseline)
        rows.append({
            "bin": int(bin_idx),
            "epoch": float(values["epoch"].mean()),
            "label": labels[int(bin_idx)],
            "selected_mean": selected_mean,
            "selected_sem": selected_sem,
            "baseline_mean": baseline_mean,
            "baseline_sem": baseline_sem,
            "n_runs": int(values["run"].nunique()),
        })
    return pd.DataFrame(rows)


def plot_primary_raw_values(raw, out_path, bin_epochs):
    """Plot representative metrics in raw units with separate y-axes."""
    available = [
        metric for metric in PRIMARY_RAW_METRICS
        if f"selected_{metric}" in raw.columns
        and f"baseline_{metric}" in raw.columns
    ]
    if not available:
        return

    fig, axes = plt.subplots(
        len(available), 1,
        figsize=(9, max(6.5, 2.5 * len(available))),
        sharex=True,
    )
    if len(available) == 1:
        axes = [axes]
    tick_values = None
    for axis, metric in zip(axes, available):
        values = _binned_run_values(
            raw,
            f"selected_{metric}",
            f"baseline_{metric}",
            bin_epochs,
        )
        if values.empty:
            continue
        tick_values = values
        x = np.arange(len(values))
        selected_mean = values["selected_mean"].to_numpy(dtype=float)
        selected_sem = values["selected_sem"].to_numpy(dtype=float)
        baseline_mean = values["baseline_mean"].to_numpy(dtype=float)
        baseline_sem = values["baseline_sem"].to_numpy(dtype=float)

        axis.plot(x, selected_mean, color="#0072B2", lw=2.0, marker="o",
                  ms=3.5, label="Selected views")
        axis.fill_between(
            x,
            selected_mean - selected_sem,
            selected_mean + selected_sem,
            color="#0072B2",
            alpha=0.2,
            label="Selected ±SEM",
        )
        axis.plot(x, baseline_mean, color="#666666", lw=1.5, ls="--",
                  label="Same-instance all-view baseline")
        axis.fill_between(
            x,
            baseline_mean - baseline_sem,
            baseline_mean + baseline_sem,
            color="#666666",
            alpha=0.12,
        )
        axis.set_ylabel("Raw value")
        axis.set_title(PRIMARY_RAW_LABELS[metric], loc="left", fontsize=10)
        axis.grid(alpha=0.25)

    if tick_values is None:
        plt.close(fig)
        return
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xticks(np.arange(len(tick_values)))
    axes[-1].set_xticklabels(
        tick_values["label"], rotation=35, ha="right")
    axes[-1].set_xlabel("Training epoch")
    fig.suptitle(
        "Selected-view projected-shape metrics "
        "(raw values, mean ±SEM across runs)",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
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
            if value_prefix in {"lift", "effect"}:
                ax.axhline(0, color="gray", ls="--", alpha=0.5)
            ax.set_ylabel(metric, fontsize=8)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Training epoch")
        fig.suptitle(f"{group.replace('_', ' ')} ({value_prefix})", y=0.995)
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"midlevel_{group}_{value_prefix}_curves.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def write_metric_range_diagnostics(view_df, summary, out_path):
    rows = []
    metrics = [m for m in PLOT_METRICS if m in view_df.columns]
    for metric in metrics:
        row = {"metric": metric}
        raw = pd.to_numeric(view_df[metric], errors="coerce").dropna()
        if len(raw):
            row.update({
                "per_view_min": float(raw.min()),
                "per_view_max": float(raw.max()),
                "per_view_std": std_metric(raw, metric),
            })
        for prefix in ("selected", "lift", "effect"):
            col = f"{prefix}_{metric}"
            if col not in summary.columns:
                continue
            vals = pd.to_numeric(summary[col], errors="coerce").dropna()
            if len(vals):
                row.update({
                    f"{prefix}_epoch_min": float(vals.min()),
                    f"{prefix}_epoch_max": float(vals.max()),
                    f"{prefix}_epoch_std": std_metric(vals, metric) if prefix == "selected" else float(vals.std(ddof=0)),
                })
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    args = parse_args()
    if args.cache_csv is None:
        args.cache_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cache",
            f"midlevel_features_{CACHE_VERSION}_{args.split}.csv",
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
    raw, summary, associations = aggregate_selected_features(
        view_df,
        args.selection_dir,
        compute_view_type_association=not args.skip_view_type_association,
    )
    raw_path = os.path.join(args.output_dir, "selected_midlevel_by_run_epoch.csv")
    summary_path = os.path.join(args.output_dir, "selected_midlevel_summary.csv")
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")
    if not summary.empty:
        range_path = os.path.join(args.output_dir, "midlevel_metric_ranges.csv")
        write_metric_range_diagnostics(view_df, summary, range_path)
        print(f"Saved: {range_path}")
        for obsolete in [
            "midlevel_lift_heatmap.png",
            "midlevel_effect_heatmap.png",
        ]:
            obsolete_path = os.path.join(args.output_dir, obsolete)
            if os.path.exists(obsolete_path):
                os.remove(obsolete_path)
        plot_primary_raw_values(
            raw,
            os.path.join(
                args.output_dir,
                "midlevel_selected_raw_primary_metrics.png",
            ),
            args.bin_epochs,
        )
        plot_group_curve_figures(summary, args.output_dir, args.bin_epochs,
                                 value_prefix="selected")
        plot_group_curve_figures(summary, args.output_dir, args.bin_epochs,
                                 value_prefix="lift")
        plot_group_curve_figures(summary, args.output_dir, args.bin_epochs,
                                 value_prefix="effect")
        if not associations.empty:
            association_path = os.path.join(
                args.output_dir, "view_type_midlevel_associations.csv")
            associations.to_csv(association_path, index=False)
            print(f"Saved: {association_path}")
            plot_view_type_midlevel_associations(
                associations, args.output_dir, args.bin_epochs)
        print(f"Saved plots under: {args.output_dir}")


if __name__ == "__main__":
    main()
