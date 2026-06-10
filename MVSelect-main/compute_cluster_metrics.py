"""Compute cluster-quality metrics from saved feature dumps, per experiment.

Walks every `meta_logs/<rep>/<exp>/<run>/feature_<epoch>.npz` that stage-2
training has dumped (the same source files `pca_tsne.py` uses) and, per
experiment and per epoch, computes:

    silhouette_class   — silhouette score over CLASS labels (cosine distance,
                          L2-normalised features). Higher = features cluster
                          more cleanly by 32-class identity.
    silhouette_view    — silhouette score over VIEW-TYPE labels (the 5
                          view-type buckets). Higher = features encode view
                          identity strongly (bad for view-invariant
                          classification — we want this LOWER over training).
    separability       — silhouette_class − silhouette_view. A single scalar
                          summarising "class-aware AND view-invariant" (HIGHER
                          over training = the model is learning the right
                          thing).

Saves a `cluster_metrics.csv` inside each experiment folder. Runs of the same
experiment are averaged epoch-wise.

Usage:
    python3 compute_cluster_metrics.py                     # all reps
    python3 compute_cluster_metrics.py --rep_list rgb
    python3 compute_cluster_metrics.py --max_samples 1500 --num_runs 3
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


FEATURE_RE = re.compile(r"feature_(\d+)\.npz$")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--root", type=str, default="meta_logs")
    p.add_argument("--rep_list", nargs="+", default=["rgb", "depth", "edge"])
    p.add_argument("--num_runs", type=int, default=None,
                   help="Cap on runs aggregated per experiment. None = use all.")
    p.add_argument("--feature_dim", type=int, default=512,
                   help="Per-view feature dim used when reshaping the saved array.")
    p.add_argument("--max_samples", type=int, default=2000,
                   help="Downsample to N points before computing silhouette "
                        "(silhouette is O(n²)). None or 0 = use all samples.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the downsample. Same seed across epochs = "
                        "comparable across the epoch axis.")
    p.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"],
                   help="Distance metric for silhouette.")
    p.add_argument("--every_n_epochs", type=int, default=None,
                   help="If set, only compute on epochs where epoch %% N == 0. "
                        "Useful when training dumped features at uneven cadence "
                        "(e.g. every epoch early, every 10 later) and you want "
                        "a regular grid for plotting.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-compute even if cluster_metrics.csv already exists.")
    return p.parse_args()


def discover_epochs(run_dir):
    epochs = []
    for fname in os.listdir(run_dir):
        m = FEATURE_RE.match(fname)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def compute_metrics_for_run(run_dir, args):
    """Iterate one run's feature_<E>.npz files; return list of per-epoch rows."""
    rows = []
    epochs = discover_epochs(run_dir)
    if args.every_n_epochs and args.every_n_epochs > 1:
        epochs = [e for e in epochs if e % args.every_n_epochs == 0]
    for e in epochs:
        npz = np.load(os.path.join(run_dir, f"feature_{e}.npz"))
        features = npz["features"].squeeze().reshape(-1, args.feature_dim)
        view_type = np.asarray(npz["view_type"])
        view_class = np.asarray(npz["view_class"])

        n = features.shape[0]
        # Downsample with a deterministic seed so the same epoch yields the
        # same subset across reruns.
        if args.max_samples and n > args.max_samples:
            rng = np.random.RandomState(args.seed + e)
            idx = rng.choice(n, args.max_samples, replace=False)
            features = features[idx]
            view_type = view_type[idx]
            view_class = view_class[idx]

        features = normalize(features, axis=1)

        unique_classes = np.unique(view_class)
        unique_views = np.unique(view_type)

        sil_class = float("nan")
        sil_view = float("nan")
        if len(unique_classes) > 1 and features.shape[0] > len(unique_classes):
            try:
                sil_class = float(silhouette_score(features, view_class, metric=args.metric))
            except Exception as ex:
                print(f"    silhouette_class failed at e={e}: {ex}")
        if len(unique_views) > 1 and features.shape[0] > len(unique_views):
            try:
                sil_view = float(silhouette_score(features, view_type, metric=args.metric))
            except Exception as ex:
                print(f"    silhouette_view failed at e={e}: {ex}")

        rows.append({
            "epoch": e,
            "n_samples": int(features.shape[0]),
            "silhouette_class": sil_class,
            "silhouette_view": sil_view,
            "separability": sil_class - sil_view if (np.isfinite(sil_class) and np.isfinite(sil_view)) else float("nan"),
        })
    return rows


def process_experiment(rep, exp, exp_root, args):
    run_dirs = sorted([
        os.path.join(exp_root, d) for d in os.listdir(exp_root)
        if os.path.isdir(os.path.join(exp_root, d))
    ])
    if args.num_runs is not None:
        run_dirs = run_dirs[:args.num_runs]
    if not run_dirs:
        return

    out_csv = os.path.join(exp_root, "cluster_metrics.csv")
    if (not args.overwrite) and os.path.exists(out_csv):
        print(f"[{rep}/{exp}] skip (exists; --overwrite to force) → {out_csv}")
        return

    # Discover whether at least one run has feature files
    has_features = any(discover_epochs(rd) for rd in run_dirs)
    if not has_features:
        return

    all_rows = []
    for run_dir in run_dirs:
        try:
            all_rows.extend(compute_metrics_for_run(run_dir, args))
        except Exception as ex:
            print(f"  run {run_dir} failed: {ex}")
    if not all_rows:
        return

    # Average across runs at each epoch (so the curve reflects a smooth
    # combination of seeded runs).
    df = pd.DataFrame(all_rows)
    df_grouped = df.groupby("epoch", as_index=False).mean(numeric_only=True)
    df_grouped["rep"] = rep
    df_grouped["experiment"] = exp
    df_grouped["n_runs"] = len(run_dirs)
    df_grouped.to_csv(out_csv, index=False)
    print(f"[{rep}/{exp}] wrote {len(df_grouped)} epoch rows → {out_csv}")


def main():
    args = parse_args()
    if not os.path.isdir(args.root):
        sys.exit(f"--root not a directory: {args.root}")
    for rep in args.rep_list:
        rep_root = os.path.join(args.root, rep)
        if not os.path.isdir(rep_root):
            print(f"Skip {rep_root}: not a directory.")
            continue
        for exp in sorted(os.listdir(rep_root)):
            exp_root = os.path.join(rep_root, exp)
            if not os.path.isdir(exp_root):
                continue
            try:
                process_experiment(rep, exp, exp_root, args)
            except Exception as ex:
                print(f"[{rep}/{exp}] ERROR: {ex}")


if __name__ == "__main__":
    main()
