"""PCA + t-SNE of training features, applied to every experiment in meta_logs/.

Walks `meta_logs/<rep>/<exp>/<run>/feature_<epoch>.npz` for every (rep, exp)
combination, auto-discovers which epochs are dumped, aggregates features
across runs of the same experiment, computes a single t-SNE per epoch, and
writes two plots per epoch into a `pca_tsne/` subfolder of the experiment:

    meta_logs/<rep>/<exp>/pca_tsne/<numcls>cls_<numruns>runs_view_type_tsne_e<E>.png
    meta_logs/<rep>/<exp>/pca_tsne/<numcls>cls_<numruns>runs_class_tsne_e<E>.png

The experiment folder name already encodes `freeze_<N>` (and other training
settings), so plots from different freeze runs are segregated automatically.

Usage:
    python pca_tsne.py                              # all reps, default settings
    python pca_tsne.py --rep_list rgb               # only the rgb rep
    python pca_tsne.py --num_classes 32             # plot all 32 classes
    python pca_tsne.py --num_runs 3                 # cap aggregation at 3 runs/exp
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import cupy as cp
from cuml.decomposition import PCA as cuPCA
from cuml.manifold import TSNE


CLASSNAMES = [
    "airplane", "can", "basket", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "bus", "cabinet", "camera", "car", "chair", "clock",
    "display", "faucet", "guitar", "helmet", "knife", "lamp", "laptop",
    "loudspeaker", "motorcycle", "mug", "pistol", "table",
    "telephone,phone,telephone set", "tower", "train,railroad train",
    "vessel,watercraft", "washer,automatic washer,washing machine",
]
VIEW_TYPES = ["Expanded", "Expanded-like", "Foreshortened",
              "Foreshortened-like", "Remainder"]

FEATURE_RE = re.compile(r"feature_(\d+)\.npz$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=str, default="meta_logs",
                   help="Root folder containing per-representation experiment folders.")
    p.add_argument("--rep_list", nargs="+", default=["rgb", "depth", "edge"],
                   help="Which representation folders under --root to walk.")
    p.add_argument("--num_classes", type=int, default=10,
                   help="If <32, only plots samples whose class_idx < this number.")
    p.add_argument("--num_runs", type=int, default=None,
                   help="Cap on runs aggregated per experiment. None = all.")
    p.add_argument("--feature_dim", type=int, default=None,
                   help="Optional feature-width validation. Default: infer from "
                        "the NPZ, allowing mixed ResNet/ViT/TinyViT experiments.")
    p.add_argument("--pca_dim", type=int, default=50,
                   help="Intermediate PCA dim before t-SNE.")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--every_n_epochs", type=int, default=None,
                   help="If set, only compute t-SNE for epochs where epoch %% N == 0. "
                        "Useful when feature dumps span an uneven epoch cadence.")
    p.add_argument("--overwrite", action="store_true",
                   help="By default, skip an experiment+epoch if both plots exist.")
    return p.parse_args()


def discover_epochs(run_dir):
    """Return sorted list of epochs E for which feature_E.npz exists in run_dir."""
    epochs = []
    for fname in os.listdir(run_dir):
        m = FEATURE_RE.match(fname)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def load_features(path, feature_dim):
    data = np.load(path)
    view_index = np.asarray(data["view_index"])
    n_rows = int(view_index.size)
    values = np.asarray(data["features"])
    if n_rows <= 0 or values.size % n_rows:
        raise ValueError(
            f"Cannot infer feature width from {values.shape} and {n_rows} labels")
    inferred_dim = values.size // n_rows
    if feature_dim is not None and inferred_dim != feature_dim:
        raise ValueError(
            f"NPZ feature width {inferred_dim} != --feature_dim {feature_dim}")
    features = values.reshape(n_rows, inferred_dim)
    return features, view_index, data["view_type"], data["view_class"]


def compute_tsne(features, pca_dim=50, perplexity=30.0):
    """L2-normalise → cuML PCA → cuML t-SNE → 2D numpy."""
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    features_gpu = cp.asarray(features)
    # PCA pre-step (kept for parity with original; cuml TSNE then runs on the
    # PCA-reduced features for stability and speed).
    pca = cuPCA(n_components=min(pca_dim, features.shape[1], features.shape[0] - 1))
    features_pca_gpu = pca.fit_transform(features_gpu)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    features_2d_gpu = tsne.fit_transform(features_pca_gpu)
    return cp.asnumpy(features_2d_gpu)


def plot_by_label(features_2d, labels, title, label_names, save_path,
                  dpi=200, point_size=5, alpha=0.6):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        idx = labels == lab
        lab_name = label_names[lab] if lab < len(label_names) else str(lab)
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                    s=point_size, alpha=alpha, label=lab_name)
    plt.legend(markerscale=3, fontsize=6, loc="upper left",
               bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"  saved: {save_path}")
    plt.close()


def filter_by_class(features, view_idx, view_type, class_idx, num_classes):
    mask = class_idx < num_classes
    return features[mask], view_idx[mask], view_type[mask], class_idx[mask]


def process_experiment(rep, exp, exp_root, args):
    """Process one meta_logs/<rep>/<exp>/ folder."""
    run_dirs = sorted([
        os.path.join(exp_root, d) for d in os.listdir(exp_root)
        if os.path.isdir(os.path.join(exp_root, d))
    ])
    if args.num_runs is not None:
        run_dirs = run_dirs[:args.num_runs]
    if not run_dirs:
        return

    all_epochs = set()
    for run_dir in run_dirs:
        all_epochs.update(discover_epochs(run_dir))
    if args.every_n_epochs and args.every_n_epochs > 1:
        all_epochs = {e for e in all_epochs if e % args.every_n_epochs == 0}
    if not all_epochs:
        return

    print(f"[{rep}/{exp}] {len(run_dirs)} run(s), {len(all_epochs)} epoch(s)")

    plots_dir = os.path.join(exp_root, "pca_tsne")
    os.makedirs(plots_dir, exist_ok=True)

    for e in sorted(all_epochs):
        prefix = f"{args.num_classes}cls_{len(run_dirs)}runs"
        view_path = os.path.join(plots_dir, f"{prefix}_view_type_tsne_e{e}.png")
        class_path = os.path.join(plots_dir, f"{prefix}_class_tsne_e{e}.png")
        if (not args.overwrite) and os.path.exists(view_path) and os.path.exists(class_path):
            continue

        features_list, view_idx_list, view_type_list, class_idx_list = [], [], [], []
        for run_dir in run_dirs:
            file_path = os.path.join(run_dir, f"feature_{e}.npz")
            if not os.path.isfile(file_path):
                continue
            try:
                f, vi, vt, ci = load_features(file_path, args.feature_dim)
            except Exception as ex:
                print(f"  skip {file_path}: {ex}")
                continue
            features_list.append(f)
            view_idx_list.append(vi)
            view_type_list.append(vt)
            class_idx_list.append(ci)
        if not features_list:
            continue

        features = np.concatenate(features_list, axis=0)
        view_idx = np.concatenate(view_idx_list, axis=0)
        view_type = np.concatenate(view_type_list, axis=0)
        class_idx = np.concatenate(class_idx_list, axis=0)

        if args.num_classes < 32:
            features, view_idx, view_type, class_idx = filter_by_class(
                features, view_idx, view_type, class_idx, args.num_classes)
        if features.shape[0] == 0:
            print(f"  e={e}: no samples after class filter, skip")
            continue

        try:
            features_2d = compute_tsne(features, pca_dim=args.pca_dim,
                                       perplexity=args.perplexity)
        except Exception as ex:
            print(f"  e={e}: t-SNE failed ({ex}), skip")
            continue

        plot_by_label(features_2d, view_type,
                      f"{rep}/{exp}  e={e}  (view type)",
                      VIEW_TYPES, view_path)
        plot_by_label(features_2d, class_idx,
                      f"{rep}/{exp}  e={e}  (class)",
                      CLASSNAMES, class_path)


def main():
    args = parse_args()
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
