"""Compute cluster-quality metrics from saved feature dumps, per experiment.

Walks every `meta_logs/<rep>/<exp>/<run>/feature_<epoch>.npz` that training
has dumped (the same source files `pca_tsne.py` uses) and, per
experiment and per epoch, computes:

    silhouette_class   — silhouette score over CLASS labels (cosine distance,
                          L2-normalised per-view features). Higher = individual
                          view features cluster more cleanly by 32-class
                          identity.
    silhouette_view    — silhouette score over VIEW-TYPE labels (the 5
                          view-type buckets). Higher = features encode view
                          identity strongly (bad for view-invariant
                          classification — we want this LOWER over training).
    silhouette_view_index
                        — silhouette score over raw camera/view-index labels
                          (e.g. 0..113 for 114-view non-roll experiments).
                          Higher = features cluster by exact camera pose.
    separability       — silhouette_class − silhouette_view. A single scalar
                          summarising "class-aware AND view-invariant" (HIGHER
                          over training = the model is learning the right
                          thing).
    silhouette_class_all_views_mean
                        — class silhouette after mean-aggregating all candidate
                          views of each object instance.
    silhouette_class_all_views_max
                        — class silhouette after max-aggregating all candidate
                          views of each object instance. This matches all-view
                          MVCNN pooling, but not the selected subset unless the
                          selected views happen to include
                          all candidate views.
    silhouette_class_selected
                        — class silhouette over the agent-selected views listed
                          in *_selection.json, pooled from saved per-view
                          features. This matches temporal_selection_test.py's
                          selected-only convention.
    silhouette_class_selected_with_init
                        — class silhouette over the exact pooled feature passed
                          into the classifier during selector testing (initial
                          view + agent-selected views). Requires newer feature
                          dumps with selected_features.

Saves a `cluster_metrics.csv` inside each experiment folder. Runs of the same
experiment are averaged epoch-wise.

Usage:
    python3 compute_cluster_metrics.py                     # all reps
    python3 compute_cluster_metrics.py --rep_list rgb
    python3 compute_cluster_metrics.py --max_samples 1500 --num_runs 3
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


FEATURE_RE = re.compile(r"feature_(\d+)\.npz$")

CLASSNAMES = [
    "airplane,aeroplane,plane",
    "ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin",
    "basket,handbasket",
    "bathtub,bathing tub,bath,tub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "bus,autobus,coach,charabanc,double-decker,jitney,motorbus,motorcoach,omnibus,passenger vehi",
    "cabinet",
    "camera,photographic camera",
    "car,auto,automobile,machine,motorcar",
    "chair",
    "clock",
    "display,video display",
    "faucet,spigot",
    "guitar",
    "helmet",
    "knife",
    "lamp",
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

DEFAULT_DATA_ROOTS = {
    "rgb": "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23",
    "edge": "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_edge_1_23_10.0",
    "depth": "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/depth_modelnet_32_60_1_23",
}


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
    p.add_argument("--aggregation", default="max", choices=["mean", "max"],
                   help="Pooling used for selected-view features reconstructed "
                        "from *_selection.json.")
    p.add_argument("--data_root", default=None,
                   help="Dataset root used to map selected filenames back to "
                        "feature rows. If omitted, uses main.py's default root "
                        "for the current rep (rgb/depth/edge).")
    p.add_argument("--split", default="test")
    p.add_argument("--test_per_cls_instances", type=int, default=5,
                   help="Must match main.py's test_set per_cls_instances.")
    p.add_argument("--non_roll", action="store_true",
                   help="Must match the training/testing run that wrote features.")
    p.add_argument("--non_like", action="store_true",
                   help="Must match the training/testing run that wrote features.")
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


def infer_num_views(run_dirs):
    for run_dir in run_dirs:
        for e in discover_epochs(run_dir):
            with np.load(os.path.join(run_dir, f"feature_{e}.npz")) as npz:
                return int(np.nanmax(npz["view_index"])) + 1
    return None


def _merge_selection_dict(merged, d):
    for ep, content in d.items():
        ep_int = int(ep)
        merged.setdefault(ep_int, {})
        for cls_str, view_dict in content.items():
            merged[ep_int].setdefault(cls_str, {})
            for vt, filenames in view_dict.items():
                merged[ep_int][cls_str].setdefault(vt, []).extend(filenames)


def load_selection_file(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    merged = {}
    _merge_selection_dict(merged, d)
    return merged


def load_selections(selection_dir):
    """Merge all *_selection.json files in one experiment folder."""
    merged = {}
    if not os.path.isdir(selection_dir):
        return merged
    for fn in sorted(os.listdir(selection_dir)):
        if not fn.endswith("_selection.json"):
            continue
        _merge_selection_dict(merged, load_selection_file(os.path.join(selection_dir, fn)))
    return merged


def _parse_view_file(fname):
    attr = fname.split(".p")[0].split("_")
    ins_id, cam = attr[0], int(attr[1])
    e, a, r = float(attr[2][1:]), float(attr[3][1:]), float(attr[4][1:])
    pose_string = f"{e}_{a}_{r}"
    is_long_like = "like" in fname and "short" not in fname
    is_short_like = "like" in fname and "short" in fname
    return ins_id, cam, r, pose_string, is_long_like, is_short_like


def resolve_data_root(args, rep):
    if args.data_root:
        return args.data_root
    return DEFAULT_DATA_ROOTS.get(rep)


def build_filename_index(data_root, num_views, args):
    """Reconstruct ModelNet40 test ordering without loading images.

    Returns:
        mapping: (class_idx, basename) -> (instance_idx, view_idx, class_idx)
        n_instances: number of test instances in the reconstructed order
    """
    if not data_root or not os.path.isdir(data_root):
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    pose_table = {}
    pose_index_counting = 0
    class_seen_ids = {cls: [] for cls in CLASSNAMES}
    instance_index = {}
    n_instances = 0
    mapping = {}

    for cls_idx, cls in enumerate(CLASSNAMES):
        pattern = os.path.join(data_root, cls, args.split, "*.png")
        for path in sorted(glob.glob(pattern)):
            fname = os.path.basename(path)
            ins_id, cam, r, pose_string, is_long_like, is_short_like = _parse_view_file(fname)
            if args.non_roll and r != 0.0:
                continue
            if args.non_like and (is_long_like or is_short_like):
                continue
            if pose_string not in pose_table:
                pose_table[pose_string] = pose_index_counting
                pose_index_counting += 1
            view_idx = pose_table[pose_string]
            if view_idx >= num_views:
                continue
            if (not args.non_roll) and cam > num_views:
                continue
            if (len(class_seen_ids[cls]) >= args.test_per_cls_instances
                    and ins_id not in class_seen_ids[cls]):
                break
            inst_key = (cls_idx, ins_id)
            if inst_key not in instance_index:
                instance_index[inst_key] = n_instances
                n_instances += 1
                class_seen_ids[cls].append(ins_id)
            mapping[(cls_idx, fname)] = (instance_index[inst_key], view_idx, cls_idx)

    return mapping, n_instances


def _silhouette_or_nan(features, labels, metric):
    if features is None or labels is None:
        return float("nan")
    labels = np.asarray(labels).reshape(-1)
    unique = np.unique(labels)
    if len(unique) <= 1 or features.shape[0] <= len(unique):
        return float("nan")
    try:
        return float(silhouette_score(features, labels, metric=metric))
    except Exception as ex:
        print(f"    silhouette failed: {ex}")
        return float("nan")


def infer_instance_feature_tensor(features, view_index, view_class):
    """Reconstruct per-instance all-view feature tensors from saved order.

    `trainer_mvcnn.test` saves features batch-by-batch in view-major order:
        view 0 for all B instances, view 1 for all B instances, ...
    The batch size can vary on the final batch, so we infer each batch boundary
    from runs of identical view_index values.

    Returns:
        feature_tensor: (num_instances, num_views, D)
        labels:        (num_instances,)
    """
    view_index = np.asarray(view_index).reshape(-1)
    view_class = np.asarray(view_class).reshape(-1)
    if features.shape[0] != view_index.shape[0]:
        raise ValueError(
            f"features/view_index length mismatch: {features.shape[0]} vs {view_index.shape[0]}"
        )

    n_views = int(np.nanmax(view_index)) + 1
    pos = 0
    feature_rows, labels = [], []
    n = len(view_index)
    while pos < n:
        if int(view_index[pos]) != 0:
            # If something unexpected slips in, advance to the next batch start
            next_zero = np.where(view_index[pos:] == 0)[0]
            if len(next_zero) == 0:
                break
            pos += int(next_zero[0])
            continue

        # Current batch size = length of the view-0 block.
        next_change = pos
        while next_change < n and int(view_index[next_change]) == 0:
            next_change += 1
        batch_size = next_change - pos
        block_len = batch_size * n_views
        end = pos + block_len
        if end > n:
            break

        expected = np.repeat(np.arange(n_views), batch_size)
        actual = view_index[pos:end].astype(int)
        if actual.shape[0] != expected.shape[0] or not np.array_equal(actual, expected):
            # Malformed or differently ordered feature dump; skip the rest
            # rather than producing a silently wrong instance-level metric.
            raise ValueError(
                "Could not infer instance groups from view_index order. "
                "Expected view-major blocks [0...0, 1...1, ...]."
            )

        feat_block = features[pos:end].reshape(n_views, batch_size, -1).transpose(1, 0, 2)
        feature_rows.append(feat_block)
        labels.append(view_class[pos:pos + batch_size])
        pos = end

    if not feature_rows:
        return None, None
    return np.concatenate(feature_rows, axis=0), np.concatenate(labels, axis=0)


def infer_all_view_aggregates(feature_tensor, labels):
    if feature_tensor is None:
        return None, None, labels
    return feature_tensor.mean(axis=1), feature_tensor.max(axis=1), labels


def build_selected_from_selection(feature_tensor, labels, filename_index, epoch_sel, aggregation):
    """Pool selected-only features listed in selection.json.

    This intentionally follows temporal_selection_test.py: filenames are grouped
    by (class, instance), duplicate selected views are removed, and the initial
    camera is not added because selection.json does not store it.
    """
    if feature_tensor is None or not filename_index or not epoch_sel:
        return None, None, 0

    by_instance = {}
    missing = 0
    for cls_str, view_dict in epoch_sel.items():
        cls_idx = int(cls_str)
        for filenames in view_dict.values():
            for fn in filenames:
                fname = os.path.basename(fn)
                hit = filename_index.get((cls_idx, fname))
                if hit is None:
                    missing += 1
                    continue
                inst_idx, view_idx, _ = hit
                if inst_idx >= feature_tensor.shape[0] or view_idx >= feature_tensor.shape[1]:
                    missing += 1
                    continue
                by_instance.setdefault(inst_idx, set()).add(view_idx)

    selected_rows, selected_labels = [], []
    for inst_idx, view_idxs in sorted(by_instance.items()):
        if not view_idxs:
            continue
        view_idxs = sorted(view_idxs)
        selected = feature_tensor[inst_idx, view_idxs, :]
        pooled = selected.mean(axis=0) if aggregation == "mean" else selected.max(axis=0)
        selected_rows.append(pooled)
        selected_labels.append(labels[inst_idx])

    if not selected_rows:
        return None, None, missing
    return np.stack(selected_rows, axis=0), np.asarray(selected_labels), missing


def compute_metrics_for_run(run_dir, args, selection_by_epoch=None, filename_index=None):
    """Iterate one run's feature_<E>.npz files; return list of per-epoch rows."""
    rows = []
    epochs = discover_epochs(run_dir)
    if args.every_n_epochs and args.every_n_epochs > 1:
        epochs = [e for e in epochs if e % args.every_n_epochs == 0]
    for e in epochs:
        npz = np.load(os.path.join(run_dir, f"feature_{e}.npz"))
        features = npz["features"].squeeze().reshape(-1, args.feature_dim)
        view_index = np.asarray(npz["view_index"])
        view_type = np.asarray(npz["view_type"])
        view_class = np.asarray(npz["view_class"])
        selected_features = None
        selected_class = None
        if "selected_features" in npz.files and "selected_class" in npz.files:
            selected_features = npz["selected_features"].squeeze().reshape(-1, args.feature_dim)
            selected_class = np.asarray(npz["selected_class"])

        feature_tensor, instance_class = infer_instance_feature_tensor(
            features, view_index=npz["view_index"], view_class=view_class
        )
        mean_features, max_features, _ = infer_all_view_aggregates(feature_tensor, instance_class)
        selected_json_features, selected_json_class, missing_selected = build_selected_from_selection(
            feature_tensor,
            instance_class,
            filename_index,
            (selection_by_epoch or {}).get(e),
            args.aggregation,
        )

        n = features.shape[0]
        # Downsample with a deterministic seed so the same epoch yields the
        # same subset across reruns.
        if args.max_samples and n > args.max_samples:
            rng = np.random.RandomState(args.seed + e)
            idx = rng.choice(n, args.max_samples, replace=False)
            features = features[idx]
            view_index = view_index[idx]
            view_type = view_type[idx]
            view_class = view_class[idx]
        if selected_features is not None and args.max_samples and selected_features.shape[0] > args.max_samples:
            rng = np.random.RandomState(args.seed + e + 100000)
            idx = rng.choice(selected_features.shape[0], args.max_samples, replace=False)
            selected_features = selected_features[idx]
            selected_class = selected_class[idx]
        if (selected_json_features is not None and args.max_samples
                and selected_json_features.shape[0] > args.max_samples):
            rng = np.random.RandomState(args.seed + e + 200000)
            idx = rng.choice(selected_json_features.shape[0], args.max_samples, replace=False)
            selected_json_features = selected_json_features[idx]
            selected_json_class = selected_json_class[idx]

        features = normalize(features, axis=1)
        mean_features = normalize(mean_features, axis=1) if mean_features is not None else None
        max_features = normalize(max_features, axis=1) if max_features is not None else None
        selected_features = normalize(selected_features, axis=1) if selected_features is not None else None
        selected_json_features = normalize(selected_json_features, axis=1) if selected_json_features is not None else None

        sil_class = _silhouette_or_nan(features, view_class, args.metric)
        sil_view = _silhouette_or_nan(features, view_type, args.metric)
        sil_view_index = _silhouette_or_nan(features, view_index, args.metric)
        sil_all_views_mean = _silhouette_or_nan(mean_features, instance_class, args.metric)
        sil_all_views_max = _silhouette_or_nan(max_features, instance_class, args.metric)
        sil_selected = _silhouette_or_nan(selected_json_features, selected_json_class, args.metric)
        sil_selected_with_init = _silhouette_or_nan(selected_features, selected_class, args.metric)

        rows.append({
            "epoch": e,
            "n_samples": int(features.shape[0]),
            "n_instances": int(mean_features.shape[0]) if mean_features is not None else 0,
            "n_selected_instances": int(selected_json_features.shape[0]) if selected_json_features is not None else 0,
            "n_selected_missing": int(missing_selected),
            "n_selected_with_init_trials": int(selected_features.shape[0]) if selected_features is not None else 0,
            "silhouette_class": sil_class,
            "silhouette_view": sil_view,
            "silhouette_view_index": sil_view_index,
            "separability": sil_class - sil_view if (np.isfinite(sil_class) and np.isfinite(sil_view)) else float("nan"),
            "silhouette_class_all_views_mean": sil_all_views_mean,
            "silhouette_class_all_views_max": sil_all_views_max,
            "silhouette_class_selected": sil_selected,
            "silhouette_class_selected_with_init": sil_selected_with_init,
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

    num_views = infer_num_views(run_dirs)
    if num_views is None:
        return

    selection_by_epoch = load_selections(exp_root)
    selection_by_run = {}
    for run_dir in run_dirs:
        run_name = os.path.basename(os.path.normpath(run_dir))
        run_sel = load_selection_file(os.path.join(exp_root, f"{run_name}_selection.json"))
        selection_by_run[run_dir] = run_sel if run_sel else selection_by_epoch

    filename_index = None
    if selection_by_epoch:
        try:
            data_root = resolve_data_root(args, rep)
            filename_index, n_index_instances = build_filename_index(data_root, num_views, args)
            print(f"[{rep}/{exp}] loaded selection index: "
                  f"{len(selection_by_epoch)} epoch(s), {n_index_instances} test instances, "
                  f"{len(filename_index)} filename entries")
        except Exception as ex:
            print(f"[{rep}/{exp}] WARNING: could not build selected-view index: {ex}")

    all_rows = []
    for run_dir in run_dirs:
        try:
            all_rows.extend(compute_metrics_for_run(
                run_dir, args,
                selection_by_epoch=selection_by_run.get(run_dir, selection_by_epoch),
                filename_index=filename_index,
            ))
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
