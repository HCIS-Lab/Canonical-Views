#!/usr/bin/env python3
"""Plot every rendered test view in the primary mid-level feature space.

The three raw coordinates are:

    x = ellipse aspect ratio
    y = bilateral symmetry
    z = edge-orientation entropy

No PCA, standardization, coordinate filtering, or point subsampling is applied.
When selection JSONs are supplied, every candidate remains visible in gray and
views selected at the requested epoch are overlaid with high-contrast markers.
Without selection JSONs, exact view family is shown through point color.
"""

import argparse
import json
import os
import shutil
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mvselect_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from midlevel_shape_features import (
    CACHE_VERSION,
    DEFAULT_DATA_ROOT,
    VIEW_BUCKET_LABELS,
    VIEW_BUCKETS,
    build_view_feature_cache,
    load_modelnet40_classnames,
)


METRICS = [
    "ellipse_aspect_ratio",
    "bilateral_symmetry",
    "edge_entropy",
]
AXIS_LABELS = {
    "ellipse_aspect_ratio": "Ellipse aspect ratio",
    "bilateral_symmetry": "Bilateral symmetry",
    "edge_entropy": "Edge entropy",
}
FAMILY_COLORS = {
    "expanded": "#007A3D",
    "Expanded-like": "#00A6A6",
    "Foreshortened": "#C43C39",
    "Foreshortened-like": "#E69F00",
    "Remainder": "#6F6F6F",
}
# Draw the majority category first so it does not hide smaller families.
DRAW_ORDER = [
    "Remainder",
    "Foreshortened-like",
    "Foreshortened",
    "Expanded-like",
    "expanded",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional class indices or aliases, e.g. --classes airplane chair.",
    )
    parser.add_argument(
        "--cache_csv",
        default=None,
        help="Default: cache/midlevel_features_<version>_<split>.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: compare/midlevel_feature_space_3d/<split>",
    )
    parser.add_argument(
        "--selection_dir",
        default=None,
        help="Optional experiment folder containing *_selection.json files.",
    )
    parser.add_argument(
        "--selection_epoch",
        type=int,
        default=100,
        help="Selection epoch highlighted when --selection_dir is supplied.",
    )
    parser.add_argument(
        "--selection_run",
        default=None,
        help=(
            "Optional selection JSON/run timestamp. Default: first saved run "
            "with feature_<selection_epoch>.npz."
        ),
    )
    parser.add_argument(
        "--initial_camera",
        type=int,
        default=0,
        help="Exact saved rollout to visualize for every instance.",
    )
    parser.add_argument(
        "--expected_selected_views",
        type=int,
        default=5,
        help="Expected agent actions after removing the initial camera.",
    )
    parser.add_argument(
        "--per_instance",
        action="store_true",
        help="With one --classes value, create one figure folder per object.",
    )
    parser.add_argument(
        "--per_instance_count",
        type=int,
        default=5,
        help=(
            "Expected number of evaluated instances in --per_instance mode. "
            "With selections, their instance IDs define the set."
        ),
    )
    parser.add_argument(
        "--one_instance_per_class",
        action="store_true",
        help=(
            "Plot one deterministic evaluated object from every requested "
            "class in one combined figure. With selections, use the first "
            "exact-rollout instance in each class."
        ),
    )
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument(
        "--limit_images",
        type=int,
        default=None,
        help="Smoke-test limit only. Omit for the dataset-wide figure.",
    )
    parser.add_argument("--marker_size", type=float, default=9.0)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--selected_marker_size", type=float, default=24.0)
    parser.add_argument("--elevation", type=float, default=22.0)
    parser.add_argument("--azimuth", type=float, default=-52.0)
    parser.add_argument("--expected_classes", type=int, default=32)
    parser.add_argument(
        "--expected_objects_per_class",
        type=int,
        default=0,
        help="Optional fixed expectation; 0 accepts every object on disk.",
    )
    parser.add_argument("--expected_views_per_object", type=int, default=114)
    parser.add_argument(
        "--skip_disk_coverage_check",
        action="store_true",
        help="Skip exact cache-vs-dataset filename comparison.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Plot finite rows and save excluded rows instead of failing when "
             "one of the three coordinates is non-finite.",
    )
    return parser.parse_args()


def display_family(value):
    return VIEW_BUCKET_LABELS.get(value, value)


def canonical_class_name(class_name):
    return str(class_name).split(",", 1)[0].strip()


def safe_path_component(value):
    text = str(value)
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in text
    ).strip("_")
    return cleaned or "unknown"


def resolve_class_filter(requested, classnames):
    """Resolve class indices, full labels, or any comma-separated alias."""
    if not requested:
        return list(range(len(classnames))), "All classes", "all_classes"

    selected = []
    for token in requested:
        normalized = str(token).strip().casefold()
        matches = []
        if normalized.isdigit():
            index = int(normalized)
            if 0 <= index < len(classnames):
                matches = [index]
        if not matches:
            for index, class_name in enumerate(classnames):
                aliases = [
                    alias.strip().casefold()
                    for alias in str(class_name).split(",")
                ]
                if (normalized == str(class_name).casefold()
                        or normalized in aliases):
                    matches.append(index)
        if len(matches) != 1:
            raise ValueError(
                f"Class '{token}' matched {len(matches)} ModelNet classes. "
                "Use a numeric class index or an exact alias.")
        if matches[0] not in selected:
            selected.append(matches[0])

    names = [canonical_class_name(classnames[index]) for index in selected]
    scope = ", ".join(names)
    slug = "_".join(
        "".join(character if character.isalnum() else "_" for character in name)
        .strip("_").lower()
        for name in names
    )
    return selected, scope, f"class_{slug}"


def resolve_rollout_feature_file(selection_dir, selection_epoch, selection_run):
    run_names = sorted(
        filename[:-len("_selection.json")]
        for filename in os.listdir(selection_dir)
        if filename.endswith("_selection.json")
    )
    if selection_run:
        wanted = os.path.basename(str(selection_run))
        if wanted.endswith("_selection.json"):
            wanted = wanted[:-len("_selection.json")]
        run_names = [run_name for run_name in run_names if run_name == wanted]
        if not run_names:
            raise FileNotFoundError(
                f"Selection run '{selection_run}' has no matching JSON in "
                f"{selection_dir}")
    candidates = [
        (run_name, os.path.join(
            selection_dir, run_name, f"feature_{selection_epoch}.npz"))
        for run_name in run_names
    ]
    available = [(run_name, path) for run_name, path in candidates
                 if os.path.isfile(path)]
    if not available:
        attempted = "\n".join(f"  {path}" for _, path in candidates)
        raise FileNotFoundError(
            "No exact-rollout feature dump was found. Expected one of:\n"
            f"{attempted}\nThe flattened selection JSON cannot recover a "
            "single five-view rollout.")
    if len(available) > 1 and selection_run is None:
        print(
            f"Found {len(available)} rollout feature dumps; using first: "
            f"{available[0][0]}. Set --selection_run to choose another.")
    return available[0]


def annotate_model_selections(view_df, selection_dir, selection_epoch,
                              selection_run, initial_camera,
                              expected_selected_views):
    """Attach one exact saved agent rollout per evaluated object."""
    frame = view_df.copy()
    frame["selection_count"] = 0
    frame["selected"] = False
    metadata = {
        "enabled": False,
        "selection_dir": None,
        "selection_epoch": None,
        "selection_run": None,
        "feature_file": None,
        "instance_identity_source": None,
        "initial_camera": None,
        "expected_agent_selected_views": None,
        "n_rollouts_in_feature_file": 0,
        "n_rollouts_for_initial_camera": 0,
        "n_rollouts_in_class_scope": 0,
        "n_unmatched_selected_views_in_class_scope": 0,
        "n_unique_selected_views": 0,
    }
    if selection_dir is None:
        return frame, metadata

    selection_dir = os.path.abspath(selection_dir)
    if not os.path.isdir(selection_dir):
        raise FileNotFoundError(
            f"Selection directory not found: {selection_dir}")
    if not any(filename.endswith("_selection.json")
               for filename in os.listdir(selection_dir)):
        raise FileNotFoundError(
            f"No *_selection.json files in {selection_dir}")
    run_name, feature_path = resolve_rollout_feature_file(
        selection_dir, selection_epoch, selection_run)
    active_classes = set(pd.to_numeric(
        frame["class_idx"], errors="raise").astype(int))
    with np.load(feature_path) as feature:
        required = {
            "selected_mask", "selected_init_cam", "selected_class",
        }
        missing = required.difference(feature.files)
        if missing:
            raise KeyError(
                f"{feature_path} is missing exact rollout arrays: "
                f"{sorted(missing)}")
        masks = np.asarray(feature["selected_mask"]).astype(bool)
        initial_cameras = np.asarray(
            feature["selected_init_cam"]).reshape(-1).astype(int)
        classes = np.asarray(
            feature["selected_class"]).reshape(-1).astype(int)
        if "selected_instance" in feature.files:
            instances = np.asarray(
                feature["selected_instance"]).reshape(-1).astype(str)
            instance_identity_source = "selected_instance"
        else:
            instances = np.full(len(classes), "", dtype=object)
            instance_identity_source = "legacy_sorted_test_loader"
    if not (len(masks) == len(initial_cameras) == len(classes)):
        raise ValueError(
            f"Exact rollout arrays have inconsistent lengths: {feature_path}")

    rollout_rows = np.flatnonzero(initial_cameras == int(initial_camera))
    if not len(rollout_rows):
        available = sorted(np.unique(initial_cameras).tolist())
        raise ValueError(
            f"Initial camera {initial_camera} is absent from {feature_path}. "
            f"Available initial cameras: {available}")
    if instance_identity_source == "legacy_sorted_test_loader":
        for class_idx in sorted(active_classes):
            class_rows = rollout_rows[classes[rollout_rows] == class_idx]
            candidate_ids = sorted(
                frame.loc[
                    pd.to_numeric(frame["class_idx"], errors="raise")
                    .astype(int) == class_idx,
                    "instance_id",
                ].astype(str).unique()
            )
            if len(candidate_ids) < len(class_rows):
                raise ValueError(
                    f"Legacy rollout has {len(class_rows)} evaluated objects "
                    f"for class {class_idx}, but the cache has only "
                    f"{len(candidate_ids)} instance IDs.")
            for row_idx, instance_id in zip(
                    class_rows, candidate_ids[:len(class_rows)]):
                instances[row_idx] = instance_id
        print(
            "Legacy feature dump has no selected_instance array; reconstructed "
            "instance IDs from the sorted five-instance test-loader order.")

    exact_keys = set()
    rollout_instances = set()
    for row_idx in rollout_rows:
        class_idx = int(classes[row_idx])
        if class_idx not in active_classes:
            continue
        instance_id = str(instances[row_idx])
        if not instance_id:
            raise ValueError(
                f"Could not reconstruct an instance ID for class {class_idx}, "
                f"rollout row {row_idx} in {feature_path}.")
        rollout_key = (class_idx, instance_id)
        if rollout_key in rollout_instances:
            raise ValueError(
                "The chosen run contains duplicate exact rollouts for "
                f"class={class_idx}, instance={instance_id}, "
                f"initial_camera={initial_camera}.")
        rollout_instances.add(rollout_key)
        agent_mask = masks[row_idx].copy()
        if not 0 <= int(initial_camera) < len(agent_mask):
            raise ValueError(
                f"Initial camera {initial_camera} is outside a mask of "
                f"length {len(agent_mask)}")
        agent_mask[int(initial_camera)] = False
        selected_indices = np.flatnonzero(agent_mask)
        if (expected_selected_views
                and len(selected_indices) != expected_selected_views):
            raise ValueError(
                f"Exact rollout for class={class_idx}, instance={instance_id} "
                f"contains {len(selected_indices)} agent-selected views after "
                f"removing initial camera {initial_camera}; expected "
                f"{expected_selected_views}.")
        exact_keys.update(
            (class_idx, instance_id, int(view_idx))
            for view_idx in selected_indices
        )

    cache_keys = list(zip(
        pd.to_numeric(frame["class_idx"], errors="raise").astype(int),
        frame["instance_id"].astype(str),
        pd.to_numeric(frame["view_index"], errors="raise").astype(int),
    ))
    frame["selection_count"] = [int(key in exact_keys) for key in cache_keys]
    frame["selected"] = frame["selection_count"] > 0
    scoped_keys = {
        key for key in exact_keys if key[0] in active_classes
    }
    scoped_rollouts = {
        key for key in rollout_instances if key[0] in active_classes
    }
    matched_views = int(frame["selected"].sum())
    unmatched_views = int(len(scoped_keys)) - matched_views
    if unmatched_views:
        raise ValueError(
            f"{unmatched_views} exact selected view indices did not match the "
            "per-view cache. Check that the feature dump and cache use the "
            "same dataset/view ordering.")
    metadata.update({
        "enabled": True,
        "selection_dir": selection_dir,
        "selection_epoch": int(selection_epoch),
        "selection_run": run_name,
        "feature_file": feature_path,
        "instance_identity_source": instance_identity_source,
        "initial_camera": int(initial_camera),
        "expected_agent_selected_views": int(expected_selected_views),
        "n_rollouts_in_feature_file": int(len(masks)),
        "n_rollouts_for_initial_camera": int(len(rollout_rows)),
        "n_rollouts_in_class_scope": int(len(scoped_rollouts)),
        "n_unmatched_selected_views_in_class_scope": unmatched_views,
        "n_unique_selected_views": matched_views,
    })
    if not frame["selected"].any():
        raise ValueError(
            "Selection JSONs were loaded, but none of their class/filename "
            "keys matched the plotted dataset rows.")
    return frame, metadata


def validate_dataset_coverage(view_df, args):
    """Reject partial/stale caches before labeling a figure as all-view."""
    if args.limit_images is not None:
        print("Smoke-test image limit is active; full-dataset coverage checks skipped.")
        return

    failures = []
    duplicate_keys = view_df.duplicated(["class_idx", "filename"])
    if duplicate_keys.any():
        failures.append(
            f"cache contains {int(duplicate_keys.sum())} duplicate "
            "class/filename rows")

    if not args.skip_disk_coverage_check:
        expected_keys = set()
        classnames = load_modelnet40_classnames()
        active_class_indices = sorted(
            pd.to_numeric(view_df["class_idx"], errors="raise")
            .astype(int).unique())
        for class_idx in active_class_indices:
            class_name = classnames[class_idx]
            split_dir = os.path.join(args.data_root, class_name, args.split)
            if not os.path.isdir(split_dir):
                failures.append(f"dataset split directory is missing: {split_dir}")
                continue
            expected_keys.update(
                (class_idx, filename)
                for filename in os.listdir(split_dir)
                if filename.lower().endswith(".png")
            )
        cache_keys = set(zip(
            pd.to_numeric(view_df["class_idx"], errors="raise").astype(int),
            view_df["filename"].astype(str),
        ))
        missing_from_cache = expected_keys.difference(cache_keys)
        extra_in_cache = cache_keys.difference(expected_keys)
        if missing_from_cache or extra_in_cache:
            failures.append(
                "cache does not exactly match PNGs on disk "
                f"({len(missing_from_cache)} missing, "
                f"{len(extra_in_cache)} extra)")

    n_classes = int(view_df["class_idx"].nunique())
    if args.expected_classes and n_classes != args.expected_classes:
        failures.append(
            f"found {n_classes} classes, expected {args.expected_classes}")

    objects_per_class = (
        view_df[["class_idx", "instance_id"]]
        .drop_duplicates()
        .groupby("class_idx")
        .size()
    )
    if args.expected_objects_per_class and (
            objects_per_class != args.expected_objects_per_class).any():
        bad = objects_per_class[
            objects_per_class != args.expected_objects_per_class]
        failures.append(
            f"{len(bad)} classes do not have "
            f"{args.expected_objects_per_class} objects")

    views_per_object = view_df.groupby(["class_idx", "instance_id"]).size()
    if args.expected_views_per_object and (
            views_per_object != args.expected_views_per_object).any():
        bad = views_per_object[
            views_per_object != args.expected_views_per_object]
        failures.append(
            f"{len(bad)} objects do not have "
            f"{args.expected_views_per_object} views "
            f"(observed range {views_per_object.min()}-{views_per_object.max()})")

    if failures:
        raise ValueError(
            "The cache does not cover the expected complete test split: "
            + "; ".join(failures)
            + ". Recompute the cache, or set the corresponding --expected_* "
              "argument to 0 only when using a deliberately different dataset. "
              "The exact disk-coverage check can be disabled explicitly with "
              "--skip_disk_coverage_check.")


def finite_plot_frame(view_df, output_dir, allow_missing):
    required = {
        "class_idx",
        "class_name",
        "instance_id",
        "view_index",
        "filename",
        "view_type",
        *METRICS,
    }
    missing_columns = sorted(required.difference(view_df.columns))
    if missing_columns:
        raise ValueError(
            "Per-view cache is missing required columns: "
            + ", ".join(missing_columns))
    unknown_families = sorted(
        set(view_df["view_type"].dropna().astype(str)).difference(VIEW_BUCKETS))
    if unknown_families:
        raise ValueError(
            "Per-view cache contains unsupported view types: "
            + ", ".join(unknown_families))

    frame = view_df.copy()
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    finite = np.isfinite(frame[METRICS].to_numpy(dtype=float)).all(axis=1)
    excluded = frame.loc[~finite].copy()
    excluded_path = os.path.join(output_dir, "excluded_nonfinite_views.csv")
    if len(excluded):
        excluded.to_csv(excluded_path, index=False)
        missing_counts = {
            metric: int((~np.isfinite(pd.to_numeric(
                excluded[metric], errors="coerce"))).sum())
            for metric in METRICS
        }
        message = (
            f"{len(excluded)} of {len(frame)} views have a non-finite 3D "
            f"coordinate ({', '.join(f'{metric}={count}' for metric, count in missing_counts.items())}). "
            f"Details: {excluded_path}")
        if not allow_missing:
            raise ValueError(message + ". Re-run with --allow_missing to omit them.")
        print(f"WARNING: {message}")
    elif os.path.exists(excluded_path):
        os.remove(excluded_path)

    plotted = frame.loc[finite].copy()
    if plotted.empty:
        raise ValueError("No views have finite values for all three coordinates")
    duplicate_key = plotted.duplicated(["class_idx", "filename"])
    if duplicate_key.any():
        duplicate = plotted.loc[duplicate_key, ["class_name", "filename"]].iloc[0]
        raise ValueError(
            "Per-view cache has duplicate class/filename rows, e.g. "
            f"{duplicate['class_name']}/{duplicate['filename']}")
    return plotted, excluded


def add_points(ax, frame, marker_size, alpha, selected_marker_size,
               selection_mode=False, show_centroids=True):
    if selection_mode:
        ax.scatter(
            frame[METRICS[0]],
            frame[METRICS[1]],
            frame[METRICS[2]],
            s=marker_size,
            c="#626B77",
            alpha=alpha,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
        )
        selected = frame[frame["selected"]]
        # A white halo keeps selected points legible over dense gray regions.
        if len(selected):
            ax.scatter(
                selected[METRICS[0]],
                selected[METRICS[1]],
                selected[METRICS[2]],
                s=selected_marker_size * 1.75,
                c="white",
                alpha=0.94,
                edgecolors="none",
                depthshade=False,
                rasterized=True,
            )
            ax.scatter(
                selected[METRICS[0]],
                selected[METRICS[1]],
                selected[METRICS[2]],
                s=selected_marker_size,
                c="#D81B60",
                alpha=0.96,
                edgecolors="#202124",
                linewidths=0.28,
                depthshade=False,
                rasterized=True,
            )
        return

    for family in DRAW_ORDER:
        block = frame[frame["view_type"] == family]
        if block.empty:
            continue
        family_alpha = alpha * (0.62 if family == "Remainder" else 1.0)
        ax.scatter(
            block[METRICS[0]],
            block[METRICS[1]],
            block[METRICS[2]],
            s=marker_size,
            c=FAMILY_COLORS[family],
            alpha=family_alpha,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
        )
        if show_centroids:
            center = block[METRICS].mean().to_numpy(dtype=float)
            ax.scatter(
                [center[0]], [center[1]], [center[2]],
                marker="X", s=78, c=FAMILY_COLORS[family],
                edgecolors="white", linewidths=0.8, depthshade=False,
            )


def style_axis(ax, elevation, azimuth, title=None):
    ax.set_xlabel(AXIS_LABELS[METRICS[0]], labelpad=9)
    ax.set_ylabel(AXIS_LABELS[METRICS[1]], labelpad=9)
    ax.set_zlabel(AXIS_LABELS[METRICS[2]], labelpad=9)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_box_aspect((1.2, 1.0, 1.0))
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, pad=12)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
        axis.pane.set_edgecolor((0.82, 0.82, 0.82, 1.0))


def legend_handles(frame, selection_epoch=None, selection_mode=False):
    if selection_mode:
        selected = frame[frame["selected"]]
        handles = [
            Line2D(
                [0], [0], marker="o", linestyle="none", markersize=5,
                markerfacecolor="#626B77", markeredgecolor="none",
                alpha=0.75,
                label=f"All candidate views (n={len(frame):,})",
            ),
        ]
        if len(selected):
            handles.append(Line2D(
                [0], [0], marker="o", linestyle="none", markersize=7,
                markerfacecolor="#D81B60", markeredgecolor="#202124",
                markeredgewidth=0.5,
                label=(
                    f"Selected at epoch {selection_epoch} "
                    f"(n={len(selected):,})"
                ),
            ))
        else:
            handles.append(Line2D(
                [0], [0], marker="o", linestyle="none", markersize=7,
                markerfacecolor="none", markeredgecolor="#D81B60",
                markeredgewidth=1.0,
                label=f"No matched selections at epoch {selection_epoch}",
            ))
        return handles

    handles = []
    for family in VIEW_BUCKETS:
        count = int((frame["view_type"] == family).sum())
        if count == 0:
            continue
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor=FAMILY_COLORS[family], markeredgecolor="none",
            label=f"{display_family(family)} (n={count:,})",
        ))
    handles.append(Line2D(
        [0], [0], marker="X", linestyle="none", markersize=7,
        markerfacecolor="black", markeredgecolor="white",
        label="Family centroid",
    ))
    return handles


def plot_angle(frame, args, output_dir, subtitle, elevation, azimuth,
               angle_title, filename, scope_label,
               compatibility_filename=None):
    fig = plt.figure(figsize=(9.8, 7.8))
    ax = fig.add_subplot(111, projection="3d")
    add_points(
        ax,
        frame,
        args.marker_size,
        args.alpha,
        args.selected_marker_size,
        selection_mode=args.selection_dir is not None,
    )
    style_axis(
        ax,
        elevation,
        azimuth,
        f"{scope_label}: {angle_title}\n{subtitle}",
    )
    ax.legend(
        handles=legend_handles(
            frame,
            args.selection_epoch,
            selection_mode=args.selection_dir is not None,
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False, fontsize=8,
    )
    fig.subplots_adjust(left=0.02, right=0.96, top=0.91, bottom=0.18)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300)
    print(f"Saved: {path}")
    if compatibility_filename:
        compatibility_path = os.path.join(output_dir, compatibility_filename)
        fig.savefig(compatibility_path, dpi=300)
        print(f"Saved: {compatibility_path}")
    plt.close(fig)


def plot_all_angles(frame, args, output_dir, subtitle, scope_label):
    views = [
        (
            args.elevation,
            args.azimuth,
            "Primary view",
            "all_test_views_midlevel_3d_primary.png",
            "all_test_views_midlevel_3d.png",
        ),
        (
            24.0,
            38.0,
            "Opposite view",
            "all_test_views_midlevel_3d_opposite.png",
            None,
        ),
        (
            38.0,
            132.0,
            "Rear view",
            "all_test_views_midlevel_3d_rear.png",
            None,
        ),
    ]
    for elevation, azimuth, title, filename, compatibility in views:
        plot_angle(
            frame,
            args,
            output_dir,
            subtitle,
            elevation,
            azimuth,
            title,
            filename,
            scope_label,
            compatibility,
        )

    fig = plt.figure(figsize=(18.6, 6.8))
    for index, (elevation, azimuth, title, _, _) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        add_points(
            ax,
            frame,
            args.marker_size,
            args.alpha,
            args.selected_marker_size,
            selection_mode=args.selection_dir is not None,
        )
        style_axis(ax, elevation, azimuth, title)
    fig.suptitle(f"{scope_label}: {subtitle}", y=0.98)
    fig.legend(
        handles=legend_handles(
            frame,
            args.selection_epoch,
            selection_mode=args.selection_dir is not None,
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.89, bottom=0.16, wspace=0.02)
    combined_path = os.path.join(
        output_dir, "all_test_views_midlevel_3d_angles.png")
    fig.savefig(combined_path, dpi=280)
    plt.close(fig)
    print(f"Saved: {combined_path}")


def plot_per_instance_folders(frame, args, output_dir):
    """Write one figure bundle for each evaluated object instance."""
    if args.selection_dir is not None:
        instance_ids = sorted(
            frame.loc[frame["selected"], "instance_id"].astype(str).unique())
        if not instance_ids:
            raise ValueError(
                "Per-instance output found no instance IDs in the selected "
                "views at the requested epoch.")
    else:
        # ModelNet40 iterates sorted filenames and keeps the first N instance
        # IDs when per_cls_instances=N, so this reproduces the test loader.
        instance_ids = sorted(frame["instance_id"].astype(str).unique())
        if args.per_instance_count:
            instance_ids = instance_ids[:args.per_instance_count]

    if (args.per_instance_count
            and len(instance_ids) != args.per_instance_count):
        raise ValueError(
            f"Per-instance scope resolved to {len(instance_ids)} objects, but "
            f"--per_instance_count={args.per_instance_count}. Check that the "
            "selection JSON corresponds to the five-instance test run.")

    active_instance_ids = set(instance_ids)
    active_dirs = {
        f"instance_{safe_path_component(instance_id)}"
        for instance_id in instance_ids
    }
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if (name.startswith("instance_") and os.path.isdir(path)
                and name not in active_dirs):
            shutil.rmtree(path)
            print(f"Removed stale per-instance folder: {path}")

    manifest_rows = []
    scoped = frame[
        frame["instance_id"].astype(str).isin(active_instance_ids)]
    for instance_id, block in scoped.groupby("instance_id", sort=True):
        block = block.copy()
        instance_slug = safe_path_component(instance_id)
        instance_dir = os.path.join(output_dir, f"instance_{instance_slug}")
        os.makedirs(instance_dir, exist_ok=True)
        class_name = canonical_class_name(block["class_name"].iloc[0])
        selected_count = int(block["selected"].sum())
        subtitle = (
            f"{len(block):,} finite candidate views; raw coordinates; "
            f"{selected_count:,} agent-selected views; initial camera "
            f"{args.initial_camera} excluded"
        )
        scope_label = f"{class_name}, instance {instance_id}"
        plot_all_angles(
            block, args, instance_dir, subtitle, scope_label)
        points_path = os.path.join(
            instance_dir, "instance_midlevel_3d_points.csv")
        block.to_csv(points_path, index=False)
        manifest_rows.append({
            "class_idx": int(block["class_idx"].iloc[0]),
            "class_name": str(block["class_name"].iloc[0]),
            "instance_id": str(instance_id),
            "output_dir": instance_dir,
            "n_finite_views": int(len(block)),
            "n_unique_selected_views": selected_count,
            "initial_camera": int(args.initial_camera),
        })
    manifest_path = os.path.join(output_dir, "per_instance_index.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(
        f"Saved {len(manifest_rows)} per-instance folders and manifest: "
        f"{manifest_path}")


def select_one_instance_per_class(frame, selection_metadata, output_dir):
    """Keep one deterministic evaluated instance from every active class."""
    selected_blocks = []
    manifest_rows = []
    selection_enabled = bool(selection_metadata["enabled"])
    for class_idx, class_block in frame.groupby("class_idx", sort=True):
        if selection_enabled:
            instance_ids = sorted(
                class_block.loc[
                    class_block["selected"], "instance_id"
                ].astype(str).unique()
            )
            if not instance_ids:
                class_name = canonical_class_name(
                    class_block["class_name"].iloc[0])
                raise ValueError(
                    "One-instance-per-class mode found no exact rollout "
                    f"instance for class {class_name} ({int(class_idx)}).")
        else:
            instance_ids = sorted(
                class_block["instance_id"].astype(str).unique())

        instance_id = instance_ids[0]
        block = class_block[
            class_block["instance_id"].astype(str) == instance_id
        ].copy()
        selected_blocks.append(block)
        manifest_rows.append({
            "class_idx": int(class_idx),
            "class_name": str(class_block["class_name"].iloc[0]),
            "instance_id": instance_id,
            "n_candidate_views": int(len(block)),
            "n_agent_selected_views": int(block["selected"].sum()),
            "instance_source": (
                "exact_rollout" if selection_enabled
                else "sorted_dataset_order"
            ),
        })

    result = pd.concat(selected_blocks, ignore_index=True)
    manifest_path = os.path.join(
        output_dir, "one_instance_per_class_index.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(
        f"Selected one instance from each of {len(manifest_rows)} classes; "
        f"manifest: {manifest_path}")

    selection_metadata = dict(selection_metadata)
    selection_metadata["one_instance_per_class"] = True
    selection_metadata["n_displayed_instances"] = len(manifest_rows)
    if selection_enabled:
        displayed_selected = int(result["selected"].sum())
        selection_metadata["n_rollouts_in_class_scope"] = len(manifest_rows)
        selection_metadata["n_unique_selected_views"] = displayed_selected
    return result, selection_metadata


def save_data_outputs(frame, excluded, output_dir, split, scope_label,
                      selection_metadata):
    columns = [
        "class_idx", "class_name", "instance_id", "view_index",
        "view_type", "filename", "image_path", "selected",
        "selection_count", *METRICS,
    ]
    columns = [column for column in columns if column in frame.columns]
    points_path = os.path.join(output_dir, "all_test_views_midlevel_3d_points.csv")
    frame[columns].to_csv(points_path, index=False)
    selected_path = os.path.join(
        output_dir, "selected_test_views_midlevel_3d_points.csv")
    frame.loc[frame["selected"], columns].to_csv(selected_path, index=False)

    summary_rows = []
    for family in VIEW_BUCKETS:
        block = frame[frame["view_type"] == family]
        if block.empty:
            continue
        row = {
            "view_type": display_family(family),
            "n_views": len(block),
            "n_objects": block[["class_idx", "instance_id"]].drop_duplicates().shape[0],
        }
        for metric in METRICS:
            values = block[metric].to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    summary_path = os.path.join(output_dir, "view_family_midlevel_3d_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    views_per_object = frame.groupby(["class_idx", "instance_id"]).size()
    audit = {
        "split": split,
        "class_filter": scope_label,
        "coordinates": {"x": METRICS[0], "y": METRICS[1], "z": METRICS[2]},
        "normalization": "none; raw metric values",
        "subsampling": "none",
        "n_plotted_views": int(len(frame)),
        "n_excluded_nonfinite_views": int(len(excluded)),
        "n_source_views": int(len(frame) + len(excluded)),
        "n_classes": int(frame["class_idx"].nunique()),
        "n_objects": int(frame[["class_idx", "instance_id"]].drop_duplicates().shape[0]),
        "min_views_per_object": int(views_per_object.min()),
        "max_views_per_object": int(views_per_object.max()),
        "selection": {
            **selection_metadata,
            "n_finite_selected_views": int(frame["selected"].sum()),
            "n_nonfinite_selected_views": int(
                excluded["selected"].sum()) if "selected" in excluded else 0,
        },
    }
    audit_path = os.path.join(output_dir, "all_test_views_midlevel_3d_audit.json")
    with open(audit_path, "w") as handle:
        json.dump(audit, handle, indent=2)
    print(f"Saved: {points_path}")
    print(f"Saved: {selected_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {audit_path}")
    return audit


def main():
    args = parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    classnames = load_modelnet40_classnames()
    class_indices, args.scope_label, class_slug = resolve_class_filter(
        args.classes, classnames)
    if args.per_instance and args.one_instance_per_class:
        raise ValueError(
            "--per_instance and --one_instance_per_class are mutually "
            "exclusive output modes.")
    if args.per_instance and len(class_indices) != 1:
        raise ValueError(
            "--per_instance requires exactly one class in --classes, e.g. "
            "--classes airplane.")
    if args.classes and args.expected_classes == 32:
        args.expected_classes = len(class_indices)
    if args.cache_csv is None:
        args.cache_csv = os.path.join(
            root_dir, "cache", f"midlevel_features_{CACHE_VERSION}_{args.split}.csv")
    if args.output_dir is None:
        output_scope = class_slug if args.classes else ""
        if args.one_instance_per_class:
            output_scope = "one_instance_per_class"
        args.output_dir = os.path.join(
            root_dir, "compare", "midlevel_feature_space_3d", args.split,
            output_scope)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    view_df = build_view_feature_cache(args)
    view_df = view_df[
        pd.to_numeric(view_df["class_idx"], errors="coerce").isin(class_indices)
    ].copy()
    if view_df.empty:
        raise ValueError(f"No cached views matched classes: {args.scope_label}")
    validate_dataset_coverage(view_df, args)
    view_df, selection_metadata = annotate_model_selections(
        view_df,
        args.selection_dir,
        args.selection_epoch,
        args.selection_run,
        args.initial_camera,
        args.expected_selected_views,
    )
    if args.one_instance_per_class:
        view_df, selection_metadata = select_one_instance_per_class(
            view_df, selection_metadata, args.output_dir)
        args.scope_label = (
            f"One evaluated instance per class "
            f"({view_df['class_idx'].nunique()} classes)")
    frame, excluded = finite_plot_frame(
        view_df, args.output_dir, args.allow_missing)
    audit = save_data_outputs(
        frame,
        excluded,
        args.output_dir,
        args.split,
        args.scope_label,
        selection_metadata,
    )
    subtitle = (
        f"{audit['n_plotted_views']:,} views, {audit['n_objects']:,} objects, "
        f"{audit['n_classes']} classes; raw coordinates")
    if selection_metadata["enabled"]:
        subtitle += (
            f"; {audit['selection']['n_finite_selected_views']:,} unique "
            f"epoch-{args.selection_epoch} selected views highlighted from "
            f"initial camera {args.initial_camera}")
    plot_all_angles(
        frame, args, args.output_dir, subtitle, args.scope_label)
    if args.per_instance:
        plot_per_instance_folders(frame, args, args.output_dir)
    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
