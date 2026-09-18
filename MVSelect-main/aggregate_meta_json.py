"""Aggregate experiment metadata and export auditable selection trajectories.

With no arguments, keeps the existing meta_logs/{rgb,depth,edge} traversal.
--experiment-dir processes a named folder directly. --selection-only explicitly
allows five-subtype logs without accuracy fields. See README for reporting details.
"""
import argparse
import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.architectures import infer_architecture_from_name
from export_selection_plot_data import load_runs
from selection_plotting import write_selection_outputs, read_priors


def aggregate_experiment(EXP_ROOT, args):
    # =========================
    # CONFIG
    # =========================
    # EXP_ROOT = "meta_logs/rgb/steps2_train_ins25_lr1e-05base1.0other1.0select_wd0.0001select0.0001_e100"   # change if needed
    SMOOTH_WINDOW = 1
    OUT_JSON = "aggregated_summary.json"
    OUT_FIG = "aggregated_view_ratios.png"

    VIEW_KEYS = [
        "expanded",
        "Expanded-like",
        "Foreshortened",
        "Foreshortened-like",
        "Remainder"
    ]

    NON_REMAINDER_KEYS = [
        "expanded",
        "Expanded-like",
        "Foreshortened",
        "Foreshortened-like"
    ]

    ACC_KEYS = [
        "accuracy",
        "all_views_accuracy",
        "random_accuracy",
        "expanded_accuracy",
        "expanded_like_accuracy",
        "foreshortened_accuracy",
        "foreshortened_like_accuracy",
        "remainder_accuracy"
    ]

    ACC_KEYS_3 = [
        "accuracy",
        "all_views_accuracy",
        "random_accuracy",
        "expanded_accuracy_3",
        "expanded_like_accuracy_3",
        "foreshortened_accuracy_3",
        "foreshortened_like_accuracy_3",
        "remainder_accuracy_3"
    ]

    ACC_KEYS_5 = [
        "accuracy",
        "all_views_accuracy",
        "random_accuracy",
        "expanded_accuracy_5",
        "expanded_like_accuracy_5",
        "foreshortened_accuracy_5",
        "foreshortened_like_accuracy_5",
        "remainder_accuracy_5"
    ]
    ACC_KEYS_MAPPING = {
        "accuracy": "Self-generated (N=5)",
        "all_views_accuracy": "All (N=114)",
        "random_accuracy": "Random (N=5)",
        "expanded_accuracy": "Random Expanded (N=1)",
        "expanded_like_accuracy": "Random Expanded-Like (N=1)",
        "foreshortened_accuracy": "Random Foreshortened (N=1)",
        "foreshortened_like_accuracy": "Random Foreshortened-Like (N=1)",
        "remainder_accuracy": "Random Remainder (N=1)"
    }

    ACC_KEYS_MAPPING_3 = {
        "accuracy": "Self-generated (N=5)",
        "all_views_accuracy": "All (N=114)",
        "random_accuracy": "Random (N=5)",
        "expanded_accuracy_3": "Random Expanded (N=3)",
        "expanded_like_accuracy_3": "Random Expanded-Like (N=3)",
        "foreshortened_accuracy_3": "Random Foreshortened (N=3)",
        "foreshortened_like_accuracy_3": "Random Foreshortened-Like (N=3)",
        "remainder_accuracy_3": "Random Remainder (N=3)"
    }

    ACC_KEYS_MAPPING_5 = {
        "accuracy": "Self-generated (N=5)",
        "all_views_accuracy": "All (N=114)",
        "random_accuracy": "Random (N=5)",
        "expanded_accuracy_5": "Random Expanded (N=5)",
        "expanded_like_accuracy_5": "Random Expanded-Like (N=5)",
        "foreshortened_accuracy_5": "Random Foreshortened (N=5)",
        "foreshortened_like_accuracy_5": "Random Foreshortened-Like (N=5)",
        "remainder_accuracy_5": "Random Remainder (N=5)"
    }
    # METRIC_KEYS = VIEW_KEYS + ["accuracy"]


    # =========================
    # HELPERS
    # =========================
    def moving_average_padded(x, w):
        """
        Moving average with same-length output.
        Pads with nearest neighbor values.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        if w <= 1 or n == 0:
            return x.copy()

        valid = np.convolve(x, np.ones(w) / w, mode="valid")
        pad_left = w // 2
        pad_right = n - len(valid) - pad_left

        left_pad = np.full(pad_left, valid[0])
        right_pad = np.full(pad_right, valid[-1])

        return np.concatenate([left_pad, valid, right_pad])

    def first_epoch(cond):
        idx = np.where(cond)[0]
        return int(idx[0]) if len(idx) > 0 else None

    # =========================
    # LOAD RUNS
    # =========================
    records, skipped = load_runs(Path(EXP_ROOT), 'legacy')
    runs = [r['_raw'] for r in records]
    run_names = [r['file'] for r in records]
    num_runs = len(runs)
    if not runs:
        print(f'{EXP_ROOT}: no files meet legacy inclusion; exclusions: {skipped}')
        return
    selection_summary = write_selection_outputs(EXP_ROOT, 'legacy', args.candidate_priors, args.epoch_start)
    required_accuracy = set(ACC_KEYS + ACC_KEYS_3 + ACC_KEYS_5 + ['per_class_acc'])
    for name, run in zip(run_names, runs):
        missing = sorted(required_accuracy - run.keys())
        if missing:
            raise ValueError(f'{name}: missing accuracy fields {missing}; use --selection-only explicitly for selection-only logs.')

    recognition_counts = {r.get('recognition_num_views') for r in runs}
    random_counts = {r.get('steps') for r in runs}
    # Do not turn missing metadata into a purportedly measured input count.
    # Explicit recognition_num_views takes precedence; steps is an action count.
    if len(recognition_counts - {None}) > 1 or len(random_counts - {None}) > 1:
        raise ValueError(f'Inconsistent recorded input counts in {EXP_ROOT}: recognition={recognition_counts}, random={random_counts}')
    recognition_n = next(iter(recognition_counts)) if len(recognition_counts)==1 else None
    random_n = next(iter(random_counts)) if len(random_counts)==1 else None
    policy_label = f'Policy-selected (N={recognition_n})' if recognition_n is not None else 'Policy-selected (N not recorded)'
    random_label = f'Random (N={random_n})' if random_n is not None else 'Random (N not recorded)'
    for mapping in [ACC_KEYS_MAPPING, ACC_KEYS_MAPPING_3, ACC_KEYS_MAPPING_5]:
        mapping['accuracy'] = policy_label
        mapping['random_accuracy'] = random_label
        for key in mapping:
            mapping[key] = mapping[key].replace('Expanded-Like', 'expanded-like').replace('Foreshortened-Like', 'foreshortened-like').replace('Random Expanded', 'Random expanded').replace('Random Foreshortened', 'Random foreshortened').replace('Random Remainder', 'Random remainder')

    # =========================
    # OVERFIT AGGREGATION
    # =========================
    overfit_flags = [bool(r.get("overfit", False)) for r in runs]
    overfit_pct = float(np.mean(overfit_flags))
    overfit_runs = [name for name, flag in zip(run_names, overfit_flags) if flag]

    # =========================
    # SMOOTH PER RUN
    # =========================
    smoothed = {k: [] for k in VIEW_KEYS}
    accs_smoothed = {k: [] for k in ACC_KEYS}
    accs_smoothed_3 = {k: [] for k in ACC_KEYS_3}
    accs_smoothed_5 = {k: [] for k in ACC_KEYS_5}
    acc_smoothed = []
    per_class_smoothed_runs = []


    for r in runs:
        for k in VIEW_KEYS:
            smoothed[k].append(moving_average_padded(r[k], SMOOTH_WINDOW))
        for acc_k in ACC_KEYS:
            accs_smoothed[acc_k].append(moving_average_padded(r[acc_k], SMOOTH_WINDOW))
        for acc_k in ACC_KEYS_3:
            accs_smoothed_3[acc_k].append(moving_average_padded(r[acc_k], SMOOTH_WINDOW))
        for acc_k in ACC_KEYS_5:
            accs_smoothed_5[acc_k].append(moving_average_padded(r[acc_k], SMOOTH_WINDOW))
        acc_smoothed.append(
            moving_average_padded(r["accuracy"], SMOOTH_WINDOW)
        )
        per_class = np.array(r["per_class_acc"]).T  # shape: (32, 100)
        per_class_smoothed = np.array([
            moving_average_padded(class_curve, SMOOTH_WINDOW)
            for class_curve in per_class
        ])  # shape: (32, epochs)
        per_class_smoothed_runs.append(per_class_smoothed)


    # Align all runs by shortest length (safety)
    min_len = min(len(arr) for k in VIEW_KEYS for arr in smoothed[k])
    min_accs_len = min(len(arr) for curves in [accs_smoothed, accs_smoothed_3, accs_smoothed_5] for arrays in curves.values() for arr in arrays)
    min_acc_len = min(len(a) for a in acc_smoothed)
    min_len_per_class = min(pc.shape[1] for pc in per_class_smoothed_runs)
    per_class_smoothed_runs = np.array([
        pc[:, :min_len_per_class] for pc in per_class_smoothed_runs
    ])  # shape: (num_runs, 32, epochs)


    for k in VIEW_KEYS:
        smoothed[k] = np.array([arr[:min_len] for arr in smoothed[k]])
    for k in ACC_KEYS:
        accs_smoothed[k] = np.array([arr[:min_accs_len] for arr in accs_smoothed[k]])
    for k in ACC_KEYS_3:
        accs_smoothed_3[k] = np.array([arr[:min_accs_len] for arr in accs_smoothed_3[k]])
    for k in ACC_KEYS_5:
        accs_smoothed_5[k] = np.array([arr[:min_accs_len] for arr in accs_smoothed_5[k]])
    acc_smoothed = np.array([a[:min_acc_len] for a in acc_smoothed])

    # =========================
    # MEAN CURVES
    # =========================
    mean_curves = {
        k: smoothed[k].mean(axis=0).tolist()
        for k in VIEW_KEYS
    }
    mean_accs_curves = {
        k: accs_smoothed[k].mean(axis=0).tolist()
        for k in ACC_KEYS
    }
    mean_accs_curves_3 = {
        k: accs_smoothed_3[k].mean(axis=0).tolist()
        for k in ACC_KEYS_3
    }
    mean_accs_curves_5 = {
        k: accs_smoothed_5[k].mean(axis=0).tolist()
        for k in ACC_KEYS_5
    }
    mean_accuracy = acc_smoothed.mean(axis=0).tolist()
    mean_per_class_accuracy = per_class_smoothed_runs.mean(axis=0)


    # =========================
    # DOMINANCE METRICS
    # =========================
    def dominance_stats(a, b):
        ever = []
        onset = []
        for i in range(num_runs):
            cond = smoothed[a][i] > smoothed[b][i]
            ep = first_epoch(cond)
            ever.append(ep is not None)
            if ep is not None:
                onset.append(ep)
        return float(np.mean(ever)), float(np.mean(onset)) if onset else None

    def over_threshold_stats(a, threshold):
        ever = []
        onset = []
        for i in range(num_runs):
            cond = smoothed[a][i] > threshold
            ep = first_epoch(cond)
            ever.append(ep is not None)
            if ep is not None:
                onset.append(ep)
        return float(np.mean(ever)), float(np.mean(onset)) if onset else None

    percentages = {}
    onsets = {}

    percentages["expanded_over_expanded-like"], onsets["expanded_over_expanded-like"] = \
        dominance_stats("expanded", "Expanded-like")

    percentages["expanded_over_remainder"], onsets["expanded_over_remainder"] = \
        dominance_stats("expanded", "Remainder")

    percentages["expanded_over_50%"], onsets["expanded_over_50%"] = \
        over_threshold_stats("expanded", 0.5)

    percentages["foreshortened_over_expanded-like"], onsets["foreshortened_over_expanded-like"] = \
        dominance_stats("Foreshortened", "Expanded-like")

    # =========================
    # REMAINDER NO LONGER DOMINATES (FIXED)
    # =========================
    ever = []
    onset = []

    for i in range(num_runs):
        remainder = smoothed["Remainder"][i]
        others_max = np.max(
            np.stack([smoothed[k][i] for k in NON_REMAINDER_KEYS]),
            axis=0
        )
        cond = remainder <= others_max
        ep = first_epoch(cond)
        ever.append(ep is not None)
        if ep is not None:
            onset.append(ep)

    percentages["remainder_no_longer_dominate"] = float(np.mean(ever))
    onsets["remainder_no_longer_dominate"] = float(np.mean(onset)) if onset else None

    # =========================
    # SAVE SUMMARY JSON (READABLE)
    # =========================
    summary = {
        "num_runs": num_runs,
        "run_names": run_names,
        "overfit_percentage": overfit_pct,
        "overfit_runs": overfit_runs,
        "mean_smoothed_curves": mean_curves,
        "percentages": percentages,
        "onset_epochs": onsets,
        "smoothing_window": SMOOTH_WINDOW,
        "mean_accuracy": mean_accuracy,
        "accuracy_alignment": {"common_points": min_accs_len, "original_lengths": {name: {key: len(run[key]) for key in sorted(required_accuracy)} for name,run in zip(run_names,runs)}},
        **selection_summary
    }

    with open(os.path.join(EXP_ROOT, OUT_JSON), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    plt.figure()
    for k in ACC_KEYS:
        plt.plot(np.arange(len(mean_accs_curves[k])) + args.epoch_start, mean_accs_curves[k], label=ACC_KEYS_MAPPING[k])

    plt.xlabel("Recorded epoch index (zero-based)" if args.epoch_start==0 else "Training epoch (index + 1)")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy Across Input Regimes (mean across runs)")
    plt.ylim(0.0, 100.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_ROOT, "aggregated_accuracy.png"))
    plt.close()

    plt.figure()
    for k in ACC_KEYS_3:
        plt.plot(np.arange(len(mean_accs_curves_3[k])) + args.epoch_start, mean_accs_curves_3[k], label=ACC_KEYS_MAPPING_3[k])

    plt.xlabel("Recorded epoch index (zero-based)" if args.epoch_start==0 else "Training epoch (index + 1)")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy Across Input Regimes (mean across runs)")
    plt.ylim(0.0, 100.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_ROOT, "aggregated_accuracy_3.png"))
    plt.close()

    plt.figure()
    for k in ACC_KEYS_5:
        plt.plot(np.arange(len(mean_accs_curves_5[k])) + args.epoch_start, mean_accs_curves_5[k], label=ACC_KEYS_MAPPING_5[k])

    plt.xlabel("Recorded epoch index (zero-based)" if args.epoch_start==0 else "Training epoch (index + 1)")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy Across Input Regimes (mean across runs)")
    plt.ylim(0.0, 100.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_ROOT, "aggregated_accuracy_5.png"))
    plt.close()


    # plt.figure()
    # plt.plot(mean_accuracy, label="Accuracy")
    # plt.xlabel("Recorded epoch index (zero-based)" if args.epoch_start==0 else "Training epoch (index + 1)")
    # plt.ylabel("Accuracy (%)")
    # plt.title("Averaged Smoothed Accuracy Across Runs")
    # plt.ylim(0.0, 100.0)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(EXP_ROOT, "aggregated_accuracy.png"))
    # plt.close()


    class_short = [
        "airplane", "ashcan", "basket", "bathtub", "bed", "bench", "bookshelf",
        "bottle", "bowl", "bus", "cabinet", "camera", "car", "chair", "clock",
        "display", "faucet", "guitar", "helmet", "knife", "lamp", "laptop",
        "loudspeaker", "motorcycle", "mug", "pistol", "table", "telephone",
        "tower", "train", "vessel", "washer",
    ]
    n_classes, n_epochs = mean_per_class_accuracy.shape
    fig, ax = plt.subplots(figsize=(max(7, 0.10 * n_epochs + 3),
                                    max(7, 0.25 * n_classes + 1)))
    im = ax.imshow(mean_per_class_accuracy, aspect="auto", cmap="viridis",
                   origin="lower", vmin=0.0, vmax=100.0)
    ax.set_xlabel("Recorded epoch index (zero-based)" if args.epoch_start==0 else "Training epoch (index + 1)")
    ax.set_ylabel("Object category")
    ax.set_title("Object-category accuracy across runs (mean)")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_short[:n_classes], fontsize=7)
    xticks = np.arange(0, n_epochs, max(1, n_epochs // 10))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t + args.epoch_start) for t in xticks])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(EXP_ROOT, "aggregated_per_class_accuracy.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('meta_logs'))
    parser.add_argument('--experiment-dir', type=Path, action='append', help='Process this folder directly; repeat for multiple conditions.')
    parser.add_argument('--selection-only', action='store_true', help='Explicitly include logs with valid selection arrays even without accuracy fields.')
    parser.add_argument('--candidate-priors', type=Path, help='JSON containing verified five-subtype candidate fractions; defaults are labelled unverified legacy constants.')
    parser.add_argument('--epoch-start', type=int, choices=[0,1], default=0, help='0 preserves original indices; set 1 only after confirming index 0 is training epoch 1.')
    args=parser.parse_args()
    read_priors(args.candidate_priors)
    experiments=args.experiment_dir or []
    if not experiments:
        for rep in ['rgb','depth','edge']:
            rep_root=args.root/rep
            if not rep_root.is_dir():
                print(f'Skipping missing representation folder: {rep_root}')
                continue
            experiments.extend(p for p in sorted(rep_root.iterdir()) if p.is_dir() and infer_architecture_from_name(p.name) in ['resnet18','vit','tinyvit'])
    for folder in experiments:
        if args.selection_only:
            write_selection_outputs(folder, 'selection', args.candidate_priors, args.epoch_start)
        else:
            aggregate_experiment(str(folder), args)
    if not experiments:
        parser.error('No experiment folders found; pass --experiment-dir /path/to/experiment.')

if __name__=='__main__':
    main()
