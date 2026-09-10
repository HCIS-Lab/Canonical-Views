import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.models.architectures import infer_architecture_from_name

# Per-object share of available views in each bucket on disk
# (modelnet_32_60_1_23). Used as the chance baseline against which agent
# selection ratios are interpreted. Sums to 1.0.
BUCKET_PRIOR = {
    "expanded": 0.035,
    "Expanded-like": 0.14,
    "Foreshortened": 0.017,
    "Foreshortened-like": 0.07,
    "Remainder": 0.738,
}
# "Family" = a pair of related view-type buckets collapsed into one curve.
# All selected views are canonical in the dataset's labelling sense; the
# family terminology here just groups buckets by geometric character
# (broadside-axis vs along-axis).
EXPANDED_FAM_PRIOR = BUCKET_PRIOR["expanded"] + BUCKET_PRIOR["Expanded-like"]
FORESHORTENED_FAM_PRIOR = BUCKET_PRIOR["Foreshortened"] + BUCKET_PRIOR["Foreshortened-like"]
REMAINDER_PRIOR = BUCKET_PRIOR["Remainder"]


def mean_sem(arr):
    arr = np.asarray(arr, dtype=float)
    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    else:
        sem = np.zeros_like(mean)
    return mean, sem


def plot_selection_share_curves(per_run, priors, curve_labels,
                                baseline_labels, colors, output_path, title,
                                line_styles=None, line_widths=None,
                                fill_alphas=None, crossover=None):
    """Plot mean selection shares, run-level SEM, and chance baselines."""
    line_styles = line_styles or {}
    line_widths = line_widths or {}
    fill_alphas = fill_alphas or {}
    means = {}
    epochs = np.arange(next(iter(per_run.values())).shape[1])

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for key, values in per_run.items():
        mean, sem = mean_sem(values)
        means[key] = mean
        color = colors[key]
        ax.plot(
            epochs,
            mean,
            color=color,
            lw=line_widths.get(key, 2.0),
            ls=line_styles.get(key, "-"),
            alpha=0.7 if key == "Remainder" else 1.0,
            label=curve_labels[key],
        )
        ax.fill_between(
            epochs,
            mean - sem,
            mean + sem,
            color=color,
            alpha=fill_alphas.get(key, 0.2),
        )

    for key, prior in priors.items():
        ax.axhline(
            prior,
            color=colors[key],
            ls=":",
            lw=1.0,
            alpha=0.5,
            label=f"{baseline_labels[key]} chance ({prior:.1%})",
        )

    if crossover is not None:
        left_key, right_key, label = crossover
        indices = np.flatnonzero(means[left_key] > means[right_key])
        if len(indices):
            epoch = int(indices[0])
            ax.axvline(
                epoch,
                color=colors[left_key],
                ls=":",
                alpha=0.6,
                label=f"{label} @ ep {epoch}",
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Share of selections")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

REP_LIST = ['rgb', 'depth', 'edge']
# , 'rgb_depth', 'rgb_edge', 'depth_edge', 'rgb_depth_edge']
# REP_LIST = ['test_rgb']
ARCH=['resnet18', 'vit', 'tinyvit']

ROOT = "meta_logs"
for REP in REP_LIST:
    REP_ROOT = os.path.join(ROOT, REP)
    for arch in ARCH:
        # ARCH_ROOT = os.path.join(REP_ROOT, arch)
        EXP_LIST = [
            name for name in os.listdir(REP_ROOT)
            if os.path.isdir(os.path.join(REP_ROOT, name))
            and infer_architecture_from_name(name) == arch
        ]
        for EXP in EXP_LIST:
            print(EXP)
            EXP_ROOT = os.path.join(REP_ROOT, EXP)
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
            runs = []
            run_names = []

            for fn in os.listdir(EXP_ROOT):
                if fn.endswith("_meta.json"):
                    with open(os.path.join(EXP_ROOT, fn), "r") as f:
                        r = json.load(f)
                        if not "per_class_acc" in r.keys():
                            continue
                        if not "expanded_accuracy_5" in r.keys():
                            continue
                        runs.append(r)
                        run_names.append(fn)
                        print(fn)

            num_runs = len(runs)
            if num_runs == 0:
                print("No runs found in " + EXP_ROOT)
                continue

            recognition_num_views = {
                int(r.get(
                    "recognition_num_views",
                    1 if r.get("active_single_view", False)
                    else int(r.get("steps", 4)) + 1,
                ))
                for r in runs
            }
            random_num_views = {
                int(r.get("steps", 1 if r.get("active_single_view", False) else 5))
                for r in runs
            }
            if len(recognition_num_views) != 1 or len(random_num_views) != 1:
                raise ValueError(
                    f"Inconsistent input-view counts across runs in {EXP_ROOT}: "
                    f"recognition={sorted(recognition_num_views)}, "
                    f"random={sorted(random_num_views)}"
                )
            recognition_num_views = recognition_num_views.pop()
            random_num_views = random_num_views.pop()
            ACC_KEYS_MAPPING["accuracy"] = (
                f"Self-generated (N={recognition_num_views})"
            )
            ACC_KEYS_MAPPING["random_accuracy"] = (
                f"Random (N={random_num_views})"
            )
            ACC_KEYS_MAPPING_3["accuracy"] = ACC_KEYS_MAPPING["accuracy"]
            ACC_KEYS_MAPPING_3["random_accuracy"] = ACC_KEYS_MAPPING["random_accuracy"]
            ACC_KEYS_MAPPING_5["accuracy"] = ACC_KEYS_MAPPING["accuracy"]
            ACC_KEYS_MAPPING_5["random_accuracy"] = ACC_KEYS_MAPPING["random_accuracy"]


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
            min_accs_len = min(len(arr) for k in ACC_KEYS for arr in accs_smoothed[k])
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
                "mean_accuracy": mean_accuracy
            }

            with open(os.path.join(EXP_ROOT, OUT_JSON), "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

            # =========================
            # EXACT VIEW-TYPE PLOT
            # =========================
            exact_colors = {
                "expanded": "#009E73",
                "Expanded-like": "#0072B2",
                "Foreshortened": "#D55E00",
                "Foreshortened-like": "#CC79A7",
                "Remainder": "#666666",
            }
            exact_labels = {
                "expanded": "Expanded",
                "Expanded-like": "Expanded-like",
                "Foreshortened": "Foreshortened",
                "Foreshortened-like": "Foreshortened-like",
                "Remainder": "Remainder",
            }
            plot_selection_share_curves(
                per_run={key: smoothed[key] for key in VIEW_KEYS},
                priors={key: BUCKET_PRIOR[key] for key in VIEW_KEYS},
                curve_labels=exact_labels,
                baseline_labels=exact_labels,
                colors=exact_colors,
                output_path=os.path.join(EXP_ROOT, OUT_FIG),
                title="Exact View Types - Selection Share "
                      "(mean ± SEM across runs)",
                line_styles={"Remainder": "--"},
                line_widths={"Remainder": 1.0},
                fill_alphas={"Remainder": 0.1},
            )

            # =========================
            # COLLAPSED FAMILY PLOT
            # expanded family       = expanded + Expanded-like
            # foreshortened family  = Foreshortened + Foreshortened-like
            # remainder             = Remainder (single bucket, drawn faint)
            #
            # All curves are share-of-selections per epoch (sums of buckets
            # already in [0, 1]). Bands are ±SEM across runs.
            # =========================
            exp_per_run = smoothed["expanded"] + smoothed["Expanded-like"]
            fore_per_run = smoothed["Foreshortened"] + smoothed["Foreshortened-like"]
            rem_per_run = smoothed["Remainder"]
            exp_mean, _ = mean_sem(exp_per_run)
            fore_mean, _ = mean_sem(fore_per_run)
            rem_mean, _ = mean_sem(rem_per_run)
            cross = first_epoch(exp_mean > fore_mean)
            family_per_run = {
                "expanded_family": exp_per_run,
                "foreshortened_family": fore_per_run,
                "Remainder": rem_per_run,
            }
            family_priors = {
                "expanded_family": EXPANDED_FAM_PRIOR,
                "foreshortened_family": FORESHORTENED_FAM_PRIOR,
                "Remainder": REMAINDER_PRIOR,
            }
            family_curve_labels = {
                "expanded_family":
                    "Expanded family (Expanded + Expanded-like)",
                "foreshortened_family":
                    "Foreshortened family (Foreshortened + Foreshortened-like)",
                "Remainder": "Remainder",
            }
            family_baseline_labels = {
                "expanded_family": "expanded-family",
                "foreshortened_family": "foreshortened-family",
                "Remainder": "remainder",
            }
            family_colors = {
                "expanded_family": "#2ca02c",
                "foreshortened_family": "#d62728",
                "Remainder": "#7f7f7f",
            }
            plot_selection_share_curves(
                per_run=family_per_run,
                priors=family_priors,
                curve_labels=family_curve_labels,
                baseline_labels=family_baseline_labels,
                colors=family_colors,
                output_path=os.path.join(
                    EXP_ROOT, "aggregated_view_family_ratios.png"),
                title="Expanded vs Foreshortened Families - Selection Share "
                      "(mean ± SEM across runs)",
                line_styles={"Remainder": "--"},
                line_widths={"Remainder": 1.0},
                fill_alphas={"Remainder": 0.1},
                crossover=(
                    "expanded_family",
                    "foreshortened_family",
                    "expanded > foreshortened",
                ),
            )

            # =========================
            # NORMALIZED ("LIFT OVER CHANCE") PLOT
            # lift = observed_share / chance_share
            # 1.0 = uniform random, >1 = over-selected, <1 = under-selected.
            # =========================
            exp_lift_per_run = exp_per_run / EXPANDED_FAM_PRIOR
            fore_lift_per_run = fore_per_run / FORESHORTENED_FAM_PRIOR
            rem_lift_per_run = rem_per_run / REMAINDER_PRIOR

            exp_lift_mean, exp_lift_sem = mean_sem(exp_lift_per_run)
            fore_lift_mean, fore_lift_sem = mean_sem(fore_lift_per_run)
            rem_lift_mean, rem_lift_sem = mean_sem(rem_lift_per_run)
            epochs = np.arange(exp_lift_mean.shape[0])

            plt.figure()
            plt.plot(epochs, exp_lift_mean, color="#2ca02c", lw=2.0,
                     label="Expanded family lift")
            plt.fill_between(epochs, exp_lift_mean - exp_lift_sem,
                             exp_lift_mean + exp_lift_sem,
                             color="#2ca02c", alpha=0.2)
            plt.plot(epochs, fore_lift_mean, color="#d62728", lw=2.0,
                     label="Foreshortened family lift")
            plt.fill_between(epochs, fore_lift_mean - fore_lift_sem,
                             fore_lift_mean + fore_lift_sem,
                             color="#d62728", alpha=0.2)
            plt.plot(epochs, rem_lift_mean, color="#7f7f7f", lw=1.0, ls="--", alpha=0.7,
                     label="Remainder lift")
            plt.fill_between(epochs, rem_lift_mean - rem_lift_sem,
                             rem_lift_mean + rem_lift_sem,
                             color="#7f7f7f", alpha=0.1)
            plt.axhline(1.0, color="black", ls=":", lw=1.0, alpha=0.6, label="chance (1.0)")
            plt.xlabel("Epoch")
            plt.ylabel("Lift over chance (observed share / available share)")
            plt.title("Expanded vs Foreshortened Families — Lift over Chance "
                      "(mean ± SEM)")
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(EXP_ROOT, "aggregated_view_family_lift.png"))
            plt.close()

            # Record the collapsed curves and crossing epoch into the summary JSON
            summary["view_family"] = {
                "bucket_prior": BUCKET_PRIOR,
                "expanded_family_prior": EXPANDED_FAM_PRIOR,
                "foreshortened_family_prior": FORESHORTENED_FAM_PRIOR,
                "remainder_prior": REMAINDER_PRIOR,
                "expanded_family_mean": exp_mean.tolist(),
                "foreshortened_family_mean": fore_mean.tolist(),
                "remainder_mean": rem_mean.tolist(),
                "expanded_family_lift_mean": exp_lift_mean.tolist(),
                "foreshortened_family_lift_mean": fore_lift_mean.tolist(),
                "remainder_lift_mean": rem_lift_mean.tolist(),
                "expanded_over_foreshortened_first_epoch": cross,
            }
            with open(os.path.join(EXP_ROOT, OUT_JSON), "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

            plt.figure()
            for k in ACC_KEYS:
                plt.plot(mean_accs_curves[k], label=ACC_KEYS_MAPPING[k])

            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title("Test Accuracy Across Input Regimes (mean across runs)")
            plt.ylim(0.0, 100.0)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(EXP_ROOT, "aggregated_accuracy.png"))
            plt.close()

            plt.figure()
            for k in ACC_KEYS_3:
                plt.plot(mean_accs_curves_3[k], label=ACC_KEYS_MAPPING_3[k])

            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title("Test Accuracy Across Input Regimes (mean across runs)")
            plt.ylim(0.0, 100.0)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(EXP_ROOT, "aggregated_accuracy_3.png"))
            plt.close()

            plt.figure()
            for k in ACC_KEYS_5:
                plt.plot(mean_accs_curves_5[k], label=ACC_KEYS_MAPPING_5[k])

            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title("Test Accuracy Across Input Regimes (mean across runs)")
            plt.ylim(0.0, 100.0)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(EXP_ROOT, "aggregated_accuracy_5.png"))
            plt.close()


            # plt.figure()
            # plt.plot(mean_accuracy, label="Accuracy")
            # plt.xlabel("Epoch")
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
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Class")
            ax.set_title("Per-class accuracy across runs (mean)")
            ax.set_yticks(range(n_classes))
            ax.set_yticklabels(class_short[:n_classes], fontsize=7)
            xticks = np.arange(0, n_epochs, max(1, n_epochs // 10))
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(t + 1) for t in xticks])
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Accuracy (%)")
            fig.tight_layout()
            fig.savefig(os.path.join(EXP_ROOT, "aggregated_per_class_accuracy.png"), dpi=150)
            plt.close(fig)
