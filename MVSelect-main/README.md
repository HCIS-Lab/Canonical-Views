# Learning to Select Camera Views: Efficient Multiview Understanding at Few Glances

## Overview
We release code for **MVSelect**, a view selection module for efficient multiview understanding. Parallel to reducing the image resolution or using lighter network backbones, the proposed approach reduces the computational cost for multiview understanding by limiting the number of views considered. 


 
## Content
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Training](#training)


## Dependencies
Please install dependencies with 
```
pip install -r requirements.txt
```

## Data Preparation

For multiview classification, we use ModelNet40 dataset with the circular 12-view setup [[link](https://github.com/jongchyisu/mvcnn_pytorch)][[download](http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz)] and the dodecahedral 20-view setup [[link](https://github.com/kanezaki/pytorch-rotationnet)][[download](https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar)]. 

For multiview detection, we use MultiviewX [[link](https://github.com/hou-yz/MultiviewX)][[download](https://1drv.ms/u/s!AtzsQybTubHfgP9BJt2g7R_Ku4X3Pg?e=GFGeVn)] and Wildtrack [[link](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)][[download](http://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip)] in this project. 

Your `~/Data/` folder should look like this
```
Data/
├── modelnet/
│   ├── modelnet40_images_new_12x/
│   │   └── ...
│   └── modelnet40v2png_ori4/
|       └── ...
├── MultiviewX/
│   └── ...
└── Wildtrack/ 
    └── ...
```


## Training
In order to train the task networks, please run the following
```shell script
# task network
python main.py
``` 
This should automatically return the full N-view results, as well as the oracle performances. 

To train the MVSelect, please run
```shell script
# MVSelect only
python main.py --step 2 --base_lr 0 --other_lr 0
# joint training
python main.py --step 2
``` 
The default dataset is Wildtrack. For other datasets, please specify with the `-d` argument.


## Pre-trained models
You can download the checkpoints at this [link](https://1drv.ms/u/s!AtzsQybTubHfhNRCxKzkaOiLCKkIIA?e=fQxfhI).

---

## Extensions: training, aggregation, and probing

This fork adds new training options, aggregation/plot outputs, and a zero-shot
probe of the stage-1 classifier. See the companion repo `human_multiview-main/`
for VGGT/Pi3/DINOv2-based probes that consume the `selection.json` files
produced here.

### Stage-1 training with random K views per batch

The default stage-1 trainer (`--steps 0`) aggregates ALL N views (e.g. N=114
under `--non_roll`) per instance through the MVCNN backbone. With the new
`--train_num_views K` flag, each training batch is instead restricted to **K
independent random views per batch item, re-sampled every batch**. This makes
stage-1 training closer to what stage 2 will encounter, and is ~20× faster
when going from N=114 → K=5.

```bash
# Default: train on all available views
python main.py --epochs 100 --non_roll --steps 0 --num_train_instances 25 --dataset rgb

# New: 5 random views per batch
python main.py --epochs 100 --non_roll --steps 0 --num_train_instances 25 --dataset rgb \
  --train_num_views 5
```

The K-view sampler **overrides dropcam** for stage-1 training (the K selected
positions are forcibly marked active in `keep_cams`), so you don't need to pass
`--dropcam 0` alongside it. Evaluation/validation passes still use all views
regardless of the flag.

### Skip stage 1 — joint training from ImageNet weights

Normally stage 2 (`--steps >0`) loads the stage-1 fine-tuned classifier
checkpoint via `logs/<dataset>/<arch>_performance.txt`. With `--skip_stage1`,
this lookup is bypassed: the MVCNN backbone keeps its default ImageNet-pretrained
weights (`models.resnet18(pretrained=True)` / `models.vit_b_16(pretrained=True)`
from torchvision), and the classifier + selector heads are trained jointly from
their default initialization.

```bash
# Joint train classifier + selector from ImageNet weights — no stage-1 needed
python main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 \
  --dataset rgb --skip_stage1
```

Practical notes:
- The backbone is still ImageNet-pretrained — only the ModelNet fine-tuning of
  the backbone+classifier (what stage 1 produces) is skipped.
- The model has more to learn at once (classification + view selection), so
  initial epochs are usually noisier. You may want a longer schedule or a
  higher backbone LR (`--base_lr_ratio` > 1) to compensate.
- `--skip_stage1` and `--resume` are independent: passing `--resume <run>`
  still loads weights from `logs/<dataset>/<run>/model.pth` and overrides
  `--skip_stage1`.

### Restrict selector candidates by view family

Stage-2 MVSelect can be forced to choose additional views only from one view
family using `--selector_view_limit`. This restriction applies only when the
selector is making an action during training/testing. The initial view is not
restricted, and the random/all-view/restricted-view baseline tests still use
their own candidate pools.

Choices:

| option | selector can choose |
|---|---|
| `all` | any currently available view (default) |
| `expanded_family` | Expanded + Expanded-like |
| `foreshortened_family` | Foreshortened + Foreshortened-like |
| `foreshortened_family_remainder` | Foreshortened + Foreshortened-like + Remainder |
| `remainder` | views outside the two families |

```bash
# Expanded-family-only selector
python main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 \
  --dataset rgb --save_feature --selector_view_limit expanded_family

# Foreshortened-family-only selector
python main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 \
  --dataset rgb --save_feature --selector_view_limit foreshortened_family

# Foreshortened-family + remainder selector
python main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 \
  --dataset rgb --save_feature --selector_view_limit foreshortened_family_remainder

# Remainder-only selector
python main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 \
  --dataset rgb --save_feature --selector_view_limit remainder
```

Non-default limits are encoded in saved paths so they do not collide with
unrestricted runs:

```text
logs/rgb/resnet18steps5_selview_expanded_family_train_ins25_..._<timestamp>/
meta_logs/rgb/resnet18steps5_selview_expanded_family_train_ins25_..._e100/
logs/rgb/resnet18steps5_selview_foreshortened_family_remainder_train_ins25_..._<timestamp>/
meta_logs/rgb/resnet18steps5_selview_foreshortened_family_remainder_train_ins25_..._e100/
```

If the restricted family is exhausted before all selector steps are consumed,
the selector repeats within the allowed pool rather than leaking to another
view family. If a custom dataset has no candidate from the requested family for
an instance, the trainer prints a warning and falls back to the original
`keep_cams` for that instance.

### Aggregated meta-log visualizations

`aggregate_meta_json.py` walks `meta_logs/<rep>/<exp>/*_meta.json`, averages
the per-epoch curves across runs, and writes the following per-experiment
plots into each experiment folder:

| File | What it shows |
|---|---|
| `aggregated_view_ratios.png` | The original 5-bucket selection-ratio curves (Expanded / Expanded-like / Foreshortened / Foreshortened-like / Remainder) over training. |
| `aggregated_view_family_ratios.png` | **NEW**. Collapsed 2-curve view: `expanded family = expanded + Expanded-like` (green) vs `foreshortened family = Foreshortened + Foreshortened-like` (red), with dashed horizontal lines at each group's chance-baseline share (combined: 17.5% / 8.7% / 73.8% of available views) and a vertical line marking the first epoch where expanded > foreshortened. ±SEM shaded across runs. |
| `aggregated_view_family_lift.png` | **NEW**. Same collapsed family comparison plotted as **lift over chance** (`observed_share / available_share`). Horizontal reference at `1.0` = uniform random. Magnitude-aware view of preference strength independent of bucket sizes. |
| `aggregated_per_class_accuracy.png` | **RESTYLED**. Per-class accuracy over epochs as a `class × epoch` viridis heatmap (was: 32 overlapping line curves). Much easier to spot which classes the agent is learning earliest. |
| `aggregated_accuracy.png`, `_3.png`, `_5.png` | Per-view-type accuracies for N=1/3/5 view sets (unchanged). |
| `aggregated_summary.json` | Adds a `view_family` block: bucket priors, mean expanded/foreshortened/remainder family curves, lift curves, and the crossing epoch. |
| `pca_tsne/` (subfolder) | Per-epoch t-SNE plots produced by `pca_tsne.py` (see below). Two PNGs per epoch: one colored by view type, one colored by class. |

The bucket priors are baked into `BUCKET_PRIOR` at the top of the file (the
dataset's per-object availability shares). Adjust if you re-generate the
ModelNet renders with a different sphere-of-views.

```bash
cd MVSelect-main
python3 aggregate_meta_json.py    # walks meta_logs/{rgb,depth,edge}/<exp>/
```

### Per-experiment t-SNE of features (`pca_tsne.py`)

`pca_tsne.py` walks every `meta_logs/<rep>/<exp>/<run>/feature_<epoch>.npz`
that stage-2 training has dumped, aggregates features across runs of the
same experiment, then computes one cuPCA → cuml-TSNE 2D embedding per epoch
and saves two PNGs per epoch:

- one colored by **view type** (Expanded / Expanded-like / Foreshortened / Foreshortened-like / Remainder)
- one colored by **class** (32 ModelNet classes; `--num_classes < 32` filters to the first N)

Plots go into a `pca_tsne/` subfolder inside each experiment directory, so
output for different `freeze_epoch` runs (or other settings) is segregated
automatically:

```
meta_logs/rgb/freeze_50_resnet18steps5_...e100/
└── pca_tsne/
    ├── 10cls_3runs_view_type_tsne_e10.png
    ├── 10cls_3runs_class_tsne_e10.png
    ├── 10cls_3runs_view_type_tsne_e20.png
    └── ...
```

```bash
# Default: all 3 reps, 10-class filter, all runs aggregated per experiment
python3 pca_tsne.py

# Only rgb, all 32 classes, cap at 3 runs per experiment
python3 pca_tsne.py --rep_list rgb --num_classes 32 --num_runs 3

# Force re-render even if plots already exist
python3 pca_tsne.py --overwrite
```

Skips experiments where no `feature_<E>.npz` files were dumped (e.g. stage-1
runs that didn't save features). Uses cupy + cuml for GPU acceleration; falls
back gracefully per-experiment if t-SNE fails on a particular epoch.

### Cluster-quality metrics over training (`compute_cluster_metrics.py` + `aggregate_cluster_metrics.py`)

`pca_tsne.py` gives a qualitative picture. To track cluster development
**quantitatively** across experiments, two new scripts compute silhouette
scores on the same per-epoch feature dumps and overlay them across the
freeze sweep (or any experiment list) in the standard plot styles
(`line`, `heatmap`, `sorted_bars`, `rank_stacked`).

Eight metrics per epoch:

| metric | meaning | desired direction over training |
|---|---|---|
| `silhouette_class` | how cleanly **individual per-view backbone features** cluster by 32-class label (cosine dist, L2-normalized) | **up** — single views becoming class-discriminative |
| `silhouette_class_selected` | class silhouette after pooling the agent-selected filenames from `*_selection.json`; selected-only, matching `temporal_selection_test.py` | **up** — selected multi-view representation becoming class-discriminative |
| `silhouette_class_selected_with_init` | class silhouette over the exact classifier input feature: initial view + agent-selected views; requires newer feature dumps with `selected_features` | **up** — classifier-input representation becoming class-discriminative |
| `silhouette_class_all_views_mean` | class silhouette after reconstructing each object instance and mean-aggregating all candidate views | **up** — all-view object representation becoming class-discriminative |
| `silhouette_class_all_views_max` | class silhouette after reconstructing each object instance and max-aggregating all candidate views; matches stage-1/all-view MVCNN pooling | **up** — best comparison to all-view MVCNN performance |
| `silhouette_view` | how cleanly features cluster by 5-bucket view-type label | **down** — features becoming view-invariant |
| `silhouette_view_index` | how cleanly features cluster by exact view index/camera pose, e.g. 0-113 in the 114-view setting | **down** — features becoming pose-index-invariant |
| `separability` | `silhouette_class − silhouette_view` | **up** — class-aware AND view-invariant |

All silhouette scores are bounded in [-1, 1]. `separability` is a per-view
single scalar that captures the "class identity + view invariance" trajectory
in one number.

Important interpretation note: `feature_<E>.npz` stores **one feature vector per
view**, because it is the same source used by `pca_tsne.py`. Therefore
`silhouette_class` can look numerically small even when the final classifier
has high accuracy: MVCNN classifies after aggregating multiple selected views,
not from each single-view feature alone. If you want the selected-view cluster
metric for stage-2 selector performance, inspect `silhouette_class_selected`.

For stage-2 selector experiments, note the distinction:

- `silhouette_class_selected` is reconstructed from existing per-view features
  plus `*_selection.json`. It follows the same convention as
  `temporal_selection_test.py`: group selected filenames by `(class, instance)`,
  dedupe them, pool those selected views, and **do not add the initial view**
  because `selection.json` does not store it.
- `silhouette_class_selected_with_init` requires newer feature dumps because it
  uses the exact `overall_feat` passed into the classifier during test, which
  includes the initial view plus agent-selected views.

New feature dumps also save:

- `selected_features` — the exact `overall_feat` passed into the classifier
  for each test-set `(instance, initial camera)` trial.
- `selected_class` — class label for each selected pooled feature.
- `selected_init_cam` — the initial camera index for that trial.
- `selected_mask` — boolean mask of the initial + agent-selected views.

If you compute cluster metrics on old feature files, `silhouette_class_selected`
can still be computed from `selection.json`, but
`silhouette_class_selected_with_init` will be empty/NaN until you rerun
testing/training with `--save_feature` to generate the selected-feature arrays.

Two-step pipeline:

```bash
cd MVSelect-main

# Step 1: walk meta_logs/, compute cluster_metrics.csv per experiment
#         (idempotent — re-runs are a no-op unless --overwrite)
python3 compute_cluster_metrics.py --rep_list rgb

# Step 2: overlay one comparison set at a time
./run_cluster_metrics_pipeline.sh                         # freeze sweep
COMPARISON_SET=selector_limit ./run_cluster_metrics_pipeline.sh
STYLE=sorted_bars ./run_cluster_metrics_pipeline.sh
METRICS="separability silhouette_class_selected silhouette_view_index" STYLE=line BIN_EPOCHS=20 ./run_cluster_metrics_pipeline.sh
OVERWRITE=1 ./run_cluster_metrics_pipeline.sh   # force recompute step 1
```

The wrapper runs step 1 then step 2 in sequence; if step 1 has already
populated CSVs from a previous run it skips them (re-aggregation alone is
fast).

Comparison sets are intentionally separate:

- `COMPARISON_SET=freeze` writes `compare/cluster_metrics_freeze_sweep/`:
  `no_freeze` vs `freeze_10` through `freeze_50`.
- `COMPARISON_SET=selector_limit` writes
  `compare/cluster_metrics_selector_limit_sweep/`: `select_all` vs
  `select_expanded`, `select_foreshortened`,
  `select_foreshortened_remainder`, and `select_remainder`.

Outputs (under `compare/<COMPARISON_NAME>/`):
- `<metric>_line.png` / `<metric>_heatmap.png` / `<metric>_sorted_bars.png` /
  `<metric>_rank_stacked.png` — one per requested metric × style.
- `aggregated.csv` — concatenated `(experiment, epoch, metric_values)` rows.

Practical notes:
- **Sample cost**: silhouette is O(n²). Default `MAX_SAMPLES=2000` keeps each
  epoch's computation under a second. Bump to 5000 for tighter estimates,
  drop to 500–1000 for a smoke test.
- **Requires saved features.** Step 1 reads `feature_<E>.npz` files, which
  only exist if training was run with `--save_feature`. If no features are
  saved for an experiment, that experiment is silently skipped.
- **Aggregate across runs**: when multiple training runs exist under one
  experiment folder, step 1 averages their per-epoch silhouette scores into
  a single row.
- **Per-view vs selected pooled vs all-view pooled**: `silhouette_class`,
  `silhouette_view`, and `silhouette_view_index` are computed on the saved
  per-view backbone features. `silhouette_view` uses the 5 semantic view-type
  buckets; `silhouette_view_index` uses the raw camera index labels
  (`0..113` for your non-roll setup).
  `silhouette_class_selected` is computed by cross-checking `*_selection.json`
  against the saved per-view features. The two `silhouette_class_all_views_*`
  metrics reconstruct all candidate views belonging to the same object instance
  from the saved feature order, then aggregate them before computing class
  silhouette.

**Epoch-density filter (`--every_n_epochs N` / `EVERY_N_EPOCHS=N`).** The
**feature dumps** (`feature_<E>.npz` files used by silhouette / t-SNE) were
trained with an uneven cadence — every epoch through 1–20, then every 10
afterwards — so their epoch grid is unevenly spaced. The
`--every_n_epochs N` flag keeps only epochs where `epoch % N == 0`,
collapsing that uneven set down to a clean 10, 20, …, 100 grid. Scripts
that read features:

- `compute_cluster_metrics.py` — input filter (skips silhouette computation
  for unneeded epochs).
- `aggregate_cluster_metrics.py` — output filter (drops rows post-load).
- `pca_tsne.py` — input filter (skips t-SNE for unneeded epochs).

The cluster-metrics bash wrapper (`run_cluster_metrics_pipeline.sh`)
defaults to `EVERY_N_EPOCHS=10`; set `EVERY_N_EPOCHS=1` (or `0`) to disable.

**Other analyses are unaffected** — `selection.json`, training-time test
accuracies, and per-epoch model checkpoints are saved every epoch through
training, so `temporal_selection_test.py`, `aggregate_temporal_tests.py`,
and `aggregate_vggt_confidence.py` already see a regular epoch grid by
construction and need no filter.

What to look for in the freeze sweep:
- `silhouette_class_selected` rising more sharply / earlier under
  `freeze_30/40/50` → those settings build an aggregated class representation
  from selected views faster.
- `silhouette_class` rising while remaining much lower than
  `silhouette_class_selected` → individual views are not as class-separated
  as the pooled multi-view object representation, which is expected.
- `silhouette_view` dropping faster under any setting → that setting
  produces more view-invariant representations.
- `silhouette_view_index` dropping faster → exact camera-pose identity is being
  suppressed more strongly than in other settings.
- `separability` curves' eventual height + crossover patterns answer "which
  setting buys the most class-aware, view-invariant representation per
  epoch?" in one chart.

### Mid-level visible-shape features of selected views

To test whether the selector is learning more than an arbitrary view convention,
use `midlevel_shape_features.py`. It computes descriptors directly from the
already-rendered ModelNet PNGs, then joins them to `*_selection.json` so each
epoch is summarized by the views the agent actually selected.

Detailed metric definitions, implementation notes, and caveats are in
[`README_midlevel_shape_features.md`](README_midlevel_shape_features.md).

You do **not** need ShapeNet meshes or dataset regeneration for this version.
These are image-derived, visible-structure measurements: "what mid-level shape
information does this camera view expose?" ShapeNet would only be necessary if
you want object-intrinsic 3D ground-truth quantities, such as mesh symmetry axes,
true 3D medial axes, or part annotations independent of viewpoint.

Metrics:

| group | metrics | interpretation |
|---|---|---|
| axis visibility | `ellipse_orientation_deg`, `ellipse_aspect_ratio`, `skeleton_length_norm`, `skeleton_elongation` | whether the silhouette exposes a clear major axis / elongated structure |
| symmetry / part organization | `bilateral_symmetry`, `medial_axis_symmetry`, `skeleton_endpoint_count`, `skeleton_branchpoint_count`, `skeleton_branch_density` | whether the visible shape exposes organized symmetric or branched structure |
| edge organization | `dominant_edge_orientation_deg`, `edge_entropy`, `edge_anisotropy` | whether edges are organized around dominant orientations or isotropic/noisy |

Orientation columns use axial circular averaging in the epoch summaries, so
0 degrees and 180 degrees are treated as the same axis. The default comparison
heatmaps emphasize scalar "amount/organization" metrics; orientation columns
remain in the CSVs for more targeted inspection.

The default comparison value is `lift`:

```text
lift_metric = mean(metric on selected views at epoch t)
              - mean(metric over all candidate views of the same object instances)
```

So a positive lift means the selector is using views with more of that visible
mid-level structure than the same objects' all-view baseline. This avoids
confusing "some object classes are more elongated/symmetric" with "the selector
prefers views that expose elongation/symmetry."

Commands:

```bash
cd MVSelect-main

# One experiment: writes <selection_dir>/midlevel_features/
python3 midlevel_shape_features.py \
  --selection_dir meta_logs/rgb/resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100 \
  --bin_epochs 10

# Freeze sweep: no_freeze vs freeze_10..freeze_50
./run_midlevel_shape_features_pipeline.sh

# Selector-limit sweep: select_all vs select_expanded/select_foreshortened/...
COMPARISON_SET=selector_limit ./run_midlevel_shape_features_pipeline.sh

# Plot raw selected values instead of selected-minus-baseline lift
VALUE=selected STYLE=both ./run_midlevel_shape_features_pipeline.sh
```

Outputs:

- `<selection_dir>/midlevel_features/per_view_midlevel_features.csv` — cached
  image descriptors for all candidate PNGs.
- `<selection_dir>/midlevel_features/selected_midlevel_by_run_epoch.csv` — one
  row per selection file and epoch.
- `<selection_dir>/midlevel_features/selected_midlevel_summary.csv` — averaged
  over selection files per epoch.
- `<selection_dir>/midlevel_features/midlevel_lift_heatmap.png` — per-experiment
  selected-minus-baseline heatmap.
- `compare/midlevel_shape_<comparison>_sweep/*_heatmap.png` — cross-experiment
  freeze or selector-limit plots.

### Temporal selection test — manipulation robustness + margin stability (`temporal_selection_test.py`)

Replay an experiment's per-epoch agent selections through the **final**
trained classifier (held fixed) and track two things over the 100 epochs of
selection history:

1. **Manipulation robustness** — accuracy drop `clean − manipulated` for three
   image perturbations:
   - rotation ±10°
   - color jitter (brightness/contrast/saturation/hue)
   - rotation + color jitter
2. **Prediction margin** — top-1 minus top-2 logit on the unperturbed
   selections, mean ± std across instances.

Holding the classifier fixed isolates the *selection-quality* signal from the
*model-improvement* signal. Any change in curves over epochs reflects how the
agent's selections evolved, not the classifier's training progress. Useful for
the question raised by the freeze-epoch sweep: "performance plateaus early but
selection keeps moving — what is the agent actually optimising for?"

```bash
# Bare experiment-folder name — auto-resolves to meta_logs/<dataset>/<exp>/
# and auto-locates logs/<dataset>/<exp>_<timestamp>/model.pth for the checkpoint.
python temporal_selection_test.py \
  --selection_dir resnet18steps5_train_ins25_lr0.0005..._e100 \
  --dataset rgb --non_roll --num_train_instances 25

# Full relative path also works
python temporal_selection_test.py \
  --selection_dir meta_logs/rgb/resnet18steps5_train_ins25_..._e100 \
  --dataset rgb --non_roll --num_train_instances 25

# Explicit checkpoint (overrides auto-location; useful when an experiment has
# multiple stage-2 logdirs and you want a specific one)
python temporal_selection_test.py \
  --selection_dir meta_logs/rgb/...e100 \
  --checkpoint   logs/rgb/resnet18steps5_..._2026-05-30_12-34-56/model.pth \
  --dataset rgb --non_roll --num_train_instances 25

# Smoke test on the first 10 epochs of the selection history
python temporal_selection_test.py \
  --selection_dir meta_logs/rgb/...e100 \
  --dataset rgb --non_roll --num_train_instances 25 \
  --max_epochs 10

# Sweep every stage-2 experiment in meta_logs/ in one go.
# Default: final classifier held fixed (uses model.pth).
./run_temporal_test_all.sh                                        # single GPU, final classifier
GPUS=4 ./run_temporal_test_all.sh                                 # 4-GPU round-robin
COMPARISON_SET=selector_limit GPUS=4 ./run_temporal_test_all.sh    # only selector-limit experiments
PER_EPOCH_CHECKPOINT=1 ./run_temporal_test_all.sh                 # classifier-at-epoch-t, needs model_e<E>.pth
```

**Path resolution.** `--selection_dir` accepts either the bare experiment
folder name (auto-prepends `meta_logs/<dataset>/`) or the full relative path.

**Checkpoint discovery.** With no `--checkpoint` argument, the script searches
`logs/<dataset>/` for directories of the form `<exp_basename>_<timestamp>/`
and picks the most-recent one. (Note: `<arch>_performance.txt` points to the
**stage-1** backbone, not the final stage-2 classifier needed here — so the
script does NOT use that file. Pass `--checkpoint` explicitly when you have
multiple stage-2 runs of the same experiment and want a specific one; the
script prints the full list of candidates when more than one matches.)

**Important: stage-2 checkpoint saving was previously skipped.** Before this
fix, `main.py` only ran `torch.save(model.state_dict(), ...)` when
`args.steps == 0`, so stage-2 training (including `--skip_stage1 --steps >0`
joint training) silently never wrote a `model.pth`. The save gate has been
removed — both stages now write `logdir/model.pth` every epoch (last-epoch
weights at run completion). **Stage-2 experiments trained before this fix
will not have a `model.pth` on disk**; re-train (or load whatever weights
are in memory if the training process is still alive) to use them with the
temporal test, the zero-shot probe, or any other downstream checkpoint-based
analysis.

Outputs (default `<selection_dir>/temporal_test/`):
- `deviation_rotate.png`, `deviation_jitter.png`, `deviation_rotate_jitter.png`
  — accuracy drop curves; expected to fall over epochs if selection improves.
- `margin_over_time.png` — prediction-margin curve; expected to rise if
  selection becomes more confident.
- `temporal_test.csv` — raw `(epoch, condition, accuracy, mean_margin,
  std_margin, n_trials)` rows.

A note on the protocol: this measures the FINAL classifier's response to each
epoch's selections. To instead measure model-at-epoch-E's confidence on its
own selections, you would need per-epoch checkpoints — which aren't saved by
default. The fixed-classifier protocol is the cleaner one for the
"is selection improving?" question because it removes the confound of model
improvement entirely.

**Two protocols available:**

| `--per_epoch_checkpoint` | What is held fixed | Curves answer |
|---|---|---|
| off (default) | the FINAL classifier | "How does the fully-trained model now see past selections?" — isolates selection-quality changes. |
| on | the selections, model varies per epoch | "What reward signal was the agent getting at epoch t?" — what was driving view-selection changes during training. |

The `--per_epoch_checkpoint` protocol requires per-epoch model snapshots,
which the default training run does NOT save. Enable them via the new
`--save_every_epoch N` flag in `main.py` (writes `model_e<E>.pth` every N
epochs in addition to the rolling `model.pth`):

```bash
# Train with per-epoch checkpoints (1 = every epoch; ~5 GB for a 100-epoch run)
python main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 \
  --dataset rgb --save_every_epoch 1

# Lighter alternative: snapshot every 5 epochs (~1 GB for a 100-epoch run)
python main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 \
  --dataset rgb --save_every_epoch 5
```

Then run the temporal test in classifier-at-epoch-t mode:

```bash
python temporal_selection_test.py \
  --selection_dir meta_logs/rgb/.../e100 \
  --dataset rgb --non_roll --num_train_instances 25 \
  --per_epoch_checkpoint
```

If `--save_every_epoch 5` was used, epochs without a saved snapshot (1, 2, 3,
4, 6, 7, ...) are simply skipped — the plots will show only the snapshotted
epochs (5, 10, 15, ...). The script prints which epochs were skipped at the
end of the run.

The plot titles indicate which protocol was used (`final classifier held
fixed` vs `classifier-at-epoch-t`) so figures from the two modes don't get
mixed up.

**Sweep-wrapper default.** `run_temporal_test_all.sh` defaults to
`PER_EPOCH_CHECKPOINT=0`, i.e. the **final classifier held fixed** protocol.
This uses each experiment's final `model.pth` and works for ordinary training
runs. To run the classifier-at-epoch-t protocol, pass
`PER_EPOCH_CHECKPOINT=1`; that requires `model_e<E>.pth` snapshots from
training with `--save_every_epoch >0`.

Use `COMPARISON_SET=freeze` or `COMPARISON_SET=selector_limit` to evaluate
only the experiments that will later be aggregated by
`run_aggregate_temporal_tests.sh`. The default `COMPARISON_SET=all` keeps the
old behavior and scans every stage-2 experiment under `meta_logs/`.

### Aggregating temporal-test runs across experiments (`aggregate_temporal_tests.py`)

`temporal_selection_test.py` outputs one set of plots per experiment, each with
its own y-axis range — making side-by-side comparison awkward. This script
reads `<exp>/temporal_test/temporal_test.csv` from any list of experiments and
overlays their curves on a single figure per metric, with a shared y-axis. No
re-running of the eval required.

```bash
# Direct: list experiments on the CLI (PATH or PATH:LABEL syntax)
python3 aggregate_temporal_tests.py \
  --exp resnet18steps3_train_ins25_lr0.0005..._e100:no_freeze \
  --exp freeze_10_resnet18steps3_train_ins25_lr0.0005..._e100:freeze_10 \
  --exp freeze_20_resnet18steps3_train_ins25_lr0.0005..._e100:freeze_20 \
  --dataset rgb \
  --output_dir compare/steps3_freeze_sweep

# Or use the bash wrapper for the standard comparison sets
./run_aggregate_temporal_tests.sh
COMPARISON_SET=selector_limit ./run_aggregate_temporal_tests.sh
```

Outputs (in `--output_dir`):
- `deviation_rotate.png`, `deviation_jitter.png`, `deviation_rotate_jitter.png`
  — one line per experiment for each manipulation type.
- `margin_over_time.png` — one line per experiment.
- `aggregated.csv` — concatenated raw rows with an `experiment` column for
  ad-hoc analysis.

Useful CLI flags: `--smooth N` applies a rolling-mean window (default 1 = off);
`--ymax_dev` / `--ymax_margin` force consistent y-axis caps when comparing
multiple comparison sets; `--title_suffix "..."` adds a second title line;
`--style {line,heatmap,both}` switches plot style — `heatmap` produces an
`experiments × epochs` grid (rows = experiments, color = deviation/margin)
which is much easier to read than overlaid lines when there are many
experiments; `--bin_epochs N` collapses the heatmap's epoch axis into N
bins for an even more compact view. The bash wrapper defaults to
`BIN_EPOCHS=10`, so 100 epoch-level rows become 10 heatmap columns by default.
Use `BIN_EPOCHS=0` to restore the full epoch-by-epoch heatmap.

**The bash wrapper defaults to `STYLE=heatmap`** because overlaid lines get
unreadable past ~4 experiments. Other options:

- `STYLE=line` — original overlaid-lines plots (`deviation_<cond>.png`).
- `STYLE=heatmap` — `experiments × epochs` grid; rows = experiments, color =
  deviation/margin (`deviation_<cond>_heatmap.png`).
- `STYLE=sorted_bars` — grouped bars per epoch sorted **left-to-right by
  value** (largest leftmost). Same experiment = same colour throughout, so
  when one experiment overtakes another, its colour block changes horizontal
  position within the epoch group. Y-axis is the actual value (not a sum), so
  the magnitudes are directly readable
  (`deviation_<cond>_sorted_bars.png`). **This is the recommended choice if
  you want both rank changes and meaningful y-axis values.**
- `STYLE=rank_stacked` — same idea but stacked **vertically** (segments
  ordered by value at each epoch). Compact, but the total bar height is the
  sum of values across experiments — not directly meaningful, so the y-axis
  reads as a derived quantity rather than a real magnitude
  (`deviation_<cond>_rank_stacked.png`).
- `STYLE=both` = line + heatmap. `STYLE=all` = every style.

Heatmap and rank_stacked outputs use `_heatmap.png` / `_rank_stacked.png`
suffixes; line-mode files keep their original names, so styles never overwrite
each other. If you previously ran in line mode and only see the old
`deviation_<cond>.png` files in the output dir, the new files are sitting
next to them under the suffixed names; or just clear the folder and re-run.

The wrapper keeps comparison families separate:

- `COMPARISON_SET=freeze` writes `compare/steps5_freeze_sweep/`:
  `no_freeze` vs `freeze_10` through `freeze_50`.
- `COMPARISON_SET=selector_limit` writes `compare/steps5_selector_limit_sweep/`:
  `select_all` vs `select_expanded`, `select_foreshortened`,
  `select_foreshortened_remainder`, and `select_remainder`.

Use env vars (`COMPARISON_NAME`, `DATASET`, `SMOOTH`, `YMAX_DEV`,
`YMAX_MARGIN`, `STYLE`) to tune the run without editing the script body.

### Zero-shot single-view classification probe

After stage-1 training (`main.py --steps 0`), MVSelect writes a
`logs/<dataset>/<arch>_performance.txt` file recording the best run. Stage 2
auto-loads its checkpoint from there. The new `zero_shot_view_type_test.py`
script reuses that mechanism to test the stage-1 view-agnostic classifier
under each view-type bucket independently:

```bash
# Tests the stage-1 model with ONE random view per instance per bucket type
python3 zero_shot_view_type_test.py --dataset rgb --non_roll --num_train_instances 25

# More runs for tighter error bars
python3 zero_shot_view_type_test.py --dataset rgb --non_roll --num_train_instances 25 --n_runs 10

# Direct checkpoint path (override the auto-locator)
python3 zero_shot_view_type_test.py --dataset rgb --non_roll \
  --checkpoint logs/rgb/<run_name>/model.pth
```

The probe loops `restricted_view_test(num_views=1, view_type=...)` for each
of the 5 buckets, then writes:
- `bar_overall.png` — 5 bars per view type, mean ± std across `n_runs`.
- `per_class_heatmap.png` — 32-class × 5-bucket accuracy heatmap.
- `per_class_grid.png` — small-multiples (1 subplot per class, 5 bars each).
- `results.csv` — raw per-run, per-class accuracies.

Outputs land in `logs/<dataset>/zero_shot_view_test/` by default. To get
per-run accuracy from existing `restricted_view_test` callsites, the function
now accepts an `n_runs=` kwarg (default 10) and returns the per-run accuracy
array in the 4th element of its return tuple — old callers ignore the extra
data.

**A note on what's actually zero-shot.** The MVCNN classifier above was
trained on ModelNet, so calling it "zero-shot" is only true relative to the
view-type *labels* (the classifier doesn't know which bucket a view is
from). For a probe that's zero-shot in the stronger sense — a model that has
never seen ModelNet at all — see the CLIP-based probe below.

### Zero-shot MOCHI oddity from MVSelect features (`mochi_zero_shot_feature_oddity.py`)

This tests whether a trained MVSelect/MVCNN representation supports the MOCHI
oddity task without training a MOCHI probe. For each trial, it extracts one
feature vector per image, computes pairwise cosine similarities, assigns each
image its mean similarity to the others, and predicts the oddity as the image
with the lowest mean similarity.

This is the closest zero-shot analogue to Bonnen et al.'s pairwise decision
rule for our checkpoints. It ignores the ModelNet classifier logits and uses
features only.

Important checkpoint note:

- `model.pth` is overwritten every epoch and therefore contains only the final
  checkpoint.
- `model_e<E>.pth` exists only if training used `--save_every_epoch <N>`.
- `feature_<E>.npz` cannot replace `model_e<E>.pth` for MOCHI, because those
  feature files were extracted from ModelNet test images, not MOCHI images.

```bash
cd MVSelect-main

# Final checkpoints: no_freeze vs freeze_10..freeze_50
./run_mochi_zero_shot_feature_oddity.sh

# Final checkpoints: select_all vs selector-view-limit runs
COMPARISON_SET=selector_limit ./run_mochi_zero_shot_feature_oddity.sh

# Model-at-epoch, only if model_e<E>.pth snapshots exist
PER_EPOCH_CHECKPOINT=1 EPOCHS=10,20,30,40,50,60,70,80,90,100 \
  ./run_mochi_zero_shot_feature_oddity.sh

COMPARISON_SET=selector_limit PER_EPOCH_CHECKPOINT=1 EPOCHS=10,20,30,40,50,60,70,80,90,100 \
  ./run_mochi_zero_shot_feature_oddity.sh
```

Outputs land under `compare/mochi_feature_oddity_<comparison>/`:
- `mochi_feature_oddity_trials.csv` — one row per trial/checkpoint.
- `mochi_feature_oddity_summary.csv` — trial accuracy, condition-macro
  chance-normalized accuracy, and mean similarity margin.
- `mochi_feature_oddity_bar.png` — final-checkpoint comparison.
- `mochi_feature_oddity_heatmap.png` and
  `mochi_feature_oddity_over_epochs.png` — per-epoch comparisons.

### CLIP zero-shot single-view classification probe (`clip_zero_shot_view_type.py`)

Same protocol as the MVCNN probe (one random view per instance per
view-type bucket, repeated N times), but the classifier is **CLIP** instead
of the stage-1 MVCNN — CLIP was trained on web image-text pairs and has
never seen ModelNet, so the result is a genuinely zero-shot reading of view
informativeness. Classification is done by encoding the image with CLIP's
image tower, encoding per-class text prompts with CLIP's text tower, and
taking cosine similarity. Top-1 and top-5 accuracy are both reported.

```bash
# Default: ViT-B/32, 5-template 3D-aware prompt ensemble, 25 instances/class × 5 runs
python clip_zero_shot_view_type.py

# Bigger CLIP — slower but stronger zero-shot
python clip_zero_shot_view_type.py --clip_model openai/clip-vit-large-patch14

# Single-template prompts ("a photo of a X") instead of the 3D ensemble
python clip_zero_shot_view_type.py --no_prompt_ensemble

# Smoke test on 10 instances
python clip_zero_shot_view_type.py --limit 10
```

Outputs (default `logs/clip_zero_shot_view_test/`):
- `bar_top1.png` / `bar_top5.png` — 5 bars per view type, mean ± std.
- `per_class_top1_heatmap.png` / `per_class_top5_heatmap.png` — view_type ×
  class accuracy heatmaps (viridis, 0–100%).
- `results.csv` — per (run, instance, view_type) row with predicted class,
  top-5 indices, and binary correctness flags. Easy to slice for ad-hoc
  analyses (e.g., per-class confusion matrices).

What to look for:
- If `bar_top1.png` orders view types **the same way** as the MVCNN probe's
  bar chart — Expanded > Expanded-like > Remainder > Foreshortened >
  Foreshortened-like (or similar) — that's strong evidence that the
  ordering is about the views' *intrinsic informativeness for a generic
  visual classifier*, not about anything ModelNet- or MVCNN-specific.
- If CLIP gives a different ordering than MVCNN, that's interesting on its
  own — the discrepancy tells you which buckets MVCNN learned to use that
  a generic vision-language model wouldn't.
- Absolute accuracy: don't expect MVCNN-level numbers. CLIP zero-shot on
  ModelNet renders is typically 30–50% top-1, 60–80% top-5 — the relative
  ordering matters more than the magnitude.

Prerequisites:
- `transformers` (already in your conda env if you've been using VGGT/Pi3).
- ~$\sim$2GB GPU memory for ViT-B/32; ~6GB for ViT-L/14.
- ~5–10 min on one GPU for a default run (~4000 CLIP forwards in batches).

### Multiple `--freeze_epoch` runs

`logs/` and `meta_logs/` directories already include a `freeze_<N>_` prefix
in the experiment folder name when `--freeze_epoch != 100`, so per-run
training artifacts are segregated automatically. Two additional files needed
freeze-aware names to avoid silent overwrites — both are now handled:

- **`logs/<dataset>/<arch>_performance.txt`** → becomes
  `<arch>_freeze<N>_performance.txt` when `--freeze_epoch != 100`. This file
  records the best stage-1 run, used by both stage-2 training (`main.py
  --steps >0`) and `zero_shot_view_type_test.py` to locate the checkpoint.
  Before this fix, multiple stage-1 runs with different freeze values would
  silently overwrite each other's pointer.
- **`zero_shot_view_type_test.py` output dir** → default
  `logs/<dataset>/zero_shot_view_test/` becomes
  `zero_shot_view_test_freeze<N>/`. Pass `--freeze_epoch <N>` to the script
  to match the stage-1 run you want to probe.

Example: train two stage-1 backbones (default freeze + `freeze_epoch=80`),
then probe each:

```bash
# Stage 1 baseline (freeze_epoch=100 default = never freeze)
python main.py --epochs 100 --non_roll --steps 0 --num_train_instances 25 --dataset rgb
# → writes logs/rgb/resnet18_performance.txt

# Stage 1 with backbone frozen after epoch 80
python main.py --epochs 100 --non_roll --steps 0 --num_train_instances 25 --dataset rgb \
  --freeze_epoch 80
# → writes logs/rgb/resnet18_freeze80_performance.txt (does NOT clobber the above)

# Probe each separately
python zero_shot_view_type_test.py --dataset rgb --non_roll --num_train_instances 25
# → outputs to logs/rgb/zero_shot_view_test/

python zero_shot_view_type_test.py --dataset rgb --non_roll --num_train_instances 25 \
  --freeze_epoch 80
# → outputs to logs/rgb/zero_shot_view_test_freeze80/

# Stage 2 on top of each — auto-loads the matching performance.txt
python main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 --dataset rgb
python main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 --dataset rgb \
  --freeze_epoch 80
```

For `aggregate_meta_json.py`: no changes needed — plots are written inside
each `meta_logs/<rep>/freeze_<N>_<arch>...` folder, which already segregates
them by the dir name.

For the `human_multiview-main/` probes (agent vs random, single-view
confidence, pair-confidence control): the `--selection_dir` you pass picks
which agent run to evaluate, but the default `--output_dir` only encodes
`--selected_view_type`. If you run probes on **two stage-2 agents with
different `--freeze_epoch`**, pass `--output_dir` explicitly per run so the
two don't collide, e.g.:

```bash
./scripts/run_pipeline.sh OUTPUT_DIR=results/views/agent_default
./scripts/run_pipeline.sh OUTPUT_DIR=results/views/agent_freeze80
```

(All three `human_multiview-main/scripts/run_*.sh` wrappers accept
`OUTPUT_DIR=...` as an env-var override.)

### Recommended pipeline

1. **Train stage 1 (view-agnostic classifier).** Either all-views or with
   `--train_num_views 5` if you want a backbone trained closer to the stage-2
   sampling regime.
2. **Train stage 2 (view selector).** `python main.py --epochs 100 --non_roll
   --steps 5 --num_train_instances 25 --dataset rgb`. This auto-loads the
   stage-1 checkpoint via `<arch>_performance.txt`.
3. **Aggregate training logs.** `python aggregate_meta_json.py` for the
   family-ratio / lift / per-class-accuracy plots above.
4. **Probe view-type informativeness.** `python zero_shot_view_type_test.py`
   for the MVCNN classifier's per-bucket accuracy.
5. **Probe model uncertainty on agent selections.** In `human_multiview-main/`:
   `./scripts/run_pipeline.sh` for the VGGT/DINOv2 agent-vs-random
   comparison, `./scripts/run_view_type_single_view.sh` for VGGT's
   single-view depth confidence per bucket, `./scripts/run_pair_confidence_control.sh`
   for the identical / random / bucket-pair sanity check.

See `human_multiview-main/README.md` for the probe scripts that consume the
`selection.json` files produced by step 2.
