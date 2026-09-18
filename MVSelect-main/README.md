# Learning to Select Camera Views: Efficient Multiview Understanding at Few Glances

Publication-ready caption templates for the generated analysis figures are
collected in [`../PAPER_FIGURE_CAPTIONS.md`](../PAPER_FIGURE_CAPTIONS.md).

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

### Backbone architectures

`--arch` accepts `resnet18`, `vit`, and `tinyvit`. The `tinyvit` option uses
timm's `tiny_vit_5m_224.dist_in22k_ft_in1k` backbone (224-pixel input,
320-dimensional pooled feature) with ImageNet-22k distillation followed by
ImageNet-1k fine-tuning. `timm==0.9.16` is included in `requirements.txt`.

All training arguments work with TinyViT:

```bash
python main.py --arch tinyvit --epochs 100 --non_roll --steps 5 \
  --num_train_instances 25 --dataset rgb --batch_size 6 --lr 0.0005 \
  --save_feature --save_every_epoch 1 --skip_stage1
```

The architecture is encoded in every training path and checkpoint lookup:

```text
logs/rgb/tinyvitsteps5_train_ins25_..._<timestamp>/
meta_logs/rgb/tinyvitsteps5_train_ins25_..._e100/
logs/rgb/tinyvit_performance.txt
logs/rgb/tinyvit_zero_shot_view_test/
downstream_logs/tinyvit/...
```

This prevents TinyViT checkpoints, feature dumps, selections, and plots from
sharing a path with ResNet. Feature analyses infer the saved feature width
from each NPZ, so `meta_logs/` may contain 512-dimensional ResNet,
768-dimensional ViT, and 320-dimensional TinyViT experiments together.

Analysis wrappers accept the same architecture through `ARCH`:

```bash
ARCH=tinyvit GPUS=4 COMPARISON_SET=freeze ./run_temporal_test_all.sh
ARCH=tinyvit COMPARISON_SET=freeze ./run_aggregate_temporal_tests.sh
ARCH=tinyvit COMPARISON_SET=freeze ./run_cluster_metrics_pipeline.sh
ARCH=tinyvit COMPARISON_SET=freeze ./run_midlevel_shape_features_pipeline.sh
ARCH=tinyvit GPU_ID=0 ./run_selector_score_analysis.sh
ARCH=tinyvit GPU_ID=0 ./run_view_contribution_pipeline.sh
ARCH=tinyvit GPU_ID=0 ./run_single_multiview_midlevel_analysis.sh
ARCH=tinyvit GPU_ID=0 COMPARISON_SET=freeze \
  ./run_mochi_zero_shot_feature_oddity.sh
python zero_shot_view_type_test.py --arch tinyvit --dataset rgb --non_roll
```

For backward compatibility, ResNet comparison folders keep their existing
names. Non-ResNet comparison folders append the architecture, for example
`compare/cluster_metrics_freeze_sweep_tinyvit/` and
`compare/steps5_freeze_sweep_tinyvit/`.

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

This fork adds selector restrictions, aggregation/plot outputs, and downstream
probes. See the companion repo `human_multiview-main/` for VGGT/Pi3/DINOv2-based
probes that consume the `selection.json` files produced here.

### Active single-view control

The active single-view control tests whether a learned view preference still
emerges when recognition never aggregates multiple views. It uses one selector
action:

| selector context | classifier input | reward |
|---|---|---|
| random initial view | selected view only (N=1) | `loss(initial) - loss(selected)` |

The initial view remains the policy's observation in both conditions. In the
active-single condition it is only an action-independent reward baseline and
does not contribute to the classification loss or classifier gradients. This
is preferable to `--steps 0`, which has no learned action policy. Training
samples the initial camera uniformly. Each per-epoch test evaluates every
possible initial camera and pools those results, so the reported bias is not
tied to one lucky initial view.

Here N=1 denotes the classifier input. The image-conditioned policy must first
observe an initial view before choosing that classifier view, so the sensing
procedure still captures a scouting image. A system limited to one captured
image in total cannot make an image-conditioned active choice; it would instead
learn an open-loop camera prior.

Train five seeds on one GPU and generate the figures:

```bash
SINGLE_GPU=0 SEEDS="0 1 2 3 4" \
  ./run_active_single_view_experiment.sh
```

Both networks train jointly from the same seeded ImageNet backbone
initialization. Useful overrides include:

```bash
# Save feature dumps in addition to selections and checkpoints.
SAVE_FEATURE=1 ./run_active_single_view_experiment.sh

# Regenerate plots without training.
./run_active_single_view_comparison.sh

# Optional matched N=2 control; not required for the single-view analysis.
TRAIN_PAIR=1 PAIR_GPU=1 SINGLE_GPU=0 INCLUDE_PAIR=1 \
  ./run_active_single_view_experiment.sh
```

`--active_single_view` is valid only with `--steps 1`. It is encoded as
`steps1_active_single_` in `logs/` and `meta_logs/`, so its checkpoints,
selections, and metadata cannot collide with the ordinary `steps1` run.

The comparison is written to `compare/active_single_view_bias/` (with an
architecture suffix for non-ResNet backbones):

| File | What it shows |
|---|---|
| `active_single_view_types.png` | Exact view-type selection shares with availability baselines and run-level SEM. |
| `active_single_families.png` | Expanded-family, foreshortened-family, and remainder shares. |
| `active_single_view_bias_magnitude.png` | Total variation distance between the selected-view distribution and dataset availability; 0 means no aggregate selection bias. |
| `active_single_recognition_accuracy.png` | Accuracy when the classifier receives only the selected view. |
| `selection_bias_runs.csv` | Run-level values underlying all plots. |

If active single-view develops a consistent directional preference and a
nonzero distance from the availability prior, multiview feature pooling is not
necessary for the bias. This control remains descriptive: repeat it across
seeds and report both direction-specific shares and overall bias magnitude.

### Controlled policy-replay representation experiment

`train_downstream.py` pools a range of selection epochs into one static top-K
dataset and cannot test policy-history effects. The controlled replacement is
`run_policy_replay_experiment.sh`, which trains recognition networks from an
exactly shared initialization under frozen-early, final-from-start, naturally
evolving, temporally shuffled, family-matched replacement, and random-view
training policies. Three additional controls use random views for the first
10, 20, or 30 epochs and then rejoin the naturally evolving no-freeze policy
at the matching absolute epoch. It also
evaluates the original jointly trained no-freeze model as a clearly marked
external reference. All conditions use identical held-out N=5 inputs, and the
output includes accuracy, class silhouette, linear CKA, prediction
disagreement, and separate matched-control/reference audits.
To avoid unreadable all-condition trajectories, aggregation also writes raw
and difference-from-evolving heatmaps plus focused policy-history,
random-warm-up, and view-identity curve panels with shared metric scales.

```bash
ARCH=resnet18 GPUS=4 SEEDS="0 1 2" \
  ./run_policy_replay_experiment.sh
```

See [`README_policy_replay.md`](README_policy_replay.md) for protocol details,
direct commands, output definitions, and interpretation.

### Restrict selector candidates by view family

MVSelect can be forced to choose additional views only from one view
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

`aggregate_meta_json.py` preserves the existing no-argument traversal of
`meta_logs/{rgb,depth,edge}/<exp>/*_meta.json`. Missing representation folders are
reported and skipped. You can also pass `--experiment-dir` directly, without
requiring an architecture prefix in that folder's name. Selection plots use
accessible colours and distinct line styles, with one legend entry per curve.
Dotted lines show candidate availability; they are not classification chance.

| File | What it shows |
|---|---|
| `aggregated_view_ratios.png` | All five subtype trajectories and availability baselines. |
| `aggregated_view_family_ratios.png` | Expanded family, foreshortened family and remainder. Subtypes are summed within each run before means and SEM are computed. |
| `aggregated_view_family_lift.png` | Enrichment relative to candidate availability; the historical filename is retained. A reference at 1 means selected share equals candidate share. |
| `aggregated_view_trajectories.png` | Two panels showing family trajectories and all five subtypes together. |
| Matching `.pdf` and `.svg` files | Vector exports of all four selection displays. |
| `selection_plot_data.json` | Compact full per-file subtype curves, means, SEM, file hashes, available metadata, exclusions, original lengths, truncated files, priors and plotted epoch indices. Send one file per condition; full metadata archives are unnecessary for reviewing these curves. |
| `aggregated_summary.json` | Existing legacy metrics and means, plus subtype/family SEM and explicit reporting metadata. Written by the full aggregation mode. |
| `aggregated_accuracy.png`, `_3.png`, `_5.png` | Existing accuracy comparisons; missing policy input counts are labelled as unrecorded rather than guessed. |
| `aggregated_per_class_accuracy.png` | Object-category accuracy heatmap. |

**Uncertainty and inclusion.** Selection curves are unsmoothed. SEM is computed
across included files, which must be audited as independent training runs. With
one file, SEM is unavailable (`null`) and no uncertainty band is drawn. Summaries
use the shortest common selection prefix; full original curves remain in the
compact JSON. Exclusions and truncation are reported. Full aggregation retains
the legacy requirement for `per_class_acc` and `expanded_accuracy_5`; it also
checks the remaining accuracy fields before aggregating them. Accuracy comparisons
use the common prefix across all three accuracy grids to avoid ragged arrays.

**Candidate denominators.** Default subtype priors remain the historical
constants (0.035, 0.14, 0.017, 0.07, 0.738), explicitly labelled as requiring
verification. They are not re-estimated from selected shares. Supply a JSON with
exactly the keys `expanded`, `Expanded-like`, `Foreshortened`,
`Foreshortened-like`, `Remainder` and verified candidate fractions using
`--candidate-priors`. Use denominators appropriate to the actual candidate pool.
The raw expanded-versus-foreshortened share crossing is no longer drawn: the two
families have different availability. The old crossing field is retained in the
summary for compatibility and explicitly marked as not a bias-onset measure.

**Epoch convention.** The default axis preserves the original zero-based indices
and says so. Set `--epoch-start 1` only after confirming index 0 represents
training epoch 1. This affects display labels; legacy onset fields remain indices.

```bash
cd MVSelect-main
python3 aggregate_meta_json.py
python3 aggregate_meta_json.py --experiment-dir /path/to/multiview/experiment
# Explicitly permit single-view or older logs containing valid selection arrays
# without the legacy accuracy fields. This changes inclusion; record that choice.
python3 aggregate_meta_json.py --experiment-dir /path/to/singleview/experiment --selection-only
# Optional verified availability and epoch origin:
python3 aggregate_meta_json.py --experiment-dir /path/to/experiment --candidate-priors /path/to/candidate_priors.json --epoch-start 1
```

To export only the compact data without plotting (Python standard library only):

```bash
python3 export_selection_plot_data.py /path/to/experiment --output /path/to/selection_export.json
# Optional explicit inclusion change for selection-only logs:
python3 export_selection_plot_data.py /path/to/experiment --output /path/to/singleview_export.json --inclusion selection
```

The exporter refuses to overwrite an existing output. Both modes reject invalid
fractions or five-subtype shares that do not sum to one within rounding tolerance.
Do not silently compare results from different inclusion rules.

For the descriptor distribution, `plot_feature_pairs.py` replaces overlapping 3D
views with three pairwise projections of the existing points table. It preserves
all finite rows, reports exclusions and requires an explicit context label:

```bash
python3 plot_feature_pairs.py --input /path/to/all_test_views_midlevel_3d_points.csv --output /path/to/feature_pairs --context 'test subset; run ID; source epoch; initial camera'
python3 -m unittest test_selection_plotting
```

Dependencies: NumPy and Matplotlib for aggregation; pandas additionally for the
pairwise descriptor plot. These utilities do not train models or modify source
metadata. Updating the local checkout does not itself update a remote server;
transfer the changed files together, including `selection_plotting.py` and
`export_selection_plot_data.py` used by the aggregator.

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

Skips experiments where no `feature_<E>.npz` files were dumped. Uses cupy +
cuml for GPU acceleration; falls back gracefully per-experiment if t-SNE fails
on a particular epoch.

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
| `silhouette_class_all_views_max` | class silhouette after reconstructing each object instance and max-aggregating all candidate views | **up** — all-view pooled object representation becoming class-discriminative |
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
metric for selector performance, inspect `silhouette_class_selected`.

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

The core comparison value is `lift`:

```text
lift_metric = mean(metric on selected views at epoch t)
              - mean(metric over all candidate views of the same object instances)
```

So a positive lift means the selector is using views with more of that visible
mid-level structure than the same objects' all-view baseline. This avoids
confusing "some object classes are more elongated/symmetric" with "the selector
prefers views that expose elongation/symmetry."

For cross-metric plots, the wrapper defaults to `VALUE=effect`, a standardized
lift:

```text
effect_metric = lift_metric / std(metric over all candidate views of the same object instances)
```

This is easier to compare across metrics whose raw units have very different
ranges. Effect heatmaps use a fixed `[-1, 1]` color scale; values outside that
range are saturated.

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

# Per-metric heatmaps use VALUE; grouped comparison curves use GROUP_VALUE.
# Defaults: VALUE=effect, GROUP_VALUE=all.
VALUE=selected STYLE=both ./run_midlevel_shape_features_pipeline.sh
GROUP_VALUE=selected ./run_midlevel_shape_features_pipeline.sh
GROUP_VALUE=lift ./run_midlevel_shape_features_pipeline.sh

# Visual diagnostics: 10 views per test instance.
python3 visualize_midlevel_extractions.py

# Raw 3D feature space: every view of every test object.
./run_midlevel_feature_space_3d.sh

# The same raw feature space for one object class.
CLASS_NAME=airplane ./run_midlevel_feature_space_3d.sh

# One primary/opposite/rear/combined figure bundle per airplane instance.
CLASS_NAME=airplane PER_INSTANCE=1 ./run_midlevel_feature_space_3d.sh

# One combined figure containing one evaluated instance from every class.
ONE_INSTANCE_PER_CLASS=1 ./run_midlevel_feature_space_3d.sh
```

Outputs:

- `<selection_dir>/midlevel_features/per_view_midlevel_features.csv` — cached
  image descriptors for all candidate PNGs.
- `<selection_dir>/midlevel_features/selected_midlevel_by_run_epoch.csv` — one
  row per selection file and epoch.
- `<selection_dir>/midlevel_features/selected_midlevel_summary.csv` — averaged
  over selection files per epoch.
- `<selection_dir>/midlevel_features/view_type_midlevel_associations.csv` —
  five-way eta-squared and signed one-vs-rest point-biserial associations for
  ellipse aspect ratio, bilateral symmetry, and edge entropy.
- `<selection_dir>/midlevel_features/midlevel_metric_ranges.csv` — per-view raw
  ranges versus epoch-level selected/lift/effect ranges for sanity checking.
- `<selection_dir>/midlevel_features/midlevel_selected_raw_primary_metrics.png`
  — per-experiment raw ellipse aspect ratio, bilateral symmetry, and edge
  entropy in separate panels, with selected-view mean ±SEM and the raw
  same-instance all-view baseline. Separate y-axes avoid comparing incompatible
  raw units through one shared heatmap scale.
- `<selection_dir>/midlevel_features/midlevel_axis_visibility_selected_curves.png`
  — selected axis-visibility metrics over epochs.
- `<selection_dir>/midlevel_features/midlevel_symmetry_part_organization_selected_curves.png`
  — selected symmetry / part-organization metrics over epochs.
- `<selection_dir>/midlevel_features/midlevel_edge_organization_selected_curves.png`
  — selected edge-organization metrics over epochs.
- `logs/midlevel_visual_samples/test_v10/<class>/<instance>_midlevel_extractions.png`
  — contact sheets showing original render, mask with Zhang-Suen skeleton,
  Sobel gradient magnitude, and Sobel gradient orientation for sampled test
  views. Row labels include `ellipse_aspect_ratio`, `bilateral_symmetry`, and
  `edge_entropy`; the same values are saved in `visualization_index.csv`.
- `compare/midlevel_feature_space_3d/test/all_test_views_midlevel_3d_primary.png`,
  `all_test_views_midlevel_3d_opposite.png`, and
  `all_test_views_midlevel_3d_rear.png` — every finite candidate test view in
  raw ellipse-aspect-ratio, bilateral-symmetry, and edge-entropy coordinates,
  with epoch-100 no-freeze selections highlighted by default over a neutral
  candidate cloud. Each angle uses a separate full-size canvas;
  `all_test_views_midlevel_3d_angles.png` retains the concatenated view and
  `all_test_views_midlevel_3d.png` aliases the primary view. With
  `CLASS_NAME=<class> PER_INSTANCE=1`, the same figure bundle is generated in
  one folder for each of the five evaluated instance IDs present in the saved
  exact rollout feature dump, rather than all physical instances in the class
  directory. Legacy dumps without `selected_instance` are supported by
  reconstructing the missing IDs from the unshuffled loader's first-five
  sorted instance order; selected camera indices still come from the exact
  saved masks. The default overlay is the five agent actions from initial
  camera 0 in the first saved run; the initial camera itself is excluded. Set
  `SELECTION_DIR` and `SELECTION_EPOCH` to change the
  overlay, or `HIGHLIGHT_SELECTED=0` to restore exact-family colors. The folder
  also contains all-point and selected-point CSVs, family summaries, and a
  count/selection audit.
- `compare/midlevel_feature_space_3d/test/one_instance_per_class/` — the same
  combined angle figures restricted to one deterministic evaluated object from
  each of the 32 classes. All 114 candidate views and the five exact selected
  views are retained per object; `one_instance_per_class_index.csv` identifies
  the chosen object in each class.
- `compare/midlevel_shape_<comparison>_sweep/*_heatmap.png` — cross-experiment
  freeze or selector-limit plots. With defaults, these are standardized
  `effect_<metric>_heatmap.png` files on a fixed `[-1, 1]` color scale.
- `compare/midlevel_shape_<comparison>_sweep/view_type_eta_squared_*`,
  `view_type_point_biserial_*`, and `view_type_raw_*` — cross-experiment
  association strength, signed view-type direction, and raw view-type means for
  the three primary mid-level features. These new association outputs are
  intentionally restricted to the `no_freeze` experiment.
- `compare/midlevel_shape_<comparison>_sweep/selected_axis_visibility_curves.png`,
  `selected_symmetry_part_organization_curves.png`, and
  `selected_edge_organization_curves.png` — cross-experiment grouped curve
  figures.
- `compare/midlevel_shape_<comparison>_sweep/lift_axis_visibility_curves.png`,
  `lift_symmetry_part_organization_curves.png`, and
  `lift_edge_organization_curves.png` — the same grouped comparisons after
  subtracting the instance-matched all-view baseline.
- `compare/midlevel_shape_<comparison>_sweep/effect_axis_visibility_curves.png`,
  `effect_symmetry_part_organization_curves.png`, and
  `effect_edge_organization_curves.png` — the same grouped comparisons in
  standardized effect-size units.

With the wrapper defaults, one comparison folder is the freeze sweep
(`no_freeze` vs `freeze_10` through `freeze_50`) and the other is the
selector-limit sweep (`select_all` vs limited selector families).

### Selector score versus projected-shape structure

`selector_score_analysis.py` replays exact test rollouts with matching
`model_e<E>.pth` checkpoints and relates every valid candidate's DQN action
value to ellipse aspect ratio, bilateral symmetry, medial-axis symmetry, and
edge entropy. The combined heatmaps include all four metrics, while separate
axis-visibility, symmetry, and edge-organization heatmaps provide focused views.
Timestamped runs are aggregated only when their complete untimestamped
experiment name matches, so freeze and `selview_*` settings are never mixed.

```bash
# Default no-freeze experiment, epochs 10,20,...,100, physical GPU 0.
GPU_ID=0 ./run_selector_score_analysis.sh

# One exact selector-limit setting; all matching timestamped runs are pooled.
EXPERIMENTS="resnet18steps5_selview_remainder_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100" \
GPU_ID=1 ./run_selector_score_analysis.sh
```

The primary figure is `selector_score_correlation_heatmap.png`: positive cells
mean candidates with more of a cue receive higher selector scores within the
same decision. `selector_choice_percentile_heatmap.png` shows whether the
greedy choice lies above or below the candidate median for each cue. Raw
reconstructed `[rollout, step, view]` score tensors are retained for audit.
Definitions, commands, outputs, and caveats are in
[`README_selector_score_analysis.md`](README_selector_score_analysis.md).

### Selected-view contribution to multiview recognition

`view_contribution_analysis.py` tests whether the mid-level structure visible
in one selected view predicts how much that view contributes to the multiview
classification. For every agent-selected set `S` and selected view `vi`, it
computes:

```text
delta_M(vi) = M(S) - M(S without vi)

delta_replace_M(vi, u) = M(S) - M((S without vi) union {u})
```

The exact selected sets come from `selected_mask` in `feature_<epoch>.npz`;
the flattened selection JSON does not preserve individual rollout membership.
The two independently reported objectives are the correct-class logit and
top-1 minus top-2 prediction margin. Positive delta means that removing the
view weakened that objective. By default, `S` contains the agent actions only;
`CONTEXT=with_initial` retains the rollout's initial input while each agent
view is removed. For replacement, `u` is sampled from views outside the
complete saved selection mask, so neither another selected view nor the initial
input is used as the substitute.

The script correlates each delta with `ellipse_aspect_ratio`,
`bilateral_symmetry`, and `edge_entropy`. Replacement reports an absolute
selected-cue analysis and a separate change-versus-change analysis using
`cue(selected) - cue(replacement)`. Exact view family is categorical, so it is
analyzed with per-family mean contributions and eta-squared rather than an
arbitrary numeric family encoding. Detailed definitions, outputs, and caveats
are in
[`README_view_contribution.md`](README_view_contribution.md).

```bash
# One experiment; default exact epochs are 10,20,...,100.
python3 view_contribution_analysis.py \
  --selection_dir meta_logs/rgb/<experiment> \
  --non_roll \
  --epoch_stride 10

# Run only the no-freeze experiment on physical GPU 2.
GPU_ID=2 ./run_view_contribution_pipeline.sh

# Use every initial camera (the wrapper defaults to 10 evenly spaced cameras).
NUM_INITIAL_CAMS=0 GPU_ID=2 ./run_view_contribution_pipeline.sh

# Draw ten deterministic unselected replacements per rollout.
REPLACEMENT_SAMPLES=10 REPLACEMENT_SEED=42 GPU_ID=2 \
  ./run_view_contribution_pipeline.sh

# Use model_e<t>.pth instead of holding the final classifier fixed.
PER_EPOCH_CHECKPOINT=1 GPU_ID=2 \
  ./run_view_contribution_pipeline.sh
```

The wrapper targets only the no-freeze experiment; override
`EXPERIMENT=<folder>` only when its folder name differs. Outputs use a tagged
subfolder such as
`<selection_dir>/view_contribution/agent_only_final_classifier_cams_10_repl5_seed42/`
so different replacement settings do not overwrite each other.

### Single- versus five-view mid-level cue importance

`single_multiview_midlevel_analysis.py` compares the no-freeze final classifier
on every single candidate view, random five-view sets, and exact agent-selected
five-view sets from epoch 100. Single-view outputs are centered by the same
object's all-view mean; five-view outputs are centered by the same object's
random-five mean. This makes object-balanced correlations and grouped
cross-validated explained variance comparable across input sizes without
equating their raw logit scales. Matching timestamped training runs are
averaged within each object, so repeated runs do not multiply the effective
object count.

```bash
GPU_ID=0 ./run_single_multiview_midlevel_analysis.sh

# Smoke test.
LIMIT_INSTANCES=10 RANDOM_SETS=20 GPU_ID=0 \
  ./run_single_multiview_midlevel_analysis.sh

# Re-render existing CSV results only and refresh all plot labels.
PLOT_ONLY=1 ./run_single_multiview_midlevel_analysis.sh

# Compute only the raw five-family cue means; skip classifier evaluation.
FAMILY_MEANS_ONLY=1 ./run_single_multiview_midlevel_analysis.sh
```

The analysis also measures the held-out R-squared gained by adding exact view
family composition after ellipse aspect ratio, bilateral symmetry, and edge
entropy. `all_candidate_view_family_midlevel_means.png` provides the simpler
raw comparison: each panel averages one cue within exact view family and object,
then reports the mean and SEM across test objects. Details and output
definitions are in
[`README_single_multiview_midlevel.md`](README_single_multiview_midlevel.md).

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
and picks the most-recent one. Pass `--checkpoint` explicitly when you have
multiple runs of the same experiment and want a specific one; the script prints
the full list of candidates when more than one matches.

**Important checkpoint note.** Older training code did not always write a
`model.pth` for selector runs. Current training writes `logdir/model.pth` every
epoch, so the file contains the last-epoch weights after completion. Experiments
trained before that fix may need to be rerun before checkpoint-based analyses.

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
ARCH=tinyvit COMPARISON_SET=freeze ./run_aggregate_temporal_tests.sh
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

This probe samples one random view per instance per view-type bucket, repeated
N times, and classifies each image with **CLIP**. CLIP was trained on web
image-text pairs and has never seen ModelNet, so the result is a genuinely
zero-shot reading of view informativeness. Classification is done by encoding
the image with CLIP's image tower, encoding per-class text prompts with CLIP's
text tower, and taking cosine similarity. Top-1 and top-5 accuracy are both
reported.

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
- If `bar_top1.png` orders view types as Expanded > Expanded-like > Remainder >
  Foreshortened > Foreshortened-like (or similar), that supports the claim that
  expanded-family views are intrinsically more informative for generic visual
  classification.
- Absolute accuracy: don't expect supervised ModelNet-level numbers. CLIP zero-shot on
  ModelNet renders is typically 30–50% top-1, 60–80% top-5 — the relative
  ordering matters more than the magnitude.

Prerequisites:
- `transformers` (already in your conda env if you've been using VGGT/Pi3).
- ~$\sim$2GB GPU memory for ViT-B/32; ~6GB for ViT-L/14.
- ~5–10 min on one GPU for a default run (~4000 CLIP forwards in batches).

### Multiple `--freeze_epoch` runs

`logs/` and `meta_logs/` directories already include a `freeze_<N>_` prefix
in the experiment folder name when `--freeze_epoch != 100`, so per-run
training artifacts are segregated automatically.

For `aggregate_meta_json.py`: no changes needed — plots are written inside
each `meta_logs/<rep>/freeze_<N>_<arch>...` folder, which already segregates
them by the dir name.

For the `human_multiview-main/` probes (agent vs random, single-view
confidence, pair-confidence control): the `--selection_dir` you pass picks
which agent run to evaluate, but the default `--output_dir` only encodes
`--selected_view_type`. If you run probes on multiple agents with different
`--freeze_epoch`, pass `--output_dir` explicitly per run so the outputs don't
collide, e.g.:

```bash
./scripts/run_pipeline.sh OUTPUT_DIR=results/views/agent_default
./scripts/run_pipeline.sh OUTPUT_DIR=results/views/agent_freeze80
```

(All three `human_multiview-main/scripts/run_*.sh` wrappers accept
`OUTPUT_DIR=...` as an env-var override.)

### Recommended pipeline

1. **Train MVSelect agents.** Example:
   `python main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 --dataset rgb`.
2. **Aggregate training logs.** `python aggregate_meta_json.py` for the
   family-ratio / lift / per-class-accuracy plots above.
3. **Probe view-type informativeness.** `python clip_zero_shot_view_type.py`
   for CLIP top-1/top-5 per-bucket accuracy.
4. **Probe model uncertainty on agent selections.** In `human_multiview-main/`:
   `./scripts/run_pipeline.sh` for the VGGT/DINOv2 agent-vs-random
   comparison, `./scripts/run_view_type_single_view.sh` for VGGT's
   single-view depth confidence per bucket, `./scripts/run_pair_confidence_control.sh`
   for the identical / random / bucket-pair sanity check.

See `human_multiview-main/README.md` for the probe scripts that consume the
`selection.json` files produced by step 2.


## Paper verification update — 13 September 2026

See [the current paper action guide](README_paper_actions.md) for the exact E01 training,
E03 equal-image-count evaluation and E12 seven-checkpoint extension commands,
and the files to return. E04 is local analysis; no author command is needed.
GPU troubleshooting and CLIP inference are complete. Older runbooks are superseded. New training defaults to `--td_target_mode legal`
and uses `tdlegal` in output names. Use `--td_target_mode legacy` only for the
explicit comparison with the historical target. Existing checkpoints are unchanged.
