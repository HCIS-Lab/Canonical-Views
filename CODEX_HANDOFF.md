# Coding and Experiment Handoff

Last updated: 2026-09-10

This document transfers the coding and experiment state of the project. Paper
text, argument structure, and current manuscript wording are being transferred
separately through ChatGPT and should be treated as the authoritative source for
paper writing. Use this document for code behavior, experimental protocols,
commands, outputs, unresolved implementation work, and reproducibility.

## First Actions

1. Work from the live primary checkout unless the user explicitly selects a
   different checkout:

   ```text
   /Users/hankkung/Documents/Documents - Hank’s MacBook Pro/GitHub/Cognitive-Inspired-View-Selection
   ```

2. On the compute server, the project is normally at:

   ```text
   /nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection
   ```

3. Read this file, `MVSelect-main/README.md`, and the analysis-specific README
   for the task before editing.
4. Run `git status --short` before doing anything destructive.
5. Do not reset, clean, checkout over, or revert the current working tree. It
   contains a large amount of uncommitted implementation work from this project.
6. The paper-writing handoff from ChatGPT takes precedence over figure-caption
   prose in this repository if the two disagree.

## Critical Repository State

The live primary checkout is on branch `main` at commit `1c67c38` and matched
`origin/main` at the time of this handoff. Before adding the handoff files, it
already had 29 modified tracked files and 24 untracked files. The diff was about
1,852 insertions and 289 deletions. Most of the extensions described below are
therefore not safely recoverable from `origin/main` yet.

Do not assume a clean clone contains this work. Review, test, commit, and push it
before changing machines or deleting local state.

The old Codex worktree is:

```text
/Users/hankkung/.codex/worktrees/f62e/Cognitive-Inspired-View-Selection
```

Its `.git` file still points to the obsolete path
`/Users/hankkung/Documents/GitHub/Cognitive-Inspired-View-Selection/...`, so
ordinary Git commands in that worktree fail. The live primary checkout above is
the source of truth. Source files in the two trees were identical at handoff
except for local-only files and the active-single accuracy-label update, which
has now also been applied to the primary checkout.

### Uncommitted tracked files at handoff

```text
MVSelect-main/README.md
MVSelect-main/README_midlevel_shape_features.md
MVSelect-main/aggregate_downstream.py
MVSelect-main/aggregate_meta_json.py
MVSelect-main/aggregate_midlevel_shape_features.py
MVSelect-main/compute_cluster_metrics.py
MVSelect-main/main.py
MVSelect-main/midlevel_shape_features.py
MVSelect-main/mochi_zero_shot_feature_oddity.py
MVSelect-main/pca_tsne.py
MVSelect-main/requirements.txt
MVSelect-main/run_aggregate_temporal_tests.sh
MVSelect-main/run_cluster_metrics_pipeline.sh
MVSelect-main/run_main.sh
MVSelect-main/run_midlevel_shape_features_pipeline.sh
MVSelect-main/run_mochi_zero_shot_feature_oddity.sh
MVSelect-main/run_temporal_test_all.sh
MVSelect-main/src/datasets/__init__.py
MVSelect-main/src/models/mvcnn.py
MVSelect-main/src/models/mvselect.py
MVSelect-main/src/trainer_mvcnn.py
MVSelect-main/temporal_selection_test.py
MVSelect-main/train_downstream.py
MVSelect-main/zero_shot_view_type_test.py
human_multiview-main/README.md
human_multiview-main/scripts/run_aggregate_vggt_confidence.sh
human_multiview-main/scripts/run_evaluation_views.py
human_multiview-main/scripts/run_pipeline.sh
human_multiview-main/scripts/run_views_pipeline_sweep.sh
```

### Important untracked source and documentation

```text
MVSelect-main/README_policy_replay.md
MVSelect-main/README_selector_score_analysis.md
MVSelect-main/README_single_multiview_midlevel.md
MVSelect-main/README_view_contribution.md
MVSelect-main/aggregate_policy_replay.py
MVSelect-main/compare_active_single_view_bias.py
MVSelect-main/evaluate_joint_policy_reference.py
MVSelect-main/experiment_arch.sh
MVSelect-main/plot_midlevel_feature_space_3d.py
MVSelect-main/run_active_single_view_comparison.sh
MVSelect-main/run_active_single_view_experiment.sh
MVSelect-main/run_midlevel_feature_space_3d.sh
MVSelect-main/run_policy_replay_experiment.sh
MVSelect-main/run_selector_score_analysis.sh
MVSelect-main/run_single_multiview_midlevel_analysis.sh
MVSelect-main/run_view_contribution_pipeline.sh
MVSelect-main/selector_score_analysis.py
MVSelect-main/single_multiview_midlevel_analysis.py
MVSelect-main/src/datasets/policy_replay_dataset.py
MVSelect-main/src/models/architectures.py
MVSelect-main/train_policy_replay.py
MVSelect-main/view_contribution_analysis.py
MVSelect-main/visualize_midlevel_extractions.py
PAPER_FIGURE_CAPTIONS.md
```

Re-run `git status --short` for the exact current list because this handoff adds
two more untracked Markdown files.

## Project Layout

### `MVSelect-main`

Training, selection logging, recognition evaluation, temporal tests,
representation analysis, mid-level image descriptors, selector-score analysis,
view contribution and replacement, policy replay, CLIP probing, and plotting.

### `human_multiview-main`

The Bonnen et al. human-multiview code plus extensions that evaluate VGGT,
DINOv2, Pi3, DUSt3R, and MASt3R on MVSelect-selected views. It contains selected
versus random view-set confidence, single-view confidence, pair controls, epoch
aggregation, and experiment-comparison plots.

### Repository root

Older data-generation and rendering utilities. `PAPER_FIGURE_CAPTIONS.md`
contains generated caption drafts and plot descriptions, but manuscript wording
must follow the separate ChatGPT handoff.

## Compute Environment and Data

Typical server environments observed during development:

- `hank3d`: MVSelect training and most PyTorch evaluation.
- `rapids`: PCA/t-SNE and some cluster analysis runs.
- `human_multiview`: VGGT and human-multiview probes.

Verify environments rather than assuming they are reproducible from the
requirements files. `MVSelect-main/requirements.txt` currently contains old and
duplicate pins, including two Pillow versions and both OpenCV packages.

Primary rendered dataset:

```text
/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23
```

Important server directories:

```text
MVSelect-main/logs/rgb/       timestamped checkpoints and run logs
MVSelect-main/meta_logs/rgb/  selections, metadata, feature dumps, analyses
MVSelect-main/compare/        cross-experiment figures and CSVs
MVSelect-main/cache/          per-view mid-level feature caches
human_multiview-main/results/ per-experiment confidence outputs
human_multiview-main/compare/ cross-experiment confidence figures
```

These generated directories are ignored or only partially tracked. Do not
expect them in a fresh Git clone. Never commit large checkpoints or datasets.

## Core Experiment Semantics

### Standard active multiview selection

With `--steps K`, the selector observes an initial random camera and chooses K
additional actions. In the ordinary model, the recognition network aggregates
the initial view and selected views. Be explicit about whether an analysis uses
agent actions only or the classifier's full input.

### Active single-view control

`--active_single_view` is valid only with `--steps 1`.

- The selector observes a random initial view.
- It chooses one action-conditioned view.
- The classifier receives only the selected view, so recognition input is N=1.
- The initial image is a scout and an action-independent reward baseline.
- Reward is `loss(initial) - loss(selected)`.
- Initial-view loss is detached and does not update the classifier.
- Training evaluates all possible initial cameras at test time and pools them.
- Paths include `steps1_active_single_`, avoiding collisions with ordinary
  `steps1` experiments.
- New feature dumps mark only the selected recognition view in `selected_mask`;
  the scout is stored separately as `selected_init_cam`.

This is an active N=1 recognition control, not a one-capture sensing system. An
image-conditioned policy must inspect something before choosing a camera.

### Selector candidate limits

`--selector_view_limit` supports:

```text
all
expanded_family
foreshortened_family
foreshortened_family_remainder
remainder
```

The restriction applies only when the learned selector is allowed to choose.
Random, fixed-view-family, and all-view test baselines are not restricted. The
experiment name includes `selview_<limit>_`, so logs and plots do not collide.

### Architectures

`--arch` supports:

```text
resnet18
vit
tinyvit
```

`tinyvit` uses the timm TinyViT-5M/224 backbone. Architecture-aware wrappers
source `MVSelect-main/experiment_arch.sh`; non-ResNet comparison output names
receive an architecture suffix.

### Joint initialization and checkpoints

`--skip_stage1` bypasses the older recognition-checkpoint initialization and
jointly trains classifier and selector from an ImageNet-pretrained backbone.
This is the protocol used by the newer active-single launcher.

`--save_every_epoch N` writes `model_e<E>.pth` at the requested cadence and
`model.pth` at completion. Many old freeze and no-freeze runs have only the final
checkpoint; do not claim classifier-at-time-t analysis unless matching
`model_e<E>.pth` files exist.

The user previously requested avoiding internal audience phrases such as
"advisor-facing summary" in public README files. Use general public wording.
The user also asked to stop calling expanded views "canonical". Use:

- expanded family = Expanded + Expanded-like
- foreshortened family = Foreshortened + Foreshortened-like
- remainder

## Experiment Naming

Shared names are defined in `MVSelect-main/experiment_arch.sh`.

For ResNet-18 defaults:

```text
Base/no-freeze:
resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100

Active pair:
resnet18steps1_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100

Active single:
resnet18steps1_active_single_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100
```

Freeze runs prepend `freeze_10_`, `freeze_20_`, ..., `freeze_50_` to the base
name. Selector-limit runs insert `selview_<limit>_` after `steps5_`.

Timestamped checkpoint directories under `logs/rgb/` append a value such as
`_2026-06-23_12-47-59`. Match the exact experiment prefix and architecture;
never pool a no-freeze checkpoint with a selector-limit or freeze run.

## View-Type Definitions

Dataset view type is encoded in the rendered filename and follows the original
MVSelect substring precedence:

- Expanded: filename contains `planar` and not `short`.
- Foreshortened: contains `short` and not `like`.
- Expanded-like: contains `like` and not `short`.
- Foreshortened-like: contains both `like` and `short`.
- Remainder: none of the above.

Availability priors currently hardcoded for the rendered dataset:

```text
Expanded               3.5%
Expanded-like         14.0%
Foreshortened          1.7%
Foreshortened-like     7.0%
Remainder             73.8%
```

Family priors are 17.5% expanded, 8.7% foreshortened, and 73.8% remainder.
If the dataset is regenerated, recompute these values instead of reusing them.

## Main Analysis Map

### Standard training diagnostics

`aggregate_meta_json.py` walks every experiment under `meta_logs` and writes
plots inside each experiment folder:

- `aggregated_view_ratios.png`: exact five-type selection shares, SEM, and
  availability baselines.
- `aggregated_view_family_ratios.png`: expanded family, foreshortened family,
  and remainder.
- `aggregated_view_family_lift.png`: observed share divided by availability.
- `aggregated_accuracy.png`: selected, all-view, random, and one-view-family
  test performance.
- `aggregated_accuracy_3.png` and `_5.png`: auxiliary fixed-family tests.
- `aggregated_per_class_accuracy.png`: class-by-epoch heatmap.

The active-single metadata stores `recognition_num_views=1`. The aggregator now
uses this metadata rather than hardcoding selected N=5, so active-single legends
show `Self-generated (N=1)` and `Random (N=1)`. In active-single selection-ratio
figures, only the chosen action is counted, not the scout view.

Command:

```bash
cd /nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/MVSelect-main
python3 aggregate_meta_json.py
```

### Active-single preference summary

`compare_active_single_view_bias.py` aggregates active-single seeds and plots
exact types, families, total-variation distance from availability, and accuracy.

```bash
./run_active_single_view_comparison.sh

# Custom experiment path:
SINGLE_DIR=/absolute/path/to/meta_logs/rgb/<active-single-exp> \
  ./run_active_single_view_comparison.sh
```

Default output: `compare/active_single_view_bias/`.

### Temporal manipulation and prediction margin

`temporal_selection_test.py` replays selections from epochs through a
classifier, measuring rotation, color jitter, combined perturbation, and top-1
minus top-2 logit margin. The margin unit is raw logit difference and includes
all predictions, not only correct predictions.

- Default protocol holds the final classifier fixed while changing selected
  views across epochs.
- `--per_epoch_checkpoint` uses classifier-at-t only when snapshots exist.
- `run_temporal_test_all.sh` computes per-experiment CSVs.
- `run_aggregate_temporal_tests.sh` compares either `freeze` or
  `selector_limit`; default plot style is heatmap with 10-epoch bins.

```bash
GPUS=4 COMPARISON_SET=freeze ./run_temporal_test_all.sh
COMPARISON_SET=freeze STYLE=heatmap BIN_EPOCHS=10 \
  ./run_aggregate_temporal_tests.sh

GPUS=4 COMPARISON_SET=selector_limit ./run_temporal_test_all.sh
COMPARISON_SET=selector_limit STYLE=heatmap BIN_EPOCHS=10 \
  ./run_aggregate_temporal_tests.sh
```

Important: this implementation merges selected filenames per instance from
`selection.json`. It is not a faithful active-single rollout analysis because
that would turn many N=1 rollouts into an artificial multiview set.

### Feature clusters

`pca_tsne.py` provides qualitative all-candidate per-view feature embeddings.
`compute_cluster_metrics.py` adds quantitative cosine-silhouette metrics:

- `silhouette_class`: clusters labeled by object class.
- `silhouette_view`: clusters labeled by five view types.
- `silhouette_view_index`: clusters labeled by camera index 0 through 113.
- `separability = silhouette_class - silhouette_view`.

Each silhouette score is computed from the same L2-normalized features using
different labels. The aggregate pipeline defaults to at most 2,000 samples
because silhouette distance computation is quadratic.

Only feature-based analyses should apply `EVERY_N_EPOCHS=10`; selection-based
analyses have complete epoch histories and must not be thinned at data loading.

```bash
COMPARISON_SET=freeze EVERY_N_EPOCHS=10 \
  ./run_cluster_metrics_pipeline.sh
COMPARISON_SET=selector_limit EVERY_N_EPOCHS=10 \
  ./run_cluster_metrics_pipeline.sh
```

Active-single caveat: the exact saved classifier-input feature is N=1 and is
numerically valid, but the legacy column name
`silhouette_class_selected_with_init` is misleading because the scout is not in
the classifier input. Prefer adding a generic
`silhouette_class_classifier_input` name before publishing active-single cluster
results. The `silhouette_class_selected` reconstruction from selection JSON is
not an exact active-single rollout representation.

### Mid-level visible-shape descriptors

Primary descriptors currently emphasized:

- Ellipse aspect ratio: axis visibility.
- Bilateral symmetry: silhouette organization.
- Medial-axis symmetry: skeleton-based symmetry proxy.
- Edge entropy: edge orientation diversity.

Implementation details and caveats are in
`MVSelect-main/README_midlevel_shape_features.md`.

- Object masks are thresholded from rendered images.
- Skeletons use binary thinning via Zhang-Suen skeletonization.
- Medial-axis symmetry mirrors the skeleton and computes overlap.
- Edges use Sobel gradients on grayscale renders.
- Bilateral symmetry computes left-right and up-down mirror IoU and uses the
  maximum, making it orientation-tolerant rather than strictly vertical-axis
  bilateral symmetry.
- Descriptor ranges may be narrow because controlled renders and silhouette
  preprocessing remove substantial appearance variation. Validate distributions
  and extraction overlays before interpreting small effects.

```bash
python3 midlevel_shape_features.py \
  --selection_dir meta_logs/rgb/<experiment> \
  --data_root /nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23

COMPARISON_SET=freeze ./run_midlevel_shape_features_pipeline.sh
COMPARISON_SET=selector_limit ./run_midlevel_shape_features_pipeline.sh
```

The direct Python analysis is compatible with active-single selections. The
comparison wrapper currently only knows `freeze` and `selector_limit`; add an
`active_single` comparison set rather than pretending it is part of steps5.

### Selector score versus visible-shape cues

Raw selector scores were not logged during the original runs. They are replayed
from every available checkpoint by `selector_score_analysis.py`. The main
statistics are within-rollout Spearman correlation between candidate score and
cue, plus selected-action score percentile. Small correlation and a high choice
percentile can coexist: the policy can rank its chosen action highly while the
tested cue explains little of the ranking over all 114 candidates.

```bash
GPU_ID=0 ./run_selector_score_analysis.sh

EXPERIMENTS="<exact-experiment-name>" EPOCHS="10 20 30 40 50 60 70 80 90 100" \
  GPU_ID=0 ./run_selector_score_analysis.sh
```

Active-single caveat: score computation itself is meaningful, but replay
validation currently reconstructs an initial-plus-action mask and compares it
with the saved action-only active-single `selected_mask`. Update validation to
read `recognition_input` and expect an action-only mask for active-single before
using this pipeline for active-single claims.

### View contribution and replacement

`view_contribution_analysis.py` measures correct-class logit and prediction
margin changes when selected views are removed or replaced. Negative
cross-entropy was removed from figures. Replacement cue correlations use:

```text
x(selected) - x(replacement)
```

against:

```text
M(selected set) - M(set with replacement)
```

Positive Spearman correlation means larger cue advantage tends to accompany a
larger performance advantage; it is not a regression slope and does not mean a
one-unit cue change causes that many logit units.

```bash
GPU_ID=0 EXPERIMENT=<base-no-freeze-experiment> \
  ./run_view_contribution_pipeline.sh
```

Leave-one-out is undefined for active-single because deleting its only
recognition view creates N=0. A selected-to-unselected N=1 replacement analysis
is meaningful, but the current pipeline needs an explicit N=1 replacement-only
mode before use.

### Single- versus multiview mid-level cue analysis

`single_multiview_midlevel_analysis.py` compares random N=1 view families,
random N=5 sets, and selected N=5 sets. It computes object-balanced correlations
and grouped cross-validated R-squared.

```bash
GPU_ID=0 ./run_single_multiview_midlevel_analysis.sh
PLOT_ONLY=1 ./run_single_multiview_midlevel_analysis.sh
FAMILY_MEANS_ONLY=1 ./run_single_multiview_midlevel_analysis.sh
```

This script explicitly requires selected N=5 and is not an active-single
analysis. Do not use its selected5 results for an active-single experiment.

### Three-dimensional mid-level feature space

`plot_midlevel_feature_space_3d.py` plots every candidate view with x, y, z set
to ellipse aspect ratio, bilateral symmetry, and edge entropy. It can highlight
exact selected views and produce primary, opposite, rear, and combined angles.

```bash
./run_midlevel_feature_space_3d.sh
CLASS_NAME=airplane ./run_midlevel_feature_space_3d.sh
CLASS_NAME=airplane PER_INSTANCE=1 PER_INSTANCE_COUNT=5 \
  ./run_midlevel_feature_space_3d.sh
ONE_INSTANCE_PER_CLASS=1 ./run_midlevel_feature_space_3d.sh
```

For active-single, override both the selection path and expected selection
count:

```bash
SELECTION_DIR=meta_logs/rgb/<active-single-experiment> \
EXPECTED_SELECTED_VIEWS=1 \
  ./run_midlevel_feature_space_3d.sh
```

The cache can contain a small number of non-finite descriptors. The wrapper
defaults to `ALLOW_MISSING=1` and records excluded views rather than failing.

### Controlled policy replay

`train_policy_replay.py` trains fresh recognizers from matched initialization
and update budgets under replayed selection histories. Conditions include:

- evolving no-freeze policy;
- selection at epoch 10, 20, or 30 fixed from the beginning;
- final selection fixed from the beginning;
- temporally shuffled evolving policy;
- random selection;
- random views for the first 10, 20, or 30 epochs followed by the evolving
  original policy;
- optional family-matched random control;
- optional jointly trained no-freeze reference.

Use labels such as `Selection at e30 fixed from start`, not `Frozen policy e30`;
the selection comes from epoch 30 of the no-freeze run, not from the separate
freeze_30 experiment.

```bash
GPUS=4 GPU_IDS="0 1 2 3" ./run_policy_replay_experiment.sh
```

See `MVSelect-main/README_policy_replay.md` for definitions and outputs. This
pipeline defaults to N=5 and is separate from active-single.

### CLIP zero-shot single-view view-type probe

`clip_zero_shot_view_type.py` samples views uniformly from all files in each
filename-defined bucket. It is independent of agent selections and evaluates
top-1/top-5 zero-shot accuracy against text labels.

```bash
python3 clip_zero_shot_view_type.py
```

It requests safetensor weights explicitly to avoid the PyTorch `<2.6`
`torch.load` CVE restriction in recent Transformers.

### MOCHI-style feature oddity

`mochi_zero_shot_feature_oddity.py` performs a pairwise feature-similarity
oddity test on MVSelect backbone features. DINO-style models use feature
similarity, whereas VGGT uses confidence in the human-multiview pipeline.

```bash
GPU_ID=0 COMPARISON_SET=freeze ./run_mochi_zero_shot_feature_oddity.sh
GPU_ID=0 COMPARISON_SET=selector_limit ./run_mochi_zero_shot_feature_oddity.sh
```

The core script can load an active-single checkpoint because it calls the
backbone feature extractor, but the wrapper has no active-single comparison set.

## Human-Multiview Confidence Analysis

The selected-view pipeline uses multiple `selection.json` files, bins epochs
into 1-10, 11-20, ..., 91-100, counts selection frequencies, and picks top-K
agent-selected views per instance for each bin. It intentionally excludes the
initial scout unless the analysis explicitly requests otherwise.

Hardcoded/common data roots are the NFS dataset above and matching
`MVSelect-main/meta_logs/rgb/<experiment>` directories.

### Per-experiment evaluation, then aggregation

The aggregate VGGT command cannot work until each experiment has an
`overall_summary.csv`. Run the sweep first:

```bash
cd /nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/human_multiview-main

GPUS=4 COMPARISON_SET=freeze ./scripts/run_views_pipeline_sweep.sh
COMPARISON_SET=freeze STYLE=heatmap \
  ./scripts/run_aggregate_vggt_confidence.sh

GPUS=4 COMPARISON_SET=selector_limit \
  ./scripts/run_views_pipeline_sweep.sh
COMPARISON_SET=selector_limit STYLE=heatmap \
  ./scripts/run_aggregate_vggt_confidence.sh
```

Pair-level confidence processes every pair separately and averages pair scores.
Set-level or joint confidence processes the full selected set in one VGGT call
and averages per-input-image confidence maps. Neither metric is an attention
score. VGGT confidence is directly predicted per pixel by its confidence head,
then averaged over object masks.

Single-view confidence is not depth-map entropy. It is the same learned
per-pixel confidence head run with N=1. Absolute values are not directly
comparable to pair or joint values because N=1 lacks cross-view context and is
not the primary calibrated regime. Within-N=1 comparisons across view types are
the defensible use.

```bash
GPUS=4 MODELS="vggt" N_RUNS=5 \
  ./scripts/run_view_type_single_view.sh

GPUS=4 MODELS="vggt" N_RUNS=5 \
  ./scripts/run_pair_confidence_control.sh
```

For the pair control, same-bucket pairs are two random views from that bucket,
not identical images. The identical condition explicitly duplicates the same
image.

Active-single caveat: pair metrics require at least two views and are therefore
undefined for the active-single recognition input. Setting `NUM_CAM=1` in the
selected-view sweep would choose the most frequent action over an epoch range,
not preserve each exact active-single rollout. Do not present that as an exact
active-single pair analysis.

## Active-Single Compatibility Matrix

### Valid now

- `aggregate_meta_json.py` selection ratios, family ratios, lift, N=1 accuracy,
  and per-class accuracy.
- `compare_active_single_view_bias.py`.
- Direct `midlevel_shape_features.py` selected-view descriptors.
- All-candidate per-view PCA/t-SNE and per-view cluster metrics when feature
  dumps exist.
- Core MOCHI feature oddity from an active-single checkpoint.
- CLIP single-view probe, though it is independent of MVSelect policy.
- 3D mid-level visualization with `EXPECTED_SELECTED_VIEWS=1`.

### Requires implementation changes

- Selector-score replay validation must understand action-only recognition
  masks.
- Cluster metric output should use a generic classifier-input label instead of
  `selected_with_init`.
- Mid-level, cluster, MOCHI, and 3D wrappers should gain explicit active-single
  presets rather than requiring path overrides.
- View contribution needs replacement-only N=1 support.
- An exact active-single temporal perturbation analysis must replay individual
  rollouts or consume exact NPZ rollout arrays, not merge selection JSON by
  instance.

### Scientifically invalid without redefining the question

- Leave-one-view-out contribution for N=1, because removal produces N=0.
- VGGT or DINO pair confidence/similarity on an N=1 recognition input.
- Existing selected-N=5 single/multiview analysis applied to active-single.
- Existing N=5 policy-replay figures called active-single evidence.

## Known Failure Modes and Resolved Bugs

- Python 3.11 `random.Random()` does not accept tuple seeds. Single-view scripts
  now join seed fields into a deterministic string.
- New Transformers may reject `.bin` CLIP weights with PyTorch `<2.6`; use
  safetensors.
- Cross-experiment specifications may use `PATH:LABEL`. Parsing was fixed in
  aggregators to avoid treating the label as part of an absolute path.
- Temporal tests originally looked for `logs/rgb/resnet18_performance.txt`,
  which is not a final selector/classifier checkpoint. They now search matching
  timestamped experiment logdirs or accept `--checkpoint`.
- `model.pth` is now saved for selector training as well as the old task-only
  mode. Historical runs may still lack it.
- Freeze-aware and selector-limit-aware names are essential. Do not merge runs
  by architecture alone.
- `run_aggregate_vggt_confidence.sh` only aggregates existing summaries; it
  cannot generate missing summaries. Run `run_views_pipeline_sweep.sh` first.
- Matplotlib colorbars hid small positive maxima behind round ticks. Cluster
  heatmaps now explicitly label true endpoints and zero where applicable.
- Rank-stacked plots preserve rank but make the stacked y-axis meaningless.
  Prefer heatmaps or sorted side-by-side bars for quantitative comparison.
- A negative silhouette-by-view-index value is plausible and means camera-index
  labels do not form compact clusters. It is not automatically a code error.
- PCA/t-SNE can look visually separated while global silhouette is low because
  t-SNE distorts distances and 85% classification accuracy does not require
  compact spherical clusters.

## Validation State

At handoff, syntax-only checks succeeded for 86 Python files and all shell
launchers in `MVSelect-main` and `human_multiview-main`. Full tests were not run
locally because the dataset, CUDA models, checkpoints, and project conda
environments live on the NFS compute servers.

Before publishing or launching a large sweep:

1. Parse-check changed Python and `bash -n` changed wrappers.
2. Run a small `LIMIT`, `LIMIT_TRIALS`, `RUN_LIMIT`, or short-epoch smoke test.
3. Inspect one CSV and one figure for units, input count, epoch count, and label
   correctness.
4. Verify the exact checkpoint prefix, architecture, selector limit, and freeze
   condition.
5. Verify whether a figure uses the final classifier or classifier-at-t.
6. Confirm whether the initial scout is included in the analyzed set.

## Collaboration Requirements

- Every time files are edited, summarize exactly which files changed.
- Use precise experimental names and state N explicitly.
- Do not call expanded or expanded-like views "canonical".
- Do not write insider phrases such as "advisor-facing" in public docs.
- Do not output metrics the user did not ask for.
- Keep freeze and selector-limit comparisons in separate figures.
- Keep ResNet, ViT, and TinyViT outputs in architecture-specific paths.
- Prefer heatmaps for many experiments and ten epoch bins for presentation.
- Ask whether an apparent relationship is correlation, effect, logit difference,
  accuracy percentage, or confidence before labeling an axis.
- Treat generated logs, checkpoints, and datasets as user data. Never delete or
  overwrite them while cleaning code.

## Recommended Next Coding Work

1. Preserve the current working tree in Git after review. This is the highest
   priority because the implementation is not on `origin/main`.
2. Finish active-single post-training compatibility:
   - selector-score replay validation;
   - generic classifier-input cluster naming;
   - N=1 replacement-only contribution;
   - explicit active-single wrapper presets.
3. Add focused tests for metadata naming, experiment-path resolution, exact
   selected-mask semantics, and checkpoint matching.
4. Replace hardcoded experiment lists with a small manifest format while
   preserving current command compatibility.
5. Reconcile or modernize environment specifications without breaking the
   working server environments.

