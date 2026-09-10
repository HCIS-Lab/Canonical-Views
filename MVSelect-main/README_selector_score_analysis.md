# Selector-Score Analysis

`selector_score_analysis.py` tests whether the learned selector increasingly
assigns higher action values to candidate views that expose particular
mid-level projected-shape properties. Unlike selected-versus-unselected or
leave-one-out analyses, it evaluates every valid candidate at the moment a
choice is made.

## Selector Output

MVSelect currently uses a DQN-style selector. For one rollout state, its
`value_head` returns one raw action value for each camera:

```text
action_value shape = [batch, number_of_views]
```

At test time the selector chooses the valid candidate with the largest action
value. These values are not probabilities. Applying softmax afterward would
create a normalized visualization, but it would not be the policy used by the
model and would introduce an arbitrary temperature. The analysis therefore
uses action-value ranks rather than calling them probabilities.

## Exact Replay

For every experiment setting, timestamped training runs are matched strictly:

```text
logs/rgb/<experiment>_<YYYY-MM-DD_HH-MM-SS>/model_e<E>.pth
meta_logs/rgb/<experiment>/<ISO timestamp>/feature_<E>.npz
```

The complete `<experiment>` string must match. A no-freeze run is therefore
not combined with a freeze run or any `selview_*` run. Multiple timestamped
runs with the same complete experiment name are treated as repeated runs and
aggregated afterward.

For each matched run, epoch, object, initial camera, and selection step, the
script:

1. Loads per-view features from `feature_<E>.npz` and selector parameters from
   `model_e<E>.pth`.
2. Reconstructs the current selected-view mask.
3. Reconstructs the valid candidate mask, including the experiment's
   `selector_view_limit` when present.
4. Computes the raw action value of every valid candidate and takes the greedy
   action used during testing.
5. Repeats for every selector step.
6. Confirms that the replayed final mask exactly equals the saved
   `selected_mask`. A mismatch stops the analysis by default.

Only epochs containing both a checkpoint and a feature dump can be analyzed.
The default `EPOCH_STRIDE=10` uses epochs 10, 20, ..., 100, matching the normal
feature-dump cadence.

## Statistics

Raw DQN score offsets and scales can differ across objects, steps, and epochs.
Pooling all raw scores into one correlation would therefore confound score
calibration with view preference. The primary statistic is instead computed
inside each individual selector decision:

```text
Spearman rho(candidate action values, candidate cue values)
```

Interpretation:

- Positive rho: candidates with a larger cue tend to receive higher selector
  values in that state.
- Negative rho: candidates with a larger cue tend to receive lower values.
- Rho near zero: no monotonic candidate ordering by that cue.

The script also records the chosen view's cue percentile among valid
candidates:

- `0.5`: the chosen view is at the candidate median for that cue.
- Above `0.5`: the choice favors larger cue values.
- Below `0.5`: the choice favors smaller cue values.
- `1.0`: the chosen view has the largest cue value, allowing for tied ranks.

Decision-level values are averaged within each timestamped run first. Figures
then show the mean across repeated runs; SEM in the summary CSV is also across
runs, not across correlated object/initial-camera decisions.

## Mid-Level Measures

The plots intentionally focus on four representative descriptors:

| Group | Measures |
|---|---|
| Axis visibility | Ellipse aspect ratio |
| Symmetry | Bilateral symmetry, medial-axis symmetry |
| Edge organization | Edge entropy |

The unsuffixed `selector_score_correlation_heatmap.png` and
`selector_choice_percentile_heatmap.png` combine all four metrics in one
overview. The category-suffixed figures isolate axis visibility, symmetry, and
edge organization. Other cached descriptors are not included in this analysis
or its plots.

## Commands

Default no-freeze experiment on physical GPU 0:

```bash
cd MVSelect-main
GPU_ID=0 ./run_selector_score_analysis.sh
```

Analyze all timestamped runs belonging to one exact selector-limit setting:

```bash
EXPERIMENTS="resnet18steps5_selview_remainder_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100" \
GPU_ID=1 ./run_selector_score_analysis.sh
```

Analyze several settings in one invocation. They are still written and
aggregated separately:

```bash
EXPERIMENTS="resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100 freeze_10_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100 resnet18steps5_selview_remainder_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100" \
GPU_ID=2 ./run_selector_score_analysis.sh
```

Smoke test one run, two epochs, and forty exact rollouts per epoch:

```bash
RUN_LIMIT=1 EPOCHS="10 100" LIMIT_TRIALS=40 GPU_ID=0 \
./run_selector_score_analysis.sh
```

Use all 114 initial-camera rollouts instead of ten evenly spaced cameras:

```bash
NUM_INITIAL_CAMS=0 GPU_ID=0 ./run_selector_score_analysis.sh
```

## Outputs

Each experiment is saved independently under:

```text
meta_logs/rgb/<experiment>/selector_score/
```

| Output | Contents |
|---|---|
| `run_manifest.csv` | Exact log run, metadata run, checkpoint, feature dump, selector limit, rollout count, and replay match rate. |
| `selector_score_decisions.csv` | One row per selector decision, with candidate count, chosen view, score margin, per-cue Spearman rho, and chosen-cue percentile. |
| `selector_score_run_summary.csv` | Decision statistics averaged within each timestamped run, separately for every step and across all steps. |
| `selector_score_epoch_summary.csv` | Mean and SEM across repeated timestamped runs. `step=0` denotes the all-step summary used by the main heatmaps. |
| `selector_score_correlation_heatmap.png` | Combined four-metric mean within-decision rank correlation between selector action value and cue over epochs. |
| `selector_choice_percentile_heatmap.png` | Combined four-metric chosen-view cue percentile over epochs. |
| `selector_score_correlation_axis_visibility_heatmap.png` | Ellipse-aspect-ratio score association. |
| `selector_score_correlation_symmetry_heatmap.png` | Bilateral- and medial-axis-symmetry score associations. |
| `selector_score_correlation_edge_organization_heatmap.png` | Edge-entropy score association. |
| `selector_choice_percentile_<group>_heatmap.png` | Corresponding category-specific chosen-cue percentile heatmaps. |
| `raw/<timestamp>/selector_scores_e<E>.npz` | Reconstructed raw `scores`, `candidate_mask`, `actions`, rollout identity, and replay validation for every candidate. |

## Caveats

1. Selector value is contextual. A camera's score depends on the object, the
   initial view, previously selected views, and selection step; it is not an
   intrinsic fixed score for that camera.
2. Selector-limit experiments are analyzed only inside their allowed candidate
   pool. They cannot reveal how a disallowed view would have ranked.
3. A positive association is descriptive evidence that the policy organizes
   choices around a cue. It does not establish that the cue caused classifier
   performance.
4. Several descriptors are correlated with each other. Interpret their
   separate heatmaps as marginal associations, not independent causal effects.
5. The descriptors come from rendered 2D silhouettes, skeleton proxies, and
   Sobel edges; they do not measure ShapeNet mesh-intrinsic geometry.
