# Selected-View Contribution and Replacement

This analysis connects image-derived mid-level cues to the recognition model's
multiview prediction. It asks whether a selected view that exposes a stronger
shape cue also makes a larger marginal contribution to the prediction made from
the selected set. It evaluates both deleting a selected view and replacing it
with a view the agent did not select.

## Definition

For one test object, let the agent-selected set be:

```text
S = {v1, v2, ..., vN}
```

`S` is reconstructed from `selected_mask` and `selected_init_cam` in the
matching `feature_<epoch>.npz` dump. These arrays preserve one exact rollout
for each `(test object, initial camera, epoch)`. The flattened
`*_selection.json` is used only to identify runs and epochs; it does not retain
the membership of an individual rollout.

By default, the initial input view is removed from the mask, so `S` contains
only the agent actions. With `--context with_initial`, the initial view remains
in the prediction context while the analysis still measures deletion of each
agent-selected view; the initial view itself is not assigned a delta.

For each selected view `vi`, the script evaluates the classifier on the full
set and on the set with that view removed:

```text
delta_M(vi) = M(S) - M(S without vi)
```

A positive delta means that removing `vi` weakened the measured prediction
objective. A negative delta means that the objective improved after removal.

The replacement intervention uses an unselected view `u`:

```text
delta_replace_M(vi, u) = M(S) - M((S without vi) union {u})
```

A positive replacement delta means that the selected view produced a stronger
objective than the unselected substitute. A negative value means the sampled
unselected view was better. Replacement candidates exclude every view in the
saved selected mask, including the initial input. The replacement therefore
never silently reintroduces the initial view when `--context agent_only` is
used.

By default, five unselected candidates are sampled without replacement for
each rollout. The same candidates are used for every selected view in that
rollout, reducing variation caused by comparing selected views against
different-quality reference sets. Sampling is deterministic under
`--replacement_seed`.

The selected view is not sampled: every agent-selected view is removed and
replaced in turn. For `replacement_view_family_<objective>_heatmap.png`, the
five replacement deltas are first averaged once per selected view, then those
per-view means are averaged by the selected view's exact family and selection
epoch. Replacement families are pooled in this figure; use
`replacement_family_pair_<objective>_heatmap.png` to separate each selected
family -> replacement-family pair. A `+0.5` correct-logit cell means that,
on average, the original selected view gave the ground-truth class a raw logit
0.5 higher than its sampled unselected substitutes. This is a logit difference,
not an accuracy percentage. The `n` column in
`replacement_view_family_contributions.csv` is the number of selected-view
means contributing to each family/epoch cell.

The script computes two objectives separately:

| Objective | Definition | Interpretation of positive delta |
|---|---|---|
| `correct_logit` | Raw logit `z_y` for the ground-truth class | The view raises evidence for the correct class. |
| `prediction_margin` | Largest logit minus second-largest logit | The view increases the model's confidence gap, whether the top class is correct or incorrect. |

The correct-class logit is target-aware. Prediction margin is not: a
confidently wrong prediction can have a large positive margin.
`full_correct` and `without_correct` are retained in the row-level CSV so this
case can be inspected directly.

Correlation heatmaps report unitless Spearman rank correlation by default.
Family heatmaps report mean intervention differences in raw logit or raw
logit-margin units. Their numeric magnitudes are therefore not directly
comparable.

## Required Saved Data

Each evaluated epoch needs a matching feature dump with these arrays:

```text
selected_mask
selected_init_cam
selected_class
features, view_index, view_class
```

The existing dumps already contain these arrays. New dumps also save
`selected_instance`, which makes the object identity explicit. For legacy
dumps, the script reconstructs object indices from the deterministic
batch-major/initial-camera ordering used by `trainer.test()` and validates the
reconstructed class labels before inference.

The old flattened selection JSON alone is insufficient for this analysis: it
contains actions accumulated across all initial cameras but not the grouping
needed to recover one five-view set.

## Efficient Evaluation

With the final-classifier protocol, the recognition backbone extracts all test
object/view features once per run checkpoint. Those features are reused across
all selected epochs and all initial-camera rollouts. With the
classifier-at-epoch-t protocol, the script uses the per-view features already
stored in `feature_<epoch>.npz`, which were generated by that epoch's model.

For either protocol, the full, leave-one-out, and replacement masks reuse the
same per-view features. This preserves the model's configured `max` or `mean`
pooling while avoiding a separate image-backbone pass for every intervention.

The model accepts a variable number of views because aggregation is performed
over the view dimension before the linear classifier.

## Mid-Level Relationships

Each selected filename is joined to the shared per-view cache generated by
`midlevel_shape_features.py`. The primary continuous cues are:

| Cue | Question |
|---|---|
| `ellipse_aspect_ratio` | Do views with a clearer elongated silhouette axis contribute more? |
| `bilateral_symmetry` | Do views exposing stronger 2D reflection symmetry contribute more? |
| `edge_entropy` | Do views with more dispersed 2D edge orientations contribute more? |

For these continuous descriptors, the default statistic is Spearman rank
correlation. Leave-one-out correlates the selected cue with deletion delta.
Replacement reports two relationships:

1. selected-view cue versus the selected view's mean delta across its sampled
   replacements;
2. `cue(selected) - cue(replacement)` versus the pair's replacement delta.

The first is intentionally an absolute-level analysis: it asks whether selected
views with a larger cue tend to outperform random unselected baselines by more,
after averaging over those baselines. It does not measure cue change. The
second is the direct change-versus-change analysis: it tests whether replacing
a view with one that has a weaker mid-level cue predicts a larger objective
drop. Spearman is used because the relationship need not be linear and the
descriptors are bounded or skewed. Use `--correlation pearson` to request a
linear correlation instead.

The exact view family is categorical:

```text
Expanded
Expanded-like
Foreshortened
Foreshortened-like
Remainder
```

Assigning those labels arbitrary numbers and correlating the numbers would be
invalid. The analysis therefore reports:

- mean delta and 95% normal-approximation interval for each exact family;
- eta-squared across the five families, measuring how much variation in delta
  is associated with family membership;
- a one-way ANOVA p-value as a descriptive companion to eta-squared.

## Checkpoint Protocols

The default protocol holds the final no-freeze model fixed,
recomputes its per-view test features, and evaluates selections from epochs 10,
20, ..., 100. This answers:

> Which views from different points in selector training are useful to the
> final recognition model?

With `--per_epoch_checkpoint`, selections from epoch `t` are evaluated by the
classifier in `model_e<t>.pth` using the matching per-view features saved at
epoch `t`. This answers:

> Which selected views were useful to the recognition model that existed at
> epoch `t`?

The second protocol requires matching per-epoch checkpoints. Experiments that
only contain `model.pth` can use the default final-classifier protocol.

When an experiment folder contains multiple timestamped selection JSON files,
the script matches each file to the same-timestamp directory under
`logs/<dataset>/`. If that checkpoint cannot be found, it falls back to the
most recent matching checkpoint and records `checkpoint_match=fallback_latest`
in the row-level CSV. Passing `--checkpoint` intentionally uses that explicit
checkpoint for every selection file.

Epoch filtering selects exact epochs; it does not average selections into
bins. `--epoch_stride 10` means epochs 10, 20, ..., 100.

The Python script uses all initial cameras by default. The no-freeze wrapper
defaults to 10 evenly spaced initial cameras to keep row counts and run time
manageable. Set `NUM_INITIAL_CAMS=0` for all 114 non-roll initial cameras.

## Commands

One experiment:

```bash
cd MVSelect-main

python3 view_contribution_analysis.py \
  --selection_dir meta_logs/rgb/resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100 \
  --non_roll \
  --epoch_stride 10
```

Use an explicit checkpoint:

```bash
python3 view_contribution_analysis.py \
  --selection_dir meta_logs/rgb/<experiment> \
  --checkpoint logs/rgb/<matching_run>/model.pth \
  --non_roll \
  --epoch_stride 10
```

Classifier-at-epoch-t protocol:

```bash
python3 view_contribution_analysis.py \
  --selection_dir meta_logs/rgb/<experiment> \
  --per_epoch_checkpoint \
  --non_roll \
  --epoch_stride 10
```

Run the no-freeze experiment on physical GPU 2:

```bash
GPU_ID=2 ./run_view_contribution_pipeline.sh

# Final analysis over every non-roll initial camera instead of the wrapper's
# default 10-camera subset.
NUM_INITIAL_CAMS=0 GPU_ID=2 ./run_view_contribution_pipeline.sh
```

Use ten unselected replacement samples per rollout:

```bash
REPLACEMENT_SAMPLES=10 REPLACEMENT_SEED=42 GPU_ID=2 \
  ./run_view_contribution_pipeline.sh

# Retain the initial input in M(.) while deleting each agent-selected view.
CONTEXT=with_initial GPU_ID=2 ./run_view_contribution_pipeline.sh
```

Smoke test on epochs 20 and 100, with five object sets per run and epoch:

```bash
EPOCHS="20 100" LIMIT_TRIALS=5 GPU_ID=0 \
  ./run_view_contribution_pipeline.sh
```

## Per-Experiment Outputs

The Python script defaults to `<selection_dir>/view_contribution/`. The wrapper
uses a tagged subfolder so different protocols and replacement samples do not
overwrite one another, for example:

```text
<selection_dir>/view_contribution/agent_only_final_classifier_cams_10_repl5_seed42/
```

Files in that output folder are:

| File | Contents |
|---|---|
| `leave_one_view_out_contributions.csv` | One row per agent-selected view and exact rollout, with initial camera, context, full/leave-one-out/delta values for both objectives, the three mid-level cues, and exact view family. |
| `contribution_correlations.csv` | Overall and per-epoch cue/delta correlations. |
| `view_family_contributions.csv` | Overall and per-epoch mean delta, standard deviation, standard error, and 95% interval for each exact family. |
| `view_family_association.csv` | Overall and per-epoch eta-squared and one-way ANOVA result. |
| `correlation_<objective>_heatmap.png` | Cue/delta correlation across selected epochs for one objective. |
| `view_family_<objective>_heatmap.png` | Mean delta by exact view family and epoch. |
| `view_family_eta_squared_heatmap.png` | Strength of family membership's association with each objective over time. |
| `overall_correlation_heatmap.png` | Dataset-wide cue/delta relationship pooled across evaluated epochs. |
| `selected_to_unselected_replacements.csv` | One row per selected/replacement pair, including both views' cues and both replacement deltas. |
| `replacement_selected_view_means.csv` | Replacement effects averaged across sampled substitutes once per selected view. |
| `replacement_correlations.csv` | Correlations for selected cues and selected-minus-replacement cue differences. |
| `replacement_view_family_contributions.csv` | Mean replacement delta grouped by the selected view's exact family. |
| `replacement_view_family_association.csv` | Eta-squared for selected-family membership and replacement contribution. |
| `replacement_family_pair_contributions.csv` | Mean delta for every selected-family to replacement-family transition. |
| `replacement_correlation_<objective>_heatmap.png` | Selected-minus-replacement cue differences correlated with prediction change over epochs, for ellipse aspect ratio, bilateral symmetry, and edge entropy. |
| `replacement_view_family_<objective>_heatmap.png` | Mean replacement delta by selected family and epoch. |
| `replacement_family_pair_<objective>_heatmap.png` | Mean replacement delta for all exact family transitions and epochs. |

## Caveats

1. Leave-one-out delta is a marginal contribution in the context of the other
   selected views, not an intrinsic score for the image. A useful but redundant
   view can have a small delta because another selected view supplies the same
   evidence.

2. This is not a Shapley-value analysis. It evaluates deletion from one full
   selected set rather than averaging a view's contribution over all possible
   subsets.

3. With max view pooling, a view only changes pooled feature channels where it
   supplies the maximum activation. Small or zero deltas for many views are
   therefore expected and are part of the trained model's aggregation behavior.

4. Correlations do not establish that the mid-level cue caused the
   contribution. View family, object class, pose, and other visible properties
   can covary with both the descriptor and delta.

5. The same object and view can appear at multiple epochs or in multiple
   selection files. The reported p-values treat rows as observations and should
   be read descriptively; effect direction, magnitude, consistency across
   epochs, and replication across experiments are more important than a pooled
   p-value.

6. Pooled `overall` correlations mix evaluated epochs. Use the per-epoch
   heatmaps to determine whether a relationship is stable or only appears late
   in training.

7. Raw logit and margin deltas depend on classifier calibration. Rank
   correlations and eta-squared are generally easier to interpret across
   epochs than raw mean logit deltas.

8. These mid-level descriptors are 2D visible-view proxies measured from the
   rendered PNGs, not mesh-intrinsic 3D measurements.

9. Replacement results depend on the sampled unselected candidates. Increase
   `REPLACEMENT_SAMPLES` and repeat with another `REPLACEMENT_SEED` to check
   robustness. The selected-view correlation averages sampled effects before
   analysis; selected-minus-replacement correlation remains pair-level.

10. The wrapper's default 10-camera subset is deterministic and evenly spaced
    over camera indices, but it is still a subset. Use `NUM_INITIAL_CAMS=0` for
    results averaged over every available initial camera.
