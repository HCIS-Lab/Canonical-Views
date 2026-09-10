# Single- and Multi-View Mid-Level Cue Analysis

This analysis asks whether ellipse aspect ratio, bilateral symmetry, and edge
entropy explain final-classifier predictions differently when the classifier
receives one view or five views. It is intentionally restricted to the
`no_freeze` experiment and always holds the final classifier fixed.

## Input Regimes

| Regime | Input | Sampling |
|---|---|---|
| `single` | `N=1` | Every candidate view is evaluated independently. Results are also grouped into Expanded, Expanded-like, Foreshortened, Foreshortened-like, and Remainder. |
| `random5` | `N=5` | By default, 100 deterministic random sets are sampled without duplicate views for every test object. |
| `selected5` | `N=5` | Exact agent-selected sets from `feature_100.npz`. The initial input view is removed, leaving the five agent actions. |

Evaluating every single candidate is preferable to drawing one random view per
type: it removes sampling noise, preserves within-family variation, and does
not add significant backbone cost because all per-view features are extracted
once and reused.

## Comparable N=1 and N=5 Quantities

Raw correct-class logits and prediction margins can differ systematically with
input-set size. The primary analysis therefore uses within-object,
set-size-matched centering.

For a single view `v`:

```text
prediction_utility_1(v)
    = M({v}) - mean_u M({u})

cue_enrichment_1(v)
    = x(v) - mean_u x(u)
```

The means are over all candidate single views `u` of the same object.

For a five-view set `S`:

```text
prediction_utility_5(S)
    = M(S) - mean_R M(R)

cue_enrichment_5(S)
    = summary_x(S) - mean_R summary_x(R)
```

`R` denotes random five-view sets of the same object. For each cue, the script
records the mean, maximum, and standard deviation across the five views. The
mean is the direct N=1/N=5 comparison; maximum and standard deviation test
whether a strong individual view or cross-view diversity is more relevant.

The two prediction objectives remain separate:

- correct-class logit;
- top-1 minus top-2 prediction margin.

Top-1 accuracy is also reported directly for each single-view family, random
five-view sets, and selected five-view sets.

## Statistical Analyses

### Object-Balanced Correlation

Spearman correlation is calculated separately within each object between cue
enrichment and prediction utility. Per-object correlations are then averaged
on Fisher's z scale. This prevents easy classes or objects with unusually large
logits from creating a pooled correlation and gives each object equal weight.
When multiple timestamped training runs match the no-freeze experiment, their
Fisher-z correlations are averaged within each object before the across-object
mean and confidence interval are calculated.

The primary `n1_n5_correlation_<objective>.png` figures compare the same three
mean cue summaries across:

```text
all single views (N=1)
random sets (N=5)
agent-selected sets (N=5)
```

### Grouped Cross-Validated R-Squared

Three linear models predict within-object prediction utility:

```text
cues_only:
    ellipse + symmetry + entropy

family_only:
    exact-family proportions

cues_plus_family:
    ellipse + symmetry + entropy + exact-family proportions
```

The prediction target is classifier-output enrichment: either centered
correct-class logit or centered top-1-minus-top-2 margin. R-squared therefore
measures how much held-out variation in the classifier output is predicted by
the listed inputs; it is not an R-squared for predicting cues or accuracy.
Each cue receives its own fitted coefficient, so ellipse aspect ratio and
symmetry can point positively while entropy points negatively. The model does
not assume that all three cues have the same direction.

Cross-validation folds are grouped by object, so views or sets from one object
never appear in both training and test folds. With five folds, the model is
fit on four groups of objects and predicts the fifth unseen group; this is
repeated until every object has a held-out prediction. The
`incremental_family_after_cues` value is:

```text
R2(cues_plus_family) - R2(cues_only)
```

A positive held-out increment means view-family composition predicts something
about the classifier output that is not captured by the three measured cues.
Negative cross-validated R-squared is valid: it means the model generalizes
worse than predicting the held-out mean.

## Command

```bash
cd MVSelect-main

GPU_ID=0 ./run_single_multiview_midlevel_analysis.sh
```

Useful overrides:

```bash
# Faster smoke test.
LIMIT_INSTANCES=10 RANDOM_SETS=20 GPU_ID=0 \
  ./run_single_multiview_midlevel_analysis.sh

# Use all initial-camera rollouts for selected N=5 sets.
NUM_INITIAL_CAMS=0 GPU_ID=1 \
  ./run_single_multiview_midlevel_analysis.sh

# Explicit final checkpoint.
CHECKPOINT=logs/rgb/<matching-run>/model.pth GPU_ID=2 \
  ./run_single_multiview_midlevel_analysis.sh

# Regenerate plots from existing CSVs without running the classifier again.
PLOT_ONLY=1 ./run_single_multiview_midlevel_analysis.sh

# Only compute the direct raw means by exact view family; no classifier pass.
FAMILY_MEANS_ONLY=1 ./run_single_multiview_midlevel_analysis.sh
```

Outputs are written to:

```text
meta_logs/rgb/<no-freeze-experiment>/single_multiview_midlevel/
```

## Outputs

| Output | Meaning |
|---|---|
| `input_regime_trials.csv` | Row-level predictions, cue summaries, family proportions, matched baselines, and enrichments. |
| `performance_by_regime.csv` | Object-macro top-1 accuracy and mean prediction outputs. |
| `midlevel_correlations.csv` | Object-balanced correlations and 95% intervals. |
| `cross_validated_r2.csv` | Held-out explained variance for cues, family, combined, and incremental-family models. |
| `standardized_coefficients.csv` | Full-data standardized coefficients from the combined descriptive model. |
| `all_candidate_view_family_midlevel_means.csv` | Raw object-macro mean, SEM, pooled mean, and sample counts for each cue and exact view family across all candidate test views. |
| `accuracy_by_input_regime.png` | Single-view family accuracy beside random and selected N=5 accuracy. |
| `all_candidate_view_family_midlevel_means.png` | Three direct raw-value panels comparing ellipse aspect ratio, bilateral symmetry, and edge entropy across the five exact view families. |
| `n1_n5_correlation_<objective>.png` | Direct comparison of mean-cue importance across N=1, random N=5, and selected N=5. |
| `n5_summary_correlation_<objective>.png` | Mean, maximum, and diversity cue summaries for random and selected N=5 sets. |
| `single_family_correlation_<objective>.png` | Within-family single-view correlations. |
| `cue_and_family_cross_validated_r2.png` | Cues-only, family-only, combined, and incremental-family held-out R-squared. |

## Caveats

1. The final classifier was trained for multiview input. `N=1` is therefore a
   diagnostic probe, not an in-distribution estimate of its intended operating
   accuracy.
2. Correct-class logit and prediction margin are not calibrated probabilities.
3. Within-object centering makes association strengths comparable but does not
   make raw N=1 and N=5 logits identical in meaning.
4. Family and mid-level cues are observational and correlated. Incremental
   cross-validated R-squared supports additional predictive information, not a
   causal claim.
5. Prediction margin is not target-aware. A confidently incorrect prediction
   can have a large margin; use correct-class logit and accuracy alongside it.
