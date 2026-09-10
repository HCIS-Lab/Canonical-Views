# Figure Caption Catalog

These captions describe the default analyses and can be copied into a paper or
adapted when a run uses non-default epochs, initial-camera counts, perturbation
strengths, or experiment sets. Replace bracketed placeholders with the exact
model, comparison, metric, or number of repeated runs used in the figure.

## View Selection and Classification

### `aggregated_view_ratios.png`

**Caption.** Exact view-type composition of the agent's selections over
training. Curves show the mean selection share for Expanded, Expanded-like,
Foreshortened, Foreshortened-like, and Remainder views; shaded regions denote
SEM across repeated runs. Dashed horizontal references indicate each type's
availability in the candidate-view pool, so departures from the reference show
selection preference rather than dataset frequency alone.

### `aggregated_view_family_ratios.png`

**Caption.** View-family composition of the agent's selections over training.
Expanded and Expanded-like views are combined as the expanded family, and
Foreshortened and Foreshortened-like views as the foreshortened family; the
Remainder is shown separately. Curves and shaded regions report mean +/- SEM
across runs, and dashed references show candidate-pool availability. The
vertical marker identifies the first epoch at which expanded-family selection
share exceeds foreshortened-family share.

### `aggregated_view_family_lift.png`

**Caption.** Selection lift of each view family relative to its availability in
the candidate pool. Lift is observed selection share divided by available
share; 1 denotes uniform sampling, values above 1 indicate over-selection, and
values below 1 indicate under-selection. Curves report the mean across runs.

### `aggregated_accuracy.png`, `aggregated_accuracy_3.png`, `aggregated_accuracy_5.png`

**Caption.** Classification accuracy by exact view type for inputs containing
[one/three/five] views. Values are aggregated across evaluation instances and
training runs; higher values indicate that inputs from the corresponding view
type support more accurate object recognition.

### `aggregated_per_class_accuracy.png`

**Caption.** Development of test accuracy for each object class over training.
Rows denote ModelNet classes, columns denote training epochs, and cell color
encodes class-specific accuracy. The heatmap exposes classes for which
recognition improves early, late, or remains difficult.

## Temporal Robustness and Prediction Stability

### `deviation_rotate.png` and `deviation_rotate_heatmap.png`

**Caption.** Sensitivity of recognition to rotating each selected view by
randomly signed 10-degree perturbations. Values are clean accuracy minus
rotated-view accuracy in percentage points, evaluated using selections saved at
each epoch. Positive values indicate that rotation lowers performance. The
caption or Methods should state whether the final classifier was held fixed or
the classifier checkpoint from the corresponding epoch was used.

### `deviation_jitter.png` and `deviation_jitter_heatmap.png`

**Caption.** Sensitivity of recognition to color jitter applied to the selected
views. Values are clean accuracy minus perturbed accuracy in percentage points
for selections saved at each epoch; positive values indicate a performance
drop. [Lines/rows] identify experiment settings.

### `deviation_rotate_jitter.png` and `deviation_rotate_jitter_heatmap.png`

**Caption.** Sensitivity of recognition to the joint rotation and color-jitter
manipulation. Values are clean minus perturbed accuracy in percentage points
for selections saved at each epoch; positive values indicate reduced
robustness under the combined perturbation.

### `margin_over_time.png` and `margin_over_time_heatmap.png`

**Caption.** Prediction stability for views selected at successive training
epochs. Stability is the mean top-1 minus top-2 classifier-logit margin on the
unperturbed input; larger values indicate more decisive predictions but do not
guarantee correctness. [Lines/rows] denote experiment settings. The caption or
Methods should state whether the final classifier was fixed or matched to each
selection epoch.

### `*_sorted_bars.png`

**Caption.** Cross-experiment comparison of [metric] at each displayed epoch.
Bars retain the metric's original y-axis and are ordered within each epoch to
make changes in experiment ranking visible. Sorting changes display order only
and does not alter values.

### `*_rank_stacked.png`

**Caption.** Rank-oriented comparison of [metric] across experiments and
epochs. Segments are ordered by value within each epoch to emphasize changes in
ranking; total bar height is a sum of experiment values and has no independent
scientific interpretation.

## Representation Clustering

### `silhouette_class_heatmap.png`

**Caption.** Class clustering of saved per-view representations over training.
For each experiment and epoch, cosine-distance silhouette score is computed on
L2-normalized view features using object-class labels. Positive values indicate
that views are, on average, closer to views of the same class than to views of
other classes; values near zero indicate overlapping class structure.

### `silhouette_view_heatmap.png`

**Caption.** Exact-family clustering of saved per-view representations over
training. Silhouette score is computed with the five view-type labels. Larger
positive values indicate stronger organization by projected view type, whereas
values near or below zero indicate weak view-type clustering.

### `silhouette_view_index_heatmap.png`

**Caption.** Camera-index clustering of saved per-view representations over
training. Silhouette score uses the exact camera index (0--113 in the 114-view
setting) as the cluster label. Positive values indicate pose-specific feature
organization; negative values indicate that a view is typically closer to
features assigned to other camera indices than to its own index.

### `separability_heatmap.png`

**Caption.** Relative class-versus-view organization of per-view
representations. Each cell is class-label silhouette minus five-way view-type
silhouette for one experiment and epoch. Positive values indicate stronger
class organization than view-type organization; negative values indicate the
reverse. This difference is a descriptive composite, not a standalone measure
of classification accuracy.

### `silhouette_class_selected_heatmap.png`

**Caption.** Class separability of object-level features reconstructed by
pooling the agent-selected views at each epoch, excluding the initial input
view. Silhouette score uses object-class labels and cosine distance; higher
values indicate more compact within-class and more separated between-class
object representations.

### `silhouette_class_selected_with_init_heatmap.png`

**Caption.** Class separability of the exact pooled representation provided to
the classifier during selector evaluation: the initial view plus the
agent-selected views. Higher cosine-distance silhouette scores indicate better
class clustering of these object-level representations.

### `silhouette_class_all_views_mean_heatmap.png` and `silhouette_class_all_views_max_heatmap.png`

**Caption.** Class separability after [mean/max] pooling all candidate-view
features of each object. Each object contributes one pooled representation,
and silhouette score is computed using class labels and cosine distance. These
figures characterize all-view pooling, not the subset chosen by the agent.

### `<N>cls_<R>runs_class_tsne_e<E>.png`

**Caption.** Two-dimensional t-SNE projection of saved representations at epoch
E, colored by object class. Nearby points have similar high-dimensional
features under the local geometry emphasized by t-SNE; apparent cluster sizes
and global distances should not be interpreted quantitatively.

### `<N>cls_<R>runs_view_type_tsne_e<E>.png`

**Caption.** Two-dimensional t-SNE projection of saved representations at epoch
E, colored by exact view type. The visualization qualitatively assesses whether
representations remain organized by projected view family; quantitative
conclusions use the corresponding silhouette analyses.

## Mid-Level Projected-Shape Measures

### `all_test_views_midlevel_3d_<primary/opposite/rear/angles>.png`

**Caption.** Distribution of every rendered view with finite measurements for
every object in the test split across three raw projected-shape measures:
ellipse aspect ratio,
bilateral silhouette symmetry, and edge-orientation entropy. Small gray points
are all candidate images; five magenta points with white halos are the agent
actions from one exact saved epoch-[E] rollout with initial camera [C]. The
initial input view is removed from the saved rollout mask. Coordinates are not standardized or
subsampled. Primary, opposite, and rear viewpoints are rendered as separate
full-size figures to reduce occlusion and retain point visibility. Views with
an undefined coordinate are not imputed and are reported separately in the
accompanying audit. The `angles` version concatenates the same three camera
angles. When generated per instance, every panel contains only the candidate
and selected views of one of the five physical objects used by the evaluation
loader. For legacy dumps without saved instance names, instance identity is
reconstructed from that loader's deterministic sorted order while selected
camera indices remain those in the exact saved rollout mask.

For the one-instance-per-class variant, the figure contains one deterministic
evaluated object from each of the 32 classes, with all candidate views and the
five exact agent-selected views retained for every object.

### `midlevel_selected_raw_primary_metrics.png`

**Caption.** Raw projected-shape properties of agent-selected views over
training. Separate panels report ellipse aspect ratio, bilateral symmetry, and
edge-orientation entropy in their native units. Curves show selected-view mean
+/- SEM across runs; the matched all-candidate baseline is computed from the
same object instances.

### `midlevel_<group>_selected_curves.png` and `selected_<group>_curves.png`

**Caption.** Raw mid-level properties of selected views for the [axis
visibility/symmetry/edge organization] group over training. Curves compare
[experiment settings] using the metric's native units; shaded regions, when
shown, denote SEM across runs.

### `midlevel_<group>_lift_curves.png` and `lift_<group>_curves.png`

**Caption.** Change in selected-view [group] properties relative to the
same-instance all-candidate baseline. Lift is selected mean minus baseline mean
in raw metric units; zero denotes no difference, and the sign indicates the
direction of selection bias.

### `midlevel_<group>_effect_curves.png`, `effect_<metric>_heatmap.png`, and `effect_<group>_curves.png`

**Caption.** Standardized selection effect for [metric/group] over training.
Effect is selected-minus-baseline mean divided by the same-instance candidate
standard deviation. Positive and negative values indicate selection above and
below the baseline, respectively; heatmap colors are clipped to [-1, 1] for
comparability across metrics.

### `selected_<metric>_heatmap.png` and `selected_<metric>_line.png`

**Caption.** Raw selected-view [metric] over training for the compared
experiment settings. [Rows/curves] identify experiments, the horizontal axis
denotes epoch, and values remain in the metric's native units. These figures
describe the selected views and do not subtract candidate-pool availability or
the same-object baseline.

### `lift_<metric>_heatmap.png` and `lift_<metric>_line.png`

**Caption.** Selected-view [metric] relative to the same-instance
all-candidate baseline over training. Lift is selected mean minus baseline mean
in raw units; positive and negative values indicate selection above and below
the baseline, respectively.

### `view_type_raw_<metric>_heatmap.png`

**Caption.** Raw mean [metric] of selected views from each exact view type over
training. Rows identify experiment and view type, columns identify epoch or
epoch bin, and color is the mean in the metric's native units. Cell values
characterize selected views only and are not normalized by type availability.

### `view_type_point_biserial_<metric>_heatmap.png`

**Caption.** Signed association between exact view-type membership and
[metric] among selected views. Each cell is the point-biserial correlation
between membership in one type and the continuous metric versus all other
types; positive values indicate larger metric values for that type.

### `view_type_midlevel_eta_squared_heatmap.png`

**Caption.** Strength of association between five-way exact view type and the
selected views' mid-level features. Eta-squared is the fraction of metric
variance associated with differences among view types; it is non-negative and
does not encode which type has larger values.

### `view_type_midlevel_point_biserial_heatmap.png`

**Caption.** Signed one-versus-rest association between each exact view type and
each selected-view mid-level feature. Positive values indicate that the type
has larger feature values than the remaining types, while negative values
indicate smaller values.

### `*_midlevel_extractions.png`

**Caption.** Visual audit of the projected-shape measurements. Rows show the
rendered view, extracted object mask and thinned silhouette skeleton, Sobel edge
magnitude, and Sobel edge orientation, together with the resulting mid-level
metric values. These diagnostics verify the image-derived proxies used in the
quantitative analyses.

## Selector Score and Choice

### `selector_score_correlation_heatmap.png`

**Caption.** Association between the learned selector's action values and
projected-shape cues over training. At each selector decision, rank correlation
is computed across all currently valid candidate views between raw DQN action
value and ellipse aspect ratio, bilateral symmetry, medial-axis symmetry, or
edge entropy. Decision-level correlations are averaged within each timestamped
run and then across matched runs. Positive values indicate that candidates with
larger cue values receive higher selector scores in the same state.

### `selector_score_correlation_axis_visibility_heatmap.png`

**Caption.** Within-decision rank correlation between candidate action value
and ellipse aspect ratio over training. Positive values indicate increasing
selector preference for views with more elongated projected silhouettes,
conditional on the currently valid candidate set.

### `selector_score_correlation_symmetry_heatmap.png`

**Caption.** Within-decision rank correlation between candidate action value
and bilateral or medial-axis symmetry over training. Positive values indicate
that more symmetric candidate projections tend to receive higher action values
within the same selector state.

### `selector_score_correlation_edge_organization_heatmap.png`

**Caption.** Within-decision rank correlation between candidate action value
and edge-orientation entropy over training. Positive values indicate higher
scores for candidates with more diverse edge orientations; negative values
indicate preference for more anisotropic edge organization.

### `selector_choice_percentile_heatmap.png`

**Caption.** Mid-level cue percentile of the greedily chosen view among the
currently valid candidates. Values above 0.5 indicate that the chosen view is
above the candidate median for that cue, values below 0.5 indicate selection
below the median, and 1 denotes a candidate-maximum cue value. Values are
averaged within runs and then across matched runs.

### `selector_choice_percentile_<group>_heatmap.png`

**Caption.** Cue percentile of the greedily selected view for the [axis
visibility/symmetry/edge organization] measure. Columns denote checkpoint
epochs; values above the candidate median (0.5) indicate preferential choice
of views with larger cue values in the current state.

## Selected-View Contribution

### `correlation_<objective>_heatmap.png`

**Caption.** Association between each selected view's mid-level features and
its leave-one-view-out contribution to [correct-class logit/prediction margin].
Contribution is the full selected-set output minus the output after removing
that view, so positive values indicate that removal weakens the objective.
Cells report rank correlation across evaluated selected views at each epoch.

### `view_family_<objective>_heatmap.png`

**Caption.** Mean leave-one-view-out contribution to [objective] by exact
selected-view family and selection epoch. Positive values indicate that
removing a view from that family lowers the full-set objective; values are in
raw [logit/logit-margin] units.

### `view_family_eta_squared_heatmap.png`

**Caption.** Strength of association between exact selected-view family and
leave-one-view-out contribution over training. Eta-squared measures the
fraction of contribution variance associated with family membership and does
not encode the direction of individual family effects.

### `overall_correlation_heatmap.png`

**Caption.** Dataset-level association between selected-view mid-level features
and leave-one-view-out contribution, pooled across the displayed selection
epochs. Rows identify cues and columns identify classifier objectives; cells
are unitless rank correlations.

### `replacement_correlation_<objective>_heatmap.png`

**Caption.** Change-versus-change association under selected-view replacement.
For each intervention, the cue difference is selected-view value minus
replacement-view value, and the objective difference is full selected-set
output minus output after substituting the unselected view. Positive
correlation means replacing a stronger-cue view with a weaker-cue view tends to
reduce [objective].

### `replacement_view_family_<objective>_heatmap.png`

**Caption.** Mean effect of replacing a selected view, grouped by the selected
view's exact family and epoch. Each selected view is replaced by sampled views
outside the saved selection; positive values mean the replacement lowers
[objective], indicating a positive contribution from the removed selected
view.

### `replacement_family_pair_<objective>_heatmap.png`

**Caption.** Mean intervention effect for each selected-family to
replacement-family transition over training. Positive values indicate that
substituting the row-family selected view with the specified replacement
family lowers [objective]. The figure separates family identity of both sides
of the intervention.

## Single- and Five-View Recognition

### `all_candidate_view_family_midlevel_means.png`

**Caption.** Raw mid-level feature means for all candidate test views grouped by
exact view family. Within each object, views of a family are averaged first;
bars then report the mean across objects and error bars denote object-level
SEM. Panels show ellipse aspect ratio, bilateral symmetry, and edge-orientation
entropy in native units.

### `accuracy_by_input_regime.png`

**Caption.** Final-classifier top-1 accuracy for individual views from each
exact family, random five-view sets, and epoch-[E] agent-selected five-view
sets. Every candidate single view is evaluated independently; random sets
contain no duplicate view, and selected sets exclude the initial input. Bars
report object-macro accuracy with SEM across unique test objects.

### `n1_n5_correlation_<objective>.png`

**Caption.** Object-balanced association between mid-level cue enrichment and
[correct-class-logit/prediction-margin] enrichment for all single views, random
five-view sets, and agent-selected five-view sets. Single-view values are
centered by the same object's all-view mean, and five-view values by the same
object's random-five mean. Rank correlations are computed within each object
and averaged on Fisher's z scale; positive values indicate that larger cues
rank with stronger classifier outputs.

### `n5_summary_correlation_<objective>.png`

**Caption.** Association between five-view cue summaries and [objective]
enrichment. Mean, maximum, and within-set standard deviation respectively test
whether average cue strength, one strong view, or cross-view cue diversity is
related to classifier output. Columns compare random and agent-selected
five-view sets; correlations are object-balanced.

### `single_family_correlation_<objective>.png`

**Caption.** Within-family single-view association between each mid-level cue
and [objective]. Correlations are computed separately within object and exact
view family before object-balanced aggregation. The figure tests variation
inside a family and may differ in sign from comparisons driven by mean
differences between families.

### `cue_and_family_cross_validated_r2.png`

**Caption.** Held-out prediction of within-object classifier-output enrichment
from three mid-level cues, exact view-family identity or proportions, and their
combination. Five-fold cross-validation is grouped by object, preventing views
or sets from the same object from appearing in both fitting and evaluation
folds. R-squared above zero indicates prediction better than the held-out mean;
the family-gain bar is combined-model R-squared minus cue-only R-squared.

## Controlled Policy Replay

### `accuracy_over_training.png`

**Caption.** Held-out recognition accuracy for networks trained using recorded
epoch-10, epoch-20, or epoch-30 selections throughout training; final-epoch
selections throughout training; naturally evolving or temporally shuffled
selections; and
family-matched replacement recorded-action policies, together with a fresh
recognizer trained on randomly resampled K-view sets. Random-warm-up controls
use random views for the first 10, 20, or 30 epochs, then replay the original
no-freeze selections at the matching absolute epoch for the remainder of
training. The dashed joint
no-freeze curve is an external reference whose selector and classifier were
trained together. All replay conditions within a seed
start from the same model state, receive the same number of optimizer updates,
and are evaluated on identical deterministic five-view inputs; curves show
means across seeds with SEM. The external reference uses the same evaluation
inputs but is not initialization- or update-matched to the replay controls.

### `margin_over_training.png`

**Caption.** Mean held-out prediction margin during controlled policy-replay
training. The margin is the top predicted logit minus the second-highest logit
for each object. All conditions use matched initialization, update count, and
evaluation inputs; curves show means across seeds with SEM.

### `<accuracy/margin>_over_training_heatmap.png`

**Caption.** Held-out [accuracy/prediction margin] across controlled
policy-replay conditions and recognition epochs. Rows are training-view
histories, columns are evaluated epochs, and each cell reports the mean across
recognition seeds. All replay controls share initialization, optimizer-update
count, and held-out inputs; the joint no-freeze row is an external reference.

### `<accuracy/margin>_over_training_delta_vs_evolving_heatmap.png`

**Caption.** Difference in held-out [accuracy/prediction margin] relative to
the naturally evolving replay at the same recognition epoch. Positive values
indicate a larger metric than evolving and negative values indicate a smaller
metric. Each cell is computed after averaging across recognition seeds; the
joint no-freeze row remains an external, non-initialization-matched reference.

### `<accuracy/margin>_over_training_<policy_history/random_warmup/identity_controls>_curves.png`

**Caption.** Focused held-out [accuracy/prediction-margin] trajectories for the
named policy-replay contrast. Curves show means across recognition seeds with
SEM, and every focused panel uses the same metric-wide y-axis limits as the
complete comparison. The dashed black joint no-freeze curve is an external
reference.

### `final_accuracy.png`

**Caption.** Final held-out recognition accuracy after training from a shared
initialization under different recorded view-policy histories. Bars show means
across seeds with SEM. Evaluation uses the same deterministic five-view input
for every condition, so differences reflect recognition-network training
rather than condition-specific test-time selections.

### `final_class_silhouette.png`

**Caption.** Class silhouette score of each final recognition network's
aggregated held-out features, computed with cosine distance and object-class
labels. Higher values indicate tighter within-class and better separated
between-class representations. Bars show means across seeds with SEM on
identical held-out inputs.

### `final_vs_initial_cka.png`

**Caption.** Linear centered-kernel-alignment similarity between each final
representation and its shared initial representation on identical held-out
objects and views. Lower values indicate greater representational change from
initialization; bars show means across seeds with SEM. The external jointly
trained reference is omitted when its genuine epoch-0 checkpoint is
unavailable.

### `final_representation_cka.png`

**Caption.** Pairwise linear centered-kernel-alignment similarity between
final recognition representations learned under different recorded policy
histories. Each cell averages seed-matched comparisons on identical held-out
objects and views; larger values indicate more similar representational
geometry. Comparisons to the external joint reference average that reference
against all replay seeds and are not seed-matched training controls.

### `final_prediction_disagreement.png`

**Caption.** Pairwise disagreement between final recognition networks trained
under different recorded policy histories. Each cell reports the percentage
of identical held-out inputs assigned different top-1 classes by the two
conditions, averaged across matched seeds.

## Zero-Shot and Feature-Oddity Probes

### `bar_overall.png`

**Caption.** Single-view classification accuracy of the trained recognition
model by exact view family. One view is sampled from each family per object and
sampling run; bars report mean accuracy with variability across runs. Because
the classifier was trained on the target dataset, this is a controlled
single-view probe rather than a strict zero-shot evaluation.

### `per_class_heatmap.png` and `per_class_grid.png`

**Caption.** Single-view classification accuracy by object class and exact view
family for the trained recognition model. The heatmap summarizes all classes
in one matrix, while the grid shows one panel per class; both expose whether
family-level differences are consistent across semantic categories.

### `bar_top1.png` and `bar_top5.png`

**Caption.** CLIP zero-shot object-recognition accuracy by exact single-view
family. A random view is sampled from each family for each object and run, and
CLIP image embeddings are compared with text embeddings of ModelNet class
names. Bars report mean [top-1/top-5] accuracy and variability across sampling
runs.

### `per_class_top1_heatmap.png` and `per_class_top5_heatmap.png`

**Caption.** CLIP zero-shot [top-1/top-5] accuracy by object class and exact
single-view family. Cell color shows accuracy across repeated random samples,
revealing class-specific differences in view informativeness.

### `mochi_feature_oddity_heatmap.png`

**Caption.** Zero-shot pairwise feature-similarity oddity performance across
training checkpoints and experiment settings. For each oddity trial, the model
selects the pair with the greatest within-pair feature similarity as the
same-object pair. Cell color reports 3D-perception oddity accuracy; chance is
one third.

### `mochi_feature_oddity_over_epochs.png`

**Caption.** Development of zero-shot pairwise feature-similarity oddity
accuracy over training. Curves identify experiment settings and report the
fraction of trials in which the same-object pair has the greatest feature
similarity; the horizontal chance reference is one third.

### `mochi_feature_oddity_bar.png`

**Caption.** Final-checkpoint zero-shot feature-similarity oddity accuracy
across experiment settings. Higher accuracy indicates that the learned
representation more reliably groups views of the same object without fitting
an oddity-task classifier.

## Multiview Geometry Confidence

### `<model>_pair_<value>_heatmap.png`

**Caption.** Pair-level geometric confidence of selected views across epochs
and experiment settings. For each selected set, all unordered view pairs are
processed separately; each pair score averages the model's two masked
per-image confidence maps, and pair scores are then averaged within the set.
Color encodes [agent confidence/random confidence/agent-minus-random delta];
higher confidence denotes greater predicted geometric precision.

### `<model>_pair_<value>_over_epochs.png`

**Caption.** Pair-level geometric confidence across training epochs for the
compared experiment settings. Curves report [agent confidence/random
confidence/agent-minus-random difference] after averaging all unordered pairs
within each selected set. Higher confidence denotes greater predicted
geometric precision.

### `<model>_set_<value>_heatmap.png`

**Caption.** Joint set-level geometric confidence across epochs and experiment
settings. All selected views are processed simultaneously, each output
confidence map is averaged over the object mask, and the resulting per-view
values are averaged over the set. Color encodes [agent confidence/random
confidence/agent-minus-random delta].

### `<model>_set_<value>_over_epochs.png`

**Caption.** Joint set-level geometric confidence across training epochs. All
views in a set are processed simultaneously, object-masked confidence is
averaged over views, and curves compare experiment settings for [agent
confidence/random confidence/agent-minus-random difference].

### `<model>_<metric>_delta_heatmap.png`

**Caption.** Agent-over-random difference in [pair-level/set-level] geometric
confidence by object class and epoch group. Positive values indicate greater
predicted geometric precision for agent-selected views than matched random
views; rows are classes and columns are epoch groups.

### `<model>_<metric>_per_class_grid.png`

**Caption.** Class-specific geometric confidence for agent-selected and matched
random views over epoch groups. Each panel shows one object class, allowing the
dataset-level confidence difference to be checked for consistency across
semantic categories.

### `<model>_<metric>_macro.png`

**Caption.** Dataset-level [pair/set] geometric confidence for agent-selected
and matched random views across epoch groups. Curves report macro means across
object classes with SEM; their separation is the agent-over-random confidence
difference.

### `<model>_bar_overall.png`

**Caption.** Single-view depth-confidence-head output by exact view family.
Each view is processed independently, its confidence map is averaged over the
object mask, and values are aggregated across objects and repeated samples.
Absolute values are not directly calibrated against multi-view confidence;
the figure compares families within the single-view regime.

### `<model>_per_class_heatmap.png`

**Caption.** Single-view confidence by object class and exact view family.
Cells report mean object-masked confidence-head output across sampled views,
showing whether family differences are consistent across semantic classes.

### `<model>_bars.png`

**Caption.** Pair-confidence control comparing identical-image pairs, random
distinct-view pairs, same-family pairs, and cross-family pairs. Pair confidence
is the mean object-masked confidence over both images. Same-family conditions
sample two views from the same family and do not duplicate the image unless
explicitly labeled identical.

### `<model>_heatmap.png`

**Caption.** Pair confidence for every exact view-family combination. Each cell
averages confidence from pairs containing one sampled view from the row family
and one from the column family; diagonal cells use two independently sampled
views from the same family. Higher values indicate greater predicted geometric
precision or correspondence compatibility.

## Active Single-View Control

### `active_single_view_types.png`

**Caption.** Exact view-type selection shares for an active recognition model
whose classifier receives only the selected view. The initial scouting view
conditions the policy but is excluded from recognition. Results pool
selections across every possible initial camera; curves show means across
training seeds, shading shows SEM, and dotted lines mark candidate-set
availability.

### `active_single_families.png`

**Caption.** View-family preferences learned without multiview recognition
pooling. The classifier receives one selected view, while an initial scouting
view only conditions the selector. Curves report mean selection shares across
training seeds with SEM and candidate-set availability baselines.

### `active_single_view_bias_magnitude.png`

**Caption.** Active single-view selection bias measured as total variation
distance between the five-type selected-view distribution and candidate-set
availability. Zero denotes selection proportional to availability; larger
values indicate stronger view preference without identifying its direction.

### `active_single_recognition_accuracy.png`

**Caption.** Test accuracy during active single-view training. For every
initial scouting camera, the policy chooses one view and the classifier makes
its prediction from that selected view alone; the initial view is not pooled
into the recognition representation.

### `active_single_vs_pair_view_types.png`

**Caption.** Exact view-type selection shares for matched one-action active
recognition models. In the active-pair condition, the classifier aggregates
the initial and selected views; in the active-single condition, only
the selected view enters the classifier while the initial view conditions the
policy. Results pool deterministic selections across every possible initial
camera. Curves show means across matched training seeds, shading shows SEM,
and dotted lines mark each view type's availability in the candidate set.

### `active_single_vs_pair_families.png`

**Caption.** View-family preferences for matched active-pair and active-single
models over training. The active-single classifier receives only its selected
view, whereas the active-pair classifier pools the initial and selected views.
Curves report selection shares across matched seeds with SEM and candidate-set
availability baselines.

### `view_bias_magnitude.png`

**Caption.** Overall view-selection bias measured as total variation distance
between the five-type selected-view distribution and the candidate-set
availability distribution. Zero denotes selection proportional to view
availability; larger values denote a stronger departure from that baseline,
independent of the direction of preference. Curves compare matched
active-pair and active-single training runs.

### `active_single_minus_pair_selection_heatmap.png`

**Caption.** Difference in selected-view share between active-single and
active-pair models across training intervals. Each cell is active-single minus
active-pair share in percentage points, averaged over epochs in the column and
matched runs; positive values indicate relatively greater selection by the
single-view recognition condition.

### `recognition_accuracy.png`

**Caption.** Recognition accuracy for matched active-pair and active-single
models during training. The figure provides performance context for the bias
comparison; absolute accuracy is not directly controlled across conditions
because the classifiers receive two and one recognition views, respectively.

## Reporting Notes

- State the number of objects, runs, random samples, initial cameras, and
  epochs in the Methods or adapt the caption when defaults were changed.
- Confidence, rank correlation, silhouette score, eta-squared, R-squared, and
  logit differences are unitless or model-scale quantities; do not label them
  as percentages. Accuracy and accuracy drop should be labeled explicitly as
  proportions or percentage points.
- A positive association is descriptive and does not establish that the cue
  causally drives selection or recognition.
- Prediction margin measures decisiveness, not correctness. Report it together
  with correct-class logit or accuracy.
- Selector-limit experiments evaluate only their allowed candidate pool; their
  selector-score correlations are conditional on that restriction.
