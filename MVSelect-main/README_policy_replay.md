# Controlled Policy-Replay Recognition Experiment

## Purpose

This experiment tests whether the selector's training history changes the
recognition network itself, rather than merely changing which images are shown
at test time. Separate recognition networks are trained from an exactly shared
initial state, with the same optimizer, update count, object instances, number
of views, and held-out evaluation inputs. Only the temporal sequence of
replayed training selections changes.

The legacy `train_downstream.py` cannot perform this experiment. It merges all
selections between `--start` and `--end` into one frequency table, chooses one
static top-K set, and never changes the selected set with recognition-training
epoch. It therefore supports a static epoch-range preference probe, not a
temporal policy replay.

## Replay protocols

For each object and source-policy epoch, all `*_selection.json` files in one
no-freeze experiment are combined. The K filenames with the largest recorded
selection counts become that epoch's replayed training input. This preserves
the top-K action-frequency convention used by the legacy downstream loader,
but keeps epochs separate.

| Condition | Policy used at recognition epoch t | Question |
|---|---|---|
| `frozen_10` (plot: `Epoch-10 selections throughout`) | top-K selections recorded at no-freeze epoch 10 for all t | What representation develops under early recorded selections? |
| `frozen_20` (plot: `Epoch-20 selections throughout`) | top-K selections recorded at no-freeze epoch 20 for all t | Same, with later recorded selections. |
| `frozen_30` (plot: `Epoch-30 selections throughout`) | top-K selections recorded at no-freeze epoch 30 for all t | Same, with later recorded selections. |
| `final_from_start` | final policy for all t | Is the final view distribution sufficient by itself? |
| `evolving` | policy epoch t at epoch t | Natural ordered curriculum. |
| `shuffled` | the same evolving policy epochs in a fixed random order | Does temporal order matter? |
| `family_matched_random` | evolving family counts, but alternate view identities | Do exact views matter beyond five-way family frequencies? |
| `random_views` | K uniformly random distinct candidate views, resampled every recognition epoch | Does any learned policy outperform unstructured view exposure? |
| `random_warmup_10` | random views for t=1..10, then no-freeze policy epoch t for t=11..100 | Does removing the first 10 epochs of structured policy exposure change the learned representation? |
| `random_warmup_20` | random views for t=1..20, then no-freeze policy epoch t for t=21..100 | Same test with a 20-epoch random warm-up. |
| `random_warmup_30` | random views for t=1..30, then no-freeze policy epoch t for t=31..100 | Same test with a 30-epoch random warm-up. |
| `joint_no_freeze_reference` | original classifier and selector trained jointly | How does the original co-developed model compare on the same fixed held-out inputs? |

The family-matched control is necessary for the exact-view-identity claim.
Temporal shuffling preserves the overall collection of policy epochs but does
not isolate exact identities from view-family composition.

This is a **recorded-action replay**, not an online execution of historical
selector checkpoints. `selection.json` flattens choices across initial-camera
rollouts, so the replay uses per-instance action frequencies rather than
recovering an individual rollout trajectory.

`family_matched_random` replaces each replayed selected filename with a
different candidate filename from the same exact view family whenever one is
available. It preserves each object's per-epoch family multiset while changing
the specific camera/view identities. The plot label therefore says
`Family-matched replacement views`; "identity" refers to the image/view
filename, not the object identity.

`random_views` trains another fresh recognizer from the same seed-specific
shared initialization as the replay controls. For every training object and
recognition epoch, it chooses K distinct candidate views uniformly by a
deterministic hash permutation. The set is resampled at the next epoch. It uses
the same objects represented in `selection.json`, but does not use their
recorded selected filenames or view-family frequencies.

The three `random_warmup_*` conditions are also fresh recognizers trained from
the shared initialization. During warm-up, K distinct random views are
resampled deterministically for every object and recognition epoch. After
warm-up they join the original no-freeze policy at the matching absolute
epoch: for example, `random_warmup_10` uses random views through epoch 10 and
the original epoch-11 selections at recognition epoch 11. It does not restart
the policy sequence from epoch 1. This isolates whether early structured view
exposure matters while holding the final 90, 80, or 70 policy epochs fixed to
the natural no-freeze history. Within a classifier seed, the deterministic
random sets at a given epoch are identical across `random_views` and every
warm-up condition that is still random at that epoch.

The joint no-freeze condition is an **external reference**, not another replay
control. Its classifier and selector were jointly optimized in the original
training run. The evaluator loads its `model_e<E>.pth` checkpoints and tests
the recognition network on the same deterministic held-out N=5 inputs as the
replay networks; it does not execute the selector at evaluation. Because the
original run did not share the replay initialization, training objects, or
optimizer-update sequence, it is excluded from `control_audit.csv` and listed
in `external_reference_audit.csv`.

## Controlled variables

For each classifier seed, all conditions load the same
`shared_initializations/seed_<S>.pth` strictly. They also use:

- the same architecture and ImageNet initialization;
- the same classifier initialization hash;
- the same recognition-training instances and K views per update;
- the same minibatch count, epoch count, optimizer, and learning-rate schedule;
- the same deterministic random held-out N=5 inputs from `test-down`;
- the same evaluation epochs and metrics.

`control_audit.csv` checks that every condition within a seed has exactly one
initialization hash, one optimizer-update count, and one held-out input hash.
Do not interpret the plots unless `same_initialization`, `same_update_count`,
`same_evaluation_inputs`, and `all_conditions_present` are all `True`.

The external joint reference is permitted to differ on initialization and
update count, but its evaluation-input hash must match the replay runs.

The source selections were recorded on the ModelNet `test` instances, so the
recognition training split defaults to `test`. Evaluation uses the disjoint
`test-down` instances. The number of recognition-training objects is the
number of object IDs actually represented in the selection files, normally
five per class for the current MVSelect test loader.

## Run the full experiment

From `MVSelect-main/`:

```bash
# ResNet-18, three recognition seeds, four GPUs.
ARCH=resnet18 GPUS=4 SEEDS="0 1 2" \
  ./run_policy_replay_experiment.sh
```

By default the wrapper auto-selects the lexically latest timestamped directory
whose prefix exactly matches the no-freeze experiment. Pin the intended
original run when several exist:

```bash
ARCH=resnet18 GPUS=4 SEEDS="0 1 2" \
JOINT_LOGDIR=logs/rgb/resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100_2026-06-23_12-47-59 \
  ./run_policy_replay_experiment.sh
```

The wrapper defaults to this no-freeze selection history:

```text
meta_logs/rgb/resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100
```

Use an explicit experiment when its hyperparameters differ:

```bash
ARCH=resnet18 GPUS=4 SEEDS="0 1 2" \
EXPERIMENT=resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100 \
  ./run_policy_replay_experiment.sh
```

Or pass an absolute selection directory:

```bash
ARCH=resnet18 GPUS=4 \
SELECTION_DIR=/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/MVSelect-main/meta_logs/rgb/<experiment> \
  ./run_policy_replay_experiment.sh
```

TinyViT uses separate experiment and output paths:

```bash
ARCH=tinyvit GPUS=4 SEEDS="0 1 2" \
  ./run_policy_replay_experiment.sh
```

Smoke test:

```bash
ARCH=resnet18 GPUS=1 SEEDS="0" EPOCHS=2 EVAL_EVERY=1 \
TRAIN_PER_CLS=1 TEST_PER_CLS=1 \
INCLUDE_RANDOM_WARMUP=0 \
INCLUDE_JOINT_REFERENCE=0 \
OUTPUT_ROOT=compare/policy_replay_smoke_resnet18 \
  ./run_policy_replay_experiment.sh
```

Useful overrides:

```bash
GPU_IDS="1 2 3 4"              # physical GPUs to use
NUM_VIEWS=5                    # replayed training set size
TEST_NUM_VIEWS=5               # common held-out input size
INCLUDE_FAMILY_MATCHED=0       # omit only the extra family control
INCLUDE_RANDOM=0               # omit fresh-recognizer random-view control
INCLUDE_RANDOM_WARMUP=0        # omit all random-then-evolving controls
RANDOM_WARMUP_EPOCHS="10 20 30" # warm-up durations to include
INCLUDE_JOINT_REFERENCE=0      # omit original joint-training reference
JOINT_LOGDIR=logs/rgb/<run>    # pin a timestamped original no-freeze run
JOINT_FINAL_EPOCH=100           # epoch represented by its model.pth
OVERWRITE=1                    # rerun completed conditions
EVAL_VIEW_SEED=2027            # common held-out view-set seed
SHUFFLE_SEED=1729              # one temporal permutation shared by all seeds
```

## Run one condition directly

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_policy_replay.py \
  --selection_dir meta_logs/rgb/resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100 \
  --protocol frozen --policy_epoch 10 \
  --arch resnet18 --epochs 100 --num_views 5 --test_num_views 5 \
  --seed 0 --gpu_id 0 \
  --shared_init compare/policy_replay_manual/shared_seed0.pth \
  --output_dir compare/policy_replay_manual/frozen_10/seed_0
```

Other protocol arguments are:

```bash
--protocol frozen --policy_epoch 20
--protocol frozen --policy_epoch 30
--protocol final
--protocol evolving
--protocol shuffled --shuffle_seed 1729
--protocol family_matched_random
--protocol random
--protocol random_then_evolving --random_warmup_epochs 10
--protocol random_then_evolving --random_warmup_epochs 20
--protocol random_then_evolving --random_warmup_epochs 30
```

Every direct condition for a given seed must use the same `--shared_init`
path. The wrapper enforces this automatically.

## Outputs

Default output root:

```text
compare/policy_replay_<arch>/<experiment>/
```

Per-condition and per-seed files:

- `metrics.csv`: held-out accuracy, loss, and prediction margin over training;
- `policy_epoch_schedule.json`: recognition epoch to source-policy epoch;
- `run_summary.json`: initialization hash, update count, final metrics, and
  fallback diagnostics;
- `initial_test_features.npz` and `final_test_features.npz`: aggregated
  features, logits, labels, and held-out instance IDs on identical inputs;
- `model.pth`: final recognition network;
- `training_curve.png`: that run's accuracy and margin curves.

Cross-condition files:

- `control_audit.csv`: verifies matched initialization and update count;
- `external_reference_audit.csv`: verifies that the original jointly trained
  reference uses identical evaluation inputs and records why it is excluded
  from the matched replay audit;
- `accuracy_over_training.png`: held-out accuracy by recognition epoch;
- `margin_over_training.png`: prediction stability by recognition epoch;
- `accuracy_over_training_heatmap.png` and
  `margin_over_training_heatmap.png`: all conditions as rows and evaluated
  epochs as columns, with the across-seed mean printed in every cell;
- `accuracy_over_training_delta_vs_evolving_heatmap.png` and
  `margin_over_training_delta_vs_evolving_heatmap.png`: the same layout after
  subtracting the naturally evolving replay at each epoch; positive cells
  have a larger metric than evolving for that epoch;
- `*_policy_history_curves.png`: fixed early/final, evolving, shuffled, and
  joint-reference conditions only;
- `*_random_warmup_curves.png`: evolving, fully random, random-warm-up
  10/20/30, and the joint reference only;
- `*_identity_controls_curves.png`: evolving, family-matched replacement,
  fully random, and the joint reference only;
- `final_accuracy.png`: final held-out accuracy with seed-level SEM;
- `final_class_silhouette.png`: class clustering of final aggregated features;
- `final_vs_initial_cka.png`: how far each representation moved from the
  shared initialization (`1` means identical up to linear CKA invariances);
- `final_representation_cka.png`: pairwise linear CKA between final
  representations on identical held-out objects and views;
- `final_prediction_disagreement.png`: percentage of held-out objects on which
  two conditions predict different classes;
- `pairwise_representation_comparison.csv`: numerical CKA and disagreement;
- `aggregated_run_summaries.csv` and `aggregated_training_metrics.csv`.

The wrapper passes every enabled replay condition to the aggregator as a
requirement. If `random_views` or another enabled condition is missing for any
seed, aggregation stops with a missing-condition error instead of silently
producing incomplete figures.

To add the random control to an output directory created by an older run,
rerun the same wrapper command with `INCLUDE_RANDOM=1` and without
`OVERWRITE=1`. Completed conditions are skipped, the missing `random_views`
runs are trained, and all comparison figures are regenerated afterward.

## Interpreting the contrasts

- `final_from_start` vs `evolving`: same endpoint distribution, different
  history. A difference supports path-dependent representation learning.
- `evolving` vs `shuffled`: same collection of policy epochs, different order.
  A difference is direct evidence for a temporal curriculum effect.
- `evolving` vs `family_matched_random`: matched family composition but
  different view identities. A difference supports information beyond family
  frequencies.
- any replay policy vs `random_views`: tests whether its structured view
  exposure changes recognition relative to random K-view exposure under the
  same initialization and optimization budget.
- `random_warmup_10/20/30` vs `evolving`: tests whether replacing only the
  first 10, 20, or 30 epochs of natural policy exposure with random views
  changes the final recognition network.
- differences among `random_warmup_10/20/30` test whether the duration of
  early unstructured exposure has a graded effect while all conditions rejoin
  the same absolute no-freeze policy timeline afterward.
- `frozen_10/20/30` vs `final_from_start`: tests which fixed policy
  distribution is sufficient when present throughout recognition training.
- replayed `frozen_10/20/30` vs the original jointly trained freeze runs:
  separates policy exposure from historical selector-recognizer coadaptation.

Accuracy alone shows that the training policy affects downstream behavior.
The CKA, class-silhouette, and disagreement figures establish that this occurs
with different learned representations under exactly matched test-time inputs.

The external joint reference appears in accuracy, margin, final silhouette,
pairwise final-representation CKA, and prediction-disagreement figures. It is
omitted from `final_vs_initial_cka.png` unless a genuine epoch-0 checkpoint is
available: `model_e1.pth` is already trained for one epoch and is not treated
as initialization.

### Reading the representation figures

- `final_prediction_disagreement.png` reports the percentage of identical
  held-out objects for which two final classifiers output different top-1
  classes. Zero means identical decisions; it is not an accuracy or error
  value because both predictions may be correct or both may be wrong.
- `final_representation_cka.png` compares the complete final pooled-feature
  matrices on identical objects and views. Linear CKA near 1 means the two
  networks organize those objects similarly up to orthogonal rotation and
  global scaling; a lower value means different representational geometry.
- `final_vs_initial_cka.png` compares each replay network's final features with
  its own pre-replay features on identical inputs. A high value means less
  geometric change from initialization and a lower value means more change;
  neither direction alone implies better accuracy or clustering.

`final_from_start` means a fresh recognizer receives the top-K views recorded
at the final selector epoch for every recognition-training epoch, beginning at
epoch 1. The final selector is not run online and its weights are not trained;
its recorded final view set acts as a fixed training distribution.

`frozen_30` is a legacy internal folder/condition identifier retained so
existing results remain readable. Its plot label is `Epoch-30 selections
throughout`: a fresh recognizer receives the top-K views recorded at epoch 30
of the no-freeze evolving run throughout all 100 recognition-training epochs.
It does **not** use, continue, or retrain the separate original `freeze_30`
classifier/selector experiment. The same distinction applies to the epoch-10
and epoch-20 conditions.
