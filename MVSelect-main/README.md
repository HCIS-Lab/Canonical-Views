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

### Aggregated meta-log visualizations

`aggregate_meta_json.py` walks `meta_logs/<rep>/<exp>/*_meta.json`, averages
the per-epoch curves across runs, and writes the following per-experiment
plots into each experiment folder:

| File | What it shows |
|---|---|
| `aggregated_view_ratios.png` | The original 5-bucket selection-ratio curves (Expanded / Expanded-like / Foreshortened / Foreshortened-like / Remainder) over training. |
| `aggregated_view_canonicality.png` | **NEW**. Collapsed 2-curve view: `canonical = Expanded + Expanded-like` (green) vs `foreshortened = Foreshortened + Foreshortened-like` (red), with dashed horizontal lines at each bucket's chance-baseline share (3.5% / 14% / 1.7% / 7% / 73.8% of available views) and a vertical line marking the first epoch where canonical > foreshortened. ±SEM shaded across runs. |
| `aggregated_view_canonicality_lift.png` | **NEW**. Same collapsed comparison plotted as **lift over chance** (`observed_share / available_share`). Horizontal reference at `1.0` = uniform random. Magnitude-aware view of preference strength independent of bucket sizes. |
| `aggregated_per_class_accuracy.png` | **RESTYLED**. Per-class accuracy over epochs as a `class × epoch` viridis heatmap (was: 32 overlapping line curves). Much easier to spot which classes the agent is learning earliest. |
| `aggregated_accuracy.png`, `_3.png`, `_5.png` | Per-view-type accuracies for N=1/3/5 view sets (unchanged). |
| `aggregated_summary.json` | Adds a `canonicality` block: bucket priors, mean canonical/foreshortened/remainder curves, lift curves, and the crossing epoch. |
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

# Sweep every stage-2 experiment in meta_logs/ in one go
./run_temporal_test_all.sh                # single GPU
GPUS=4 ./run_temporal_test_all.sh         # round-robin across 4 GPUs
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
   canonicality / lift / per-class-accuracy plots above.
4. **Probe view-type informativeness.** `python zero_shot_view_type_test.py`
   for the MVCNN classifier's per-bucket accuracy.
5. **Probe model uncertainty on agent selections.** In `human_multiview-main/`:
   `./scripts/run_pipeline.sh` for the VGGT/DINOv2 agent-vs-random
   comparison, `./scripts/run_view_type_single_view.sh` for VGGT's
   single-view depth confidence per bucket, `./scripts/run_pair_confidence_control.sh`
   for the identical / random / bucket-pair sanity check.

See `human_multiview-main/README.md` for the probe scripts that consume the
`selection.json` files produced by step 2.
