<div align="center">
<h1>Human-level 3D shape perception emerges from multi-view learning</h1>

<a href="https://tzler.github.io/human_multiview/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://huggingface.co/datasets/tzler/MOCHI"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MOCHI-blue" alt="Dataset"></a>
<a href="https://github.com/tzler/human_multiview"><img src="https://img.shields.io/badge/GitHub-Code-black?logo=github" alt="Code"></a>
<a href="https://arxiv.org/abs/2602.17650"><img src="https://img.shields.io/badge/arXiv-2602.17650-b31b1b" alt="arXiv"></a>


[Tyler Bonnen](https://tzler.github.io/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/)

**UC Berkeley**

</div>

<p align="center">
  <img src="https://tzler.github.io/human_multiview/static/media/main_results.png" width="80%" alt="Main results">
</p>

## Overview

We demonstrate that multi-view transformers are the first vision models to match human performance on 3D shape inference tasks. First we develop a zero-shot evaluation framework (i.e., no task-specific training, fine tuning, or linear decoders) to determine the perceptual abilities of these multi-view models, then we evaluate on the [MOCHI](https://huggingface.co/datasets/tzler/MOCHI) benchmark. [VGGT](https://github.com/facebookresearch/vggt) matches human accuracy, error patterns, and reaction times.


This repository provides the complete analysis pipeline to evaluate models, reproduce results  figures, and explore cross-image attention patterns.

## Quick Start

```bash
# clone with submodules
git clone --recursive https://github.com/tzler/human_multiview.git

# move into the repo and reate conda environment
conda env create -f environment.yml
conda activate human_multiview

# Install this package
pip install -e .
```

## Usage

The repo is a complete pipeline: **scripts generate data, notebooks visualize data**.

### 1. Run model evaluations

```bash
# All models on all MOCHI trials
python scripts/run_evaluation.py --gpu_id 0

# Specific models
python scripts/run_evaluation.py --models vggt dinov2 --gpu_id 0

# Quick test (10 trials)
python scripts/run_evaluation.py --models vggt --n_trials 10 --gpu_id 0
```

### 2. Extract attention data (optional, for Fig 4)

```bash
python scripts/run_attention.py --gpu_id 0
```

### 3. Generate figures

| Notebook | Figures | GPU required |
|----------|---------|:---:|
| `notebooks/01_main_results.ipynb` | Fig 3, S1–S6 | No |
| `notebooks/02_layer_analysis.ipynb` | Fig 2 (bottom), S7 | Yes (for Fig 2) |
| `notebooks/03_attention_visualization.ipynb` | Fig 4, S8, S9 | Yes |

## Models

| Model | Type | Evaluation metric |
|-------|------|-------------------|
| [VGGT-1B](https://github.com/facebookresearch/vggt) | Multi-view | Depth confidence |
| [DUST3R](https://github.com/naver/dust3r) | Multi-view | Depth confidence |
| [MAST3R](https://github.com/naver/mast3r) | Multi-view | Descriptor confidence |
| [Pi3](https://github.com/jjordanoc/Pi3) | Multi-view | Depth confidence |
| DINOv2-Large | Single-image | Cosine similarity |
| VGGT-1B (untrained) | Multi-view | Depth confidence / layer similarity |

All models are evaluated zero-shot using a pairwise encoding strategy: for each trial, we encode all image pairs, extract confidence/similarity scores, and predict the oddity as the image with the lowest mean pairwise score.

An untrained (randomly initialized) VGGT baseline is also available to confirm that performance is attributable to learned representations rather than architecture or input features. It performs at chance on all metrics and layers. See `models/vggt_untrained.py` and run with `--models vggt_untrained --output_dir results/untrained/`.

<details>
<summary>Project structure</summary>

```
human_multiview/          Python package
├── config.py             Constants and model registry
├── data.py               MOCHI dataset loading
├── models/               Model wrappers (one per model)
│   ├── base.py           Abstract base class
│   ├── vggt.py           VGGT-1B
│   ├── dust3r.py         DUST3R
│   ├── mast3r.py         MAST3R
│   ├── pi3.py            Pi3
│   ├── dinov2.py         DINOv2 baseline
│   └── vggt_untrained.py VGGT-1B (random init)
├── evaluate.py           Oddity prediction, confidence margins, solution layers
├── attention.py          Cross-image attention extraction and visualization
└── plotting.py           Figure styling

scripts/                  Data generation
├── run_evaluation.py     Run all models → CSVs
└── run_attention.py      Extract attention maps → NPZs

notebooks/                Figure generation
├── 01_main_results.ipynb
├── 02_layer_analysis.ipynb
└── 03_attention_visualization.ipynb

repos/                    Model repos (git submodules)
├── vggt/
├── dust3r/
├── mast3r/
└── pi3/
```

</details>

---

## MVSelect probing extensions

This fork adds probes that use VGGT / Pi3 / DUSt3R / MASt3R / DINOv2 to evaluate
view selections produced by the MVSelect agent (see `MVSelect-main/`), and to
characterize view-type informativeness intrinsic to a ModelNet dataset.

The original paper's Fig. 3-left is a **MOCHI oddity-task accuracy** figure:
`scripts/run_evaluation.py` loads MOCHI and predicts the odd image from
pairwise model scores. The MVSelect extension below is a related but different
analysis: it keeps the dataset as ModelNet and asks whether the views selected
by a trained selector produce higher geometric confidence than random views.
Use the MOCHI command when evaluating perception models such as VGGT/Pi3/DINOv2;
use the MVSelect sweep when comparing selector policies such as `no_freeze`,
`freeze_10`, or `selview_expanded_family`.

All scripts share two hardcoded defaults that point at the local data layout:
- `--data_root /nfs/wattrel/.../modelnet_32_60_1_23`
- `--selection_dir /nfs/wattrel/.../MVSelect-main/meta_logs/rgb/...` (when needed)

Override either on the CLI when relevant. All scripts support `--shard i/N` for
multi-GPU parallelism, plus a `--from_csv` plot-only mode for merging shard CSVs
into final figures.

### 1. Agent vs random — view-set uncertainty pipeline

For each ModelNet instance, build (a) the **agent's top-K selected views** and
(b) a **same-size random subset** of the instance's views, then run each through
VGGT / DINOv2 (and optionally Pi3, DUSt3R, MASt3R) to compute set-level
uncertainty. The random subset is deterministic per instance — same for every
epoch group — so any trend over epochs is attributable to the agent's policy,
not random-baseline noise.

Pipeline:

```bash
cd human_multiview-main

# 4-GPU parallel run, full 10 epoch groups, --selected_view_type 01234 (all buckets)
./scripts/run_pipeline.sh

# Smoke test on 2 GPUs with 5 trials
GPUS=2 LIMIT=5 ./scripts/run_pipeline.sh

# Different selected_view_type filter
VIEW_TYPE=01 ./scripts/run_pipeline.sh         # expanded-family buckets only
VIEW_TYPE=23 ./scripts/run_pipeline.sh         # foreshortened-family buckets only
```

To compare several MVSelect training conditions, use the sweep wrapper. It
keeps the freeze sweep and selector-limit sweep separate so the plots do not
mix conceptually different manipulations.

```bash
GPUS=4 MODELS="vggt dinov2" ./scripts/run_views_pipeline_sweep.sh
COMPARISON_SET=selector_limit GPUS=4 MODELS="vggt dinov2" ./scripts/run_views_pipeline_sweep.sh
```

Internally the wrapper runs the three steps in sequence:

```bash
# (a) Evaluate — sharded across GPUs. Writes per-shard CSVs.
python scripts/run_evaluation_views.py --models vggt dinov2 \
  --selected_view_type 01234 \
  --epoch_groups "1-10,11-20,21-30,31-40,41-50,51-60,61-70,71-80,81-90,91-100" \
  --num_cam 5 --split test --per_cls_instances 5 --shard 0/4

# (b) Aggregate — concat shards, group by (epoch_group, dataset).
python scripts/aggregate_views.py --csv results/views/v01234/*_views_shard*.csv \
  --output_dir results/views/v01234/summary/

# (c) Plot — three figures per (model, metric) with bucket-proportion subtitle.
python scripts/plot_epoch_trends.py \
  --summary results/views/v01234/summary/per_class_summary.csv \
  --output_dir results/views/v01234/plots/
```

Outputs land under `results/views/v{selected_view_type}/`:

```
v01234/
├── vggt_views_shard*.csv               # per-instance, per-epoch-group raw rows
├── dinov2_views_shard*.csv
├── summary/
│   ├── per_class_summary.csv           # aggregated by (epoch_group, class)
│   └── overall_summary.csv             # aggregated by epoch_group
└── plots/
    ├── vggt_pair_mean_v01234_per_class_grid.png      # 32 subplots, agent vs random over epochs
    ├── vggt_pair_mean_v01234_delta_heatmap.png       # class × epoch_group Δ heatmap
    ├── vggt_pair_mean_v01234_macro.png               # macro-mean across classes
    └── vggt_image_mean_v01234_* (VGGT/Pi3 only)
```

**Metrics emitted by `run_evaluation_views.py`** (all averaged over object masks
via `get_object_mask`, threshold-based silhouette):

| metric column | model regime | meaning |
|---|---|---|
| `<model>_agent_pair_mean` / `_random_pair_mean` | pairwise (all models) | Mean over C(N,2) pair-confidences, where each pair-conf = avg of the two per-view confidence maps. The paper's metric. |
| `<model>_agent_pair_min` / `_random_pair_min` | pairwise | Worst pair score — useful for "did the agent avoid any bad pairings?" |
| `<model>_agent_image_mean` / `_random_image_mean` | joint (VGGT, Pi3 only) | One forward pass on all N views; mean of per-view confidences. The set-level metric. |
| `<model>_agent_image_min` / `_random_image_min` | joint | Worst per-view confidence in the joint pass. |
| `dinov2_agent_sim_mean` / `_random_sim_mean` | DINOv2 only | Mean pairwise cosine similarity over patch tokens. |

For each CSV row we also write `agent_n_<bucket>` for the five view-type
buckets — totals of how many of the agent's selected views fell in each bucket
across all instances. The plot script renders this as a one-line subtitle on
every figure (e.g. `Agent selections (v01234): Expanded 32.4% (n=2147), ...`).

### 2. View-type single-view confidence probe

Per ModelNet view-type bucket — Expanded, Expanded-like, Foreshortened,
Foreshortened-like, Remainder — sample one random view per instance, run VGGT
(or Pi3) in **single-view mode** (N=1 input), and read the mean depth-precision
over the object mask. Repeats per (instance, bucket) for an error bar.

The bucket assignment uses MVSelect's filename rules (`'planar'` / `'short'` /
`'like'` substrings) — see `human_multiview/data.py::_classify_view_by_filename` —
so sampling is uniform over **all views of each type on disk**, independent of
which views the agent ever selected.

```bash
# 4-GPU parallel run, VGGT only, 5 runs, 25 instances per class
./scripts/run_view_type_single_view.sh

# More runs, more models
GPUS=4 MODELS="vggt pi3" N_RUNS=10 ./scripts/run_view_type_single_view.sh
```

Outputs to `results/single_view/`:
- `<model>_bar_overall.png` — 5 bars, mean ± std confidence per view type.
- `<model>_per_class_heatmap.png` — view_type × class heatmap of mean confidence.
- `<model>_per_class_grid.png` — small-multiples with one subplot per class.
- `<model>_single_view_confidence.csv` — raw per-run per-instance data.

**Interpretation caveat**: with N=1, VGGT's cross-view attention has no other
view to attend to, so `depth_conf` reflects a *learned monocular prior over
correspondence-friendliness* rather than a calibrated cross-view precision.
The within-regime ordering across buckets is meaningful; absolute values are
inflated relative to pair/joint regimes and not directly comparable.

### 3. Pair-confidence control test

For each instance, evaluate three categories of pair input to VGGT and compare
the resulting pair confidence:

- **Identical** — same view paired with itself (degenerate upper-bound check).
- **Random** — two distinct random views from the instance's full pool.
- **Bucket pair** — one view from bucket `b1` + one from bucket `b2`. Covers
  all 5 same-bucket and 10 cross-bucket combinations including
  Expanded + Remainder.

```bash
./scripts/run_pair_confidence_control.sh
```

Outputs to `results/pair_control/`:
- `<model>_bars.png` — 17 bars (identical | 5 same-bucket | 10 cross-bucket | random), color-coded.
- `<model>_heatmap.png` — 5 × 5 bucket × bucket pair-confidence heatmap with numeric annotations.
- `<model>_pair_control.csv` — raw data.

A "same-bucket pair" is two **different** views drawn from the same view-type
bucket, not the same view twice. The Identical column is the only condition
that feeds two identical images to the model.

### 4. Comparing VGGT confidence across experiments (e.g., a freeze sweep)

**Two-step workflow.** The final overlay plots from
`aggregate_vggt_confidence.py` consume **`overall_summary.csv` files that
must already exist for each experiment you want to compare.** Those files are
produced by running the per-experiment VGGT pipeline (step A below). Then the
aggregator (step B) overlays the curves onto shared figures.

This step requires that the MVSelect repo's `meta_logs/<dataset>/<exp>/`
folders already exist — i.e. stage-2 training has been run for each
experiment, producing the `*_selection.json` files the VGGT pipeline reads.

#### A. Run the per-experiment VGGT pipeline (once per MVSelect experiment)

For each MVSelect experiment you want to compare, the per-experiment pipeline
needs to write its own `overall_summary.csv` into a unique folder. Two ways:

**A.1 (recommended) — sweep wrapper.** Run one comparison family at a time.
The wrapper invokes
`run_pipeline.sh` per experiment with the right `SELECTION_DIR` /
`OUTPUT_DIR`, populating

```
results/views/v<VIEW_TYPE>/<exp_label>/
├── vggt_views.csv (and _shard*.csv)
├── dinov2_views.csv
├── summary/
│   ├── per_class_summary.csv
│   └── overall_summary.csv      ← what step B reads
├── plots/                       ← per-experiment plots
└── logs/                        ← per-shard logs
```

for every experiment.

```bash
cd human_multiview-main

# Freeze sweep: no_freeze vs freeze_10..freeze_50
./scripts/run_views_pipeline_sweep.sh

# Selector-limit sweep: select_all vs restricted selector policies
COMPARISON_SET=selector_limit ./scripts/run_views_pipeline_sweep.sh

# Smoke test on the first 10 trials per experiment
LIMIT=10 ./scripts/run_views_pipeline_sweep.sh
```

Each experiment runs sequentially and uses all `GPUS` GPUs internally via
sharding. With ~6 experiments and `GPUS=4`, expect roughly N × 15–20 minutes.
The selector-limit sweep reuses the unrestricted experiment output under
`results/views/v01234/no_freeze/`; the aggregation step labels it as
`select_all`.

**A.2 — manual one-experiment-at-a-time.** If you want to launch experiments
individually (e.g., on different machines), call `run_pipeline.sh` with the
new `SELECTION_DIR` and `OUTPUT_DIR` env vars:

```bash
SELECTION_DIR=/abs/path/to/MVSelect-main/meta_logs/rgb/<exp_folder> \
OUTPUT_DIR=results/views/v01234/no_freeze \
  ./scripts/run_pipeline.sh

SELECTION_DIR=/abs/path/to/MVSelect-main/meta_logs/rgb/freeze_10_<exp_folder> \
OUTPUT_DIR=results/views/v01234/freeze_10 \
  ./scripts/run_pipeline.sh
# ... once per experiment
```

The default `--selection_dir` baked into `run_evaluation_views.py` only
covers one specific path — every additional experiment **must** set
`SELECTION_DIR` explicitly, otherwise the wrapper just re-evaluates the same
default experiment under different output folders.

#### B. Overlay the experiments' curves

Once every experiment has a populated `summary/overall_summary.csv`, run the
aggregator. It reads each experiment's `overall_summary.csv` and overlays
the pair-level and set-level confidence curves:

```bash
# Freeze sweep: no_freeze vs freeze_10..freeze_50
./scripts/run_aggregate_vggt_confidence.sh

# Selector-limit sweep: select_all vs restricted selector policies
COMPARISON_SET=selector_limit ./scripts/run_aggregate_vggt_confidence.sh

# Many experiments → heatmap is much more readable than overlaid lines
STYLE=heatmap ./scripts/run_aggregate_vggt_confidence.sh

# Plot agent-over-random advantage instead of raw confidence
VALUE=delta_mean ./scripts/run_aggregate_vggt_confidence.sh

# Or call the python script directly
python3 scripts/aggregate_vggt_confidence.py \
  --summary results/views/v01234/no_freeze/summary:no_freeze \
  --summary results/views/v01234/freeze_10/summary:freeze_10 \
  --summary results/views/v01234/freeze_20/summary:freeze_20 \
  --model vggt \
  --output_dir compare/vggt_confidence_freeze_sweep
```

Outputs (in `--output_dir`):
- `<model>_pair_<value>_over_epochs.png` — pair-level VGGT confidence per
  experiment (or `_heatmap.png` for the heatmap version).
- `<model>_set_<value>_over_epochs.png` — joint set-level confidence
  (VGGT/Pi3 only — equivalent of the `image_mean` metric).
- `aggregated.csv` — concatenated raw rows for ad-hoc analysis.

Useful flags:
- `--value {agent_mean | delta_mean | macro_agent | macro_delta | ...}` —
  default `agent_mean` plots VGGT's confidence on the agent's selections;
  `delta_mean` plots agent-minus-random advantage.
- `--style {line,heatmap,sorted_bars,rank_stacked,both,all}` — `heatmap`
  makes many-experiment comparisons much more readable. **The bash wrapper
  defaults to `STYLE=heatmap`**; pass `STYLE=line` to fall back to line plots,
  or `STYLE=all` for every style.
  - `sorted_bars` is the **recommended rank-aware option**: grouped bars per
    epoch sorted left→right by value, with the y-axis showing the actual
    value (not a sum). Rank flips show up as a colour changing its horizontal
    position within an epoch group.
  - `rank_stacked` is the same idea but stacked vertically — more compact but
    the y-axis becomes a sum of values, less directly meaningful.
  - Each style writes its own filename suffix (`_heatmap.png`,
    `_sorted_bars.png`, `_rank_stacked.png`) so they never overwrite each other.
- `--bin_epochs N` — heatmap epoch axis collapsed to N bins.
- `--smooth N` — line-plot rolling mean window.
- `--title_suffix "..."` — extra title line.

#### Troubleshooting "no overall_summary.csv at ..."

If the aggregator prints `SKIP <label>: no overall_summary.csv at <path>`,
step A hasn't been run for that experiment yet (or was run with a different
`OUTPUT_DIR`). Two things to check:

1. Does `<path>/overall_summary.csv` exist? If not, run step A for that
   experiment — `run_views_pipeline_sweep.sh` is the most foolproof way.
2. Does `run_aggregate_vggt_confidence.sh`'s `SUMMARIES` list match the
   folder names that step A actually wrote? The default in this repo expects
   `results/views/v01234/<exp_label>/summary/` (matching the sweep wrapper).
   If you used a different layout, edit one of the two to match.

### Reproducing trend plots

The complete narrative for advisors is built from three independent probes:

| probe | script | what it measures |
|---|---|---|
| CLIP single-view classification | `MVSelect-main/clip_zero_shot_view_type.py` | semantic informativeness per view type |
| Single-view confidence (VGGT) | `view_type_single_view_confidence.py` | geometric informativeness per view type |
| Pair-confidence by composition | `pair_confidence_control_test.py` | calibration of the confidence metric itself |

If all three order the five buckets the same way
(`Expanded > Expanded-like > Remainder > Foreshortened > Foreshortened-like`),
that's the cleanest "expanded-family informativeness" story across two
independent models (one classification, one geometry) and an independent
calibration probe.

### Extension layout

```
human_multiview-main/
├── human_multiview/
│   ├── data.py
│   │   ├── load_modelnet_selected()       # agent vs random trial loader
│   │   ├── build_view_type_index()        # per-instance bucket → file list
│   │   ├── _classify_view_by_filename()   # mirrors MVSelect's labelling
│   │   └── _instance_rng()                # deterministic per-instance RNG
│   ├── models/
│   │   ├── vggt.py    → extract_single_view_confidence()
│   │   └── pi3.py     → extract_single_view_confidence()
│   └── evaluate.py
│       ├── evaluate_set_pairwise_confidence()
│       ├── evaluate_set_joint_confidence()
│       └── evaluate_set_dinov2_similarity()
└── scripts/
    ├── run_evaluation_views.py            # agent vs random eval (+ --shard, --epoch_groups)
    ├── aggregate_views.py                 # merge + per-class summaries
    ├── plot_epoch_trends.py               # 3 plots per (model, metric)
    ├── view_type_single_view_confidence.py
    ├── pair_confidence_control_test.py
    ├── run_pipeline.sh                    # multi-GPU wrapper (agent vs random)
    ├── run_view_type_single_view.sh
    └── run_pair_confidence_control.sh
```

---

## Citation

```bibtex
@misc{bonnen2026human,
      title={Human-level 3D shape perception emerges from multi-view learning}, 
      author={Tyler Bonnen and Jitendra Malik and Angjoo Kanazawa},
      year={2026},
      url={https://arxiv.org/abs/2602.17650}, 
}
```

## License

MIT
