# E18: why selected experience helps or hinders learning

Prepared September 16, 2026. **New experiment, not a result.** Local validation covers input construction, checkpoint compatibility and synthetic CPU execution. The actual dataset/GPU run must still be executed on the server.

## What this tests

Two explanations are distinguished, without assuming either is true:

1. **Insufficient coverage:** late selections repeatedly expose too narrow a range of appearances. Change their camera spread while preserving each object's five operational view-family counts and the number of unique images.
2. **Learner dependence:** the same image set induces different useful learning updates at different recipient stages. Branch from the same recipient state within each stage; compare fresh recipients with recipients given 30 epochs of identical random-view training.

The implementation combines the corrected individual-source replay requested in E16 with the learner-stage comparison in E13 and a new mechanism intervention, E18. It does **not** complete E05 (restricted active training with matched candidate pools), E10 (fixed focal image with changed companions), or E17 (matched random training in the full original active-learning setup).

## Install in the actual server repository

The new code is installed in the actual local MVSelect-main repository. Copy `paper_mechanism_20260916.zip` to this existing server directory:

`/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/MVSelect-main/`

Activate the same working `hank3d` environment used for the completed corrected-target models. From that directory:

```bash
cd /nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/MVSelect-main
unzip -o paper_mechanism_20260916.zip
python3 -m unittest test_paper_mechanism.py -v
```

The ZIP installs only new mechanism utilities and compact source-image lists. It does not change `main.py`, the selector, the original replay scripts, any old checkpoint, or any old result folder. It needs no additional Python library beyond the existing PyTorch/torchvision/Pillow environment. These tests use tiny synthetic inputs, not scientific data.

## Step 1 — verify the existing files (no GPU training)

```bash
python3 run_paper_mechanism.py --job-id 1 --preflight --output compare/paper_e18_preflight_20260916
```

This verifies image availability, camera directions, split separation and checkpoint hashes. The data root and exact corrected-source checkpoint paths are already supplied; no manual JSON inspection or metadata export is required. A failure names the exact missing/mismatching file. Send that error plus `preflight.json` if it exists; do not regenerate data or silently substitute checkpoints.

## Step 2 — one GPU software check

On **one allocated GPU**, using its logical device index 0:

```bash
python3 run_paper_mechanism.py --job-id 1 --gpu-id 0 --smoke-test --execute --output compare/paper_e18_smoke_20260916
```

This runs all six conditions at two tiny recipient stages, on a small subset for one adaptation epoch. It is explicitly labeled **not a scientific result**. It also checks strict loading of the existing source weights, source-recognizer scoring, branch restoration, gradient diagnostics and output assembly. The full summarizer refuses smoke results.

If it fails, return the console error and `compare/paper_e18_smoke_20260916/FAILED.json` if present. If it passes, continue to Step 3. No result interpretation or hyperparameter selection is needed before the full run. Existing output folders are never overwritten; do not rerun a completed command into the same folder.

## Step 3 — 15 independent GPU jobs

Each command below occupies **one GPU** and runs its 12 branches sequentially. The models do not communicate across machines. Your servers share the files, so install once. Use a different job number in each allocated job. Do not launch two commands on the same allocated GPU at the same time. Logical `--gpu-id 0` respects the scheduler's existing `CUDA_VISIBLE_DEVICES`; the script does not rewrite it.

Each full job comprises 30 warm-up epochs plus **6 conditions × 2 stages × 100 adaptation epochs**, with 160 training objects and 27 updates per epoch at batch size 6. This is 32,400 branch updates plus 810 warm-up updates per job. It is a substantial experiment, not a quick evaluation. The smoke test does not give a reliable full runtime estimate because its dataset is smaller.

Jobs 1–5 cover source seeds 0–4 with recipient seed 0; jobs 6–10 use recipient seed 1; jobs 11–15 use recipient seed 2. Across 12 GPU allocations, start jobs 1–12, then jobs 13–15 as allocations become available. Do not rerun the old E01 training.

```bash
python3 run_paper_mechanism.py --job-id 1 --gpu-id 0 --execute --output compare/paper_e18_20260916/job01
python3 run_paper_mechanism.py --job-id 2 --gpu-id 0 --execute --output compare/paper_e18_20260916/job02
python3 run_paper_mechanism.py --job-id 3 --gpu-id 0 --execute --output compare/paper_e18_20260916/job03
python3 run_paper_mechanism.py --job-id 4 --gpu-id 0 --execute --output compare/paper_e18_20260916/job04
python3 run_paper_mechanism.py --job-id 5 --gpu-id 0 --execute --output compare/paper_e18_20260916/job05
python3 run_paper_mechanism.py --job-id 6 --gpu-id 0 --execute --output compare/paper_e18_20260916/job06
python3 run_paper_mechanism.py --job-id 7 --gpu-id 0 --execute --output compare/paper_e18_20260916/job07
python3 run_paper_mechanism.py --job-id 8 --gpu-id 0 --execute --output compare/paper_e18_20260916/job08
python3 run_paper_mechanism.py --job-id 9 --gpu-id 0 --execute --output compare/paper_e18_20260916/job09
python3 run_paper_mechanism.py --job-id 10 --gpu-id 0 --execute --output compare/paper_e18_20260916/job10
python3 run_paper_mechanism.py --job-id 11 --gpu-id 0 --execute --output compare/paper_e18_20260916/job11
python3 run_paper_mechanism.py --job-id 12 --gpu-id 0 --execute --output compare/paper_e18_20260916/job12
python3 run_paper_mechanism.py --job-id 13 --gpu-id 0 --execute --output compare/paper_e18_20260916/job13
python3 run_paper_mechanism.py --job-id 14 --gpu-id 0 --execute --output compare/paper_e18_20260916/job14
python3 run_paper_mechanism.py --job-id 15 --gpu-id 0 --execute --output compare/paper_e18_20260916/job15
```

These are separate job commands, **not a block to paste into one terminal expecting parallel execution**. If fewer GPUs are available, run a later command after the previous command finishes. An interrupted job remains incomplete and is not silently resumed or counted as evidence. Retain its folder and send its error before relaunching under a new output name.

## Step 4 — summarize and return one archive

After all 15 jobs finish, run once (no GPU needed):

```bash
python3 summarize_paper_mechanism.py --root compare/paper_e18_20260916 --output compare/paper_e18_summary_20260916
```

Copy this exact file back to the Mac project-data `compare/` folder:

`compare/paper_e18_summary_20260916/E18_results_to_return.tar.gz`

The archive contains every job's raw per-object predictions, learning curves, source-recognizer scores, update diagnostics, input lists, provenance, branch audits and descriptive summaries. Keep the checkpoint `.pth` files on the server. Do not send the original approximately 1 GB selection files again.

## Fixed protocol

All sources are the five completed corrected-target unrestricted models, evaluated at source epochs 10 and 100. Image lists are reconstructed independently per source model from recorded action frequencies over initial cameras: the five most frequent distinct images per object, ties broken by filename. They are **not five-action rollouts** and exclude the initial observation, as in the earlier replay construction. Frequency cutoff ties are recorded. No source models are pooled.

Six conditions:

| Condition | Experience for each training object |
|---|---|
| `early` | Fixed five images ranked by the epoch-10 source model |
| `late` | Fixed five images ranked by the epoch-100 source model |
| `late_spread` | Five images with the late set's exact operational family counts and greater or equal mean angular separation |
| `late_cluster` | Five images with the same counts and smaller or equal angular separation |
| `random_fixed` | Five randomly chosen images, reused throughout training |
| `random_resampled` | Five random images, redrawn each training epoch |

The spread interventions search 256 random, family-matched sets per object and retain the widest/narrowest, including the actual late set among candidates. This is a bounded search, not the global optimum. Alternatives may share images with the actual set. Neither model accuracy nor test outcomes choose the sets. The original historical family parser is preserved. Those operational labels do not all correspond to planar geometry.

All fixed-set conditions use the same number of objects, images per object, total image presentations, object order and updates. The family counts match **per object**, not merely on average. Random-resampled versus random-fixed intentionally changes exposure to distinct images over time. It does not isolate repetition from cumulative coverage; both are recorded by the deterministic schedule in the input manifest and code.

Recipients use ImageNet-initialized ResNet-18 backbones from the verified epoch-0 source file and new category-classifier initializations with seeds 0–2. No trained source classifier or selector initializes a recipient. Within a recipient seed, initial weights and random warm-up images are shared across sources. At stages 0 and 30, all six branches restore identical weights, normalization buffers and complete Adam states. Adaptation uses max pooling of five images, batch size 6, Adam learning rate 5e-5, weight decay .01 and a matched 100-epoch cosine schedule. The 30-epoch warm-up uses the same Adam settings and constant learning rate. The learning rate restarts for every branch. Optimizer state is part of recipient stage; stage dependence does not isolate weights from optimizer history.

## Data separation and evaluation

- Replay training: the same 160 objects represented by source selection records, in the dataset's `test` directory. They were held out from original active training but are **training data for the fresh replay recipients**.
- Diagnostic objects: first two sorted objects per category among the five `test-down` objects (64).
- Endpoint evaluation: remaining three per category (96). Three fixed random five-image sets per object are shared by all conditions and jobs.
- Object mesh IDs are checked for overlap among these three roles. Evaluation conditions have identical image counts.
- These `test-down` objects were part of historical replay evaluation. The endpoint set is held out from this new training and diagnostics, **not a previously unexamined external dataset**. No final-test outcome chooses an image set, learning rate, checkpoint or stopping point.

### Why the update diagnostic is mechanistic, and its limit

At each recipient stage, compute the full-recognizer gradient `g_S` for one fixed training object per category (32 objects), separately for each image condition. Compute `g_V` on the 64 diagnostic objects under common random views. For a small plain SGD step, expected diagnostic loss reduction is approximately `eta * dot(g_V, g_S)`.

Record both gradient norms, the unnormalized dot product and cosine similarity. Apply a disposable step at **both** 1e-5 and 1e-4 and measure the actual loss reduction. All parameters and normalization buffers are restored afterward. Batch-normalization statistics are fixed during this diagnostic so changing running statistics cannot masquerade as a gradient effect. Full branch training updates normalization normally and uses Adam. Therefore the diagnostic is a local, plain-SGD test; it is **not an exact explanation of 100-epoch Adam behavior**. Diagnostic gradients never guide training or select examples.

Source recognizers at epochs 10 and 100 are also scored on the same five-image sets. These scores assess immediate recognition on the replay image banks, not their original six-image rollouts. They do not measure learning progress.

## Planned comparisons and interpretation

1. **Evidence connection (E16):** early versus late replay at adaptation epoch 100 in fresh recipients. Report all five individual sources and all three recipient seeds, whether the historical ordering replicates or reverses.
2. **Coverage intervention (E18):** `late_spread - late_cluster` at epoch 100, with `late_spread - late` as a rescue comparison. Inspect realized spread for each object. Improvement after widening the sets supports a role for the alternative visual evidence selected by this rule. It does not prove angular distance itself, rather than correlated visible shape/texture, is the unique mechanism.
3. **Learner-stage dependence (E13):** the stage-30 minus stage-0 change in the late-versus-early advantage, primarily using common-view cross-entropy at **20 additional epochs**, with accuracy secondary. Compare gains from each branch's own starting point; do not compare raw mature versus fresh accuracy. Epoch-100 contrasts are also saved. Inspect floor/ceiling effects and baseline competence; 30 warm-up epochs is a prescribed stage, not assumed maturity.
4. **Local explanation:** examine whether loss reduction and gradient agreement change with recipient stage and spread intervention. A matching long-horizon intervention effect is stronger evidence than a correlation alone.
5. **Exposure control:** random-resampled versus random-fixed at epoch 100 describes effects of continued variety under an equal image/update budget. It is not the matched full active-versus-random training experiment E17.

Multiple mechanisms can operate together. A stage interaction alone does not exclude coverage, saturation or optimizer effects. No interaction does not prove universal learner independence. A spread rescue is evidence about this intervention and model, not proof that all angular diversity helps. Null, mixed and reversed results are retained.

The summarizer first averages the three recipient contrasts within each source and reports the mean and sample SD across five source models. It retains all individual contrasts. Recipients are crossed with sources; do not treat 15 jobs or hundreds of object/view rows as 15 independent source models. It checks branch/input equality and records cross-source warm-up mismatches. It supplies **descriptive summaries**, not automatically selected significance tests or a mechanism conclusion. Final uncertainty and multiplicity handling remain assistant-owned analysis after receiving the results.
