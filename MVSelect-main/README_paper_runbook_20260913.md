# Superseded execution instructions

**Use [README_paper_actions.md](README_paper_actions.md) for current commands and return files.** GPU troubleshooting, CLIP and perturbation epochs 10/30/100 are complete. Do not repeat the old diagnostic or retry steps below. The text below is retained only as a historical record.

---

# Exact next actions — 13 September 2026

The A09 preflight passed. Its first inference attempt failed while initializing
CUDA, before evaluating a model. No new scientific result is recorded.

## 1. Install this update

Transfer `paper_runbook_update_20260913.zip` to the remote MVSelect-main folder.
These changes are also installed in the actual local repository. From the remote
repository, run:

```bash
cd /nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/MVSelect-main
unzip paper_runbook_update_20260913.zip
python3 paper_runbook_update_20260913/install_update.py --repo .
```

The installer checks existing files, backs up files it changes, and refuses to
overwrite an unexpected version. It does not alter data, checkpoints or results.
If it reports a version conflict, send that message; do not force the overwrite.

**Training change:** future `main.py` runs use `--td_target_mode legal` by default.
Their folder names contain `tdlegal`. `--td_target_mode legacy` explicitly retains
the old future-credit calculation. Existing saved models are unaffected. The
verification launcher records the chosen mode and runs in separate directories.

## 2. Diagnose the CUDA failure first

In the same `hank3d` environment and GPU allocation you use for training:

```bash
python3 diagnose_paper_gpu.py --gpu-id 0 --output compare/paper_gpu_diagnostic_20260913.json
```

Send `compare/paper_gpu_diagnostic_20260913.json`. This records the NVIDIA GPU
status, the Python/PyTorch environment, visible devices and a tiny CUDA tensor
test. It changes no driver or device-visibility setting.

- If the tensor test passes, proceed below.
- If NVIDIA cannot communicate with a GPU, this needs the server/GPU allocation
  to be restored; changing the paper's evaluation settings cannot repair it.
- If NVIDIA sees GPUs but PyTorch fails, the report distinguishes build,
  visibility and initialization information. Send it before changing packages.
- If your server uses a scheduler, run inside the GPU allocation. Keep the
  scheduler's `CUDA_VISIBLE_DEVICES` value. `--gpu-id 0` means the first GPU visible
  to that process, which need not be physical GPU 0.

The warning's suggestion about changing `CUDA_VISIBLE_DEVICES` is one possible
cause, not a diagnosis. The failed evaluator does not modify that variable.
Official references: [PyTorch CUDA environment variables](https://docs.pytorch.org/docs/main/cuda_environment_variables.html),
[NVIDIA device enumeration](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).

## 3. A09 / E12: resume the perturbation evaluation

The old failed attempt may have created its output directory. Leave that record
in place. First run this small test, which evaluates only one checkpoint and two
object/draw trials:

```bash
python3 evaluate_paper_perturbations.py --gpu-id 0 --smoke-test --output compare/paper_perturbations_smoke_20260913
```

If it completes, run the full evaluation:

```bash
python3 evaluate_paper_perturbations.py --gpu-id 0 --output compare/paper_perturbations_retry1_20260913
```

Return these three files from `compare/paper_perturbations_retry1_20260913/`:

- `trial_results.csv`
- `run_summary.csv`
- `evaluation_plan.json`

This is new inference from existing models: 20 trained runs, classifier epochs
10/30/100, identical random five-image inputs, and identical perturbations across
models. It does not retrain or reproduce the old pooled-input perturbation plot.
The smoke output is labelled as a test and must not enter the paper's results.

## 4. E01: correct future credit and measure whether it changes the main findings

Example: a selector limited to side views must not receive future credit for
choosing a forbidden front view. The old trainer can do that because its
future-value maximum includes every camera. The new calculation uses the same
available-camera rule as action selection. Already used cameras are excluded
until a permitted pool is exhausted; the existing repeat-within-pool rule is
preserved. Terminal transitions receive zero future credit.

The correction is in `paper_td_target.py`, called at both training sites in
`src/trainer.py`; `main.py` records the mode. No additional source inspection by
the author is needed. The local and supplied server source both contain the old
unmasked calculation; the exact historical job revision is still unarchived.

Run the CPU checks in your research environment; no GPU is required:

```bash
python3 test_paper_td_target.py
```

Then inspect the exact six-job plan without starting training:

```bash
python3 run_paper_training_checks.py --suite e01 --seeds 0 --conditions free expanded_family freeze_10 --modes legacy legal --output compare/paper_e01_pilot_20260913
```

Once CUDA and the tests pass, start that plan:

```bash
python3 run_paper_training_checks.py --suite e01 --seeds 0 --conditions free expanded_family freeze_10 --modes legacy legal --gpu-id 0 --output compare/paper_e01_pilot_20260913 --execute
```

**This command trains six new models for 100 epochs each, sequentially.** It uses
the paper's 32 categories, 25 training instances per category, ImageNet ResNet-18,
five actions, original learning rates and penalties, and identical explicit seed
and deterministic settings within each corrected/original pair. It changes only
the target mode within a pair. It saves initial weights and checkpoints every
10 epochs; it does not save redundant per-image feature arrays.

Each job has `console.log` under its own folder, for example:
`compare/paper_e01_pilot_20260913/free_legacy_seed0/console.log`.

Return:
`compare/paper_e01_pilot_20260913/results_to_return.tar.gz`.
This small archive includes metadata, console logs, source snapshots, the full
plan and checkpoint records, including a tensor-content hash of initial weights.
Keep checkpoints and large `*_selection.json` files on the server.
If a job fails, the launcher stops; send its `console.log` and `training_plan.json`.

A single seed tests execution and provides a first effect estimate. It cannot
establish that the correction is harmless. After auditing that batch, the same
comparison can extend to four additional seeds with a new output directory:

```bash
python3 run_paper_training_checks.py --suite e01 --seeds 1 2 3 4 --conditions free expanded_family freeze_10 --modes legacy legal --gpu-id 0 --output compare/paper_e01_followup_20260913 --execute
```

**Do not start this 24-model follow-up yet.** First return the pilot. If the
correction changes conclusions, all retained affected restriction/freezing
conditions need corrected results; the first three conditions alone do not
validate the whole paper. The launcher supports all original restriction pools
and freezing epochs 10, 20, 30, 40 and 50. We will use those results to set the
remaining scope, rather than issuing an automatic 50-model rerun now.

## 5. E03: equal image counts, using existing checkpoints

This addresses a specific claim about selected versus random images: the
standard policy supplies its initial image plus five actions, whereas the old
random baseline has five images total. Changing test-image count can change
accuracy even though training is unchanged.

The new evaluator compares both methods at **six images total**, with the same
initial image. It also compares them at **five images total** using the first
four policy actions. The latter is a shorter evaluation of a five-action policy,
not a new policy trained for four actions. It uses all five unrestricted final
checkpoints and three initial-view draws per held-out object. Both methods see
the same objects and initial images; random additional views are shared across
models. It does not equalize feature-extraction compute or prove efficiency.

```bash
python3 evaluate_paper_view_budget.py --preflight --output compare/paper_view_budget_preflight_20260913
python3 evaluate_paper_view_budget.py --gpu-id 0 --smoke-test --output compare/paper_view_budget_smoke_20260913
```

If both succeed:

```bash
python3 evaluate_paper_view_budget.py --gpu-id 0 --output compare/paper_view_budget_20260913
```

Return `trial_results.csv`, `run_summary.csv`, `evaluation_plan.json` from
`compare/paper_view_budget_20260913/`. Run this if retaining an equal-budget
policy-versus-random claim. It is useful but lower priority than E01. Comparisons
between free, frozen and restricted models under the existing common random-five
test already have equal test-input counts.

## 6. Every other experiment ID: what to do now

| ID | Author action now | What remains / why |
|---|---|---|
| E02 | None; resolved. | The selected-image-only control is verified in all 15 existing runs. |
| E04 | No new command and no extra core seeds. | I will calculate uncertainty from the five existing core runs and three replay seeds. Old multiview seeds are unspecified, so those conditions cannot be presented as paired by seed. E01 reruns address the target correction, not a missing-repetitions request. |
| E05 | Defer; no command assigned. | Matching candidate-pool size and angular coverage would test whether geometry itself causes a restriction effect. Required only for that stronger claim; the present intervention compares the actual sampling regimes. |
| E06 | Keep the new `model_e0.pth` files produced by E01. | They preserve initialization. A frozen-backbone comparison remains optional. Do not use `--freeze_backbone` alone as a validated E06 command: the current active-training branch can re-enable backbone gradients. An untrained greedy selector also has tied initial action values, so a first-camera tie is not a learned viewing preference. |
| E07 | Defer; no command assigned. | Withhold camera directions during training only if claiming transfer to directions unavailable during learning. Current results concern different viewing conditions within the candidate grid. |
| E08 | Defer; no server command assigned. | Angular-distance analysis describes where errors occur. It is an optional analysis, not another training requirement. |
| E09 | None now; existing replay records are received. | The fixed-mixture control is unrun. Existing shuffled and random-warm-up results must first be fully used; warm-up is not missing new work. |
| E10 | Defer; no command assigned. | A companion-view intervention would test complementarity directly. It is needed to claim that mechanism, not to report the observed bias and transfer differences. |
| E11 | Defer; no command assigned. | Additional architectures/datasets broaden the scope. No result is implied by available TinyViT code. |
| E12 | Run A09 above once. | This is the same matched perturbation evaluation, not a second assignment. Severity curves and geometric-error tests remain optional. |
| E13 | Recommended next central experiment, after E01; not implemented or assigned yet. | Cross an early/late recognizer with early/late sampling experience, then compare learning gains on identical test views. This can distinguish learner-stage compatibility from a late policy being poor for every learner. A valid implementation must branch from shared recipient states and match optimizer/update budgets; the existing replay wrapper does not already do this. |
| E14 | Optional runnable alternative below; do not launch automatically. | One selector action in both conditions separates the effect of aggregating two images from taking more actions. |
| E15 | None; completed locally from existing selections. | Geometric flags have already been counted independently of historical family labels. |

If choosing E14 later, this exact command trains five matched seed pairs (ten new
100-epoch models); omit `--execute` to print the plan first:

```bash
python3 run_paper_training_checks.py --suite e14 --seeds 0 1 2 3 4 --gpu-id 0 --output compare/paper_e14_readout_20260913 --execute
```

Return `compare/paper_e14_readout_20260913/results_to_return.tar.gz`.
Do not call this a reciprocal switching experiment: it compares training with
one selected classifier image against initial-plus-selected classifier images.

## 7. Other information still needed

- A10–CLIP, only if retaining the SI probe: after CUDA works, run
  `python3 run_paper_clip_probe.py --gpu-id 0 --output compare/paper_clip_probe_20260913`.
  Return `results.csv`, `probe_config.json`, `probe.log` from that directory.
- A11 remains deferred: Hank's research affiliation/present address,
  correspondence, contributions, funding, competing interests and release links.
- No additional A01–A08, original image, CKA or VGGT file transfer is requested.
- Main/SI numerical reconciliation, figures, source-data assembly and final word
  limit editing remain my work. The latest assembled manuscript is unchanged.

## Validation and limits

Local CPU checks pass for legal targets, both trainer call sites, camera ordering,
deterministic equal-budget draws, matched training plans and synthetic ResNet-18
recognizer forward/checkpoint loading. Python 3.9 syntax is checked. End-to-end
server GPU inference and full training are not tested here and remain pending.
The diagnostic can identify a GPU problem; this code update cannot promise to
repair the server's driver or allocation state.
