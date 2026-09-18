# Superseded execution instructions

**Use [README_paper_actions.md](README_paper_actions.md) for current commands and return files.** GPU troubleshooting, CLIP and perturbation epochs 10/30/100 are complete. Do not repeat the old diagnostic or retry steps below. The text below is retained only as a historical record.

---

# Paper verification tools — 13 September 2026

Run from the remote repository in the same Python/GPU environment used for the
original experiments:

```bash
cd /nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/MVSelect-main
```

All files in this bundle are installed in the actual local MVSelect-main
repository. Transfer the bundle's contents into the remote MVSelect-main
directory. The earlier collector is already complete; do not run it again.

## A03 and A07

No command is needed. Original RGB examples, the server generator snapshot,
full-test descriptors and all five run trajectories have been received.
One object per category remains appropriate for an illustrative scatterplot.

## A09: a matched test of image perturbations

The old script pools historical choices into a varying number of images and
tests one final classifier. The new script instead tests all four core
conditions (unrestricted; freeze 10, 20, 30), five trained runs each, at saved
classifier epochs 10, 30 and 100. It uses the same 160 test objects, three random
five-image draws per object, and identical perturbations for every model and
checkpoint. It does not train a model or reproduce the old pooled-input figure.

First check the 60 checkpoint paths and the image files without inference:

```bash
python3 evaluate_paper_perturbations.py --preflight --output compare/paper_perturbations_preflight_20260913
```

If that succeeds, run the new evaluation:

```bash
python3 evaluate_paper_perturbations.py --gpu-id 0 --output compare/paper_perturbations_20260913
```

Settings are explicit in the saved plan: five images; three sampling repeats;
seed 42; in-plane rotation uniformly within ±10°; brightness/contrast/saturation
factors within 0.6–1.4; hue shift within ±0.1; rotation uses a white fill.
The four test conditions are clean, rotation, colour jitter, and both.
This is a specific perturbation control, not a test of unseen camera directions
or generic robustness. No model chooses its test images in this protocol.

Return these files from `compare/paper_perturbations_20260913/`:

- `trial_results.csv`
- `run_summary.csv`
- `evaluation_plan.json`

If preflight fails, return its exact error. Do not substitute another checkpoint
or rename an old result. To use another GPU, change only `--gpu-id`.

## A10–CLIP: new documented inference if the SI probe is retained

The original CLIP CSV was not found in the supplied server file index. This
command produces a new documented evaluation; it does not recover missing
historical numbers. It uses the existing CLIP script with its base-patch32 model,
the five-template prompt ensemble, 25 test objects per category, five sampling
repeats and seed 42. This matches the external VGGT probe's broader object scope,
not the core recognizer's five test objects per category.

```bash
python3 run_paper_clip_probe.py --gpu-id 0 --output compare/paper_clip_probe_20260913
```

Return `results.csv`, `probe_config.json` and `probe.log` from that directory.
No CLIP training occurs. Until these new values are available, keep the previous
CLIP numbers marked as awaiting their numerical source.

## Taxonomy reporting — already audited locally

`audit_paper_taxonomy.py` reads the collector's full filename manifest and only
the requested epochs of the large selection JSONs. It outputs separate
principal-planar, near-planar and major-axis-alignment flags while preserving
the original view-family labels. This prevents a label repair from silently
changing the pools used in historical training.

`paper_candidate_priors_test.json` contains the measured candidate fractions for
the original five labels. Use it with the existing plotting tool's
`--candidate-priors` option when rebuilding those plots. Do not replace the
five-label restricted-training results with a newly relabelled experiment.
No additional taxonomy command is requested from the author now.

## Validation and limits

The scripts pass Python 3.9 syntax checks and fixture checks covering requested
epoch extraction, missing-epoch rejection, deterministic distinct-view sampling,
perturbation bounds, all 160 test objects in preflight, overwrite rejection and
the CLIP configuration plan. The perturbation recognizer matches the repository's
ResNet-18 → spatial average → view maximum → category-linear-layer architecture;
all recognition checkpoint keys must load strictly. Selector weights are unused
because this evaluates common fixed inputs.

GPU inference has not been tested locally: this desktop runtime lacks PyTorch
and torchvision. Remote results remain unrun until the author returns outputs.
These evaluations do not resolve the separate E01 training-target concern.
