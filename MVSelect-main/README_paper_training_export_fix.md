# Continue E01 after the final plotting failure

The received E01 archives show that all three original-target models finish 100 training epochs. The final plotting routine then requests a camera-table filename that is absent from the isolated run folder. The exception prevents the metadata export and stops the launcher before the corrected-target model starts. The original weights are retained. **Do not repeat the original-target training.** E03 and E12 are completed and independently audited; do not rerun them.

The fix uses the dataset’s actual camera mapping, writes the numerical summary before final plots, and records exact measurements after every epoch. A diagnostic plotting failure is reported in `plot_error.txt` without discarding the numerical summary. The training loss, target calculation, optimizer, seed, batch size and training schedule are unchanged by this export fix. Existing run snapshots remain untouched.

## Install

Transfer `paper_training_export_fix_20260914.zip` into the remote MVSelect-main repository. In the usual GPU environment:

```bash
cd /nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/MVSelect-main
unzip paper_training_export_fix_20260914.zip
python3 paper_training_export_fix_20260914/install_update.py --repo .
python3 test_paper_training_exports.py
```

The short tests exercise the actual metadata-writing block with a forced plotting exception, full-precision epoch recording, and real plots made without the old pose-table file. Run training only if they pass. The installer backs up known versions and stops before changing files if it finds unexpected edits; send that error instead of forcing an overwrite.

## Train only the three corrected-target models

Run one command per terminal if three GPUs are visible as indices 0, 1 and 2. Each command trains one model for 100 epochs, using seed 0 and `--modes legal`. If only one GPU is available, run the same three commands sequentially, using `--gpu-id 0` each time. A scheduler allocation that exposes only one GPU also uses logical index 0.

Terminal 1:

```bash
python3 run_paper_training_checks.py --suite e01 --seeds 0 --conditions free --modes legal --gpu-id 0 --output compare/paper_e01_legal_free_20260914 --execute
```

Terminal 2:

```bash
python3 run_paper_training_checks.py --suite e01 --seeds 0 --conditions expanded_family --modes legal --gpu-id 1 --output compare/paper_e01_legal_expanded_family_20260914 --execute
```

Terminal 3:

```bash
python3 run_paper_training_checks.py --suite e01 --seeds 0 --conditions freeze_10 --modes legal --gpu-id 2 --output compare/paper_e01_legal_freeze_10_20260914 --execute
```

Return these three archives, rather than the whole run folders:

- `compare/paper_e01_legal_free_20260914/results_to_return.tar.gz`
- `compare/paper_e01_legal_expanded_family_20260914/results_to_return.tar.gz`
- `compare/paper_e01_legal_freeze_10_20260914/results_to_return.tar.gz`

Keep checkpoints and large selection files on the server. If a run fails, return its archive if created, or its `training_plan.json` and job `console.log`. The output folders above are new; they preserve the existing original-target runs. If an output folder already contains a completed legal run, return its archive instead of training a duplicate.

## Recovery and remaining interpretation

The original runs’ console files contain all 100 epochs of policy, all-view, random and restricted-input accuracy, rounded to 0.1 percentage point. Those are saved separately and labelled as rounded; they are not presented as a recovered full-precision `_meta.json`. Exact selection proportions can be recovered from the original selection files. The checkpoints allow new, explicitly labelled evaluations if needed. The old full-precision per-epoch accuracy arrays are not saved, so they cannot be recreated exactly from rounded console text.

The three legal runs complete the diagnostic pilot; they do not establish a five-seed replication. Decide further repetitions after comparing the corrected and original versions. No further E03, E12, CLIP, rendering or GPU-diagnostic command is assigned.
