# E01: 12 replications across servers or separate GPU jobs

The author confirms that the 12 GPUs are spread across servers or separate allocations. Run one independent model per allocated GPU. These are the same 12 previously assigned corrected-target replications, not additional experiments. Training settings are unchanged. Seed 0 and the original-target models are already complete.

## Install once on the shared filesystem

Transfer paper_e01_parallel_20260914.zip to the repository directory and use the GPU environment that completed the pilot. On any server sharing this repository:

```bash
cd /nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/MVSelect-main
unzip -n paper_e01_parallel_20260914.zip
python3 test_paper_e01_parallel.py
```

Continue when the tests report OK. They use fake CPU jobs, including an intentional failure; these are software tests, not model-training results. The package adds a parallel wrapper, its test and this guide to the existing repository. It does not change the existing training launcher or model code.

## Launch one numbered run in each single-GPU allocation

The commands below assume each allocation exposes one GPU, so its logical index is 0. If you instead have several GPUs visible on one server, assign a different visible --gpus index to each simultaneously running command on that server, or use the grouped example below. GPU indices restart on each server/allocation. Do not paste all 12 commands into one single-GPU terminal; that would run them sequentially. Start each in its own allocated GPU session or scheduler job. The wrapper does not reserve GPUs, connect to other servers or submit scheduler jobs.

Run from MVSelect-main in every session. Use each job number exactly once across all servers. The output directory identifies the job on the shared filesystem.

### Job 01 — free, seed 1

```bash
python3 run_paper_e01_parallel.py --job-ids 1 --gpus 0 --output compare/paper_e01_job01_20260914 --execute
```

### Job 02 — free, seed 2

```bash
python3 run_paper_e01_parallel.py --job-ids 2 --gpus 0 --output compare/paper_e01_job02_20260914 --execute
```

### Job 03 — free, seed 3

```bash
python3 run_paper_e01_parallel.py --job-ids 3 --gpus 0 --output compare/paper_e01_job03_20260914 --execute
```

### Job 04 — free, seed 4

```bash
python3 run_paper_e01_parallel.py --job-ids 4 --gpus 0 --output compare/paper_e01_job04_20260914 --execute
```

### Job 05 — expanded_family, seed 1

```bash
python3 run_paper_e01_parallel.py --job-ids 5 --gpus 0 --output compare/paper_e01_job05_20260914 --execute
```

### Job 06 — expanded_family, seed 2

```bash
python3 run_paper_e01_parallel.py --job-ids 6 --gpus 0 --output compare/paper_e01_job06_20260914 --execute
```

### Job 07 — expanded_family, seed 3

```bash
python3 run_paper_e01_parallel.py --job-ids 7 --gpus 0 --output compare/paper_e01_job07_20260914 --execute
```

### Job 08 — expanded_family, seed 4

```bash
python3 run_paper_e01_parallel.py --job-ids 8 --gpus 0 --output compare/paper_e01_job08_20260914 --execute
```

### Job 09 — freeze_10, seed 1

```bash
python3 run_paper_e01_parallel.py --job-ids 9 --gpus 0 --output compare/paper_e01_job09_20260914 --execute
```

### Job 10 — freeze_10, seed 2

```bash
python3 run_paper_e01_parallel.py --job-ids 10 --gpus 0 --output compare/paper_e01_job10_20260914 --execute
```

### Job 11 — freeze_10, seed 3

```bash
python3 run_paper_e01_parallel.py --job-ids 11 --gpus 0 --output compare/paper_e01_job11_20260914 --execute
```

### Job 12 — freeze_10, seed 4

```bash
python3 run_paper_e01_parallel.py --job-ids 12 --gpus 0 --output compare/paper_e01_job12_20260914 --execute
```

All 12 can run at the same time. Leave the current corrected-target training recipe unchanged. The launcher checks GPU access, records progress in parallel_plan.json, logs each run and collects the result archive. It refuses existing output folders and detects already-started requested replications in nearby paper_e01_* folders.

## Return one combined archive after the jobs finish

Each numbered output folder gets e01_parallel_results_to_return.tar.gz. From MVSelect-main, on any server sharing the filesystem, run this once after all 12 jobs have stopped:

```bash
tar -czf compare/paper_e01_12jobs_results_20260914.tar.gz -C compare \
  paper_e01_job01_20260914/e01_parallel_results_to_return.tar.gz \
  paper_e01_job02_20260914/e01_parallel_results_to_return.tar.gz \
  paper_e01_job03_20260914/e01_parallel_results_to_return.tar.gz \
  paper_e01_job04_20260914/e01_parallel_results_to_return.tar.gz \
  paper_e01_job05_20260914/e01_parallel_results_to_return.tar.gz \
  paper_e01_job06_20260914/e01_parallel_results_to_return.tar.gz \
  paper_e01_job07_20260914/e01_parallel_results_to_return.tar.gz \
  paper_e01_job08_20260914/e01_parallel_results_to_return.tar.gz \
  paper_e01_job09_20260914/e01_parallel_results_to_return.tar.gz \
  paper_e01_job10_20260914/e01_parallel_results_to_return.tar.gz \
  paper_e01_job11_20260914/e01_parallel_results_to_return.tar.gz \
  paper_e01_job12_20260914/e01_parallel_results_to_return.tar.gz
```

Return exactly:

```text
compare/paper_e01_12jobs_results_20260914.tar.gz
```

This packages only the small result archives. Checkpoints and large selection exports remain on the server. No additional copy of the original expanded-family selection file is needed; its supplied compressed copy is verified and that request is closed.

If a job fails, retain its archive and send the error; do not rerun all jobs or reuse the output directory. If an interrupted job has no archive, collect its existing outputs after it has stopped, for example for job 01:

```bash
python3 run_paper_e01_parallel.py --output compare/paper_e01_job01_20260914 --collect-only
```

Substitute the actual numbered output folder. This only packages saved records; it never starts training. Collection returns a nonzero exit status for an incomplete model but still writes an archive. If it cannot collect, send that job's parallel_plan.json and *_launcher.log. A GPU preflight failure occurs before an output folder is made; send the terminal error in that case.

## Optional grouping when one server allocation exposes several GPUs

For example, if one allocation exposes four GPUs, these four jobs can use one command:

```bash
python3 run_paper_e01_parallel.py --job-ids 1 2 3 4 --gpus 0 1 2 3 --output compare/paper_e01_groupA_20260914 --execute
```

Assign disjoint remaining job IDs to the other allocations, with their own visible GPU indices and distinct group output folders. This replaces the individual commands for those IDs; do not launch both. For grouped execution return each group's e01_parallel_results_to_return.tar.gz, preserving the parent folder names. The 12-folder collection command above applies only to the numbered single-job layout.

## Expected speed and verification

Twelve simultaneously allocated GPUs can run the 12 models in one wave. Compared with three GPUs each running four seeds sequentially, the ideal wall time is about one quarter. This is an estimate, not a measured benchmark. The slowest model and queue delays determine completion time; shared storage and CPU/RAM availability can limit throughput. Each RGB run uses eight data-loader workers. If several models share one server, allow for those workers and their memory use.

The wrapper respects CUDA_VISIBLE_DEVICES without changing it. There is no distributed training within a model. Omitting --execute prints a plan and starts nothing. Local CPU tests cover assignment, queueing, failed-job collection and duplicate-run rejection; no new GPU experiment has been run by the assistant.
