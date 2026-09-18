# E01: inspect existing replications before launching replacements

The author reports that the earlier four-seed commands were started and have all stopped. The new job-01 launcher finds `compare/paper_e01_legal_replication_free_20260914/free_legal_seed1` and stops before its GPU check. This guard checks for an existing run directory; it does not establish whether that model finished, failed or exported all of its results.

Do not delete/rename that directory or bypass the guard. It may contain reusable training results. The old three-condition schedule and the new twelve-job schedule request the same models.

## Your next command — CPU only

Transfer paper_e01_status_20260915.zip into MVSelect-main on the shared server filesystem. Install it once, then inspect from any server that can read the same compare directory. The checker uses only the Python standard library and needs no GPU allocation.

```bash
cd /nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/MVSelect-main
unzip -n paper_e01_status_20260915.zip
python3 test_paper_e01_status.py
python3 inspect_paper_e01_status.py --root compare --output compare/paper_e01_existing_work_20260915.json
```

The tests should report OK. Then the inspection command prints a 12-row status table and writes one report. **Return `compare/paper_e01_existing_work_20260915.json`.** You may also paste the table, but the report contains the exact paths and log tails needed to determine what to reuse.

No further original expanded-family selection export, checkpoint copy or new training command is requested now. Leave existing training files where they are. The report only reads plans, small metadata, epoch summaries and the last 6,000 bytes of each relevant console log. It does not read checkpoint tensors or the large selection JSONs. It creates only the requested report and refuses to overwrite an existing report.

## How the statuses are interpreted

| Status | Meaning |
|---|---|
| finished_export_recorded | The plan records exit code 0 and the final metadata reports 100 epochs with matching seed and corrected target. This is a completion indicator, not the full scientific audit. |
| failed_exit_recorded | The plan records a nonzero exit code. A failure after training may still leave useful completed checkpoints/results; inspect the log before deciding to retrain. |
| no_exit_recorded | A work directory exists but the plan has no exit code. The author has said the old commands stopped; the report will show how far their saved epochs extend. |
| exit_zero_export_needs_review | The process exited successfully but its exported metadata is incomplete, missing or inconsistent. |
| listed_not_started | The old batch plan lists this job but its work directory is absent. Such jobs may have been queued behind an earlier job; they are not automatically treated as new work to launch. |
| wrapper_record_needs_review | A parallel wrapper record exists without a readable child training plan. |
| directory_without_plan_needs_review | A job folder exists without a corresponding readable plan. Preserve it for inspection. |
| multiple_attempts_review_required | More than one attempt exists for the same condition/seed. Do not silently select or overwrite an attempt. |
| not_found | The inspected layouts contain no entry for this job. |
| scan_incomplete | A saved plan could not be read; absence of an entry cannot establish that the job is missing. |

Saved files do not prove process liveness across multiple servers. The current stop status comes from the author's explicit reply, not a local process check. This script never starts, stops or resumes a model and does not automatically issue retry commands.

After the report is returned, the assistant will identify completed records to retain, export-only repairs if needed, and the exact remaining jobs/commands. The twelve-model target remains three conditions × seeds 1–4; the number requiring further training is currently unknown. Seed 0 remains complete.
