#!/usr/bin/env python3
"""Read existing E01 records without starting, stopping or modifying training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

ROSTER = [dict(job_number=i+1, id=f'{condition}_legal_seed{seed}',
               condition=condition, seed=seed)
          for i, (condition, seed) in enumerate(
              (c, s) for c in ('free', 'expanded_family', 'freeze_10')
              for s in (1, 2, 3, 4))]
JSON_LIMIT = 16 * 1024 * 1024
LOG_TAIL_BYTES = 6000


def read_json(path):
    if path.stat().st_size > JSON_LIMIT:
        raise ValueError('JSON exceeds 16 MB inspection limit; not loaded.')
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError('Expected a JSON object.')
    return data


def epoch_summary(path):
    # Metrics are small. Never open *_selection.json or model checkpoints.
    before = path.stat()
    if before.st_size > JSON_LIMIT:
        return dict(path=str(path), error='Metrics exceed 16 MB inspection limit.')
    epochs, invalid, seeds, modes = [], 0, set(), set()
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
                epoch = row['epoch']
                if not isinstance(epoch, int):
                    raise ValueError('non-integer epoch')
                epochs.append(epoch)
                seeds.add(str(row.get('seed')))
                modes.add(str(row.get('td_target_mode')))
            except (ValueError, KeyError, TypeError):
                invalid += 1
    after = path.stat()
    return dict(path=str(path), rows=len(epochs), unique_epochs=len(set(epochs)),
                last_epoch=max(epochs) if epochs else None,
                covers_epochs_1_to_100=set(range(1, 101)).issubset(epochs),
                invalid_or_partial_lines=invalid, seeds=sorted(seeds), modes=sorted(modes),
                changed_during_read=(before.st_mtime_ns, before.st_size) !=
                                    (after.st_mtime_ns, after.st_size))


def inspect_attempt(row, plan_path, group, work):
    attempt = dict(plan=str(plan_path), work_directory=str(work),
                   directory_exists=work.is_dir(), returncode=row.get('returncode'),
                   group_completed=group.get('completed'),
                   condition=row.get('condition'), seed=row.get('seed'),
                   target_mode=row.get('target_mode'),
                   metadata=[], metrics=[], logs=[], errors=[])
    archive = plan_path.parent / 'results_to_return.tar.gz'
    attempt['result_archive'] = str(archive) if archive.is_file() else None
    attempt['source_sha256'] = group.get('source_sha256', {})
    attempt['arguments'] = row.get('arguments', [])
    for path in sorted(work.rglob('*_meta.json')) if work.is_dir() else []:
        try:
            meta = read_json(path)
            accuracy = meta.get('accuracy', [])
            attempt['metadata'].append(dict(path=str(path),
                accuracy_epochs=len(accuracy) if isinstance(accuracy, list) else None,
                seed=meta.get('seed'), td_target_mode=meta.get('td_target_mode'),
                selector_view_limit=meta.get('selector_view_limit'),
                recognition_num_views=meta.get('recognition_num_views')))
        except (OSError, ValueError) as error:
            attempt['errors'].append(f'{path}: {error}')
    for path in sorted(work.rglob('paper_epoch_metrics.jsonl')) if work.is_dir() else []:
        try:
            attempt['metrics'].append(epoch_summary(path))
        except OSError as error:
            attempt['errors'].append(f'{path}: {error}')
    for path in sorted(work.rglob('console.log')) if work.is_dir() else []:
        try:
            stat = path.stat()
            with path.open('rb') as stream:
                stream.seek(max(0, stat.st_size - LOG_TAIL_BYTES))
                tail = stream.read(LOG_TAIL_BYTES).decode('utf-8', errors='replace')
            attempt['logs'].append(dict(path=str(path), bytes=stat.st_size,
                modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(), tail=tail))
        except OSError as error:
            attempt['errors'].append(f'{path}: {error}')
    rc = row.get('returncode')
    if rc is not None and rc != 0:
        status = 'failed_exit_recorded'
    elif rc == 0:
        meta = attempt['metadata']
        ready = (len(meta) == 1 and meta[0]['accuracy_epochs'] == 100
                 and meta[0]['seed'] == row.get('seed')
                 and meta[0]['td_target_mode'] == 'legal')
        status = 'finished_export_recorded' if ready else 'exit_zero_export_needs_review'
    elif work.is_dir():
        status = 'no_exit_recorded'
    else:
        status = 'listed_not_started'
    attempt['status'] = status
    return attempt


def inspect(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Results directory does not exist: {root}')
    rows = {r['id']: dict(r, attempts=[]) for r in ROSTER}
    errors, groups, covered = [], [], set()
    # Old four-seed layout and new wrapper's single-run child layout.
    plan_paths = sorted(set(root.glob('paper_e01_*/training_plan.json')) |
                        set(root.glob('paper_e01_*/*/training_plan.json')))
    for path in plan_paths:
        try:
            group = read_json(path)
            jobs = group.get('jobs', [])
            if not isinstance(jobs, list):
                raise ValueError('jobs is not a list')
            requested = [j for j in jobs if isinstance(j, dict) and j.get('id') in rows]
            if not requested:
                continue
            groups.append(dict(plan=str(path), completed=group.get('completed'),
                               relevant_jobs=[j['id'] for j in requested],
                               recorded_failed_jobs=[j.get('id') for j in jobs
                                   if isinstance(j, dict) and j.get('returncode') not in (None, 0)]))
            for row in requested:
                work = path.parent / row['id']
                rows[row['id']]['attempts'].append(inspect_attempt(row, path, group, work))
                covered.add(work)
        except (OSError, ValueError) as error:
            errors.append(f'{path}: {error}; may be a partial write. Do not infer jobs are missing.')
    for path in sorted(root.glob('paper_e01_*/parallel_plan.json')):
        try:
            group = read_json(path)
            jobs = group.get('jobs', [])
            if not isinstance(jobs, list):
                raise ValueError('jobs is not a list')
            for row in jobs:
                if not isinstance(row, dict) or row.get('id') not in rows:
                    continue
                output = path.parent / row['id']
                if (output / 'training_plan.json').is_file():
                    continue  # Child was inspected above (or its read error is recorded).
                rows[row['id']]['attempts'].append(dict(plan=str(path),
                    work_directory=str(output), directory_exists=output.is_dir(),
                    status='wrapper_record_needs_review',
                    wrapper_status=row.get('status'), launcher_pid=row.get('launcher_pid'),
                    returncode=row.get('returncode'), error=row.get('error'),
                    note='PID may belong to another server. Process liveness is not established.'))
                covered.add(output)
        except (OSError, ValueError) as error:
            errors.append(f'{path}: {error}; do not infer jobs are missing.')
    # An orphan directory also needs review; absence of a plan is not permission to overwrite.
    for row in rows.values():
        candidates = set(root.glob('paper_e01_*/' + row['id'])) | set(root.glob('paper_e01_*/*/' + row['id']))
        for work in sorted(candidates):
            if work.is_dir() and work not in covered and not (work / 'training_plan.json').exists():
                row['attempts'].append(dict(work_directory=str(work),
                    directory_exists=True, status='directory_without_plan_needs_review'))
        attempts = row['attempts']
        if not attempts:
            row['status'] = 'not_found' if not errors else 'scan_incomplete'
        elif len(attempts) > 1:
            row['status'] = 'multiple_attempts_review_required'
        else:
            row['status'] = attempts[0]['status']
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        inspected_from_host=socket.gethostname(), compare_root=str(root),
        training_started_or_stopped=False, existing_files_modified=False,
        process_liveness_verified=False,
        interpretation='This is a filesystem snapshot, not proof that a process is running or stopped. '
          'Older serial launchers reserve future seeds in their plan; a missing job folder may still be queued. '
          'Do not delete, rename or bypass existing records to restart training. '
          'Finished export recorded is not a scientific result audit.',
        groups=groups, jobs=list(rows.values()), read_errors=errors)
    result['status_counts'] = dict(Counter(r['status'] for r in rows.values()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('compare'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Report already exists. Use another report filename; old reports are preserved.')
    try:
        report = inspect(args.root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print('Job  Condition           Seed  Saved-record status')
    for row in report['jobs']:
        print(f"{row['job_number']:02d}   {row['condition']:<19} {row['seed']}     {row['status']}")
    print(f'Return: {args.output.resolve()}')
    print('No training started/stopped. Existing run files are unchanged.')
    if report['read_errors']:
        print('Some plans could not be read; the report records the errors. Do not infer missing work.')


if __name__ == '__main__':
    main()
