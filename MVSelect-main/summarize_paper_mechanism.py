#!/usr/bin/env python3
"""Audit all 15 completed E18 jobs; keep source and recipient variation separate."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics
import tarfile
from collections import defaultdict
from paper_mechanism_core import CONDITIONS, job_spec, sha


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, output):
    if output.exists():
        raise ValueError('Use a new summary folder')
    endpoints, contrasts, jobs, branch_starts, signatures = [], [], [], [], set()
    common_eval_hash = None
    for job_id in range(1, 16):
        folder = root/f'job{job_id:02d}'
        plan = json.loads((folder/'run_plan.json').read_text())
        spec = job_spec(job_id)
        if (not plan['completed'] or not plan['scientific_result'] or plan['software_smoke_test']
                or any(plan[k] != v for k, v in spec.items())
                or plan['stages'] != [0, 30] or plan['adaptation_epochs'] != 100
                or plan['conditions'] != list(CONDITIONS)):
            raise ValueError(f'Incomplete, incompatible or smoke-only job: {folder}')
        signature = {k: plan[k] for k in ('protocol', 'batch_size', 'lr', 'weight_decay',
                      'optimizer', 'branch_schedule', 'warmup', 'recognition_images',
                      'pooling', 'training_bn', 'diagnostic_bn', 'source_code_sha256')}
        signatures.add(json.dumps(signature, sort_keys=True))
        if len(signatures) != 1:
            raise ValueError('Jobs use different settings or source code')
        eval_hash = plan['endpoint_viewsets_sha256']
        common_eval_hash = common_eval_hash or eval_hash
        if eval_hash != common_eval_hash:
            raise ValueError('Endpoint images differ between jobs')
        rows = list(csv.DictReader((folder/'endpoint_metrics.csv').open()))
        required = {(s, c) for s in (0, 30) for c in CONDITIONS}
        if len(rows) != 12 or {(int(r['stage']), r['condition']) for r in rows} != required:
            raise ValueError(f'Missing/duplicate endpoint cells in {folder}')
        for r in rows:
            if int(r['n_objects']) != 96 or int(r['n_view_sets']) != 288:
                raise ValueError('Wrong endpoint replication unit')
            endpoints.append({**spec, **r})
        audit = list(csv.DictReader((folder/'branch_audit.csv').open()))
        if len(audit) != 12 or {(int(r['stage']), r['condition']) for r in audit} != required:
            raise ValueError('Missing branch audits')
        for stage in (0, 30):
            sub = [r for r in audit if int(r['stage']) == stage]
            if len({r['initial_state_sha256'] for r in sub}) != 1:
                raise ValueError('Conditions did not branch from identical parameters/buffers')
            if len({r['initial_optimizer_steps'] for r in sub}) != 1:
                raise ValueError('Conditions did not use matched optimizer ages')
            if len({r['initial_optimizer_sha256'] for r in sub}) != 1:
                raise ValueError('Conditions did not use identical optimizer states')
            for r in sub:
                if int(r['actual_updates']) != int(r['expected_updates']):
                    raise ValueError('An update budget is incomplete')
            branch_starts.append({**spec, 'stage': stage,
                                  'state_sha256': sub[0]['initial_state_sha256']})
        with gzip.open(folder/'input_manifest.json.gz', 'rt') as f:
            manifest = json.load(f)
        if manifest['training_design_sha256'] != plan['training_design_sha256']:
            raise ValueError('Training design changed')
        contrasts.extend({**spec, **r} for r in csv.DictReader((folder/'contrasts.csv').open()))
        jobs.append({**spec, 'run_plan_sha256': sha(folder/'run_plan.json')})
    # Recipient seeds are crossed with source seeds. Three recipients do not
    # turn five source models into fifteen independent source models.
    groups = defaultdict(list)
    for r in contrasts:
        for metric in ('accuracy_difference_pp', 'loss_advantage'):
            groups[r['source_seed'], int(r['horizon_epochs']), str(r['stage']), r['contrast'], metric].append(float(r[metric]))
    source_rows = []
    for (source, horizon, stage, contrast, metric), values in sorted(groups.items()):
        if len(values) != 3:
            raise ValueError('Expected three recipient seeds per source contrast')
        source_rows.append(dict(source_seed=source, horizon_epochs=horizon, stage=stage, contrast=contrast,
                          metric=metric, n_recipient_seeds=3, mean_difference=statistics.mean(values),
                          sd_across_recipient_seeds=statistics.stdev(values)))
    grand = defaultdict(list)
    for r in source_rows:
        grand[r['horizon_epochs'], r['stage'], r['contrast'], r['metric']].append(r['mean_difference'])
    summary = [dict(horizon_epochs=h, stage=s, contrast=c, metric=m, n_source_models=len(v),
                    recipients_per_source=3, mean_difference=statistics.mean(v),
                    sd_across_source_means=statistics.stdev(v))
               for (h, s, c, m), v in sorted(grand.items())]
    start_groups = defaultdict(set)
    for r in branch_starts:
        start_groups[r['recipient_seed'], r['stage']].add(r['state_sha256'])
    audit_summary = dict(completed_jobs=15, endpoint_objects=96, endpoint_draws_per_object=3,
                         identical_branch_states_within_each_job=True,
                         cross_source_warmup_states_match=all(len(s) == 1 for s in start_groups.values()),
                         inference='Descriptive source-model means and SD. Recipient seeds are crossed; object/view rows are not model replicates. No automated significance or mechanism claim.',
                         jobs=jobs)
    output.mkdir(parents=True)
    write_csv(output/'all_endpoints.csv', endpoints)
    write_csv(output/'all_contrasts.csv', contrasts)
    write_csv(output/'source_model_contrasts.csv', source_rows)
    write_csv(output/'contrast_summary.csv', summary)
    write_csv(output/'branch_start_hashes.csv', branch_starts)
    (output/'audit.json').write_text(json.dumps(audit_summary, indent=2)+'\n')
    archive = output/'E18_results_to_return.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for job_id in range(1, 16):
            folder = root/f'job{job_id:02d}'
            for path in sorted(folder.iterdir()):
                if path.is_file() and (path.suffix in ('.csv', '.json', '.py', '.md') or path.name.endswith('.json.gz')):
                    tar.add(path, arcname=f'job{job_id:02d}/'+path.name)
        for path in sorted(output.iterdir()):
            if path.suffix in ('.csv', '.json'):
                tar.add(path, arcname='summary/'+path.name)
    print(f'Return {archive}. No checkpoints or large source selection JSONs are needed.')
    return audit_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)
