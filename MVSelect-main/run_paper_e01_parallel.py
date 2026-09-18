#!/usr/bin/env python3
"""Run independent E01 replications, at most one model per allocated GPU.

Without --execute this prints a plan and starts nothing. This wrapper calls
the existing single-GPU training launcher without altering its model settings.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import threading


ROSTER = [
    dict(job_number=i + 1, condition=condition, seed=seed,
         id=f'{condition}_legal_seed{seed}')
    for i, (condition, seed) in enumerate(
        (condition, seed)
        for condition in ('free', 'expanded_family', 'freeze_10')
        for seed in (1, 2, 3, 4))
]


def make_plan(output, gpus, numbers, runner=None):
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError('GPU indices must be distinct nonnegative logical indices.')
    if not numbers or len(set(numbers)) != len(numbers) or any(n not in range(1, 13) for n in numbers):
        raise ValueError('Job numbers must be distinct integers from 1 through 12.')
    if len(gpus) > len(numbers):
        raise ValueError('Specify no more GPUs than selected jobs.')
    runner = Path(runner or Path(__file__).with_name('run_paper_training_checks.py')).resolve()
    output = Path(output).resolve()
    jobs = []
    for i, number in enumerate(numbers):
        row = dict(ROSTER[number - 1])
        row['gpu_id'] = gpus[i % len(gpus)]
        row['output'] = str(output / row['id'])
        row['command'] = [sys.executable, str(runner), '--suite', 'e01',
                          '--seeds', str(row['seed']), '--conditions', row['condition'],
                          '--modes', 'legal', '--gpu-id', str(row['gpu_id']),
                          '--output', row['output'], '--execute']
        row['status'] = 'not_started'
        jobs.append(row)
    return dict(plan_only=True, execution_started=False, completed=False,
                output=str(output), runner=str(runner), gpus=list(gpus), jobs=jobs,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                model_settings_changed=False, distributed_training=False)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def reject_existing_jobs(plan):
    output = Path(plan['output'])
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Collect its results or use a new group output for only missing jobs.')
    if not Path(plan['runner']).is_file():
        raise ValueError('Place this script beside run_paper_training_checks.py in MVSelect-main.')
    requested = {j['id'] for j in plan['jobs']}
    # Detect the previously supplied batch commands and groups made by this wrapper.
    for directory in output.parent.glob('paper_e01_*'):
        if not directory.is_dir():
            continue
        for name in ('training_plan.json', 'parallel_plan.json'):
            path = directory / name
            if not path.exists():
                continue
            saved = json.loads(path.read_text())
            for job in saved.get('jobs', []):
                if job.get('id') in requested and (directory / job['id']).exists():
                    raise ValueError(f"Existing replication found: {directory / job['id']}. Do not duplicate it; select only missing --job-ids.")


def check_gpus(gpus):
    # Respect the scheduler's visibility mask. Do not overwrite CUDA_VISIBLE_DEVICES.
    probe = (
        'import sys,torch; ids=[int(x) for x in sys.argv[1:]]; '
        'n=torch.cuda.device_count(); '
        'assert ids and all(0<=i<n for i in ids), '
        'f"Requested logical GPUs {ids}, but only {n} are visible"; '
        '[(torch.cuda.set_device(i),torch.ones(1,device="cuda:"+str(i)).sum().item()) for i in ids]; '
        'print("GPU allocations verified:",ids)'
    )
    subprocess.run([sys.executable, '-c', probe, *map(str, gpus)], check=True)


def job_complete(job):
    directory = Path(job['output'])
    path = directory / 'training_plan.json'
    archive = directory / 'results_to_return.tar.gz'
    if not path.is_file() or not archive.is_file():
        return False
    saved = json.loads(path.read_text())
    jobs = saved.get('jobs', [])
    return bool(saved.get('completed') and len(jobs) == 1
                and jobs[0].get('id') == job['id']
                and jobs[0].get('returncode') == 0
                and len(jobs[0].get('metadata', [])) == 1)


def collect(output):
    output = Path(output).resolve()
    plan = json.loads((output / 'parallel_plan.json').read_text())
    entries = []
    for job in plan['jobs']:
        archive = output / job['id'] / 'results_to_return.tar.gz'
        entries.append(dict(id=job['id'], job_number=job['job_number'],
                            completed=job_complete(dict(job, output=str(output / job['id']))),
                            archive=str(archive.relative_to(output)) if archive.exists() else None,
                            sha256=hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None))
    receipt = dict(jobs=entries, completed_jobs=sum(x['completed'] for x in entries),
                   expected_jobs=len(entries), all_completed=all(x['completed'] for x in entries))
    write_json(output / 'collection_manifest.json', receipt)
    bundle = output / 'e01_parallel_results_to_return.tar.gz'
    # Each inner archive contains the exact single-run records and code snapshot.
    with tarfile.open(bundle, 'w:gz') as tar:
        for name in ('parallel_plan.json', 'collection_manifest.json'):
            tar.add(output / name, arcname=name)
        for job in plan['jobs']:
            directory = output / job['id']
            log = output / (job['id'] + '_launcher.log')
            if log.is_file():
                tar.add(log, arcname=log.name)
            archive = directory / 'results_to_return.tar.gz'
            if archive.is_file():
                tar.add(archive, arcname=str(archive.relative_to(output)))
            else:
                # Retain evidence from a failed/interrupted job without collecting weights.
                files = list(directory.rglob('console.log')) if directory.exists() else []
                if (directory / 'training_plan.json').is_file():
                    files.append(directory / 'training_plan.json')
                for path in sorted(files):
                    tar.add(path, arcname=str(path.relative_to(output)))
    print(f"Completed {receipt['completed_jobs']}/{receipt['expected_jobs']} jobs. Return {bundle}", flush=True)
    return receipt


def launch_jobs(plan):
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    plan.update(plan_only=False, execution_started=True)
    write_json(output / 'parallel_plan.json', plan)
    lock = threading.Lock()
    stopped = threading.Event()
    processes = {}

    def worker(gpu):
        for job in (j for j in plan['jobs'] if j['gpu_id'] == gpu):
            if stopped.is_set():
                break
            print(f"Starting job {job['job_number']}: {job['id']} on logical GPU {gpu}", flush=True)
            try:
                with (output / (job['id'] + '_launcher.log')).open('w') as log:
                    with lock:
                        if stopped.is_set():
                            return
                        process = subprocess.Popen(job['command'], stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        processes[job['id']] = process
                        job['status'] = 'running'
                        job['launcher_pid'] = process.pid
                        write_json(output / 'parallel_plan.json', plan)
                    code = process.wait()
                complete = code == 0 and job_complete(job)
                with lock:
                    processes.pop(job['id'], None)
                    job.update(returncode=code, status='completed' if complete else 'failed')
                    write_json(output / 'parallel_plan.json', plan)
                print(f"{job['id']}: {job['status']}", flush=True)
            except Exception as error:
                with lock:
                    job.update(status='failed', error=str(error))
                    write_json(output / 'parallel_plan.json', plan)

    pool = ThreadPoolExecutor(max_workers=len(plan['gpus']))
    futures = [pool.submit(worker, gpu) for gpu in plan['gpus']]
    try:
        for future in as_completed(futures):
            future.result()
    except KeyboardInterrupt:
        stopped.set()
        with lock:
            active = list(processes.values())
        for process in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        print('Stopped this group; completed outputs are retained.', flush=True)
    finally:
        pool.shutdown(wait=True)
    plan['completed'] = all(j['status'] == 'completed' for j in plan['jobs'])
    write_json(output / 'parallel_plan.json', plan)
    return collect(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='Logical GPU indices visible within this allocation.')
    parser.add_argument('--job-ids', type=int, nargs='+', default=list(range(1, 13)), help='Subset of fixed jobs 1–12 for this server/allocation.')
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--collect-only', action='store_true', help='Package existing outputs without starting training.')
    args = parser.parse_args()
    if args.collect_only:
        receipt = collect(args.output)
        return 0 if receipt['all_completed'] else 1
    try:
        plan = make_plan(args.output, args.gpus, args.job_ids)
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return 0
        reject_existing_jobs(plan)
        check_gpus(plan['gpus'])
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    receipt = launch_jobs(plan)
    return 0 if receipt['all_completed'] else 1


if __name__ == '__main__':
    sys.exit(main())
