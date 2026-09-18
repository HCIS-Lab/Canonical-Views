#!/usr/bin/env python3
"""Plan or run isolated training checks. Nothing trains without --execute."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

CONDITIONS = {
    'free': [],
    'expanded_family': ['--selector_view_limit', 'expanded_family'],
    'foreshortened_family': ['--selector_view_limit', 'foreshortened_family'],
    'foreshortened_family_remainder': ['--selector_view_limit', 'foreshortened_family_remainder'],
    'remainder': ['--selector_view_limit', 'remainder'],
    **{f'freeze_{e}': ['--freeze_epoch', str(e)] for e in [10, 20, 30, 40, 50]},
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def commands(args):
    jobs = []
    common = ['--arch', 'resnet18', '--dataset', 'rgb', '--epochs', '100',
              '--non_roll', '--num_train_instances', '25', '--batch_size', '6',
              '--lr', '0.0005', '--select_lr', '0.0001', '--select_wd', '0.0001',
              '--base_lr_ratio', '1.0', '--other_lr_ratio', '1.0',
              '--weight_decay', '0.001', '--aggregation', 'max', '--dropcam', '0.5',
              '--gamma', '0.99', '--deterministic', 'true', '--skip_stage1',
              '--save_every_epoch', '10', '--paper_record_initial']
    if args.suite == 'e01':
        cells = [(c, m, ['--steps', '5', '--td_target_mode', m] + CONDITIONS[c])
                 for c in args.conditions for m in args.modes]
    else:
        cells = [('single', 'legal', ['--steps', '1', '--td_target_mode', 'legal', '--active_single_view']),
                 ('pair', 'legal', ['--steps', '1', '--td_target_mode', 'legal'])]
    for seed in args.seeds:
        for c, mode, flags in cells:
            jobs.append(dict(id=f'{c}_{mode}_seed{seed}', condition=c, target_mode=mode,
                             seed=seed, arguments=common + flags + ['--seed', str(seed)]))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['e01', 'e14'], default='e01')
    p.add_argument('--conditions', choices=list(CONDITIONS), nargs='+', default=['free', 'expanded_family', 'freeze_10'])
    p.add_argument('--modes', choices=['legacy', 'legal'], nargs='+', default=['legacy', 'legal'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--gpu-id', type=int, default=0, help='Logical index within inherited CUDA_VISIBLE_DEVICES.')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    jobs = commands(args)
    if len({j['id'] for j in jobs}) != len(jobs):
        p.error('Duplicate jobs; use distinct seeds, conditions and modes.')
    root = Path(__file__).resolve().parent
    if not args.execute:
        print(json.dumps(dict(new_training=False, plan_only=True, jobs=jobs), indent=2))
        return
    if args.output.exists():
        p.error('Use a new output folder. Failed/finished jobs are not overwritten.')
    # This check runs in a separate process, before importing training modules.
    check = subprocess.run([sys.executable, '-c',
        'import torch,sys; d=int(sys.argv[1]); torch.cuda.set_device(d); '
        'torch.ones(1,device="cuda:"+str(d)).sum().item(); print("CUDA allocation passed")',
        str(args.gpu_id)])
    if check.returncode:
        p.error('GPU allocation failed. Run diagnose_paper_gpu.py first.')
    output = args.output.resolve()
    output.mkdir(parents=True)
    source = output / 'source'
    source.mkdir()
    shutil.copytree(root / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['main.py', 'paper_td_target.py', 'run_paper_training_checks.py']:
        shutil.copy2(root / name, source / name)
    versions = {str(x.relative_to(source)): sha(x) for x in source.rglob('*.py')}
    plan = dict(suite=args.suite, jobs=jobs, source_sha256=versions,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                logical_gpu=args.gpu_id, completed=False)
    (output / 'training_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    # main.py uses bare .cuda(); choose the logical device before executing it,
    # without rewriting the scheduler's CUDA_VISIBLE_DEVICES allocation.
    bootstrap = ('import torch,runpy,sys; torch.cuda.set_device(int(sys.argv[1])); '
                 'main=sys.argv[2]; sys.path.insert(0,str(__import__("pathlib").Path(main).parent)); '
                 'sys.argv=sys.argv[2:]; runpy.run_path(main,run_name="__main__")')
    failed = False
    for job in jobs:
        work = output / job['id']
        work.mkdir()
        command = [sys.executable, '-u', '-c', bootstrap, str(args.gpu_id), str(source / 'main.py')] + job['arguments']
        job['command'] = command
        print(f'Starting {job["id"]}; progress: {work}/console.log', flush=True)
        with (work / 'console.log').open('w') as log:
            result = subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT)
        job['returncode'] = result.returncode
        job['metadata'] = [str(x) for x in work.rglob('*_meta.json')]
        job['checkpoints'] = [dict(path=str(x), sha256=sha(x)) for x in work.rglob('model_e*.pth')]
        initial = list(work.rglob('model_e0.pth'))
        if len(initial) == 1:
            import torch
            state = torch.load(initial[0], map_location='cpu', weights_only=True)
            h = hashlib.sha256()
            for name, tensor in sorted(state.items()):
                h.update(name.encode()); h.update(str(tensor.dtype).encode())
                h.update(str(tuple(tensor.shape)).encode()); h.update(tensor.cpu().numpy().tobytes())
            job['initial_weights_sha256'] = h.hexdigest()
        (output / 'training_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        if result.returncode or len(job['metadata']) != 1:
            failed = True
            print(f'Stopped at {job["id"]}; send its console.log.', flush=True)
            break
        print(f'Completed {job["id"]}', flush=True)
    plan['completed'] = not failed
    (output / 'training_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    # Return compact numerical results and exact source. Keep weights and large
    # per-camera selection files on the server until a specific request.
    bundle = output / 'results_to_return.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        for path in sorted(output.rglob('*')):
            if path.is_file() and (path.name in ('training_plan.json', 'console.log', 'paper_epoch_metrics.jsonl', 'plot_error.txt')
                or path.name.endswith('_meta.json') or source in path.parents):
                tar.add(path, arcname=str(path.relative_to(output)))
    print(f'Return {bundle}. Keep checkpoints and *_selection.json files on the server.')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
