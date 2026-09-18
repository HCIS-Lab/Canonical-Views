#!/usr/bin/env python3
"""Read-only GPU diagnosis. Does not change device visibility or drivers."""
import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-id', type=int, default=0, help='Logical visible CUDA index.')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new report filename.')
    report = dict(host=platform.node(), python=sys.executable, python_version=sys.version,
                  requested_logical_gpu=args.gpu_id, cuda_allocation_passed=False)
    report['environment'] = {k: os.environ.get(k) for k in (
        'CONDA_DEFAULT_ENV', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
        'SLURM_JOB_ID', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'PBS_JOBID')}
    for name, command in [('nvidia_smi', ['nvidia-smi']), ('nvidia_smi_list', ['nvidia-smi', '-L'])]:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            report[name] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except Exception as ex:
            report[name] = dict(error=str(ex))
    try:
        import torch
        report['torch_version'] = torch.__version__
        report['torch_path'] = torch.__file__
        report['torch_cuda_build'] = torch.version.cuda
        report['cuda_available'] = torch.cuda.is_available()
        report['visible_device_count'] = torch.cuda.device_count()
        torch.cuda.init()
        torch.cuda.set_device(args.gpu_id)
        report['gpu_name'] = torch.cuda.get_device_name(args.gpu_id)
        x = torch.ones((4, 4), device=f'cuda:{args.gpu_id}')
        report['matrix_check_sum'] = float((x @ x).sum().item())
        torch.cuda.synchronize()
        report['cuda_allocation_passed'] = report['matrix_check_sum'] == 64.0
    except Exception:
        report['torch_error'] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(f'Report: {args.output}')
    print('CUDA tensor test: ' + ('PASS' if report['cuda_allocation_passed'] else 'FAIL'))
    sys.exit(0 if report['cuda_allocation_passed'] else 1)


if __name__ == '__main__':
    main()
