#!/usr/bin/env python3
"""Read-only CUDA driver checks in separate processes. No driver changes."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import traceback


def read(path):
    try:
        return Path(path).read_text()
    except (OSError, UnicodeError) as ex:
        return str(ex)


def worker(kind):
    result = {'python': sys.executable, 'kind': kind}
    try:
        if kind == 'torch':
            import torch
            result.update(torch_version=torch.__version__, torch_cuda_build=torch.version.cuda)
        lib = ctypes.CDLL('libcuda.so.1')
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        version = ctypes.c_int()
        result['cuDriverGetVersion_code'] = lib.cuDriverGetVersion(ctypes.byref(version))
        result['driver_api_version'] = version.value
        code = lib.cuInit(0)
        result['cuInit_code'] = code
        for func, field in [('cuGetErrorName', 'cuInit_name'), ('cuGetErrorString', 'cuInit_description')]:
            value = ctypes.c_char_p()
            fn = getattr(lib, func)
            fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            fn.restype = ctypes.c_int
            if fn(code, ctypes.byref(value)) == 0 and value.value:
                result[field] = value.value.decode(errors='replace')
        if code == 0:
            count = ctypes.c_int()
            result['cuDeviceGetCount_code'] = lib.cuDeviceGetCount(ctypes.byref(count))
            result['driver_device_count'] = count.value
            if kind == 'torch':
                x = torch.ones((4, 4), device='cuda:0')
                result['cuda_matrix_sum'] = float((x @ x).sum().item())
                result['cuda_tensor_passed'] = result['cuda_matrix_sum'] == 64.
    except Exception:
        result['error'] = traceback.format_exc()
    maps = read('/proc/self/maps').splitlines()
    result['loaded_cuda_libraries'] = sorted({line.split()[-1] for line in maps
        if any(x in line for x in ['libcuda.', 'libcudart.', 'libnvidia-'])})
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', choices=['driver', 'torch'], help=argparse.SUPPRESS)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.output is None or args.output.exists():
        p.error('Provide a new --output JSON filename.')
    result = dict(host=platform.node(), kernel=platform.release(), uid=os.getuid(),
        gid=os.getgid(), groups=os.getgroups(),
        environment={k: os.environ.get(k) for k in ['CONDA_PREFIX', 'CUDA_VISIBLE_DEVICES',
            'LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDA_HOME', 'SLURM_JOB_ID', 'PBS_JOBID']},
        nvidia_kernel_version=read('/proc/driver/nvidia/version'),
        nvidia_uvm_module_present=Path('/sys/module/nvidia_uvm').exists(),
        nvidia_uvm_initstate=read('/sys/module/nvidia_uvm/initstate'))
    result['device_nodes'] = []
    for path in sorted(Path('/dev').glob('nvidia*')):
        try:
            s = path.stat()
            result['device_nodes'].append(dict(path=str(path), mode=stat.filemode(s.st_mode),
                uid=s.st_uid, gid=s.st_gid, readable=os.access(path, os.R_OK),
                writable=os.access(path, os.W_OK), is_character_device=stat.S_ISCHR(s.st_mode),
                major=os.major(s.st_rdev), minor=os.minor(s.st_rdev)))
        except OSError as ex:
            result['device_nodes'].append(dict(path=str(path), error=str(ex)))
    result['process_checks'] = []
    programs = [(sys.executable, 'driver'), (sys.executable, 'torch')]
    system_python = Path('/usr/bin/python3')
    if system_python.is_file() and system_python.resolve() != Path(sys.executable).resolve():
        programs.append((str(system_python), 'driver'))
    for python, kind in programs:
        try:
            cp = subprocess.run([python, str(Path(__file__).resolve()), '--worker', kind],
                capture_output=True, text=True, timeout=40)
            entry = dict(python=python, kind=kind, returncode=cp.returncode, stderr=cp.stderr)
            try:
                entry['report'] = json.loads(cp.stdout)
            except ValueError:
                entry['stdout'] = cp.stdout
            result['process_checks'].append(entry)
        except Exception as ex:
            result['process_checks'].append(dict(python=python, kind=kind, error=str(ex)))
    # Query only NVIDIA-related kernel messages. No sudo; inaccessible logs are
    # recorded as such rather than attempting to change access permissions.
    if shutil.which('journalctl'):
        try:
            cp = subprocess.run(['journalctl', '-k', '-b', '--no-pager', '-n', '40',
                '--grep=NVRM|nvidia|Xid'], capture_output=True, text=True, timeout=15)
            result['nvidia_kernel_messages'] = dict(returncode=cp.returncode, stdout=cp.stdout, stderr=cp.stderr)
        except Exception as ex:
            result['nvidia_kernel_messages'] = dict(error=str(ex))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(f'Return {args.output}. No driver, package, allocation or device setting was changed.')


if __name__ == '__main__':
    main()
