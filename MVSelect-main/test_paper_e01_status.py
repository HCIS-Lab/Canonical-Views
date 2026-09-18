#!/usr/bin/env python3
"""CPU-only checks for conservative saved-run inspection."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import inspect_paper_e01_status as inspector


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def batch(root, jobs, name='paper_e01_legal_replication_free_20260914'):
    path = root / name / 'training_plan.json'
    write(path, {'completed': False, 'jobs': jobs, 'source_sha256': {'main.py': 'test-fixture'}})
    return path


def job(seed, **extras):
    return dict(id=f'free_legal_seed{seed}', seed=seed, condition='free', target_mode='legal', **extras)


def exported(work, seed, n=100, mode='legal'):
    write(work / 'meta_logs/rgb/test_meta.json', {'accuracy': [0] * n, 'seed': seed, 'td_target_mode': mode})
    (work / 'console.log').write_text('test fixture\n')
    metrics = work / 'paper_epoch_metrics.jsonl'
    metrics.write_text(''.join(json.dumps({'epoch': e, 'seed': seed, 'td_target_mode': mode})+'\n' for e in range(1, n+1)))
    (work / 'never_read_selection.json').write_text('invalid JSON; this file must not be opened')
    (work / 'model_e100.pth').write_bytes(b'checkpoint fixture; not loaded')


class StatusTests(unittest.TestCase):
    def test_serial_completed_failed_and_queued(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = batch(root, [job(1, returncode=0), job(2, returncode=1), job(3), job(4)])
            exported(path.parent/'free_legal_seed1', 1)
            exported(path.parent/'free_legal_seed2', 2, n=17)
            report = inspector.inspect(root)
            self.assertEqual([r['status'] for r in report['jobs'][:4]],
                ['finished_export_recorded','failed_exit_recorded','listed_not_started','listed_not_started'])
            self.assertEqual(report['jobs'][4]['status'], 'not_found')
            self.assertFalse(report['process_liveness_verified'])
            self.assertEqual(report['jobs'][0]['attempts'][0]['metrics'][0]['last_epoch'], 100)
            self.assertEqual(report['read_errors'], [])

    def test_no_exit_or_mismatched_export_not_completed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);path=batch(root,[job(1),job(2,returncode=0),job(3,returncode=0)])
            exported(path.parent/'free_legal_seed1',1)
            exported(path.parent/'free_legal_seed2',99)
            exported(path.parent/'free_legal_seed3',3,n=80)
            status=[r['status'] for r in inspector.inspect(root)['jobs'][:3]]
            self.assertEqual(status,['no_exit_recorded','exit_zero_export_needs_review','exit_zero_export_needs_review'])

    def test_nested_parallel_and_orphan(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);group=root/'paper_e01_job01_20260914'
            write(group/'parallel_plan.json',{'jobs':[dict(job(1),status='completed')]})
            child=batch(root,[job(1,returncode=0)],'paper_e01_job01_20260914/free_legal_seed1')
            exported(child.parent/'free_legal_seed1',1)
            (root/'paper_e01_unfinished/free_legal_seed2').mkdir(parents=True)
            write(root/'paper_e01_job03_20260914/parallel_plan.json',{'jobs':[dict(job(3),status='not_started')]})
            rows=inspector.inspect(root)['jobs']
            self.assertEqual(rows[0]['status'],'finished_export_recorded')
            self.assertEqual(len(rows[0]['attempts']),1)
            self.assertEqual(rows[1]['status'],'directory_without_plan_needs_review')
            self.assertEqual(rows[2]['status'],'wrapper_record_needs_review')

    def test_bad_plan_and_multiple_attempts(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=batch(root,[job(1)])
            (p.parent/'free_legal_seed1').mkdir()
            q=batch(root,[job(1)],'paper_e01_another')
            (q.parent/'free_legal_seed1').mkdir()
            bad=root/'paper_e01_bad/training_plan.json';bad.parent.mkdir();bad.write_text('{"jobs":')
            report=inspector.inspect(root)
            self.assertEqual(report['jobs'][0]['status'],'multiple_attempts_review_required')
            self.assertEqual(report['jobs'][1]['status'],'scan_incomplete')
            self.assertEqual(len(report['read_errors']),1)

    def test_cli_preserves_all_inputs_and_reports(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=batch(root,[job(1,returncode=0)])
            exported(p.parent/'free_legal_seed1',1)
            original={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*') if f.is_file()}
            output=root/'status.json'
            command=[sys.executable,inspector.__file__,'--root',str(root),'--output',str(output)]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('finished_export_recorded',result.stdout)
            for f,digest in original.items():self.assertEqual(hashlib.sha256(Path(f).read_bytes()).hexdigest(),digest)
            contents=output.read_bytes()
            second=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual(contents,output.read_bytes())


if __name__=='__main__':unittest.main(verbosity=2)
