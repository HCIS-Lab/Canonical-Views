#!/usr/bin/env python3
"""CPU-only checks of routing, concurrency, failure collection and preservation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import run_paper_e01_parallel as parallel


class ParallelChecks(unittest.TestCase):
    def test_preview_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'never-created'
            printed=subprocess.check_output([sys.executable,parallel.__file__,'--gpus',*map(str,range(12)),'--output',str(output)],text=True)
            plan=json.loads(printed)
            self.assertFalse(output.exists())
            self.assertFalse(plan['execution_started'])
            self.assertEqual(len({j['id'] for j in plan['jobs']}),12)
            self.assertEqual({j['gpu_id'] for j in plan['jobs']},set(range(12)))
            self.assertNotIn(0,{j['seed'] for j in plan['jobs']})

    def test_allocation_errors_and_existing_work(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);runner=base/'runner.py';runner.write_text('')
            for gpus,ids in [([0,0],[1,2]),([0],[1,1]),([0],[13]),([-1],[1])]:
                with self.assertRaises(ValueError):parallel.make_plan(base/'new',gpus,ids,runner)
            old=base/'paper_e01_legal_replication_free_20260914';old.mkdir()
            (old/'free_legal_seed1').mkdir()
            (old/'training_plan.json').write_text(json.dumps({'jobs':[{'id':'free_legal_seed1'}]}))
            with self.assertRaisesRegex(ValueError,'Existing replication'):
                parallel.reject_existing_jobs(parallel.make_plan(base/'paper_e01_new',[0],[1],runner))
            parallel.reject_existing_jobs(parallel.make_plan(base/'paper_e01_new',[0],[2],runner))

    def test_real_subprocess_concurrency_failure_and_archive(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);runner=root/'fake_runner.py'
            runner.write_text('''import argparse,json,time,tarfile,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--seeds',type=int);p.add_argument('--conditions');p.add_argument('--gpu-id',type=int);a,_=p.parse_known_args()
out=Path(a.output);out.mkdir();start=time.monotonic();time.sleep(.25);end=time.monotonic()
(out/'interval.json').write_text(json.dumps({'gpu':a.gpu_id,'start':start,'end':end}))
id=f'{a.conditions}_legal_seed{a.seeds}'
ok=a.seeds!=3
(out/'training_plan.json').write_text(json.dumps({'completed':ok,'jobs':[{'id':id,'returncode':0 if ok else 1,'metadata':['sample_meta.json'] if ok else []}]}))
with tarfile.open(out/'results_to_return.tar.gz','w:gz') as t:t.add(out/'training_plan.json',arcname='training_plan.json')
sys.exit(0 if ok else 1)
''')
            plan=parallel.make_plan(root/'paper_e01_group',[0,1],[1,2,3,4],runner)
            parallel.reject_existing_jobs(plan)
            receipt=parallel.launch_jobs(plan)
            self.assertEqual(receipt['completed_jobs'],3)
            self.assertFalse(receipt['all_completed'])
            intervals=[json.loads((Path(j['output'])/'interval.json').read_text()) for j in plan['jobs']]
            for gpu in [0,1]:
                group=sorted((i for i in intervals if i['gpu']==gpu),key=lambda i:i['start'])
                self.assertGreaterEqual(group[1]['start'],group[0]['end'])
            self.assertTrue(any(a['gpu']!=b['gpu'] and max(a['start'],b['start'])<min(a['end'],b['end']) for a in intervals for b in intervals))
            with tarfile.open(root/'paper_e01_group/e01_parallel_results_to_return.tar.gz') as t:
                self.assertEqual(len([n for n in t.getnames() if n.endswith('/results_to_return.tar.gz')]),4)
            with patch.object(parallel.subprocess,'Popen',side_effect=AssertionError('collect must not launch')):
                collected=parallel.collect(root/'paper_e01_group')
            self.assertEqual(collected['completed_jobs'],3)
            with self.assertRaises(ValueError):parallel.reject_existing_jobs(plan)


if __name__=='__main__':unittest.main(verbosity=2)
