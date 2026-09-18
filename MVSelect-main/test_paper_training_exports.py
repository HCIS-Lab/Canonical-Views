#!/usr/bin/env python3
"""Regression checks for the completed-training export failure; no GPU needed."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parent


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


records = load('experiment_records', 'src/utils/experiment_records.py')
curve = load('paper_curve', 'src/utils/draw_curve.py')


class Scalar(float):
    def item(self): return float(self)


class Counts:
    def __init__(self, values): self.values = values
    def cpu(self): return self.values


class Exports(unittest.TestCase):
    def test_epoch_journal_preserves_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'epochs.jsonl'
            records.append_epoch_record(path, {'epoch':1,'accuracy':np.float64(83.125),'counts':np.array([2,3])})
            records.append_epoch_record(path, {'epoch':2,'accuracy':84.375})
            data=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(data,[{'epoch':1,'accuracy':83.125,'counts':[2,3]},{'epoch':2,'accuracy':84.375}])

    def test_actual_main_exports_before_plot_failure(self):
        tree=ast.parse((ROOT/'main.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        training=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=='not args.eval')
        start=next(i for i,n in enumerate(training.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='remainder_ratio_hist')
        tail=training.body[start:]
        self.assertTrue(isinstance(tail[-1],ast.Expr) and ast.unparse(tail[-1].value.func)=='plot_after_records')
        # Execute the real final metadata-writing block, with a deliberate plot failure.
        with tempfile.TemporaryDirectory() as tmp:
            env={'np':np,'os':os,'json':json,'Path':Path,'plot_after_records':records.plot_after_records}
            args=SimpleNamespace(steps=5,epochs=10,non_like=False,non_roll=True,seed=0,td_target_mode='legal',
                aggregation='max',skip_stage1=True,selector_view_limit='all',active_single_view=False)
            env.update(args=args,meta_log=tmp,exp_name='test',logdir=tmp,test_set=SimpleNamespace(pose_table={'0_0_0':0}))
            for name in ('longest_ratio_hist','short_ratio_hist','longest_like_ratio_hist','short_like_ratio_hist'):
                env[name]=[.1]*10
            # Supply every history consumed by the real block.
            for n in ast.walk(ast.Module(body=tail,type_ignores=[])):
                if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load):
                    if n.id.endswith(('_prec_s','_prec_s_3','_prec_s_5')): env[n.id]=[Scalar(80+i/10) for i in range(10)]
                    elif 'per_class_acc' in n.id: env[n.id]=[[80.0]*32 for _ in range(10)]
                    elif n.id.endswith(('_count_hist','_deg_bar_hist')):env[n.id]=[1]*10
            env.update(sel_counts_total=Counts(np.array([10])),sel_counts_hist=[np.array([1])]*10,total_selected_hist=[10]*10)
            def fail(*args,**kwargs): raise FileNotFoundError('simulated diagnostic failure')
            env['plot']=fail
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                exec(compile(ast.Module(body=tail,type_ignores=[]),str(ROOT/'main.py'),'exec'),env)
            meta=json.loads((Path(tmp)/'test_meta.json').read_text())
            self.assertEqual(len(meta['accuracy']),10)
            self.assertAlmostEqual(meta['Remainder'][0],.6)
            self.assertIn('simulated diagnostic failure',(Path(tmp)/'plot_error.txt').read_text())

    def test_plot_uses_mapping_without_pose_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous=os.getcwd();os.chdir(tmp)
            try:
                mapping={'0_0_0':0,'0_90_0':1}
                ratios=[.1]*10;counts=[np.array([1,1])]*10;bars=[[.1]*10]*10
                remainder=curve.plot(tmp,[80]*10,np.array([10,10]),counts,[10]*10,
                    ratios,ratios,ratios,ratios,[1]*10,[1]*10,[1]*10,[1]*10,bars,bars,10,5,
                    non_roll=True,pose_mapping=mapping)
                self.assertEqual(len(remainder),10)
                self.assertTrue((Path(tmp)/'selection_counts_heatmap.png').is_file())
                self.assertFalse((Path(tmp)/'non_roll_pose_table.json').exists())
            finally:os.chdir(previous)

    def test_camera_mapping_rejects_missing_index(self):
        with self.assertRaises(ValueError):curve.pose_list_from_mapping({'0_0_0':1})


if __name__=='__main__':unittest.main(verbosity=2)
