import copy
import csv
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from paper_mechanism_core import (CONDITIONS, check_disjoint, contrast_values, direction,
 family_counts, job_spec, make_design, matched_spread_sets, spread, training_views)


def names(oid='object'):
    return [f'{oid}_{i}_e0.0_a{a:.1f}_r0.0.png' for i,a in enumerate(range(-180,180,15))]


class DesignTests(unittest.TestCase):
    def test_direction(self):
        self.assertAlmostEqual(direction('obj_0_e90_a0_r0.png')[1],1)
        self.assertAlmostEqual(direction('obj_0_e0_a90_r0.png')[0],1)
        with self.assertRaises(ValueError):direction('obj_0_e0_a0_r10.png')

    def test_matched_intervention(self):
        ns=names()
        # Some candidates in a separate family: must preserve the exact histogram.
        ns[0]=ns[0].replace('.png','_planar.png')
        ref=ns[:5]
        high,low=matched_spread_sets(ns,ref,'unit')
        self.assertEqual(family_counts(high),family_counts(ref))
        self.assertEqual(family_counts(low),family_counts(ref))
        self.assertGreaterEqual(spread(high)+1e-9,spread(ref))
        self.assertLessEqual(spread(low),spread(ref)+1e-9)
        self.assertEqual(len(set(high)),5)
        self.assertEqual((high,low),matched_spread_sets(ns,ref,'unit'))

    def test_schedule(self):
        ns=names();obj={'object_id':'object','class_idx':0,'candidates':ns,'early':ns[:5],'late':ns[-5:]}
        o=make_design({'objects':[obj]},42)[0]
        for c in CONDITIONS:
            first=training_views(o,c,1,0); second=training_views(o,c,2,0)
            self.assertEqual(len(set(first)),5)
            if c!='random_resampled':self.assertEqual(first,second)
        self.assertNotEqual(training_views(o,'random_resampled',1,0),training_views(o,'random_resampled',2,0))
        self.assertEqual([job_spec(i)['source_seed'] for i in range(1,6)],list(range(5)))
        self.assertEqual(job_spec(15)['recipient_seed'],2)
        with self.assertRaises(ValueError):job_spec(0)

    def test_leakage_and_interaction(self):
        with self.assertRaises(ValueError):check_disjoint([{'object_id':'x'}],[{'object_id':'x'}],[])
        rows=[{'stage':s,'condition':c,'accuracy_pct':v} for s,c,v in
              [(0,'early',20),(0,'late',10),(30,'early',50),(30,'late',60)]]
        interaction=[r for r in contrast_values(rows) if r['stage']=='interaction'][0]
        self.assertEqual(interaction['accuracy_difference_pp'],20)


class SummaryTests(unittest.TestCase):
    def test_summary_rejects_smoke_missing_and_unequal_branches(self):
        import summarize_paper_mechanism as sm
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(FileNotFoundError):sm.summarize(root,root/'missing')
            folder=root/'job01';folder.mkdir()
            (folder/'run_plan.json').write_text(json.dumps({'completed':True,'scientific_result':False}))
            with self.assertRaises(ValueError):sm.summarize(root,root/'smoke')

    def test_full_crossed_summary(self):
        import gzip
        import summarize_paper_mechanism as sm
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for j in range(1,16):
                folder=root/f'job{j:02d}';folder.mkdir();spec=job_spec(j)
                plan={**spec,'completed':True,'scientific_result':True,'software_smoke_test':False,
                      'stages':[0,30],'adaptation_epochs':100,'conditions':list(CONDITIONS),
                      'endpoint_viewsets_sha256':'common','training_design_sha256':'design',
                      **{k:'fixed' for k in ('protocol','batch_size','lr','weight_decay','optimizer',
                      'branch_schedule','warmup','recognition_images','pooling','training_bn',
                      'diagnostic_bn','source_code_sha256')}}
                (folder/'run_plan.json').write_text(json.dumps(plan))
                ends=[dict(stage=s,condition=c,accuracy_pct=50+spec['source_seed'],loss=1.,n_objects=96,n_view_sets=288)
                      for s in (0,30) for c in CONDITIONS]
                audits=[dict(stage=s,condition=c,initial_state_sha256=f'{spec["recipient_seed"]}_{s}',
                             initial_optimizer_steps=str(s),initial_optimizer_sha256=f'opt{s}',actual_updates=2700,expected_updates=2700)
                        for s in (0,30) for c in CONDITIONS]
                sm.write_csv(folder/'endpoint_metrics.csv',ends)
                sm.write_csv(folder/'branch_audit.csv',audits)
                sm.write_csv(folder/'contrasts.csv',[dict(horizon_epochs=h,**r) for h in (20,100) for r in contrast_values(ends)])
                with gzip.open(folder/'input_manifest.json.gz','wt') as f:json.dump({'training_design_sha256':'design'},f)
            report=sm.summarize(root,root/'good')
            self.assertTrue(report['cross_source_warmup_states_match'])
            rows=list(csv.DictReader((root/'good/contrast_summary.csv').read_text().splitlines()))
            self.assertTrue(all(r['n_source_models']=='5' for r in rows))
            self.assertTrue(all(float(r['mean_difference'])==0 for r in rows))
            # Corrupt one starting state. The output must fail closed.
            p=root/'job01/branch_audit.csv';rows=list(csv.DictReader(p.read_text().splitlines()))
            rows[0]['initial_optimizer_sha256']='different';sm.write_csv(p,rows)
            with self.assertRaises(ValueError):sm.summarize(root,root/'bad')


class TensorTests(unittest.TestCase):
    def test_virtual_step_direction_and_restore(self):
        import torch
        import run_paper_mechanism as run
        # A direct finite difference verifies the sign and magnitude convention.
        theta=torch.nn.Parameter(torch.tensor([.25,-.5],dtype=torch.float64))
        source_target=torch.tensor([1.,1.],dtype=torch.float64)
        probe_target=torch.tensor([2.,0.],dtype=torch.float64)
        loss=((theta-source_target)**2).sum()/2
        gs=torch.autograd.grad(loss,theta)[0]
        vl=((theta-probe_target)**2).sum()/2
        gv=torch.autograd.grad(vl,theta)[0]
        eta=1e-6
        after=(((theta-eta*gs)-probe_target)**2).sum()/2
        self.assertAlmostEqual(float((vl-after)/eta),float(gs@gv),places=4)

    def test_complete_synthetic_smoke(self):
        import torch
        from PIL import Image
        import run_paper_mechanism as run
        torch.set_num_threads(1)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base=torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.BatchNorm2d(4),torch.nn.ReLU())
                self.classifier=torch.nn.Linear(4,32)
            def forward(self,x):
                b,n,c,h,w=x.shape
                z=self.base(x.reshape(b*n,c,h,w)).mean((2,3)).reshape(b,n,4).max(1).values
                return self.classifier(z)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);train=[];probe=[];test=[]
            for role,seq in [('test',train),('probe',probe),('test-down',test)]:
                for cls in (0,1):
                    oid=f'{role}{cls}'; ns=names(oid)
                    directory=root/f'category{cls}'/role; directory.mkdir(parents=True)
                    for j,n in enumerate(ns):
                        Image.new('RGB',(12,12),((j*7)%255,30+cls*100,120)).save(directory/n)
                    seq.append(dict(class_idx=cls,category=f'category{cls}',split=role,object_id=oid,
                                    candidates=ns,early=ns[:5],late=ns[-5:]))
            train=make_design({'objects':train},9000)
            model=Toy();state=model.state_dict();cp=root/'cp.pth';torch.save(state,cp)
            manifest={'spec':job_spec(1),'source_bank_sha256':'test','training_design_sha256':'test',
                      'endpoint_viewsets_sha256':'test','train_objects':train,'diagnostic_objects':probe,
                      'endpoint_objects':test,'checkpoints':{str(e):{'path':str(cp)} for e in (10,100)},
                      'recipient_backbone':{'path':str(cp)}}
            args=SimpleNamespace(output=root/'out',smoke_test=True,cpu_smoke=True,gpu_id=0,
                 workers=0,batch_size=2,cpu_threads=1,recipient_seed=0,job_id=1,data_root=root)
            original=run.make_model
            run.make_model=Toy
            try:run.run(args,manifest)
            finally:run.make_model=original
            plan=json.loads((args.output/'run_plan.json').read_text())
            self.assertTrue(plan['completed']);self.assertFalse(plan['scientific_result'])
            rows=list(csv.DictReader((args.output/'branch_audit.csv').read_text().splitlines()))
            self.assertEqual(len(rows),12)
            for stage in ('0','1'):
                sub=[r for r in rows if r['stage']==stage]
                self.assertEqual(len({r['initial_state_sha256'] for r in sub}),1)
                self.assertEqual(len({r['initial_optimizer_steps'] for r in sub}),1)
                self.assertEqual(len({r['initial_optimizer_sha256'] for r in sub}),1)
                self.assertTrue(all(r['expected_updates']==r['actual_updates'] for r in sub))
            diag=list(csv.DictReader((args.output/'update_diagnostics.csv').read_text().splitlines()))
            self.assertEqual(len(diag),24)
            self.assertTrue(all(math.isfinite(float(r['actual_probe_loss_reduction'])) for r in diag))
            self.assertTrue((args.output/'results_to_return.tar.gz').is_file())


if __name__=='__main__':unittest.main()
