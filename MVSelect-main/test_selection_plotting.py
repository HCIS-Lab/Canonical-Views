"""Synthetic regression checks; no research metadata or training is used."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from export_selection_plot_data import export, KEYS
from selection_plotting import read_priors, write_selection_outputs
from aggregate_meta_json import aggregate_experiment

def record(expanded=.1, expanded_like=.2, length=4):
    r={key:[value]*length for key,value in zip(KEYS,[expanded,expanded_like,.1,.1,.5])}
    for key in ['accuracy','all_views_accuracy','random_accuracy']:
        r[key]=[60+i for i in range(length)]
    for base in ['expanded','expanded_like','foreshortened','foreshortened_like','remainder']:
        for suffix in ['', '_3', '_5']:r[base+'_accuracy'+suffix]=[50+i for i in range(length)]
    r['per_class_acc']=[[float(i+j) for j in range(32)] for i in range(length)]
    r.update(recognition_num_views=6,steps=5,seed=1)
    return r

class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
    def put(self,name,r):
        (self.folder/name).write_text(json.dumps(r))
    def test_family_sem_preserves_covariance_and_original_lengths(self):
        self.put('a_meta.json',record(.1,.2,4));self.put('b_meta.json',record(.2,.1,3))
        self.put('old_meta.json',{k:[.2]*3 for k in KEYS})
        d=export(self.folder)
        np.testing.assert_allclose(d['family_summary']['Expanded family']['sem'],0,atol=1e-15)
        np.testing.assert_allclose(d['subtype_summary']['expanded']['sem'],.05)
        self.assertEqual([r['original_length'] for r in d['runs']],[4,3])
        self.assertEqual(d['array_indices'],[0,1,2]);self.assertEqual(len(d['skipped_files']),1)
        self.assertEqual(export(self.folder,'selection')['included_files'],3)
    def test_single_run_uncertainty_is_unavailable(self):
        self.put('a_meta.json',record())
        self.assertEqual(export(self.folder)['subtype_summary']['expanded']['sem'],[None]*4)
    def test_invalid_shares_and_priors_are_rejected(self):
        r=record();r['expanded'][0]=.6;self.put('a_meta.json',r)
        with self.assertRaisesRegex(ValueError,'sum to one'):export(self.folder)
        p=self.folder/'priors.json';p.write_text(json.dumps(dict.fromkeys(KEYS,.1)))
        with self.assertRaisesRegex(ValueError,'sum to one'):read_priors(p)
    def test_legacy_accuracy_and_selection_outputs(self):
        a=record(.1,.2,4);b=record(.2,.1,3)
        b['expanded_accuracy_5']=b['expanded_accuracy_5'][:2]
        self.put('a_meta.json',a);self.put('b_meta.json',b)
        args=argparse.Namespace(candidate_priors=None,epoch_start=0)
        aggregate_experiment(str(self.folder),args)
        result=json.loads((self.folder/'aggregated_summary.json').read_text())
        np.testing.assert_allclose(result['mean_smoothed_curves']['expanded'],[.15]*3)
        np.testing.assert_allclose(result['mean_accuracy'],[60,61,62])
        self.assertEqual(result['accuracy_alignment']['common_points'],2)
        self.assertEqual(result['selection_reporting']['truncated_files'],['a_meta.json'])
        self.assertTrue((self.folder/'aggregated_view_trajectories.pdf').is_file())
        self.assertTrue((self.folder/'aggregated_view_family_lift.svg').is_file())
        source=json.loads((self.folder/'selection_plot_data.json').read_text())
        self.assertEqual(source['runs'][0]['original_length'],4)
        self.assertEqual(source['plotted_epochs'],[0,1,2])
    def test_explicit_single_view_mode_custom_priors_and_epoch_origin(self):
        self.put('sv_meta.json',{k:[.2,.2] for k in KEYS})
        p=self.folder/'priors.json';p.write_text(json.dumps(dict.fromkeys(KEYS,.2)))
        with patch('selection_plotting.save'):
            result=write_selection_outputs(self.folder,'selection',p,1)
        import matplotlib.pyplot as plt
        plt.close('all')
        self.assertEqual(result['selection_reporting']['plotted_epochs'],[1,2])
        self.assertEqual(result['sem_smoothed_curves']['expanded'],[None,None])
        np.testing.assert_allclose(result['view_family']['expanded_family_lift_mean'],[1,1])

if __name__=='__main__':unittest.main()
