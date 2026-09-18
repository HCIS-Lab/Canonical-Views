"""Independent, read-only verification of the returned E18 records."""
import csv, gzip, hashlib, itertools, json, math, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True,help='Directory containing job01 through job15')
parser.add_argument('--output',type=Path,required=True,help='New derived-data directory')
parser.add_argument('--code-dir',type=Path,default=Path(__file__).resolve().parent,help='Repository directory with E18 code and assets')
args=parser.parse_args()
ROOT=args.root;OUT=args.output;OUT.mkdir(parents=True,exist_ok=True);CODE=args.code_dir
sys.path.insert(0,str(CODE))
from paper_mechanism_core import CONDITIONS,canonical_hash,evaluation_views,family_counts,spread,training_views,job_spec
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return pd.read_csv(p)
all_frames={k:[] for k in ['learning_curves','update_diagnostics','source_recognizer_scores','branch_audit','warmup_metrics']}
all_contrasts=[]; jobs=[]; geometry=[]; object_endpoints=[]; input_hashes=[]
max_acc=max_loss=max_probability=0.
for ji in range(1,16):
    d=ROOT/f'job{ji:02d}'; plan=json.loads((d/'run_plan.json').read_text()); spec=job_spec(ji)
    assert all(plan[k]==v for k,v in spec.items()) and plan['completed'] and plan['scientific_result'] and not plan['software_smoke_test']
    assert plan['conditions']==list(CONDITIONS) and plan['stages']==[0,30] and plan['adaptation_epochs']==100
    for name,h in plan['source_code_sha256'].items(): assert sha(d/name)==sha(CODE/name)==h
    m=json.load(gzip.open(d/'input_manifest.json.gz','rt'))
    assert m['source_bank_sha256']==sha(CODE/f'paper_mechanism_assets/source_seed{spec["source_seed"]}.json')
    assert m['registry_sha256']==sha(CODE/'paper_mechanism_assets/registry.json')
    train,probe,test=[m[k] for k in ('train_objects','diagnostic_objects','endpoint_objects')]
    assert [len(x) for x in (train,probe,test)]==[160,64,96]
    oidsets=[set(o['object_id'] for o in a) for a in (train,probe,test)]
    assert not any(a&b for a,b in itertools.combinations(oidsets,2))
    assert canonical_hash(train)==m['training_design_sha256']==plan['training_design_sha256']
    assert canonical_hash([(o['object_id'],[evaluation_views(o,r) for r in range(3)]) for o in test])==plan['endpoint_viewsets_sha256']
    input_hashes.append(dict(**spec, order_hash=canonical_hash([o['object_id'] for o in train]), warmup_hash=canonical_hash([[training_views(o,'warmup',e,spec['recipient_seed']) for o in train] for e in range(1,31)]),fixed_hash=canonical_hash([o['sets']['random_fixed'] for o in train]), endpoint_hash=plan['endpoint_viewsets_sha256']))
    for o in train:
        for c,names in o['sets'].items():
            assert len(names)==len(set(names))==5 and set(names)<=set(o['candidates'])
            assert family_counts(names)==o['set_statistics'][c]['family_counts']
            angle=spread(names); assert abs(angle-o['set_statistics'][c]['mean_pair_angle_deg'])<1e-10
            if c in ('late_spread','late_cluster'): assert family_counts(names)==family_counts(o['sets']['late'])
            geometry.append(dict(**spec,object_id=o['object_id'],class_idx=o['class_idx'],condition=c,mean_pair_angle_deg=angle,overlap_with_late=len(set(names)&set(o['sets']['late']))))
        assert spread(o['sets']['late_cluster'])<=spread(o['sets']['late'])+1e-10<=spread(o['sets']['late_spread'])+2e-10
    frames={k:read(d/(k+'.csv')) for k in all_frames}
    lc=frames['learning_curves']; br=frames['branch_audit']; dg=frames['update_diagnostics']
    assert len(lc)==144 and len(br)==12 and len(dg)==24
    assert not lc.duplicated(['stage','condition','epoch']).any()
    assert set(lc.epoch)=={0,1,*range(10,101,10)}
    assert set(br.actual_updates)==set(br.expected_updates)=={2700}
    for stage,sub in br.groupby('stage'):
        assert sub.initial_state_sha256.nunique()==sub.initial_optimizer_sha256.nunique()==sub.initial_optimizer_steps.nunique()==1
        assert set(dg[dg.stage==stage].initial_state_sha256)==set(sub.initial_state_sha256)
    assert np.max(abs(dg.gradient_inner_product-dg.train_gradient_norm*dg.probe_gradient_norm*dg.gradient_cosine))<1e-8
    assert np.max(abs(dg.predicted_probe_loss_reduction-dg.step_size*dg.gradient_inner_product))<1e-12
    assert np.max(abs(dg.actual_probe_loss_reduction-(dg.before_probe_loss-dg.after_probe_loss)))<1e-12
    pr=read(d/'endpoint_predictions.csv'); assert len(pr)==38016
    assert not pr.duplicated(['stage','condition','epoch','object_id','repeat']).any()
    assert set(pr.object_id)==oidsets[2] and set(pr.repeat)=={0,1,2}
    assert (pr.correct==(pr.class_idx==pr.prediction).astype(int)).all()
    max_probability=max(max_probability,float(np.max(abs(np.exp(-pr.loss)-pr.true_class_probability))))
    assert max_probability<1e-6
    recal=pr.groupby(['stage','condition','epoch']).agg(loss=('loss','mean'),accuracy_pct=('correct',lambda a:100*a.mean()),n=('correct','size')).reset_index()
    assert set(recal.n)=={288}
    merged=recal.merge(lc,on=['stage','condition','epoch'],suffixes=('_raw','_summary'),validate='one_to_one')
    max_acc=max(max_acc,float(np.max(abs(merged.accuracy_pct_raw-merged.accuracy_pct_summary))))
    max_loss=max(max_loss,float(np.max(abs(merged.loss_raw-merged.loss_summary))))
    assert max_acc<1e-10 and max_loss<1e-10
    ep=read(d/'endpoint_metrics.csv').merge(lc[lc.epoch==100],on=['stage','condition'],suffixes=('_endpoint','_curve'),validate='one_to_one')
    for k in ('loss','accuracy_pct'): assert np.max(abs(ep[k+'_endpoint']-ep[k+'_curve']))<1e-10
    pairs=[('early','late'),('late_spread','late_cluster'),('late_spread','late'),('random_resampled','random_fixed')]
    contrasts=[]
    for epoch in (20,100):
        z=lc[lc.epoch==epoch].set_index(['stage','condition'])
        for stage in (0,30):
            for a,b in pairs:
                contrasts.append(dict(horizon_epochs=epoch,stage=str(stage),contrast=a+'_minus_'+b,accuracy_difference_pp=z.loc[(stage,a),'accuracy_pct']-z.loc[(stage,b),'accuracy_pct'],loss_advantage=z.loc[(stage,b),'loss']-z.loc[(stage,a),'loss']))
        contrasts.append(dict(horizon_epochs=epoch,stage='interaction',contrast='(late-early)_stage30_minus_stage0',accuracy_difference_pp=(z.loc[(30,'late'),'accuracy_pct']-z.loc[(30,'early'),'accuracy_pct'])-(z.loc[(0,'late'),'accuracy_pct']-z.loc[(0,'early'),'accuracy_pct']),loss_advantage=(z.loc[(30,'early'),'loss']-z.loc[(30,'late'),'loss'])-(z.loc[(0,'early'),'loss']-z.loc[(0,'late'),'loss'])))
    old=read(d/'contrasts.csv'); old.stage=old.stage.astype(str)
    new=pd.DataFrame(contrasts); cmp=new.merge(old,on=['horizon_epochs','stage','contrast'],suffixes=('_new','_old'),validate='one_to_one');assert len(cmp)==18
    for k in ['accuracy_difference_pp','loss_advantage']:assert np.max(abs(cmp[k+'_new']-cmp[k+'_old']))<1e-10
    all_contrasts.extend(dict(**spec,**r) for r in contrasts)
    for k,df in frames.items():
        for key,v in spec.items():df[key]=v
        all_frames[k].append(df)
    pp=pr[pr.epoch.isin([20,100])].groupby(['stage','condition','epoch','object_id','class_idx']).agg(accuracy_pct=('correct',lambda v:100*v.mean()),loss=('loss','mean')).reset_index()
    for k,v in spec.items():pp[k]=v
    object_endpoints.append(pp)
    jobs.append(dict(**spec,gpu=plan['gpu_name'],torch=plan['torch_version'],cuda=plan['cuda_version'],initial_hash=plan['recipient_initial_sha256']))

for k,v in all_frames.items():
    all_frames[k]=pd.concat(v,ignore_index=True); all_frames[k].to_csv(OUT/(k+'.csv'),index=False)
pd.DataFrame(jobs).to_csv(OUT/'job_environment.csv',index=False)
ih=pd.DataFrame(input_hashes);ih.to_csv(OUT/'input_hashes.csv',index=False)
for seed,rows in ih.groupby('recipient_seed'):
    for col in ('order_hash','warmup_hash','fixed_hash','endpoint_hash'):assert rows[col].nunique()==1
assert ih.endpoint_hash.nunique()==1
pd.DataFrame(geometry).to_csv(OUT/'view_set_geometry.csv',index=False)
pd.concat(object_endpoints,ignore_index=True).to_csv(OUT/'object_endpoints.csv',index=False)
ct=pd.DataFrame(all_contrasts);ct.to_csv(OUT/'contrasts_recomputed.csv',index=False)

# Intervals describe seed variation conditional on these objects. Crossed
# source/recipient resampling avoids pretending there are 15 independent sources.
rng=np.random.default_rng(20260916);B=50000
si=rng.integers(0,5,(B,5));ri=rng.integers(0,3,(B,3))
summary=[]
for key,g in ct.groupby(['horizon_epochs','stage','contrast']):
    for metric in ['accuracy_difference_pp','loss_advantage']:
        x=g.pivot(index='source_seed',columns='recipient_seed',values=metric).values
        bs=x[si[:,:,None],ri[:,None,:]].mean(axis=(1,2)); sm=x.mean(axis=1)
        unit=x.mean(axis=0) if key[2].startswith('random_') else sm
        signs=np.array(list(itertools.product([-1,1],repeat=len(unit))))
        exact=np.mean(abs(signs@unit/len(unit))>=abs(unit.mean())-1e-12)
        summary.append(dict(horizon_epochs=key[0],stage=key[1],contrast=key[2],metric=metric,mean=x.mean(),source_mean_sd=sm.std(ddof=1),recipient_mean_sd=x.mean(axis=0).std(ddof=1),source_min=sm.min(),source_max=sm.max(),bootstrap_low=np.quantile(bs,.025),bootstrap_high=np.quantile(bs,.975),conditional_signflip_p=exact,signflip_unit='recipient seed' if key[2].startswith('random_') else 'source mean conditional on three recipients',n_sources=5,n_recipient_seeds=3))
pd.DataFrame(summary).to_csv(OUT/'contrast_statistics.csv',index=False)
lc=all_frames['learning_curves']
sm=lc.groupby(['source_seed','stage','condition','epoch'])[['accuracy_pct','loss']].mean().reset_index();sm.to_csv(OUT/'source_learning_curves.csv',index=False)
cs=sm.groupby(['stage','condition','epoch']).agg(accuracy_mean=('accuracy_pct','mean'),accuracy_sd=('accuracy_pct','std'),loss_mean=('loss','mean'),loss_sd=('loss','std')).reset_index();cs.to_csv(OUT/'condition_summary.csv',index=False)
br=all_frames['branch_audit'];starts=br.groupby(['job_id','source_seed','recipient_seed','stage']).first().reset_index()
starts.to_csv(OUT/'branch_starts.csv',index=False)
state_report=[]
for (seed,stage),g in starts.groupby(['recipient_seed','stage']):
    state_report.append(dict(recipient_seed=int(seed),stage=int(stage),distinct_states=int(g.initial_state_sha256.nunique()),distinct_optimizer_states=int(g.initial_optimizer_sha256.nunique()),min_accuracy=float(g.initial_accuracy_pct.min()),max_accuracy=float(g.initial_accuracy_pct.max())))
diagn=all_frames['update_diagnostics']; rel=abs(diagn.actual_probe_loss_reduction-diagn.predicted_probe_loss_reduction)/abs(diagn.predicted_probe_loss_reduction)
audit=dict(completed_jobs=15,recipient_models=180,raw_prediction_rows=570240,source_models=5,recipient_seeds=3,all_expected_cells_present=True,all_code_and_input_hashes_match=True,all_within_job_branches_matched=True,equal_update_budgets=True,train_probe_endpoint_objects=[160,64,96],endpoint_draws_per_object=3,raw_accuracy_max_error=max_acc,raw_loss_max_error=max_loss,probability_max_error=max_probability,between_source_baselines=state_report,gradient_relative_error_median=float(np.median(rel)),gradient_relative_error_max=float(rel.max()),gradient_signs_agree=bool((np.sign(diagn.actual_probe_loss_reduction)==np.sign(diagn.predicted_probe_loss_reduction)).all()),limitations=['Intervals resample five source and three recipient seeds, conditional on the same endpoint objects; descriptive, not population-level guarantees.','Three recipient seeds are crossed, not 15 independent source-model replicates.','Across source jobs, warmup states can differ despite identical intended images and settings; comparisons are matched within job.','Reference random controls repeat the same input designs across sources; they do not provide five independent data-policy replications.','No retained checkpoint or original image pixels are in this archive; hashes and logged numeric outputs are audited, not GPU execution rerun.'])
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2));print(cs[cs.epoch==100].to_string(index=False))
print(pd.DataFrame(summary).query('metric=="accuracy_difference_pp"').to_string(index=False))
