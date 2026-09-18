"""Exploratory crossed source/recipient/object intervals, preserving category."""
from pathlib import Path
import numpy as np,pandas as pd,argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-root',type=Path,required=True);P=parser.parse_args().data_root
d=pd.read_csv(P/'object_endpoints.csv')
objs=d[['class_idx','object_id']].drop_duplicates().sort_values(['class_idx','object_id']);assert len(objs)==96
oindex=objs.object_id.tolist(); index=pd.MultiIndex.from_product([range(5),range(3),oindex],names=['source_seed','recipient_seed','object_id'])
def array(epoch,stage,condition,metric):
    return d[(d.epoch==epoch)&(d.stage==stage)&(d.condition==condition)].set_index(['source_seed','recipient_seed','object_id'])[metric].reindex(index).to_numpy().reshape(5,3,96)
rng=np.random.default_rng(20260916);B=20000
ss=rng.integers(0,5,(B,5));rr=rng.integers(0,3,(B,3)); oo=(np.arange(32)[None,:,None]*3+rng.integers(0,3,(B,32,3))).reshape(B,96)
out=[]
for epoch in (20,100):
 for metric in ('accuracy_pct','loss'):
  for stage in (0,30,'interaction'):
   pairs=[('early','late'),('late_spread','late_cluster'),('late_spread','late'),('random_resampled','random_fixed')] if stage!='interaction' else [('late','early')]
   for a,b in pairs:
    if stage=='interaction': x=(array(epoch,30,a,metric)-array(epoch,30,b,metric))-(array(epoch,0,a,metric)-array(epoch,0,b,metric));contrast='(late-early)_stage30_minus_stage0'
    else:x=array(epoch,stage,a,metric)-array(epoch,stage,b,metric);contrast=a+'_minus_'+b
    if metric=='loss':x=-x
    draws=[]
    for i in range(0,B,1000):
     z=x[ss[i:i+1000,:,None,None],rr[i:i+1000,None,:,None],oo[i:i+1000,None,None,:]]
     draws.append(z.mean(axis=(1,2,3)))
    bs=np.concatenate(draws)
    out.append(dict(horizon_epochs=epoch,stage=str(stage),contrast=contrast,metric='loss_advantage' if metric=='loss' else 'accuracy_difference_pp',mean=x.mean(),crossed_object_ci_low=np.quantile(bs,.025),crossed_object_ci_high=np.quantile(bs,.975)))
out=pd.DataFrame(out);out.to_csv(P/'crossed_object_intervals.csv',index=False)
print(out.to_string(index=False))
