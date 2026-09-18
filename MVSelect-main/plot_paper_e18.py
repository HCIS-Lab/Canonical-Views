#!/usr/bin/env python3
"""Render E18 paper figures from independently audited CSVs; no new training.

python3 plot_paper_e18.py --data-root PATH_TO_DERIVED_CSVS --output figures
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COND=['early','late','late_spread','late_cluster','random_fixed','random_resampled']
LABEL=['Epoch-10 selections','Epoch-100 selections','Widely spaced replacements','Closely spaced replacements','Fixed random views','Resampled random views']
COL=['#0072B2','#D55E00','#882255','#CC79A7','#777777','#222222']
STYLES=['-','--','-.',':','--','-'];MARKERS=['o','s','^','D','P','X']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
def clean(ax,letter,title):
 ax.set_title(title,loc='left',pad=11)
 ax.text(-.12,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=13)
 ax.grid(axis='x',color='#e6e6e6',lw=.5);ax.set_axisbelow(True)
def save(fig,out,name):
 for ext in ['png','svg','pdf']:fig.savefig(out/(name+'.'+ext),dpi=240,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def build(root,out):
 out.mkdir(parents=True,exist_ok=True)
 def read(name):
  path=root/name
  return pd.read_csv(path if path.exists() else root/('e18_'+name))
 lc=read('learning_curves.csv');ss=read('source_learning_curves.csv')
 geo=read('view_set_geometry.csv');ct=read('contrasts_recomputed.csv');ct.stage=ct.stage.astype(str)
 dg=read('update_diagnostics.csv');source=read('source_recognizer_scores.csv')
 intervals=read('crossed_object_intervals.csv');intervals.stage=intervals.stage.astype(str)
 fig,axs=plt.subplots(2,2,figsize=(8.5,7.25),gridspec_kw={'height_ratios':[1.15,1]})
 fig.subplots_adjust(left=.26,right=.98,bottom=.09,top=.94,wspace=.5,hspace=.66)
 for ax,stage,letter,ttl in zip(axs[0],[0,30],['a','b'],['New classifier','After 30 epochs on random views']):
  for y,(c,color) in enumerate(zip(COND,COL)):
   z=lc[(lc.stage==stage)&(lc.epoch==100)&(lc.condition==c)]
   means=z.groupby('source_seed').accuracy_pct.mean()
   # Every job shown, faint; source means shown as circles. Random controls
   # repeat across source jobs, so show three recipient means as squares.
   if c.startswith('random'):
    vals=z.groupby('recipient_seed').accuracy_pct.mean().values;marker='s'
   else: vals=means.values;marker='o'
   ax.scatter(vals,y+np.linspace(-.16,.16,len(vals)),color=color,s=18,marker=marker,alpha=.9,zorder=3)
   ax.plot([z.accuracy_pct.mean()]*2,[y-.27,y+.27],color=color,lw=2.3)
  ax.set_yticks(range(6),LABEL if stage==0 else ['']*6);ax.invert_yaxis();ax.set_xlim(40,72);ax.set_xticks([40,50,60,70]);ax.set_xlabel('Test accuracy (%)')
  clean(ax,letter,ttl)
 ax=axs[1,0]
 cg=geo.groupby(['source_seed','condition']).mean_pair_angle_deg.mean().unstack()
 for s in range(5):ax.plot([0,1,2],cg.loc[s,['late_cluster','late','late_spread']],color='#aaaaaa',lw=.9,alpha=.7)
 for x,c in enumerate(['late_cluster','late','late_spread']):
  ax.scatter([x]*5,cg[c],s=20,color=COL[COND.index(c)],zorder=4)
 ax.set_xticks([0,1,2],['Closely\nspaced','Epoch-100\nselections','Widely\nspaced']);ax.set_ylabel('Mean pairwise camera angle (°)');ax.set_ylim(25,115);ax.grid(axis='y',color='#e6e6e6',lw=.5)
 clean(ax,'c','Same view types, different images')
 ax=axs[1,1]
 z=ct[(ct.horizon_epochs==20)&(ct.contrast=='early_minus_late')];p=z.groupby(['source_seed','stage']).loss_advantage.mean().unstack()
 # loss_advantage for early-minus-late is final loss minus early loss;
 # negate so positive means final selections have lower loss.
 for s in range(5):ax.plot([0,1],[-p.loc[s,'0'],-p.loc[s,'30']],color='#b3b3b3',lw=.8,marker='o',ms=3)
 m=[-p['0'].mean(),-p['30'].mean()];ax.plot([0,1],m,color='#0072B2',lw=2,marker='D',ms=5)
 ax.axhline(0,color='#777777',lw=.8,ls='--');ax.set_xticks([0,1],['Before category\ntraining','After random-view\ntraining']);ax.set_xlim(-.3,1.3);ax.set_ylabel('Cross-entropy difference');ax.set_ylim(-.17,.20)
 clean(ax,'d','Prior training changes the benefit')
 ax.text(.02,.98,'Epoch-10 minus epoch-100 selections\n20 replay epochs',transform=ax.transAxes,va='top',fontsize=7.4)
 save(fig,out,'figure_5')

 fig,axs=plt.subplots(3,2,figsize=(8.4,8.7));fig.subplots_adjust(left=.10,right=.98,bottom=.11,top=.95,wspace=.35,hspace=.55)
 for col,stage in enumerate([0,30]):
  for row,metric in enumerate(['accuracy_pct','loss']):
   ax=axs[row,col]
   for ci,(c,color,lab) in enumerate(zip(COND,COL,LABEL)):
    z=ss[(ss.stage==stage)&(ss.condition==c)].groupby('epoch')[metric].mean()
    ax.plot(z.index,z.values,color=color,lw=1.5,ls=STYLES[ci],marker=MARKERS[ci],ms=2.2,label=lab)
   ax.set_xlabel('Additional training epoch');ax.set_xlim(0,100);ax.set_xticks([0,20,40,60,80,100]);ax.set_ylabel('Accuracy (%)' if row==0 else 'Cross-entropy')
   ax.set_ylim((0,75) if row==0 else (1.1,4.1))
   clean(ax,'abcd'[row*2+col],('New classifier' if stage==0 else 'After 30 epochs on random views'))
 for ax,ep,letter in zip(axs[2],[10,100],['e','f']):
  z=source[source.source_epoch==ep].groupby(['source_seed','condition']).accuracy_pct.mean().unstack()
  for s in range(5):ax.plot(range(4),z.loc[s,COND[:4]],color='#c7c7c7',lw=.7)
  for x,c in enumerate(COND[:4]):ax.scatter([x]*5,z[c],color=COL[x],s=13,zorder=3)
  ax.set_xticks(range(4),['Epoch 10','Epoch 100','Widely\nspaced','Closely\nspaced']);ax.set_ylabel('Selecting-model accuracy (%)');ax.set_ylim(45,92)
  clean(ax,letter,f'Selecting model at epoch {ep}')
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.5,.008))
 save(fig,out,'extended_data_9')

 fig=plt.figure(figsize=(8.4,6.7));gs=fig.add_gridspec(2,2,height_ratios=[1,1.05],left=.11,right=.98,bottom=.09,top=.94,hspace=.62,wspace=.4)
 for col,stage in enumerate([0,30]):
  ax=fig.add_subplot(gs[0,col]);z=dg[(dg.stage==stage)&(dg.step_size==.0001)].groupby(['source_seed','condition']).gradient_cosine.mean().unstack()
  for x,c in enumerate(COND):ax.scatter(x+np.linspace(-.13,.13,5),z[c],color=COL[x],s=14)
  ax.axhline(0,color='#888888',lw=.7);ax.set_xticks(range(6),['Epoch 10','Epoch 100','Widely\nspaced','Closely\nspaced','Fixed\nrandom','Resampled\nrandom'],rotation=25,ha='right');ax.set_ylabel('Gradient cosine');ax.set_ylim(-.35,.85)
  clean(ax,'ab'[col],'New classifier' if stage==0 else 'After 30 epochs on random views')
 ax=fig.add_subplot(gs[1,:]);colors={.00001:'#0072B2',.0001:'#D55E00'}
 for step,z in dg.groupby('step_size'):
  ax.scatter(z.predicted_probe_loss_reduction,z.actual_probe_loss_reduction,color=colors[step],s=13,marker='o' if step==.00001 else '^',alpha=.65,label=f'Step size {step:g}')
 lim=[min(dg.predicted_probe_loss_reduction.min(),dg.actual_probe_loss_reduction.min())-.0005,max(dg.predicted_probe_loss_reduction.max(),dg.actual_probe_loss_reduction.max())+.0005];ax.plot(lim,lim,color='#777777',lw=.8,ls='--');ax.set_xlim(lim);ax.set_ylim(lim)
 ax.set_xlabel('Predicted probe-loss reduction');ax.set_ylabel('Actual probe-loss reduction');ax.legend(frameon=False,fontsize=8,loc='upper left')
 clean(ax,'c','Disposable SGD updates test the local approximation')
 save(fig,out,'supplementary_6')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.data_root,a.output)
