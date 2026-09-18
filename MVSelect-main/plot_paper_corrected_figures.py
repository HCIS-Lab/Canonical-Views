#!/usr/bin/env python3
"""Rebuild review figures from the audited, portable source_data CSV directory.
No raw-data modification or model inference. Run with --data-dir and --output.
"""
from pathlib import Path
import argparse,json,string
from PIL import Image
from matplotlib.path import Path as MplPath
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon,FancyBboxPatch,FancyArrowPatch
P=argparse.ArgumentParser();P.add_argument('--data-dir',type=Path,required=True);P.add_argument('--output',type=Path,required=True);P.add_argument('--assets-dir',type=Path,help='Object illustrations; defaults to the assets folder beside source_data');A=P.parse_args();D=A.data_dir;O=A.output;O.mkdir(parents=True,exist_ok=True);AS=A.assets_dir or (D.parent/'assets' if (D.parent/'assets/object_examples.json').exists() else Path(__file__).resolve().parent/'paper_review_assets')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
C=['#0072B2','#E69F00','#882255','#CC79A7','#666666','#56B4E9','#D55E00','#332288']
TC=['#0077BB','#AA3377','#EE7733','#009988','#555555']
TSTYLE=['-','--',':','-.',(0,(5,2,1,2))];TMARK=['o','^','s','D','x']
MARK=['o','s','^','D','v','P','X','h']
F=['Expanded','Expanded-like','Foreshortened','Foreshortened-like','Remainder'];M=['expanded','Expanded-like','Foreshortened','Foreshortened-like','Remainder']
CL={'free':'Unrestricted','expanded_family':'Expanded family','foreshortened_family':'Foreshortened family','foreshortened_family_remainder':'Foreshortened family + remainder','remainder':'Remainder','freeze_10':'Freeze 10','freeze_20':'Freeze 20','freeze_30':'Freeze 30','freeze_40':'Freeze 40','freeze_50':'Freeze 50'}
RP=['frozen_10','frozen_20','frozen_30','final_from_start','evolving','shuffled','family_matched_random','random_views']
RL=['Epoch 10 throughout','Epoch 20 throughout','Epoch 30 throughout','Epoch 100 throughout','Recorded evolving','Temporally shuffled','Family-matched replacement','Random views']
training=pd.read_csv(D/'training_trajectories.csv');free=training[training.condition_id=='free'];geom=pd.read_csv(D/'geometric_selection.csv');replay=pd.read_csv(D/'replay_runs.csv')
corrected=pd.read_csv(D/'corrected_trajectories.csv')
manifest=[];plotted=[]
def save(fig,name):
 group_note=('Five training runs per condition' if name in ['figure_1','figure_3','figure_4','extended_data_8'] else 'Original selection-learning rule | three classifier initializations' if name in ['figure_5','extended_data_7','supplementary_2','supplementary_5'] else 'External pretrained probes' if name=='supplementary_3' else 'Original selection-learning rule (Methods)')
 if name=='figure_1':group_note='Five training runs | all candidate views available'
 fig.text(.5,.965 if name=='figure_1' else 1.065,group_note,ha='center',fontsize=9,color='#444')
 fig.savefig(O/(name+'.png'),dpi=260,bbox_inches='tight');fig.savefig(O/(name+'.svg'),bbox_inches='tight');plt.close(fig);manifest.append(name)
def tag(ax,s):ax.text(-.13,1.09,s,transform=ax.transAxes,fontsize=12,fontweight='bold',va='bottom')
def curve(ax,g,x,y,label,color,scale=1,ls='-',band=True):
 z=g.groupby(x)[y].agg(['mean','std','count']);xx=z.index.to_numpy();v=z['mean'].to_numpy()*scale;sd=z['std'].fillna(0).to_numpy()*scale
 # Redundant style/shape cues keep series identifiable without hue alone.
 style,marker=ls,'o'
 if label in F:
  idx=F.index(label);color=TC[idx];style=TSTYLE[idx];marker=TMARK[idx]
 elif current in ['figure_1','extended_data_3'] and label in ['Expanded family','Foreshortened family','Remainder']:
  idx=['Expanded family','Foreshortened family','Remainder'].index(label);color=[TC[0],TC[2],TC[4]][idx];style=['-','--','-.'][idx];marker=['o','s','^'][idx]
 elif label in CL.values():
  idx=list(CL.values()).index(label);style=['-','--','-.',':'][idx%4];marker=MARK[idx%len(MARK)]
 elif label in RL:
  idx=RL.index(label);style=['-','--','-.',':'][idx%4];marker=MARK[idx]
 elif label in ['Policy selected (6)','All candidates (114)','Random (5)']:
  marker=['o','s','^'][['Policy selected (6)','All candidates (114)','Random (5)'].index(label)]
 elif label.endswith('(1)'):
  idx=F.index(label[:-4]);color=TC[idx];style=['--',':','--',':','-.'][idx];marker=TMARK[idx]
 elif label in ['Planar','Near-planar']:style='-' if label=='Planar' else '--';marker='o' if label=='Planar' else 's'
 ax.plot(xx,v,label=label,color=color,lw=1.6,ls=style,marker=marker,markersize=3,markevery=max(1,len(xx)//6),markerfacecolor='white',markeredgewidth=.8)
 if band:ax.fill_between(xx,v-sd,v+sd,color=color,alpha=.12,lw=0)
 for xi,vi,si,ni in zip(xx,v,sd,z['count']):plotted.append(dict(figure=current,panel=ax.get_label(),series=label,x=xi,mean=vi,sd=si,n=ni))
 ax.set_xlabel('Training epoch');ax.set_xlim(min(xx),max(xx));ax.grid(axis='y',alpha=.15)
# Corrected data for the primary preference figure only.
free_historical=free.copy();geom_historical=geom.copy()
free=corrected[corrected.condition=='free'].copy();free['training_epoch']=free.epoch
for out,source in zip(M,['expanded_share_pct','expanded_like_share_pct','foreshortened_share_pct','foreshortened_like_share_pct','remainder_share_pct']):free[out]=free[source]/100
free['expanded_family']=free.expanded+free['Expanded-like'];free['foreshortened_family']=free.Foreshortened+free['Foreshortened-like']
geom=pd.concat([pd.DataFrame({'epoch':free.epoch,'metric':label,'selected_fraction':free[col]/100,'candidate_fraction':prior}) for col,label,prior in [('planar_share_pct','exact_planar',6/114),('near_planar_share_pct','near_planar',24/114)]])
# Fig 1: actual supplied object examples, multiview feedback and measured trajectories.
current='figure_1';fig=plt.figure(figsize=(8.3,8.7));gs=fig.add_gridspec(4,2,height_ratios=[1.12,1.15,1.3,1.3],hspace=.85,wspace=.45)
asset=json.loads((AS/'object_examples.json').read_text())
a=fig.add_subplot(gs[0,:]);a.axis('off');tag(a,'a');a.set_title('Same object, different views',loc='left',pad=12)
for i,v in enumerate(asset['views']):
 example=np.asarray(Image.open(AS/v['file']).convert('RGB')); x0,y0,x1,y1=asset['display_viewport']['pixel_bounds'];example=example[y0:y1,x0:x1];sub=a.inset_axes([i/5+.003,.12,.194,.65]);sub.imshow(example);sub.axis('off')
 a.text(i/5+.10,.97,v['name'],ha='center',va='top',fontsize=8.1,transform=a.transAxes)
 a.text(i/5+.10,-.03,v['description'].replace(' planar example','\nplanar example').replace(' from','\nfrom').replace(' extends','\nextends').replace(' points','\npoints').replace(' viewed','\nviewed'),ha='center',va='top',fontsize=8,transform=a.transAxes)
b=fig.add_subplot(gs[1,:]);b.axis('off');b.set_xlim(-.2,10.3);b.set_ylim(-.7,2.2);tag(b,'b')
# Multi-image experience is explicit; sequence-level mechanics belong in Methods.
for x,w,txt in [(0,2.25,'Learned view\nselection'),(7.0,3.0,'Combine views\nRecognize object category')]:
 b.add_patch(FancyBboxPatch((x,.7),w,1.0,boxstyle='round,pad=.035',fc='#eef3f6',ec='#555',lw=1));b.text(x+w/2,1.2,txt,ha='center',va='center',fontsize=9)
b.text(4.55,2.04,'Multiple views of one object',ha='center',fontsize=9)
for i,v in enumerate([asset['views'][j] for j in [0,2,4]]):
 example=np.asarray(Image.open(AS/v['file']).convert('RGB')); x0,y0,x1,y1=asset['display_viewport']['pixel_bounds'];example=example[y0:y1,x0:x1];thumb=b.inset_axes([3.0+i*1.1,.72,1.05,1.0],transform=b.transData);thumb.imshow(example);thumb.axis('off')
for x0,x1 in [(2.3,2.9),(6.4,6.9)]:b.add_patch(FancyArrowPatch((x0,1.2),(x1,1.2),arrowstyle='-|>',mutation_scale=11,lw=1,color='#333'))
# The return path exits below the recognizer, clears both boxes, and points up.
path=MplPath([(8.5,.62),(8.5,-.03),(1.125,-.03),(1.125,.62)],[MplPath.MOVETO,MplPath.LINETO,MplPath.LINETO,MplPath.LINETO])
b.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=13,color=C[0],lw=1.6))
b.text(4.9,-.46,'Recognition feedback updates selection; selected images train recognition',ha='center',fontsize=8,color=C[0])
c=fig.add_subplot(gs[2,:],label='c');tag(c,'c');c.set_title('Planar and near-planar views',loc='left')
for metric,label,color in [('exact_planar','Planar',C[0]),('near_planar','Near-planar',C[1])]:
 g=geom[geom.metric==metric];curve(c,g,'epoch','selected_fraction',label,color,100);c.axhline(g.candidate_fraction.iloc[0]*100,color=color,ls=':',lw=1)
for line in c.lines:
 if line.get_linestyle()=='-':line.set_marker('o');line.set_markersize(3);line.set_markevery([0,9,29,49,69,99])
c.set_xticks([1,20,40,60,80,100]);c.set_ylabel('Selected views (%)');c.set_ylim(0,65);c.legend(loc='upper left',ncol=2,frameon=False,fontsize=8);c.text(.98,.95,'Dotted: available proportions',transform=c.transAxes,ha='right',va='top',fontsize=8)
e=fig.add_subplot(gs[3,0],label='d');tag(e,'d');e.set_title('View families',loc='left')
for m,l,col,p in [('expanded_family','Expanded family',C[0],19.5285),('foreshortened_family','Foreshortened family',TC[2],13.6294),('Remainder','Remainder',C[4],66.8421)]:
 curve(e,free,'training_epoch',m,l,col,100);e.axhline(p,color=col,ls=':',lw=.8)
e.set_ylabel('Selected views (%)');e.set_ylim(0,100);e.legend(fontsize=7,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.38),ncol=1,handlelength=1.5,labelspacing=.35)
f=fig.add_subplot(gs[3,1],label='e');tag(f,'e');f.set_title('All five view types',loc='left')
for m,l,col in zip(M,F,C):curve(f,free,'training_epoch',m,l,col,100,band=False)
f.set_ylim(0,90);f.set_ylabel('Selected views (%)');f.legend(fontsize=7,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.38),ncol=2,columnspacing=.8,handlelength=1.4,labelspacing=.35)
save(fig,current)
free=free_historical;geom=geom_historical
# Fig 2: upper panels retain tick labels, with one figure-level legend above axes.
current='figure_2';ds=pd.read_csv(D/'descriptor_trajectories.csv');fig,axes=plt.subplots(3,1,figsize=(7.3,6.6),sharex=True)
for i,(m,l) in enumerate([('ellipse_aspect_ratio','Ellipse aspect ratio'),('bilateral_symmetry','Bilateral symmetry'),('edge_entropy','Edge-orientation entropy')]):
 ax=axes[i];ax.set_label(string.ascii_lowercase[i]);tag(ax,string.ascii_lowercase[i]);curve(ax,ds,'epoch','selected_'+m,'Mean of selected views',C[0]);ax.axhline(ds['baseline_'+m].iloc[0],color='#555',ls='--',label='Mean of all candidate views');ax.set_ylabel(l);ax.set_xlabel('Training epoch' if i==2 else '');ax.set_xticks([1,20,40,60,80,100]);ax.tick_params(axis='x',labelbottom=True);ax.tick_params(axis='y',labelleft=True)
h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,frameon=False,ncol=2,loc='upper right',bbox_to_anchor=(.99,1.005),fontsize=8.5)
fig.subplots_adjust(left=.12,right=.98,top=.90,bottom=.09,hspace=.43);save(fig,current)
# Corrected restriction: compare training conditions within each fixed evaluation regime.
current='figure_3';fig,axes=plt.subplots(1,3,figsize=(9.1,3.3),sharey=True)
for j,(metric,title) in enumerate([('policy_accuracy_pct','Policy-selected inputs (6)'),('random_five_accuracy_pct','Random inputs (5)'),('all_views_accuracy_pct','All candidate inputs (114)')]):
 ax=axes[j];ax.set_label(string.ascii_lowercase[j]);tag(ax,string.ascii_lowercase[j]);ax.set_title(title);ax.set_ylim(0,100);ax.tick_params(axis='y',labelleft=True);ax.set_yticks([0,20,40,60,80,100])
 for i,cid in enumerate(['free','expanded_family']):curve(ax,corrected[corrected.condition==cid],'epoch',metric,CL[cid],C[i])
axes[0].set_ylabel('Object-category accuracy (%)');h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.12));fig.subplots_adjust(wspace=.25,bottom=.19);save(fig,current)
# Corrected freezing: show continued improvement together with the different preference.
current='figure_4';fig,axes=plt.subplots(2,2,figsize=(7.4,6.2))
for j,(metric,title) in enumerate([('random_five_accuracy_pct','Random inputs (5)'),('planar_share_pct','Planar selection'),('policy_accuracy_pct','Policy-selected inputs (6)'),('all_views_accuracy_pct','All candidate inputs (114)')]):
 ax=axes.flat[j];ax.set_label(string.ascii_lowercase[j]);tag(ax,string.ascii_lowercase[j]);ax.set_title(title)
 for i,cid in enumerate(['free','freeze_10']):curve(ax,corrected[corrected.condition==cid],'epoch',metric,CL[cid],C[i])
 ax.axvline(10,color='#777',ls='--',lw=.8)
 ax.set_ylabel('Selected views (%)' if j==1 else 'Object-category accuracy (%)');ax.set_ylim(0,40 if j==1 else 100)
 if j==1:ax.axhline(100*6/114,color='#555',ls=':',lw=1,label='Candidate availability')
h,l=axes.flat[1].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.025));fig.tight_layout(h_pad=2,w_pad=2);save(fig,current)
# Replay paired runs, no unmatched original-model line.
current='figure_5';fig,axes=plt.subplots(1,2,figsize=(9.2,5),sharey=True)
for j,(metric,title) in enumerate([('final_accuracy_pct','Object-category accuracy (%)'),('final_prediction_margin','Gap between highest scores')]):
 ax=axes[j];tag(ax,string.ascii_lowercase[j]);ax.set_xlabel(title)
 for seed in [0,1,2]:
  vals=[replay[(replay.condition==c)&(replay.seed==seed)][metric].iloc[0] for c in RP];ax.plot(vals,range(8),lw=.65,color='#bbb',zorder=1)
 for i,c in enumerate(RP):
  vals=replay[replay.condition==c][metric];ax.scatter(vals,np.full(len(vals),i)+np.linspace(-.07,.07,len(vals)),s=22,color=C[i],zorder=3);ax.errorbar(vals.mean(),i,xerr=vals.std(ddof=1),fmt='D',mfc='white',mec='black',ecolor='black',ms=5,capsize=3,zorder=4)
 ax.set_yticks(range(8),RL);ax.grid(axis='x',alpha=.18)
axes[0].invert_yaxis();fig.tight_layout(w_pad=2);save(fig,current)
# ED1 contextual replacement, equal-run mean and final-run estimates.
current='extended_data_1';rep=pd.read_csv(D/'replacement_runs.csv');fig,axes=plt.subplots(2,2,figsize=(9,5.8),gridspec_kw={'height_ratios':[1,1.2]})
for j,(metric,title) in enumerate([('delta_replace_correct_logit','Correct-category score'),('delta_replace_prediction_margin','Gap between highest scores')]):
 ax=axes[0,j];tag(ax,string.ascii_lowercase[j]);z=rep.groupby(['view_family','epoch'])[metric].mean().unstack().reindex(F);im=ax.imshow(z,vmin=-.6,vmax=.6,cmap='RdBu_r',aspect='auto');ax.set_yticks(range(5),F);ax.set_xticks(range(10),range(10,101,10));ax.set_xlabel('Source-selection epoch');ax.set_title(title)
 for y in range(5):
  for x in range(10):ax.text(x,y,f'{z.iloc[y,x]:.2f}',ha='center',va='center',fontsize=6.7,color='white' if abs(z.iloc[y,x])>.38 else 'black')
 ax=axes[1,j];tag(ax,string.ascii_lowercase[j+2]);ax.axvline(0,color='#888',lw=.8);ax.set_yticks(range(5),F);ax.set_xlabel('Original minus replacement at source epoch 100')
 for i,fam in enumerate(F):
  v=rep[(rep.epoch==100)&(rep.view_family==fam)][metric];ax.scatter(v,np.full(len(v),i)+np.linspace(-.08,.08,len(v)),s=18,color=TC[i],marker=TMARK[i]);ax.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt='D',mfc='white',mec='black',ecolor='black',ms=4,capsize=3)
 ax.invert_yaxis()
fig.tight_layout(h_pad=2,w_pad=3);save(fig,current)
# ED2 completed checkpoint matched tests.
current='extended_data_2';pt=pd.read_csv(D/'perturbation_runs.csv');pt.condition=pt.condition.replace({'unrestricted':'free'});fig,axes=plt.subplots(2,2,figsize=(8.3,6.5))
for j,(test,label,metric) in enumerate([('clean','Clean images','accuracy_pct'),('rotation','In-plane rotation ±10°','accuracy_drop_percentage_points'),('colour_jitter','Color jitter','accuracy_drop_percentage_points'),('rotation_and_colour_jitter','Rotation and color jitter','accuracy_drop_percentage_points')]):
 # actual exported test labels are resolved without guessing a value.
 lookup={'colour_jitter':'colour_jitter','rotation_and_colour_jitter':'rotation_colour_jitter'}
 names=pt.test_condition.unique().tolist()
 if test not in names:
  candidates=[n for n in names if ('jitter' in n and (('rotation' in n)==('rotation' in test)))];assert len(candidates)==1,(test,names);test=candidates[0]
 ax=axes.flat[j];tag(ax,string.ascii_lowercase[j]);ax.set_label(string.ascii_lowercase[j]);ax.set_title(label)
 for i,cid in enumerate(['free','freeze_10','freeze_20','freeze_30']):curve(ax,pt[(pt.condition==cid)&(pt.test_condition==test)],'epoch',metric,CL[cid],C[i])
 ax.set_ylabel('Accuracy (%)' if j==0 else 'Accuracy drop (percentage points)');ax.set_xticks([10,30,50,70,100]);
 if j:ax.axhline(0,color='#555',ls=':',lw=.8)
h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,frameon=False);fig.tight_layout(rect=(0,.04,1,1));save(fig,current)
# ED3: the single-image setting uses the same learning rates as multiview.
current='extended_data_3';fig,axes=plt.subplots(2,2,figsize=(8.2,6.5))
for i,(cid,title) in enumerate([('free','Multiview recognition'),('single_primary','Single-image recognition')]):
 ax=axes.flat[i];tag(ax,string.ascii_lowercase[i]);ax.set_label(string.ascii_lowercase[i]);ax.set_title(title);g=training[training.condition_id==cid]
 for m,l,col,p in [('expanded_family','Expanded family',TC[0],19.5285),('foreshortened_family','Foreshortened family',TC[2],13.6294),('Remainder','Remainder',TC[4],66.8421)]:curve(ax,g,'training_epoch',m,l,col,100);ax.axhline(p,color=col,ls=':',lw=.8)
 ax.set_ylim(0,100);ax.set_ylabel('Selected views (%)')
ax=axes.flat[2];tag(ax,'c');ax.set_label('c');ax.set_title('Planar views')
curve(ax,geom[geom.metric=='exact_planar'],'epoch','selected_fraction','Multiview: planar',C[0],100)
u=pd.read_csv(D/'single_planar_upper_bound.csv');curve(ax,u[u.condition_id=='single_primary'],'training_epoch','principal_planar_upper_bound_pct','Single image: maximum possible',C[1],ls='--');ax.axhline(100*6/114,color='#333',ls=':',label='Planar availability');ax.set_ylabel('Selected views (%)');ax.legend(frameon=False,fontsize=6.8)
ax=axes.flat[3];tag(ax,'d');ax.set_label('d');ax.set_title('Accuracy on selected inputs')
for cid,label,color,style in [('free','Multiview (6 images)',C[0],'-'),('single_primary','Single image',C[1],'--')]:curve(ax,training[training.condition_id==cid],'training_epoch','accuracy',label,color,ls=style)
ax.set_ylabel('Object-category accuracy (%)');ax.set_ylim(0,100);ax.legend(frameon=False,fontsize=7)
h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False);fig.tight_layout(rect=(0,.04,1,1));save(fig,current)
# ED4/5 complete original evaluation grids. 8 regimes, individual type labels remain operational.
EV=[('accuracy','Policy selected (6)'),('all_views_accuracy','All candidates (114)'),('random_accuracy','Random (5)'),('expanded_accuracy','Expanded (1)'),('expanded_like_accuracy','Expanded-like (1)'),('foreshortened_accuracy','Foreshortened (1)'),('foreshortened_like_accuracy','Foreshortened-like (1)'),('remainder_accuracy','Remainder (1)')]
for name,conds in [('extended_data_4',['free','expanded_family','foreshortened_family','foreshortened_family_remainder','remainder']),('extended_data_5',['free','freeze_10','freeze_20','freeze_30','freeze_40','freeze_50'])]:
 current=name;fig,axes=plt.subplots(3,2,figsize=(8.1,8.2),sharex=True,sharey=True)
 for i,cid in enumerate(conds):
  ax=axes.flat[i];ax.set_label(string.ascii_lowercase[i]);tag(ax,string.ascii_lowercase[i]);ax.set_title(CL[cid]);ax.set_ylim(0,100);ax.tick_params(axis='x',labelbottom=True);ax.tick_params(axis='y',labelleft=True)
  for j,(m,l) in enumerate(EV):curve(ax,training[training.condition_id==cid],'training_epoch',m,l,C[j],ls='-' if j<3 else '--',band=j<3)
  if i%2==0:ax.set_ylabel('Accuracy (%)')
 if len(conds)==5:axes.flat[5].axis('off')
 h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,frameon=False,fontsize=7);fig.tight_layout(rect=(0,.06,1,1));save(fig,current)
# ED6 equal-budget paired effects.
current='extended_data_6';eb=pd.read_csv(D/'equal_budget_runs.csv');fig,axes=plt.subplots(1,2,figsize=(7,3.7),sharey=True)
for j,budget in enumerate([5,6]):
 ax=axes[j];tag(ax,string.ascii_lowercase[j]);ax.set_title(f'{budget} classifier inputs');g=eb[eb.budget==budget]
 for i,(run,r) in enumerate(g.groupby('run')):
  v=r.set_index('sampling').loc[['random','policy'],'accuracy_pct'];ax.plot([0,1],v,'-',color=C[i],lw=1,ms=4,marker=MARK[i],label=f'Run {i+1}')
 ax.set_xticks([0,1],['Random','Policy selected']);ax.set_xlim(-.3,1.3);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Object-category accuracy (%)');axes[1].legend(frameon=False,fontsize=7);axes[1].tick_params(axis='y',labelleft=True);fig.tight_layout();save(fig,current)
# SI1 per-source-epoch pooled descriptor correlations.
current='supplementary_1';co=pd.read_csv(D/'replacement_correlations.csv');co=co[(co.scope=='epoch')&(co.cue_basis=='selected_minus_replacement')&(co.method=='spearman')];fig,axes=plt.subplots(2,1,figsize=(8,4.2))
for j,obj in enumerate(['correct_logit','prediction_margin']):
 ax=axes[j];tag(ax,string.ascii_lowercase[j]);z=co[co.objective==obj].pivot(index='feature',columns='epoch',values='correlation').reindex(['ellipse_aspect_ratio','bilateral_symmetry','edge_entropy']);ax.imshow(z,vmin=-.2,vmax=.2,cmap='RdBu_r',aspect='auto');ax.set_yticks(range(3),['Ellipse aspect ratio','Bilateral symmetry','Edge-orientation entropy']);ax.set_xticks(range(10),range(10,101,10));ax.set_title(['Correct-category score','Gap between highest scores'][j]);ax.set_xlabel('Source-selection epoch')
 for y in range(3):
  for x in range(10):ax.text(x,y,f'{z.iloc[y,x]:.3f}',ha='center',va='center',fontsize=8,color='white' if abs(z.iloc[y,x])>.14 else 'black')
fig.tight_layout(h_pad=1.5);save(fig,current)
# SI2 CKA.
current='supplementary_2';ck=pd.read_csv(D/'cka_runs.csv');fig,ax=plt.subplots(figsize=(7.3,4.5))
for i,c in enumerate(RP):
 v=ck[ck.condition==c].cka_recomputed;ax.scatter(v,np.full(len(v),i)+np.linspace(-.07,.07,len(v)),color=C[i],s=25);ax.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt='D',mfc='white',mec='black',ecolor='black',ms=5,capsize=3)
ax.set_yticks(range(8),RL);ax.invert_yaxis();ax.set_xlabel('Linear CKA: final versus shared initial features');ax.set_xlim(.4,.82);ax.grid(axis='x',alpha=.2);fig.tight_layout();save(fig,current)
# SI3 external probes: repeat means as dots; repeats are sampling, not models.
current='supplementary_3';fig,axes=plt.subplots(1,2,figsize=(8.5,4.2));clip=pd.read_csv(D/'clip_sampling_repeats.csv');clip.view_type=clip.view_type.replace({'expanded':'Expanded'});vg=pd.read_csv(D/'vggt_sampling_repeats.csv');vg.view_type=vg.view_type.replace({'expanded':'Expanded'})
for j,(df,metric,label) in enumerate([(clip,'accuracy','CLIP accuracy (%)'),(vg,'mean','VGGT object-mask confidence')]):
 ax=axes[j];tag(ax,string.ascii_lowercase[j]);ax.set_ylabel(label)
 for i,fam in enumerate(F):
  v=df[df.view_type==fam][metric];assert len(v)==5,(fam,df.view_type.unique());ax.bar(i,v.mean(),width=.62,color=TC[i],alpha=.3,hatch=['','//','','//','..'][i],edgecolor=TC[i]);ax.scatter(i+np.linspace(-.13,.13,5),v,s=17,color=TC[i],marker=TMARK[i]);ax.text(i,v.max()+(.9 if j==0 else .23),f'{v.mean():.2f}',ha='center',fontsize=8)
 ax.set_xticks(range(5),F,rotation=38,ha='right');ax.set_ylim(0,65 if j==0 else 16)
fig.tight_layout();save(fig,current)
# SI4 pairwise descriptors on illustration subset with clear scope.
current='supplementary_4';v=pd.read_csv(D/'descriptor_illustration.csv');fig,axes=plt.subplots(1,3,figsize=(9,3.1));features=['ellipse_aspect_ratio','bilateral_symmetry','edge_entropy'];labels=['Ellipse aspect ratio','Bilateral symmetry','Edge-orientation entropy']
for j,(x,y) in enumerate([(0,1),(0,2),(1,2)]):
 ax=axes[j];tag(ax,string.ascii_lowercase[j]);ax.scatter(v[features[x]],v[features[y]],s=4,c='#bbbbbb',alpha=.25,rasterized=True);sel=v[v.selected==True];ax.scatter(sel[features[x]],sel[features[y]],s=17,facecolors='none',edgecolors=C[0],lw=.8);ax.set_xlabel(labels[x]);ax.set_ylabel(labels[y])
fig.tight_layout();save(fig,current)
# SI5 already-run warmups and quantitative class silhouette.
current='supplementary_5';conds=['evolving','random_warmup_10','random_warmup_20','random_warmup_30','random_views'];labels=['Recorded evolving','10-epoch random warm-up','20-epoch random warm-up','30-epoch random warm-up','Random throughout'];fig,axes=plt.subplots(1,2,figsize=(8.5,3.7),sharey=True)
for j,(metric,label) in enumerate([('final_accuracy_pct','Object-category accuracy (%)'),('final_class_silhouette','Class silhouette score')]):
 ax=axes[j];tag(ax,string.ascii_lowercase[j]);ax.set_xlabel(label)
 for i,c in enumerate(conds):
  val=replay[replay.condition==c][metric];ax.scatter(val,np.full(len(val),i)+np.linspace(-.07,.07,len(val)),s=22,color=C[i]);ax.errorbar(val.mean(),i,xerr=val.std(ddof=1),fmt='D',mfc='white',mec='black',ecolor='black',ms=4,capsize=3)
 ax.set_yticks(range(5),labels);ax.grid(axis='x',alpha=.15)
axes[0].invert_yaxis();fig.tight_layout();save(fig,current)
# ED7 replay trajectories, separating static history and schedule manipulations.
current='extended_data_7';rt=pd.read_csv(D/'replay_trajectories.csv');rt=rt[rt.phase=='test'];fig,axes=plt.subplots(2,2,figsize=(8.4,6.1))
for row,(metric,title) in enumerate([('accuracy_pct','Object-category accuracy (%)'),('prediction_margin','Gap between highest scores')]):
 for col,inds in enumerate([range(4),range(4,8)]):
  ax=axes[row,col];tag(ax,string.ascii_lowercase[row*2+col]);ax.set_label(string.ascii_lowercase[row*2+col]);ax.set_title('Static source epochs' if col==0 else 'Changing image sets')
  for i in inds:curve(ax,rt[rt.condition==RP[i]],'epoch',metric,RL[i],C[i])
  ax.set_ylabel(title);ax.set_xlabel('Fresh-classifier training epoch')
  if row==0:ax.legend(fontsize=7,frameon=False,loc='lower right');ax.set_ylim(0,100)
fig.tight_layout(h_pad=1.5);save(fig,current)
# Evidence for the scope of the freezing comparison.
current='extended_data_8';fig,axes=plt.subplots(1,2,figsize=(8.5,3.8))
for seed in range(5):
 g=corrected[corrected.seed==seed].pivot(index='epoch',columns='condition',values='random_five_accuracy_pct')
 axes[0].plot(g.index[:10],(g.free-g.freeze_10).iloc[:10],color=C[seed],marker=MARK[seed],ls=['-','--','-.',':',(0,(5,1,1,1))][seed],markersize=3,label=f'Run {seed+1}')
 vals=corrected[(corrected.seed==seed)&(corrected.epoch==100)].set_index('condition').loc[['free','freeze_10'],'random_five_accuracy_pct']
 axes[1].plot([0,1],vals,color=C[seed],marker=MARK[seed],lw=1)
axes[0].axhline(0,color='#777',lw=.8);axes[0].set_xlabel('Training epoch before freezing');axes[0].set_ylabel('Free minus freeze-10 accuracy (pp)');axes[0].set_xticks([1,3,5,7,10]);axes[0].legend(frameon=False,fontsize=8,ncol=2)
axes[1].set_xticks([0,1],['Unrestricted','Freeze 10']);axes[1].set_ylabel('Final random-five accuracy (%)');axes[1].set_xlim(-.25,1.25)
for i,ax in enumerate(axes):tag(ax,string.ascii_lowercase[i]);ax.grid(axis='y',alpha=.15)
fig.tight_layout(w_pad=2);save(fig,current)
pd.DataFrame(plotted).to_csv(O/'plotted_curve_values.csv',index=False)
(O/'figure_build.json').write_text(json.dumps({'figures':manifest,'source_dir':str(D.resolve()),'uncertainty':'sample SD across independently trained models except external probes show sampling-repeat means; descriptor illustration has no error bars','new_training':False,'new_model_inference':False,'object_examples':asset},indent=2)+'\n')
print('Built',len(manifest),'figures')
