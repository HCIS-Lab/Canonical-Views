"""Auditable selection trajectories used by aggregate_meta_json.py.

Inputs remain unchanged. All summary bands use file-level SEM, after summing
subtypes within each file for family curves. Independent seeds require a run audit.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from export_selection_plot_data import KEYS, FAMILIES, export

LEGACY_PRIORS = dict(zip(KEYS, [0.035, 0.14, 0.017, 0.07, 0.738]))
EXACT_LABELS = dict(zip(KEYS, ['Expanded', 'Expanded-like', 'Foreshortened', 'Foreshortened-like', 'Remainder']))
EXACT_COLORS = dict(zip(KEYS, ['#009E73', '#0072B2', '#D55E00', '#CC79A7', '#666666']))
FAMILY_COLORS = dict(zip(FAMILIES, ['#0072B2', '#D55E00', '#666666']))
STYLES = ['-', '--', '-.', (0, (3, 1, 1, 1, 1, 1)), ':']

def read_priors(path=None):
    priors = LEGACY_PRIORS.copy() if path is None else json.loads(Path(path).read_text())
    if not isinstance(priors,dict) or set(priors)!=set(KEYS):
        raise ValueError('Candidate-prior JSON must contain exactly the five metadata subtype keys.')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not 0<x<1 for x in priors.values()):
        raise ValueError('Candidate priors must be finite positive fractions below one.')
    if abs(sum(priors.values())-1)>1e-6:
        raise ValueError('Candidate priors must sum to one; do not use selected shares as priors.')
    source='legacy constants (verify against candidate manifest)' if path is None else str(Path(path).resolve())
    return priors,source

def draw(ax, summaries, priors, labels, colors, epochs, enrichment=False):
    for i,(key,s) in enumerate(summaries.items()):
        mean=np.asarray(s['mean'],float)
        divisor=priors[key] if enrichment else 1
        label=labels[key] if enrichment else f'{labels[key]} (available {priors[key]:.1%})'
        ax.plot(epochs,mean/divisor,color=colors[key],ls=STYLES[i],lw=1.8,label=label)
        if s['n_files']>1:
            sem=np.asarray(s['sem'],float)
            ax.fill_between(epochs,(mean-sem)/divisor,(mean+sem)/divisor,color=colors[key],alpha=.14)
        if not enrichment:
            ax.axhline(priors[key],color=colors[key],ls=':',lw=.8,alpha=.45)
    if enrichment:
        ax.axhline(1,color='#333333',ls=':',lw=1)
        ax.set_ylabel('Enrichment relative to\ncandidate availability')
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(0,1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel('Selected share (%)')
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,fontsize=8)
    ax.grid(axis='y',alpha=.15)

def save(fig,path):
    for extension in ['png','pdf','svg']:
        fig.savefig(Path(path).with_suffix('.'+extension),dpi=300,facecolor='white')
    plt.close(fig)

def write_selection_outputs(folder, inclusion='legacy', prior_file=None, epoch_start=0):
    folder=Path(folder)
    data=export(folder,inclusion)
    priors,prior_source=read_priors(prior_file)
    family_priors={k:sum(priors[v] for v in values) for k,values in FAMILIES.items()}
    epochs=np.asarray(data['array_indices'])+epoch_start
    xlabel='Recorded epoch index (zero-based)' if epoch_start==0 else 'Training epoch (index + 1; author-specified)'
    n=data['included_files']
    uncertainty=f'Mean ± SEM across {n} files; independent-run status requires verification.' if n>1 else 'One file; between-run uncertainty unavailable. No SEM band drawn.'
    availability='Dotted lines: candidate availability.'
    prior_note='Availability uses legacy constants; verify against the candidate manifest.' if prior_file is None else 'Availability supplied in candidate-prior JSON.'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    exact=data['subtype_summary'];family=data['family_summary']
    family_labels={k:k for k in family}
    for name,sums,p,labels,col,title,enrich in [
        ('aggregated_view_ratios',exact,priors,EXACT_LABELS,EXACT_COLORS,'Five view subtypes',False),
        ('aggregated_view_family_ratios',family,family_priors,family_labels,FAMILY_COLORS,'View families',False),
        ('aggregated_view_family_lift',family,family_priors,family_labels,FAMILY_COLORS,'View-family enrichment',True),
    ]:
        fig,ax=plt.subplots(figsize=(10,4.8))
        draw(ax,sums,p,labels,col,epochs,enrich)
        ax.set_xlabel(xlabel);ax.set_title(title,loc='left')
        fig.subplots_adjust(left=.1,right=.69,top=.88,bottom=.24)
        fig.text(.1,.07,uncertainty,fontsize=8)
        fig.text(.1,.025,('Reference at 1: selected share equals candidate share.' if enrich else availability)+' '+prior_note,fontsize=7.5)
        save(fig,folder/name)
    fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=True)
    for i,(ax,sums,p,labels,col,title) in enumerate([
        (axes[0],family,family_priors,family_labels,FAMILY_COLORS,'View families'),
        (axes[1],exact,priors,EXACT_LABELS,EXACT_COLORS,'Five view subtypes'),
    ]):
        draw(ax,sums,p,labels,col,epochs)
        ax.set_title(chr(97+i)+'  '+title,loc='left')
    axes[1].set_xlabel(xlabel)
    fig.subplots_adjust(left=.1,right=.69,top=.94,bottom=.14,hspace=.25)
    fig.text(.1,.055,uncertainty,fontsize=8)
    fig.text(.1,.025,availability+' '+prior_note,fontsize=7.5)
    save(fig,folder/'aggregated_view_trajectories')
    data.update({'candidate_priors':priors,'candidate_prior_source':prior_source,'plotted_epochs':epochs.tolist(),'epoch_start':epoch_start,
                 'uncertainty_definition':uncertainty,'truncated_files':[r['file'] for r in data['runs'] if r['original_length']>len(epochs)]})
    (folder/'selection_plot_data.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
    view_family={'bucket_prior':priors,'candidate_prior_source':prior_source,
                 'expanded_family_prior':family_priors['Expanded family'],'foreshortened_family_prior':family_priors['Foreshortened family'],'remainder_prior':family_priors['Remainder']}
    for key,name in [('expanded_family','Expanded family'),('foreshortened_family','Foreshortened family'),('remainder','Remainder')]:
        s=family[name];prior=family_priors[name]
        view_family[key+'_mean']=s['mean'];view_family[key+'_sem']=s['sem']
        view_family[key+'_lift_mean']=[x/prior for x in s['mean']]
        view_family[key+'_lift_sem']=[x/prior if x is not None else None for x in s['sem']]
    # Retain the old raw-share crossing field for consumers, but never label it
    # as onset of preference: these families have different candidate base rates.
    cross=np.flatnonzero(np.asarray(family['Expanded family']['mean'])>np.asarray(family['Foreshortened family']['mean']))
    view_family['expanded_over_foreshortened_first_epoch']=int(cross[0]) if len(cross) else None
    view_family['crossing_note']='Legacy zero-based raw-share crossing; not enrichment or bias onset.'
    result={'view_family':view_family,'sem_smoothed_curves':{k:s['sem'] for k,s in exact.items()},
            'selection_reporting':{k:data[k] for k in ['inclusion_rule','included_files','skipped_files','truncated_files','array_indices','plotted_epochs','epoch_start','uncertainty_definition','candidate_prior_source']},
            'selection_source_data':'selection_plot_data.json'}
    print(f'{folder}: {n} files included; {len(data["skipped_files"])} excluded; {len(data["truncated_files"])} shortened to {len(epochs)} points.')
    if prior_file is None:print('  Candidate availability uses legacy constants; verify before publication.')
    return result
