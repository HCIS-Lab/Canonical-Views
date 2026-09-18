"""Plot the exported candidate/selected descriptor table without recomputation.

Input: all_test_views_midlevel_3d_points.csv produced by the existing pipeline.
No synthetic observations, sampling, clipping or inferred selection flags.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METRICS = ['ellipse_aspect_ratio', 'bilateral_symmetry', 'edge_entropy']
LABELS = ['Ellipse aspect ratio', 'Bilateral symmetry', 'Edge-orientation entropy']

def prepare_frame(frame):
    required = ['class_idx', 'instance_id', 'view_index', 'selected', *METRICS]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    frame = frame.copy()
    keys = ['class_idx', 'instance_id', 'view_index']
    if frame[keys].isna().any().any():
        raise ValueError('Object/view keys must be complete.')
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate object/view keys: export one explicitly defined selection context.')
    flags = frame['selected'].astype(str).str.strip().str.lower()
    if not flags.isin(['true', 'false', '1', '0']).all():
        raise ValueError('Selected must explicitly contain true/false or 1/0.')
    frame['selected'] = flags.isin(['true', '1'])
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors='coerce')
    valid = np.isfinite(frame[METRICS].to_numpy()).all(axis=1)
    retained, excluded = frame.loc[valid].copy(), frame.loc[~valid].copy()
    if retained.empty:
        raise ValueError('No finite observations.')
    if not retained['selected'].any():
        raise ValueError('No finite selected observations; check the source context.')
    return retained, excluded

def plot_pairs(frame, output, context):
    retained, excluded = prepare_frame(frame)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.65))
    selected = retained.loc[retained['selected']]
    for k, (i, j) in enumerate([(0,1),(0,2),(1,2)]):
        ax = axes[k]
        ax.scatter(retained[METRICS[i]], retained[METRICS[j]],
                   s=9, c='#b5b5b5', alpha=.35, linewidths=0, rasterized=True)
        ax.scatter(selected[METRICS[i]], selected[METRICS[j]],
                   s=25, facecolors='none', edgecolors='#0072B2',
                   marker='D', linewidths=.85, zorder=3)
        ax.set(xlabel=LABELS[i], ylabel=LABELS[j])
        ax.text(-.16,1.035,'abc'[k],transform=ax.transAxes,weight='bold',fontsize=13)
        ax.spines[['top','right']].set_visible(False)
        ax.margins(.04)
    handles = [Line2D([],[],marker='o',linestyle='',color='#aaa',markersize=5,
                      label=f'All candidate views (n={len(retained):,})'),
               Line2D([],[],marker='D',linestyle='',markerfacecolor='none',
                      color='#0072B2',markersize=5,
                      label=f'Selected views (n={len(selected):,})')]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False)
    fig.suptitle(context,fontsize=10,y=.99)
    fig.subplots_adjust(left=.07,right=.985,bottom=.26,top=.84,wspace=.36)
    for ext in ['png','pdf','svg']:
        fig.savefig(output/f'feature_pairs.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    retained.to_csv(output/'feature_pairs_source.csv',index=False)
    excluded.to_csv(output/'feature_pairs_excluded.csv',index=False)
    audit={'context':context,'n_input':len(frame),'n_finite':len(retained),
           'n_excluded_nonfinite':len(excluded),'n_selected':len(selected),
           'n_objects':int(retained[['class_idx','instance_id']].drop_duplicates().shape[0]),
           'n_categories':int(retained['class_idx'].nunique()),
           'normalization':'none; original raw descriptors',
           'subsampling':'none','inference':'descriptive candidate-set diagnostic only'}
    (output/'feature_pairs_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--context',required=True,
                   help='Explicit split/object subset/run/source epoch/initial camera.')
    args=p.parse_args()
    print(json.dumps(plot_pairs(pd.read_csv(args.input),args.output,args.context),indent=2))
