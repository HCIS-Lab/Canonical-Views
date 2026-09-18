"""Export small selection-only records without copying full metadata or images.
Usage: python3 export_selection_plot_data.py RUN_FOLDER --output OUTPUT.json
Defaults to the legacy aggregation inclusion rule and no smoothing.
Use --inclusion selection explicitly for logs with selection arrays only.
This script does not train a model or modify any input file.
"""
import argparse, hashlib, json, math, statistics
from pathlib import Path

KEYS = ['expanded', 'Expanded-like', 'Foreshortened', 'Foreshortened-like', 'Remainder']
FAMILIES = {'Expanded family':KEYS[:2], 'Foreshortened family':KEYS[2:4], 'Remainder':KEYS[4:]}

def summarize(curves):
    n=len(curves)
    return {'mean':[statistics.mean(c) for c in zip(*curves)],
            'sem':[statistics.stdev(c)/math.sqrt(n) if n>1 else None for c in zip(*curves)],
            'n_files':n}

def load_runs(folder, inclusion='legacy'):
    if not folder.is_dir():
        raise ValueError(f'Experiment folder does not exist: {folder}')
    if inclusion not in ('legacy', 'selection'):
        raise ValueError(f'Unknown inclusion rule: {inclusion}')
    kept=[]; skipped=[]
    for path in sorted(folder.glob('*_meta.json')):
        raw=path.read_bytes(); r=json.loads(raw)
        if not isinstance(r,dict):raise ValueError(f'{path.name}: expected a JSON object')
        required=KEYS + (['per_class_acc','expanded_accuracy_5'] if inclusion=='legacy' else [])
        missing=[k for k in required if k not in r]
        if missing:
            skipped.append({'file':path.name,'reason':f'Excluded by {inclusion} inclusion rule: missing '+', '.join(missing)})
            continue
        curves={k:r[k] for k in KEYS}
        sizes={len(v) for v in curves.values()}
        if len(sizes)!=1 or not next(iter(sizes)):
            raise ValueError(f'{path.name}: unequal or empty subtype arrays')
        for k,values in curves.items():
            if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1 for x in values):
                raise ValueError(f'{path.name}: {k} has invalid shares; expected finite fractions in [0,1]')
        if any(abs(sum(vals)-1)>0.002 for vals in zip(*curves.values())):
            raise ValueError(f'{path.name}: five shares do not sum to one; check denominators before plotting')
        kept.append({'file':path.name,'sha256':hashlib.sha256(raw).hexdigest(),
                     'metadata':{k:r[k] for k in ['seed','active_single_view','recognition_num_views','steps'] if k in r},
                     'original_length':next(iter(sizes)), 'subtype_shares':curves, '_raw':r})
    return kept,skipped

def export(folder, inclusion='legacy'):
    kept,skipped=load_runs(folder,inclusion)
    if not kept: raise ValueError(f'No files meet the {inclusion} inclusion rule. Exclusions: {skipped}')
    minimum=min(r['original_length'] for r in kept)
    exact={k:summarize([r['subtype_shares'][k][:minimum] for r in kept]) for k in KEYS}
    family={k:summarize([[sum(r['subtype_shares'][s][i] for s in subs) for i in range(minimum)] for r in kept]) for k,subs in FAMILIES.items()}
    for r in kept:r.pop('_raw')
    return {'schema_version':2,'inclusion_rule':inclusion,'smoothing_window':1,'array_indices':list(range(minimum)),
            'epoch_note':'Indices reproduce the current zero-based plot. Verify the actual training-epoch mapping before relabelling.',
            'unit_note':'SEM is across included files; independent seeds/runs still require author confirmation. Null means SEM is unavailable for a single file.',
            'base_rate_note':'Candidate proportions are deliberately not inferred from selection shares. Supply or confirm the candidate-label denominators separately.',
            'alignment_note':'Summaries use the shortest common prefix, matching the current aggregator. Full per-file subtype curves are retained below.',
            'included_files':len(kept),'skipped_files':skipped,'subtype_summary':exact,'family_summary':family,'runs':kept}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_folder',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inclusion',choices=['legacy','selection'],default='legacy',help='selection explicitly permits logs without accuracy fields')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(f'Output exists: {a.output}; choose another output name.')
    result=export(a.run_folder,a.inclusion)
    with a.output.open('x') as f:json.dump(result,f,separators=(',',':'),allow_nan=False)
    print(f"Exported {result['included_files']} files; {len(result['skipped_files'])} excluded; {a.output.stat().st_size:,} bytes. Inputs unchanged.")
if __name__=='__main__':main()
