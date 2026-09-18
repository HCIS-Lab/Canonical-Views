#!/usr/bin/env python3
"""Audit geometric filename flags without changing historical training labels.

Reads collector manifest CSV(.gz), and optionally selected checkpoint records
from pretty-printed *_selection.json files. Uses read-only memory mapping to
locate epoch boundaries, then decodes only requested epoch objects. No model
training, rendering, inference or source-data mutation is performed.
"""
import argparse
from collections import Counter, defaultdict
import csv, gzip, json, mmap, re
from pathlib import Path

EPOCH_HEADER = re.compile(rb'^    "([0-9]+)": \{', re.MULTILINE)

def epoch_objects(path, wanted):
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        matches = [(int(m.group(1)), m.end()-1, m.start()) for m in EPOCH_HEADER.finditer(mm)]
        if not matches or len({x[0] for x in matches}) != len(matches):
            raise ValueError(f'{path}: expected unique four-space-indented epoch keys; no silent fallback to guessed parsing.')
        missing = set(wanted) - {m[0] for m in matches}
        if missing:
            raise ValueError(f'{path}: missing requested epochs {sorted(missing)}')
        for i,(epoch,start,_) in enumerate(matches):
            if epoch not in wanted:
                continue
            stop = matches[i+1][2] if i+1<len(matches) else len(mm)
            obj,_ = json.JSONDecoder().raw_decode(mm[start:stop].decode('utf-8'))
            yield epoch,obj

def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def audit(manifest, selection_dir, epochs, out):
    counts=Counter();lookup={};conflicts=[];objects=defaultdict(set)
    opener=gzip.open if manifest.suffix=='.gz' else open
    with opener(manifest,'rt',newline='') as f:
        for r in csv.DictReader(f):
            if r['split'] not in ('train','test') or int(r['included_by_current_paper_rule'])!=1:
                continue
            name=Path(r['relative_path']).name
            flags=(int(r['planar_flag']),int(r['like_flag']),int(r['short_flag']))
            objects[r['split']].add(r['object_id'])
            label=r['parser_view_type']
            counts[(r['split'],label,*flags)] += 1
            if r['split']=='test':
                if name in lookup and lookup[name]!=(label,flags):conflicts.append(name)
                lookup[name]=(label,flags)
    if conflicts:raise ValueError('Conflicting test filename labels: '+str(conflicts[:3]))
    rows=[dict(split=s,legacy_view_type=l,planar_flag=p,near_planar_flag=n,major_axis_flag=a,count=v)
          for (s,l,p,n,a),v in sorted(counts.items())]
    write_csv(out/'candidate_flag_crosswalk.csv',rows,list(rows[0]))
    totals=Counter()
    for (s,l,p,n,a),v in counts.items():totals[s]+=v
    priors=Counter()
    for (s,l,p,n,a),v in counts.items():
        if s!='test':continue
        priors['exact_planar']+=p*v;priors['near_planar']+=n*v
        priors['planar_or_near']+=int(bool(p or n))*v
        priors['nonplanar_axis_aligned']+=int(bool(a and not p and not n))*v
    selected=[];missing=[]
    if selection_dir:
        for path in sorted(selection_dir.glob('*_selection.json')):
            for epoch,obj in epoch_objects(path,set(epochs)):
                c=Counter();nselected=0
                for category,buckets in obj.items():
                    if not isinstance(buckets,dict):raise ValueError(f'Bad category structure: {path} {epoch}')
                    for legacy,names in buckets.items():
                        if not isinstance(names,list):raise ValueError(f'Bad selection list: {path} {epoch}')
                        for name,n in Counter(names).items():
                            if name not in lookup:
                                missing.append(dict(run=path.name,epoch=epoch,filename=name,count=n));continue
                            label,(p,near,a)=lookup[name]
                            if legacy.lower()!=label.lower():raise ValueError(f'Legacy label disagreement: {name} {legacy} {label}')
                            nselected+=n;c['legacy:'+label]+=n
                            c['exact_planar']+=p*n;c['near_planar']+=near*n
                            c['planar_or_near']+=int(bool(p or near))*n
                            c['nonplanar_axis_aligned']+=int(bool(a and not p and not near))*n
                for metric,n in sorted(c.items()):
                    prior_count=priors.get(metric)
                    prior=prior_count/totals['test'] if prior_count is not None else None
                    selected.append(dict(run=path.name,epoch=epoch,metric=metric,selected_count=n,
                                         selected_total=nselected,selected_fraction=n/nselected,
                                         candidate_fraction=prior,
                                         enrichment=(n/nselected)/prior if prior else None))
            print(f'Audited requested epochs: {path.name}',flush=True)
    if missing:
        write_csv(out/'missing_selected_filenames.csv',missing,list(missing[0]))
        raise ValueError(f'{len(missing)} selected filenames missing from included test manifest; summaries not released.')
    if selected:write_csv(out/'selected_geometry_by_run_epoch.csv',selected,list(selected[0]))
    info=dict(manifest=str(manifest.resolve()),selection_dir=str(selection_dir) if selection_dir else None,
              requested_epochs=epochs,objects={k:len(v) for k,v in objects.items()},denominators=dict(totals),
              geometry_candidate_fractions={k:v/totals['test'] for k,v in priors.items()},
              historical_labels_changed=False,new_training=False,
              qualification='Geometric overlay on actual filenames; inclusion is reconstructed from current first-25/train and first-5/test rule. Retain historical training pools. Near-planar uses the renderer flag, not a newly defined continuous angular threshold.')
    (out/'taxonomy_audit.json').write_text(json.dumps(info,indent=2)+'\n')
    return info

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--selection-dir',type=Path)
    p.add_argument('--epochs',nargs='+',type=int,default=[1,10,20,30,50,70,100])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Output exists; use a new directory.')
    args.output.mkdir(parents=True)
    print(json.dumps(audit(args.manifest,args.selection_dir,args.epochs,args.output),indent=2))

if __name__=='__main__':main()
