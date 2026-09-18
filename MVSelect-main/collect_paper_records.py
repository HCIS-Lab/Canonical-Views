#!/usr/bin/env python3
"""Collect existing paper records without training, inference, or reading image pixels.

Run from MVSelect-main. Uses only the Python standard library. Original files
are read only; a new directory and .tar.gz archive are created. Object inclusion
is reconstructed from the current loader's first-25/train, first-5/test rule,
not represented as proof of which objects a historical run actually used.
"""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fnmatch
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tarfile

PATTERN = re.compile(r"^(?P<object_id>[^_]+)_(?P<camera>\d+)_e(?P<e>[-+\d.]+)_a(?P<a>[-+\d.]+)_r(?P<r>[-+\d.]+)_")
SMALL_RECORDS = [
    'run_summary.json', 'metrics.csv', 'policy_epoch_schedule.json',
    '*audit*.csv', 'aggregated_run_summaries.csv', 'aggregated_training_metrics.csv',
    'pairwise_representation_comparison.csv', '*cka*.csv', 'cluster_metrics.csv',
    'temporal_test.csv', '*config*.json', '*settings*.json', '*args*.json',
    '*manifest*.json', '*analysis_summary*.json', '*single_view_confidence*.csv',
    'results.csv', 'overall_summary.csv', 'per_class_summary.csv',
    'replacement_selected_view_means.csv', 'replacement_view_family_contributions.csv',
    'replacement_family_pair_contributions.csv',
]


def dump_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classes_from_loader(repo):
    tree = ast.parse((repo / 'src/datasets/modelnet40.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'classnames' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Cannot find classnames in the dataset loader.')


def parse_image(name):
    match = PATTERN.match(name)
    if not match:
        return None
    d = match.groupdict()
    e, a, r = (float(d[k]) for k in ('e', 'a', 'r'))
    planar, short, like = ('planar' in name, 'short' in name, 'like' in name)
    flags = [planar and not short, like and not short, short and not like, short and like]
    names = ['expanded', 'expanded-like', 'foreshortened', 'foreshortened-like']
    label = next((n for n, v in zip(names, flags) if v), 'remainder')
    er, ar = math.radians(e), math.radians(a)
    return dict(object_id=d['object_id'], filename_camera_id=int(d['camera']),
                elevation_deg=e, azimuth_deg=a, roll_deg=r,
                direction_x=math.cos(er)*math.sin(ar), direction_y=math.sin(er),
                direction_z=math.cos(er)*math.cos(ar),
                planar_flag=int(planar), short_flag=int(short), like_flag=int(like),
                parser_view_type=label, overlapping_parser_flags=int(sum(flags)>1))


def collect_dataset(root, repo, out):
    fields = ['category', 'split', 'relative_path', 'included_by_current_paper_rule',
              'object_id', 'filename_camera_id', 'elevation_deg', 'azimuth_deg', 'roll_deg',
              'direction_x', 'direction_y', 'direction_z', 'planar_flag', 'short_flag',
              'like_flag', 'parser_view_type', 'overlapping_parser_flags']
    counts, objects, malformed, object_roles = Counter(), [], [], defaultdict(set)
    num_rows = 0
    with gzip.open(out / 'rendered_image_manifest.csv.gz', 'wt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for category in classes_from_loader(repo):
            for split in ('train', 'test', 'val', 'test-down'):
                folder = root / category / split
                if not folder.is_dir():
                    continue
                files = sorted(folder.glob('*.png'))
                by_object = defaultdict(list)
                for p in files:
                    record = parse_image(p.name)
                    if record is None:
                        malformed.append(str(p.relative_to(root)))
                    else:
                        by_object[record['object_id']].append((p, record))
                cap = {'train': 25, 'test': 5}.get(split, 0)
                usable = [oid for oid, items in by_object.items() if any(r['roll_deg'] == 0 for _, r in items)]
                included = set(usable[:cap])
                for oid, items in by_object.items():
                    object_roles[oid].add((category, split))
                    objects.append(dict(category=category, split=split, object_id=oid,
                                        images=len(items), non_roll_images=sum(r['roll_deg']==0 for _, r in items),
                                        included_by_current_paper_rule=int(oid in included)))
                    for p, r in items:
                        take = oid in included and r['roll_deg'] == 0
                        writer.writerow(dict(category=category, split=split, relative_path=str(p.relative_to(root)),
                                             included_by_current_paper_rule=int(take), **r))
                        if take:
                            counts[(split, r['parser_view_type'])] += 1
                        num_rows += 1
    dump_csv(out / 'object_manifest.csv', objects,
             ['category','split','object_id','images','non_roll_images','included_by_current_paper_rule'])
    totals = Counter()
    for (split, label), n in counts.items():
        totals[split] += n
    dump_csv(out / 'candidate_counts_reconstructed.csv',
             [dict(split=s, view_type=v, count=n, denominator=totals[s], fraction=n/totals[s])
              for (s,v),n in sorted(counts.items())],
             ['split','view_type','count','denominator','fraction'])
    overlaps = {oid: sorted(roles) for oid,roles in object_roles.items()
                if any(s=='train' for _,s in roles) and any(s=='test' for _,s in roles)}
    return dict(dataset_root=str(root), image_rows=num_rows, object_split_rows=len(objects),
                malformed_filenames=malformed, train_test_object_id_overlaps=overlaps,
                qualification='Membership reconstructed from present filenames and first 25 train / 5 test objects per category; verify against historical run dataset. Unit camera directions come from filename angles; filename camera IDs are not assumed to be selector indices.')


def collect_records(repo, out):
    records, copied_bytes, errors = [], 0, []
    roots = [(repo, 'MVSelect-main'), (repo.parent/'human_multiview-main', 'human_multiview-main')]
    for base, label in roots:
        for sub in ('logs', 'meta_logs', 'compare', 'results'):
            start = base/sub
            if not start.is_dir():
                continue
            for directory, dirs, files in os.walk(start, followlinks=False):
                dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not (Path(directory)/d).is_symlink())
                for name in sorted(files):
                    p = Path(directory)/name
                    if p.is_symlink() or p.suffix.lower() not in ('.json','.csv','.npz','.pth','.log','.txt','.py','.sh'):
                        continue
                    try:
                        size = p.stat().st_size
                        wanted = any(fnmatch.fnmatch(name.lower(), pat) for pat in SMALL_RECORDS)
                        copy = wanted and size <= 5*1024**2 and copied_bytes+size <= 64*1024**2
                        status = 'copied' if copy else ('size_budget_or_file_limit' if wanted else 'indexed_only')
                        if copy:
                            dest = out/'existing_records'/label/p.relative_to(base)
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(p, dest)
                            copied_bytes += size
                        records.append(dict(path=str(p), bytes=size, status=status))
                    except OSError as exc:
                        errors.append(dict(path=str(p), error=str(exc)))
    dump_csv(out/'existing_file_index.csv', records, ['path','bytes','status'])
    snapshots = []
    candidates = [repo/'main.py', repo/'src/trainer.py', repo/'src/trainer_mvcnn.py',
                  repo/'src/models/mvselect.py', repo/'src/models/mvcnn.py',
                  repo/'src/datasets/modelnet40.py', repo.parent/'modelnet_generation.py',
                  repo/'run_main.sh', repo/'run_active_single_view_experiment.sh',
                  repo/'run_policy_replay_experiment.sh', repo/'run_temporal_test_all.sh',
                  repo/'temporal_selection_test.py', repo/'run_view_contribution_pipeline.sh']
    candidates += sorted(repo.glob('*pose_table.json'))
    for p in candidates:
        if p.is_file():
            rel = p.relative_to(repo.parent)
            dest = out/'current_source_snapshots'/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p,dest)
            snapshots.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return dict(indexed_files=len(records), copied_bytes=copied_bytes, errors=errors,
                current_source_snapshots=snapshots,
                source_warning='Current source snapshots do not prove which source version ran previously. No historical trainer is inferred from them.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--dataset-root', type=Path, default=Path('/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    if not (repo/'src/datasets/modelnet40.py').is_file():
        parser.error('Use --repo-root to specify MVSelect-main.')
    if not args.dataset_root.is_dir():
        parser.error('Dataset directory is missing. Supply its actual path with --dataset-root.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = (args.output or repo/f'paper_records_{stamp}').resolve()
    archive = Path(str(out)+'.tar.gz')
    if out.exists() or archive.exists():
        parser.error('Output already exists; choose a new --output name.')
    out.mkdir(parents=True)
    report = {'created_utc':stamp, 'new_model_training':False, 'new_model_evaluation':False,
              'dataset':collect_dataset(args.dataset_root,repo,out), 'records':collect_records(repo,out)}
    (out/'collection_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(out,arcname=out.name)
    print(f'Collected existing records only. Images, checkpoints and large arrays were not copied.\nReturn this archive: {archive}\nSee collection_summary.json for omissions and reconstructed-membership qualifications.')


if __name__ == '__main__':
    main()
