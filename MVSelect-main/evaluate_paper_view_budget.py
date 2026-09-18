#!/usr/bin/env python3
"""E03: compare policy and random images with the same initial image and budget.

Uses the five saved unrestricted final models. Tests budgets five and six.
No training. The five-image policy is a four-action prefix of the trained policy.
"""
import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def camera_order(paths, pose_table):
    ordered = {}
    for path in paths:
        fields = path.stem.split('_')
        pose = tuple(float(fields[i][1:]) for i in [2, 3, 4])
        if pose not in pose_table:
            raise ValueError(f'Unknown camera pose: {path}')
        index = pose_table[pose]
        if index in ordered:
            raise ValueError(f'Duplicate camera index {index}: {path}')
        ordered[index] = path
    if set(ordered) != set(range(114)):
        raise ValueError('An object does not contain exactly the saved 114 camera positions.')
    return [str(ordered[i]) for i in range(114)]


def draw(seed, cls, oid, repeat):
    rng = random.Random(f'{seed}|{cls}|{oid}|{repeat}|budget')
    initial = rng.randrange(114)
    others = rng.sample([i for i in range(114) if i != initial], 5)
    return initial, others


def main():
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    p.add_argument('--registry', type=Path, default=here / 'paper_core_checkpoints.json')
    p.add_argument('--instances', type=Path, default=here / 'paper_test_instances.csv')
    p.add_argument('--pose-table', type=Path, default=here / 'paper_pose_table.json')
    p.add_argument('--data-root', type=Path, default=Path('/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23'))
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--image-batch-size', type=int, default=24)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--smoke-test', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new output folder.')
    if args.repeats < 1 or args.image_batch_size < 1:
        p.error('Counts must be positive.')
    models = [x for x in json.loads(args.registry.read_text())['models'] if x['condition'] == 'unrestricted']
    if len(models) != 5:
        p.error('Expected all five unrestricted models.')
    for m in models:
        m['checkpoint'] = m['checkpoints']['100']
        if not Path(m['checkpoint']).is_file():
            p.error(f'Missing checkpoint: {m["checkpoint"]}')
    pose_table = {tuple(map(float, k.split('_'))): v for k, v in json.loads(args.pose_table.read_text()).items()}
    if len(pose_table) != 114 or set(pose_table.values()) != set(range(114)):
        p.error('Invalid saved pose table.')
    objects = list(csv.DictReader(args.instances.open()))
    if len(objects) != 160 or len({x['object_id'] for x in objects}) != 160:
        p.error('Expected 160 distinct held-out objects.')
    for obj in objects:
        paths = [x for x in (args.data_root / obj['category'] / 'test').glob(obj['object_id'] + '_*.png')
                 if float(x.stem.split('_')[4][1:]) == 0.0]
        obj['paths'] = camera_order(paths, pose_table)
    if args.smoke_test:
        models = models[:1]
        objects = objects[:1]
    if not args.preflight:
        import torch
        if not torch.cuda.is_available():
            p.error('CUDA unavailable. Run diagnose_paper_gpu.py first.')
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f'cuda:{args.gpu_id}')
        torch.ones(1, device=device).sum().item()
    args.output.mkdir(parents=True)
    plan = dict(protocol='shared_initial_policy_vs_random_budgets_5_6', completed=False,
                smoke_test=args.smoke_test, new_training=False,
                settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                script_sha256=digest(Path(__file__)), pose_table_sha256=digest(args.pose_table),
                registry_sha256=digest(args.registry), instances_sha256=digest(args.instances),
                models=models, objects=objects,
                scope='Saved unrestricted classifiers; budget five uses a four-action prefix, budget six uses all five actions.')
    (args.output / 'evaluation_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    if args.preflight:
        print(f'Preflight passed: {len(models)} models, {len(objects)} objects.')
        return
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import resnet18
    from torchvision import transforms
    from PIL import Image
    from src.models.mvselect import CamSelect
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Sequential(*list(resnet18(weights=None).children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(512, 32)
            self.select_module = CamSelect(114, 512, aggregation='max')

    model = Model().to(device).eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    fields = ['run', 'checkpoint_sha256', 'object_id', 'class_idx', 'repeat', 'budget', 'sampling',
              'initial_camera', 'camera_indices', 'prediction', 'correct', 'category_loss']
    summaries = []
    with (args.output / 'trial_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for job in models:
            state = torch.load(job['checkpoint'], map_location='cpu', weights_only=True)
            state = state.get('state_dict', state)
            model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
            sha = digest(Path(job['checkpoint']))
            job['checkpoint_sha256'] = sha
            measurements = {(n, s): [] for n in [5, 6] for s in ['policy', 'random']}
            with torch.inference_mode():
                for obj in objects:
                    feats = []
                    for offset in range(0, 114, args.image_batch_size):
                        imgs = []
                        for path in obj['paths'][offset:offset + args.image_batch_size]:
                            with Image.open(path) as im:
                                imgs.append(transform(im.convert('RGB')))
                        feats.append(model.avgpool(model.base(torch.stack(imgs).to(device))))
                    feat = torch.cat(feats).unsqueeze(0)
                    cls = int(obj['class_idx'])
                    for repeat in range(args.repeats):
                        initial, random_others = draw(args.seed, cls, obj['object_id'], repeat)
                        selected = torch.zeros(1, 114, dtype=torch.bool, device=device)
                        selected[0, initial] = True
                        actions = [initial]
                        policies = {}
                        for step in range(5):
                            _, (_, _, action, _) = model.select_module(feat, selected)
                            chosen = int(action.long().argmax(1).item())
                            if chosen in actions:
                                raise RuntimeError('Unexpected repeated camera in unrestricted selection.')
                            actions.append(chosen)
                            selected |= action
                            if step in [3, 4]:
                                policies[step + 2] = list(actions)
                        for budget in [5, 6]:
                            for sampling, indices in [('policy', policies[budget]), ('random', [initial] + random_others[:budget - 1])]:
                                logits = model.classifier(feat[:, indices].max(1).values.flatten(1))
                                prediction = int(logits.argmax(1).item())
                                loss = F.cross_entropy(logits, torch.tensor([cls], device=device)).item()
                                hit = int(prediction == cls)
                                measurements[budget, sampling].append((hit, loss))
                                writer.writerow(dict(run=job['run'], checkpoint_sha256=sha,
                                    object_id=obj['object_id'], class_idx=cls, repeat=repeat, budget=budget,
                                    sampling=sampling, initial_camera=initial, camera_indices=json.dumps(indices),
                                    prediction=prediction, correct=hit, category_loss=loss))
            for (budget, sampling), values in measurements.items():
                summaries.append(dict(run=job['run'], budget=budget, sampling=sampling,
                    n_objects=len(objects), n_sampling_repeats=args.repeats,
                    accuracy_pct=100 * statistics.mean(x[0] for x in values),
                    mean_category_loss=statistics.mean(x[1] for x in values)))
            f.flush()
            print(f'Completed {job["run"]}', flush=True)
    with (args.output / 'run_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    plan.update(completed=True, completed_scope='smoke_test_only' if args.smoke_test else 'full_protocol', torch_version=torch.__version__)
    (args.output / 'evaluation_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Completed: {args.output}. Use trained runs as replicates.')


if __name__ == '__main__':
    main()
