#!/usr/bin/env python3
"""E18: crossed recipient stages, family-matched spread interventions and updates.

Default is a read-only plan. --preflight checks inputs without training.
--execute trains recipients only; all source models and old results are read-only.
See README_paper_mechanism.md for estimands, limitations and exact launch commands.
"""
import argparse
import copy
import csv
import gzip
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tarfile
import time

from paper_mechanism_core import (CONDITIONS, canonical_hash, check_disjoint,
    contrast_values, direction, evaluation_views, job_spec, make_design, sha,
    split_evaluation, training_views)

DEFAULT_DATA = '/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23'


def write_json(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_hash(state):
    import hashlib
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def optimizer_hash(value):
    import hashlib
    import torch
    h = hashlib.sha256()

    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(str(v.dtype).encode())
            h.update(str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(repr(k).encode()); visit(v[k])
        elif isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def load_recognizer_state(path, model):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    state = payload.get('state_dict', payload)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    # Exclusion is explicit: no selector parameters are used to train recipients.
    state = {k: v for k, v in state.items() if not k.startswith('select_module.')}
    model.load_state_dict(state, strict=True)


def make_model():
    import torch
    from torchvision.models import resnet18

    class Recognizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Sequential(*list(resnet18(weights=None).children())[:-2])
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = torch.nn.Linear(512, 32)

        def forward(self, images):
            b, n, c, h, w = images.shape
            feat = self.avgpool(self.base(images.reshape(b*n, c, h, w)))
            feat = feat.reshape(b, n, 512).max(dim=1).values
            return self.classifier(feat)
    return Recognizer()


class ImageGroups:
    def __init__(self, root, objects, condition=None, epoch=1, seed=0, repeats=1):
        from torchvision import transforms
        self.root, self.objects = Path(root), objects
        self.condition, self.epoch, self.seed = condition, epoch, seed
        self.repeats = repeats
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225))])

    def __len__(self):
        return len(self.objects) * self.repeats

    def __getitem__(self, index):
        import torch
        from PIL import Image
        obj = self.objects[index // self.repeats]
        repeat = index % self.repeats
        names = (evaluation_views(obj, repeat) if self.condition is None else
                 training_views(obj, self.condition, self.epoch, self.seed))
        imgs = []
        for name in names:
            with Image.open(self.root / obj['category'] / obj['split'] / name) as im:
                imgs.append(self.transform(im.convert('RGB')))
        return torch.stack(imgs), obj['class_idx'], obj['object_id'], repeat


def loader(root, objects, args, condition=None, epoch=1, shuffle=False, repeats=1):
    import torch
    # The same object order is used in every treatment at each adaptation epoch.
    generator = torch.Generator().manual_seed(40000 + args.recipient_seed*1000 + epoch)
    return torch.utils.data.DataLoader(
        ImageGroups(root, objects, condition, epoch, args.recipient_seed, repeats),
        batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
        pin_memory=args.device.startswith('cuda'), generator=generator)


def evaluate(model, batches, device):
    import torch
    import torch.nn.functional as F
    model.eval()
    rows = []
    with torch.no_grad():
        for images, labels, ids, repeats in batches:
            logits = model(images.to(device))
            target = labels.to(device)
            loss = F.cross_entropy(logits, target, reduction='none')
            prob = logits.softmax(1).gather(1, target[:, None]).squeeze(1)
            for oid, rep, y, pred, l, p in zip(ids, repeats.tolist(), labels.tolist(),
                    logits.argmax(1).tolist(), loss.tolist(), prob.tolist()):
                rows.append(dict(object_id=oid, repeat=rep, class_idx=y,
                                 prediction=pred, correct=int(y == pred), loss=l,
                                 true_class_probability=p))
    return {'loss': sum(r['loss'] for r in rows)/len(rows),
            'accuracy_pct': 100*sum(r['correct'] for r in rows)/len(rows),
            'n_objects': len(set(r['object_id'] for r in rows)),
            'n_view_sets': len(rows)}, rows


def gradient(model, batches, device):
    """Full recognizer mean-loss gradient; BN buffers fixed for this diagnostic."""
    import torch
    import torch.nn.functional as F
    model.eval()
    model.zero_grad(set_to_none=True)
    n = len(batches.dataset)
    loss_sum = 0.
    for images, labels, _, _ in batches:
        loss = F.cross_entropy(model(images.to(device)), labels.to(device), reduction='sum')
        (loss/n).backward()
        loss_sum += float(loss.detach())
    gradients = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                 for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    return loss_sum/n, gradients


def diagnostic(model, train, probe, root, args, stage):
    """No diagnostic gradient updates a retained model or chooses an image set."""
    import torch
    baseline = cpu_state(model)
    baseline_hash = state_hash(baseline)
    probe_loader = loader(root, probe, args)
    before, reference = gradient(model, probe_loader, args.device)
    # One object per category, fixed before outcomes; matched across conditions.
    sample = [next(o for o in train if o['class_idx'] == c)
              for c in sorted({o['class_idx'] for o in train})]
    ref_sq = sum(float((g.double()**2).sum()) for g in reference)
    rows = []
    for condition in CONDITIONS:
        model.load_state_dict(baseline)
        train_loss, grads = gradient(model, loader(root, sample, args, condition), args.device)
        dot = sum(float((a.double()*b.double()).sum()) for a, b in zip(reference, grads))
        norm_sq = sum(float((g.double()**2).sum()) for g in grads)
        cosine = dot/math.sqrt(ref_sq*norm_sq) if ref_sq*norm_sq > 0 else None
        for step_size in (1e-5, 1e-4):
            model.load_state_dict(baseline)
            with torch.no_grad():
                for p, g in zip(model.parameters(), grads):
                    p.add_(g, alpha=-step_size)
            after, _ = evaluate(model, probe_loader, args.device)
            rows.append(dict(stage=stage, condition=condition, diagnostic='plain_sgd_fixed_bn',
                             step_size=step_size, training_loss=train_loss,
                             train_gradient_norm=math.sqrt(norm_sq),
                             probe_gradient_norm=math.sqrt(ref_sq),
                             gradient_inner_product=dot, gradient_cosine=cosine,
                             predicted_probe_loss_reduction=step_size*dot,
                             actual_probe_loss_reduction=before-after['loss'],
                             before_probe_loss=before, after_probe_loss=after['loss'],
                             training_objects=len(sample), probe_objects=len(probe),
                             initial_state_sha256=baseline_hash))
        del grads
    model.load_state_dict(baseline)
    if state_hash(model.state_dict()) != baseline_hash:
        raise RuntimeError('Diagnostic altered the retained branch state')
    return rows


def train_epoch(model, optimizer, batches, device):
    import torch.nn.functional as F
    model.train()
    loss_sum, correct, n = 0., 0, 0
    for images, labels, _, _ in batches:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        size = len(labels)
        loss_sum += float(loss.detach())*size
        correct += int((logits.argmax(1) == labels).sum())
        n += size
    return loss_sum/n, 100*correct/n, len(batches)


def source_diagnostic(model, checkpoints, train, root, args):
    rows = []
    for epoch in (10, 100):
        load_recognizer_state(checkpoints[str(epoch)]['path'], model)
        for condition in CONDITIONS:
            metrics, _ = evaluate(model, loader(root, train, args, condition), args.device)
            rows.append(dict(source_epoch=epoch, condition=condition, **metrics))
    return rows


def resolve_inputs(args):
    spec = job_spec(args.job_id)
    args.recipient_seed = spec['recipient_seed']
    registry = json.loads((args.assets/'registry.json').read_text())
    entry = registry['sources'][str(spec['source_seed'])]
    bank_path = args.assets / entry['bank']
    if sha(bank_path) != entry['bank_sha256']:
        raise ValueError('Source image-bank hash mismatch')
    bank = json.loads(bank_path.read_text())
    if bank['source_seed'] != spec['source_seed'] or bank['target_mode'] != 'legal':
        raise ValueError('Incorrect source bank or training target')
    train = make_design(bank, 9000+spec['recipient_seed'])
    evaluation = []
    categories = {o['class_idx']: o['category'] for o in train}
    if set(categories) != set(range(32)) or len(train) != 160:
        raise ValueError('Expected five replay-training objects in each of 32 categories')
    from collections import Counter
    if set(Counter(o['class_idx'] for o in train).values()) != {5}:
        raise ValueError('Unbalanced replay-training categories')
    for cls, category in sorted(categories.items()):
        grouped = {}
        directory = args.data_root / category / 'test-down'
        for path in sorted(directory.glob('*.png')):
            fields = path.stem.split('_')
            if len(fields) < 5 or float(fields[4][1:]) != 0:
                continue
            grouped.setdefault(fields[0], []).append(path.name)
        # Reproduce the existing replay's first-five rule, explicitly recorded.
        for oid in sorted(grouped)[:5]:
            names = grouped[oid]
            if len(names) != 114 or len(set(names)) != 114:
                raise ValueError(f'Expected 114 non-roll images: {category}/{oid}')
            evaluation.append(dict(class_idx=cls, category=category, split='test-down',
                                   object_id=oid, candidates=names))
    probe, test = split_evaluation(evaluation)
    check_disjoint(train, probe, test)
    for obj in train + probe + test:
        if len(obj['candidates']) != 114:
            raise ValueError('Incomplete candidate grid')
        directions = {tuple(round(x, 6) for x in direction(n)) for n in obj['candidates']}
        if len(directions) != 114:
            raise ValueError('Duplicate camera direction in bank')
        for name in obj['candidates']:
            if not (args.data_root/obj['category']/obj['split']/name).is_file():
                raise FileNotFoundError(str(args.data_root/obj['category']/obj['split']/name))
    # Resolve exact recorded checkpoint paths relative to the server repository.
    repo = Path(__file__).resolve().parent
    cps = copy.deepcopy(entry['checkpoints'])
    initial = copy.deepcopy(registry['recipient_backbone'])
    for cp in list(cps.values()) + [initial]:
        cp['path'] = str(repo / cp['relative_path'])
        if not Path(cp['path']).is_file():
            raise FileNotFoundError(cp['path'])
        if sha(cp['path']) != cp['sha256']:
            raise ValueError('Checkpoint differs from audited source: '+cp['path'])
    full_counts = dict(train=len(train), diagnostic=len(probe), endpoint=len(test))
    if args.smoke_test:
        # Keep separate folders and label these as software tests, never evidence.
        train = train[:6]
        probe = probe[:4]
        test = test[:6]
    manifest = {'spec': spec, 'source_bank_sha256': entry['bank_sha256'],
                'registry_sha256': sha(args.assets/'registry.json'),
                'checkpoints': cps, 'recipient_backbone': initial,
                'full_object_counts': full_counts,
                'train_objects': train, 'diagnostic_objects': probe,
                'endpoint_objects': test,
                'endpoint_viewsets_sha256': canonical_hash([
                    (o['object_id'], [evaluation_views(o, r) for r in range(3)]) for o in test]),
                'training_design_sha256': canonical_hash(train)}
    return manifest


def archive_results(output):
    target = output/'results_to_return.tar.gz'
    with tarfile.open(target, 'w:gz') as tar:
        for path in sorted(output.rglob('*')):
            if path.is_file() and (path.suffix in ('.json', '.csv', '.py', '.md') or
                                   path.name.endswith('.json.gz')):
                tar.add(path, arcname=str(path.relative_to(output)))
    return target


def run(args, manifest):
    import torch
    import torchvision
    args.output.mkdir(parents=True, exist_ok=False)
    stages = (0, 1) if args.smoke_test else (0, 30)
    epochs = 1 if args.smoke_test else 100
    train, probe, test = [manifest[k] for k in ('train_objects', 'diagnostic_objects', 'endpoint_objects')]
    args.device = 'cpu' if args.cpu_smoke else f'cuda:{args.gpu_id}'
    if args.cpu_smoke and not args.smoke_test:
        raise ValueError('--cpu-smoke requires --smoke-test')
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in this allocation')
        torch.cuda.set_device(args.gpu_id)
        torch.ones(1, device=args.device).sum().item()
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    seed_everything(12000+args.recipient_seed)
    model = make_model().to(args.device)
    plan = dict(protocol='E18_v1', completed=False, scientific_result=not args.smoke_test,
                software_smoke_test=args.smoke_test, **manifest['spec'], stages=list(stages),
                adaptation_epochs=epochs, conditions=list(CONDITIONS),
                batch_size=args.batch_size, lr=5e-5, weight_decay=.01,
                optimizer='Adam', branch_schedule='cosine decay over 100 adaptation epochs',
                warmup='random five views resampled each epoch; 30 epochs; constant LR',
                recognition_images=5, pooling='max', training_bn='updating', diagnostic_bn='fixed',
                workers=args.workers, cpu_threads=args.cpu_threads,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu_name=torch.cuda.get_device_name(args.gpu_id) if args.device.startswith('cuda') else 'CPU',
                torch_version=torch.__version__, torchvision_version=torchvision.__version__,
                python_version=platform.python_version(), cuda_version=torch.version.cuda,
                source_bank_sha256=manifest['source_bank_sha256'],
                training_design_sha256=manifest['training_design_sha256'],
                endpoint_viewsets_sha256=manifest['endpoint_viewsets_sha256'])
    write_json(args.output/'run_plan.json', plan)
    with gzip.open(args.output/'input_manifest.json.gz', 'wt') as f:
        json.dump(manifest, f, indent=2)
    for name in ('run_paper_mechanism.py', 'paper_mechanism_core.py', 'README_paper_mechanism.md'):
        path = Path(__file__).with_name(name)
        if path.exists():
            (args.output/name).write_bytes(path.read_bytes())
    plan['source_code_sha256'] = {n: sha(args.output/n) for n in
                                 ('run_paper_mechanism.py', 'paper_mechanism_core.py')}
    write_json(args.output/'run_plan.json', plan)

    save_csv(args.output/'source_recognizer_scores.csv',
             source_diagnostic(model, manifest['checkpoints'], train, args.data_root, args))
    # All source policies share this ImageNet backbone. Recipient classifier seeds
    # are crossed with sources and never initialized from a trained category head.
    load_recognizer_state(manifest['recipient_backbone']['path'], model)
    seed_everything(12000+args.recipient_seed)
    model.classifier.reset_parameters()
    initial = cpu_state(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=.01)
    states = {0: (initial, copy.deepcopy(optimizer.state_dict()))}
    warmup_rows = []
    for epoch in range(1, stages[-1]+1):
        metrics = train_epoch(model, optimizer, loader(args.data_root, train, args,
                              'warmup', epoch, shuffle=True), args.device)
        warmup_rows.append(dict(epoch=epoch, loss=metrics[0], accuracy_pct=metrics[1], updates=metrics[2]))
    states[stages[-1]] = (cpu_state(model), copy.deepcopy(optimizer.state_dict()))
    save_csv(args.output/'warmup_metrics.csv', warmup_rows)
    audits, trajectory, diagnostics, predictions, endpoints, short_metrics = [], [], [], [], [], []
    test_loader = loader(args.data_root, test, args, repeats=3)
    probe_loader = loader(args.data_root, probe, args)
    started = time.monotonic()
    for stage in stages:
        baseline, optimizer_state = states[stage]
        model.load_state_dict(baseline)
        baseline_hash = state_hash(baseline)
        torch.save({'model': baseline, 'optimizer': optimizer_state}, args.output/f'recipient_stage{stage}.pth')
        diagnostics.extend(diagnostic(model, train, probe, args.data_root, args, stage))
        save_csv(args.output/'update_diagnostics.csv', diagnostics)
        baseline_eval, _ = evaluate(model, test_loader, args.device)
        for condition in CONDITIONS:
            model.load_state_dict(baseline)
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=.01)
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            for group in optimizer.param_groups:
                group['lr'] = 5e-5
            seed_everything(25000+args.recipient_seed*100+stage)
            if state_hash(model.state_dict()) != baseline_hash:
                raise RuntimeError('A branch did not start from the same state')
            initial_opt_steps = sorted(set(int(v['step']) for v in optimizer.state.values()))
            audit = dict(stage=stage, condition=condition, initial_state_sha256=baseline_hash,
                         initial_optimizer_sha256=optimizer_hash(optimizer.state_dict()),
                         initial_optimizer_steps=str(initial_opt_steps), initial_accuracy_pct=baseline_eval['accuracy_pct'],
                         expected_updates=epochs*math.ceil(len(train)/args.batch_size), actual_updates=0)
            trajectory.append(dict(stage=stage, condition=condition, epoch=0, training_loss=None,
                                   training_accuracy_pct=None, **baseline_eval))
            for epoch in range(1, epochs+1):
                # Match historical 100-epoch replay schedule without inheriting a
                # near-zero rate from the common recipient warm-up.
                batches = loader(args.data_root, train, args, condition, epoch, shuffle=True)
                model.train()
                import torch.nn.functional as F
                loss_sum, correct, n = 0., 0, 0
                for images, labels, _, _ in batches:
                    progress = audit['actual_updates'] / audit['expected_updates']
                    for group in optimizer.param_groups:
                        group['lr'] = 5e-5*.5*(1+math.cos(math.pi*progress))
                    images, labels = images.to(args.device), labels.to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    optimizer.step()
                    audit['actual_updates'] += 1
                    n += len(labels)
                    loss_sum += float(loss.detach())*len(labels)
                    correct += int((logits.argmax(1) == labels).sum())
                if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                    result, trials = evaluate(model, test_loader, args.device)
                    trajectory.append(dict(stage=stage, condition=condition, epoch=epoch,
                                           training_loss=loss_sum/n, training_accuracy_pct=100*correct/n, **result))
                    predictions.extend(dict(stage=stage, condition=condition, epoch=epoch, **r) for r in trials)
                    if epoch == 20:
                        short_metrics.append(dict(stage=stage, condition=condition, **result,
                             gain_from_branch_start_pp=result['accuracy_pct']-baseline_eval['accuracy_pct'],
                             loss_reduction_from_branch_start=baseline_eval['loss']-result['loss']))
                        save_csv(args.output/'short_horizon_metrics.csv', short_metrics)
                    save_csv(args.output/'learning_curves.csv', trajectory)
                    print(f'job {args.job_id:02d} stage {stage} {condition}: epoch {epoch}/{epochs}, accuracy {result["accuracy_pct"]:.2f}%', flush=True)
            if audit['actual_updates'] != audit['expected_updates']:
                raise RuntimeError('Unequal update budget')
            final_probe, _ = evaluate(model, probe_loader, args.device)
            audit['final_state_sha256'] = state_hash(model.state_dict())
            audits.append(audit)
            endpoints.append(dict(stage=stage, condition=condition, **result,
                                  gain_from_branch_start_pp=result['accuracy_pct']-baseline_eval['accuracy_pct'],
                                  diagnostic_probe_loss=final_probe['loss']))
            torch.save(cpu_state(model), args.output/f'model_stage{stage}_{condition}.pth')
            save_csv(args.output/'branch_audit.csv', audits)
            save_csv(args.output/'endpoint_metrics.csv', endpoints)
            save_csv(args.output/'endpoint_predictions.csv', predictions)
    contrasts = [dict(horizon_epochs=epochs, **r) for r in contrast_values(endpoints)]
    contrasts += [dict(horizon_epochs=20, **r) for r in contrast_values(short_metrics)]
    save_csv(args.output/'contrasts.csv', contrasts)
    plan.update(completed=True, elapsed_seconds=time.monotonic()-started,
                n_training_objects=len(train), n_diagnostic_objects=len(probe),
                n_endpoint_objects=len(test), recipient_initial_sha256=state_hash(initial))
    write_json(args.output/'run_plan.json', plan)
    archive = archive_results(args.output)
    print(f'Completed. Return {archive}; keep checkpoints on the server.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--assets', type=Path, default=Path(__file__).with_name('paper_mechanism_assets'))
    parser.add_argument('--data-root', type=Path, default=Path(DEFAULT_DATA))
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--cpu-smoke', action='store_true', help='Only for explicitly labeled software tests')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.cpu_threads < 1:
        parser.error('Invalid worker/batch/thread count')
    spec = job_spec(args.job_id)
    if not args.execute and not args.preflight:
        print(json.dumps(dict(**spec, new_training=False, plan_only=True,
              full_protocol='6 conditions x 2 recipient stages x 100 adaptation epochs',
              conditions=CONDITIONS, output=str(args.output)), indent=2))
        return
    if args.output.exists():
        parser.error('Use a new output folder; old/failed/completed runs are never overwritten')
    if args.cpu_smoke and not args.smoke_test:
        parser.error('--cpu-smoke requires --smoke-test')
    manifest = resolve_inputs(args)
    if args.preflight:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output/'preflight.json', dict(passed=True, new_training=False, **manifest))
        print(f'Preflight passed; no training. Report: {args.output}/preflight.json')
        return
    # Set before CUDA initialization; do not change the scheduler's device mask.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    try:
        run(args, manifest)
    except Exception as exc:
        if args.output.exists():
            write_json(args.output/'FAILED.json', dict(completed=False, error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
