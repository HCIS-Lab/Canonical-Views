"""Pure-Python design and audit utilities for the E18 replay experiment."""
import copy
import hashlib
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

CONDITIONS = ('early', 'late', 'late_spread', 'late_cluster',
              'random_fixed', 'random_resampled')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                         separators=(',', ':')).encode()).hexdigest()


def stable_sample(values, k, token):
    return random.Random(str(token)).sample(sorted(values), k)


def family(name):
    # Preserve historical operational labels, including nonplanar along-axis views.
    if 'short' in name and 'like' not in name:
        return 'foreshortened'
    if 'planar' in name and 'short' not in name:
        return 'expanded'
    if 'like' in name and 'short' in name:
        return 'foreshortened_like'
    if 'like' in name and 'short' not in name:
        return 'expanded_like'
    return 'remainder'


def direction(name):
    fields = Path(name).stem.split('_')
    elev, azim, roll = [float(fields[i][1:]) for i in (2, 3, 4)]
    if roll != 0:
        raise ValueError('E18 requires the non-roll candidate grid')
    e, a = math.radians(elev), math.radians(azim)
    return (math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a))


def angle(a, b):
    return math.degrees(math.acos(max(-1., min(1., sum(x*y for x, y in zip(a, b))))))


def spread(names):
    vectors = [direction(n) for n in names]
    return sum(angle(a, b) for a, b in itertools.combinations(vectors, 2)) / math.comb(len(names), 2)


def family_counts(names):
    return dict(sorted(Counter(map(family, names)).items()))


def matched_spread_sets(candidates, reference, token, draws=256):
    """Choose extrema among random family-matched sets; include reference.

    This is a finite randomized search, NOT a global optimization. Keeping the
    reference guarantees high >= reference >= low. Sets may share images with
    the reference. No accuracy, image descriptor or held-out label is consulted.
    """
    buckets = {}
    for name in candidates:
        buckets.setdefault(family(name), []).append(name)
    counts = family_counts(reference)
    rng = random.Random(str(token))
    vectors = {name: direction(name) for name in candidates}
    distances = {}
    for a, b in itertools.combinations(sorted(candidates), 2):
        distances[a, b] = angle(vectors[a], vectors[b])

    def score(names):
        return sum(distances[tuple(sorted((a, b)))] for a, b in
                   itertools.combinations(names, 2)) / math.comb(len(names), 2)

    low = high = tuple(sorted(reference))
    low_score = high_score = score(reference)
    for _ in range(draws):
        names = tuple(sorted(n for fam, count in counts.items()
                             for n in rng.sample(sorted(buckets[fam]), count)))
        s = score(names)
        if (s, names) < (low_score, low):
            low_score, low = s, names
        if (s, names) > (high_score, high):
            high_score, high = s, names
    assert family_counts(low) == family_counts(high) == counts
    return list(high), list(low)


def make_design(bank, design_seed):
    objects = copy.deepcopy(bank['objects'])
    for obj in objects:
        token = f'{design_seed}|{obj["class_idx"]}|{obj["object_id"]}'
        sets = {c: list(obj[c]) for c in ('early', 'late')}
        sets['late_spread'], sets['late_cluster'] = matched_spread_sets(
            obj['candidates'], sets['late'], token)
        sets['random_fixed'] = stable_sample(obj['candidates'], 5, 'fixed|' + token)
        obj['sets'] = sets
        obj['set_statistics'] = {c: {'mean_pair_angle_deg': spread(names),
                                   'family_counts': family_counts(names)}
                                 for c, names in sets.items()}
        if any(len(v) != 5 or len(set(v)) != 5 for v in sets.values()):
            raise ValueError('Every fixed set must contain five distinct images')
    return objects


def training_views(obj, condition, epoch, seed):
    if condition in ('random_resampled', 'warmup'):
        token = f'{condition}|{seed}|{epoch}|{obj["class_idx"]}|{obj["object_id"]}'
        return stable_sample(obj['candidates'], 5, token)
    return obj['sets'][condition]


def job_spec(job_id):
    if not 1 <= job_id <= 15:
        raise ValueError('job-id must be 1..15')
    # First five jobs cover all source models, not five repeats of one source.
    return {'job_id': job_id, 'source_seed': (job_id - 1) % 5,
            'recipient_seed': (job_id - 1) // 5}


def split_evaluation(objects):
    """Exactly two diagnostic and three endpoint objects per category."""
    probe, test = [], []
    for cls in range(32):
        members = sorted([o for o in objects if o['class_idx'] == cls],
                         key=lambda o: o['object_id'])
        if len(members) != 5:
            raise ValueError(f'Expected five test-down objects in category {cls}')
        probe.extend(members[:2])
        test.extend(members[2:])
    if {o['object_id'] for o in probe} & {o['object_id'] for o in test}:
        raise ValueError('Diagnostic/endpoint mesh overlap')
    return probe, test


def check_disjoint(train, probe, test):
    sets = [{o['object_id'] for o in rows} for rows in (train, probe, test)]
    if any(a & b for a, b in itertools.combinations(sets, 2)):
        raise ValueError('Training, diagnostic and endpoint meshes must be disjoint')


def evaluation_views(obj, repeat):
    token = f'eval20260916|{obj["class_idx"]}|{obj["object_id"]}|{repeat}'
    return stable_sample(obj['candidates'], 5, token)


def contrast_values(rows):
    """Per-job endpoint contrasts; sources, not rows, are the replication unit."""
    lookup = {(r['stage'], r['condition']): r for r in rows}
    out = []
    for stage in (0, 30):
        for a, b in [('early', 'late'), ('late_spread', 'late_cluster'),
                     ('late_spread', 'late'), ('random_resampled', 'random_fixed')]:
            if (stage, a) in lookup and (stage, b) in lookup:
                out.append({'stage': stage, 'contrast': a + '_minus_' + b,
                            'accuracy_difference_pp': lookup[stage, a]['accuracy_pct'] - lookup[stage, b]['accuracy_pct'],
                            'loss_advantage': (lookup[stage, b].get('loss', 0) - lookup[stage, a].get('loss', 0))})
    needed = [(s, c) for s in (0, 30) for c in ('early', 'late')]
    if all(k in lookup for k in needed):
        out.append({'stage': 'interaction', 'contrast': '(late-early)_stage30_minus_stage0',
                    'accuracy_difference_pp':
                    (lookup[30, 'late']['accuracy_pct'] - lookup[30, 'early']['accuracy_pct']) -
                    (lookup[0, 'late']['accuracy_pct'] - lookup[0, 'early']['accuracy_pct']),
                    'loss_advantage':
                    (lookup[30, 'early'].get('loss', 0) - lookup[30, 'late'].get('loss', 0)) -
                    (lookup[0, 'early'].get('loss', 0) - lookup[0, 'late'].get('loss', 0))})
    return out
