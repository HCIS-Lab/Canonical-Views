"""MOCHI dataset loading and trial helpers."""

import hashlib
import json
import os
import random
from collections import defaultdict

from PIL import Image
from datasets import load_dataset

from .config import MOCHI_DATASET


def _instance_rng(random_seed, cls_idx, ins_id):
    """RNG seeded deterministically per (random_seed, cls_idx, ins_id).

    Used so that the random-view baseline for a given instance is stable
    across epoch-group calls — same instance, same random subset, every time.
    Python's built-in hash() is process-randomised, so we use md5 for stability.
    """
    h = hashlib.md5(f"{random_seed}|{cls_idx}|{ins_id}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "little"))


MODELNET40_CLASSES = [
    'airplane,aeroplane,plane',
    'ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin',
    'basket,handbasket',
    'bathtub,bathing tub,bath,tub',
    'bed',
    'bench',
    'bookshelf',
    'bottle',
    'bowl',
    'bus,autobus,coach,charabanc,double-decker,jitney,motorbus,motorcoach,omnibus,passenger vehi',
    'cabinet',
    'camera,photographic camera',
    'car,auto,automobile,machine,motorcar',
    'chair',
    'clock',
    'display,video display',
    'faucet,spigot',
    'guitar',
    'helmet',
    'knife',
    'lamp',
    'laptop,laptop computer',
    'loudspeaker,speaker,speaker unit,loudspeaker system,speaker system',
    'motorcycle,bike',
    'mug',
    'pistol,handgun,side arm,shooting iron',
    'table',
    'telephone,phone,telephone set',
    'tower',
    'train,railroad train',
    'vessel,watercraft',
    'washer,automatic washer,washing machine',
]

VIEW_TYPE_LIST = ['Expanded', 'Expanded-like', 'Foreshortened', 'Foreshortened-like', 'Remainder']


def load_mochi():
    """Load the MOCHI dataset from HuggingFace.

    Returns:
        dataset: HuggingFace dataset with all trials.
    """
    return load_dataset(MOCHI_DATASET)["train"]


def get_trial_images(trial):
    """Parse a MOCHI trial into canonical (A, A', B) ordering.

    Args:
        trial: A single trial dict from the MOCHI dataset.

    Returns:
        img_A: First image of the matching object.
        img_A_prime: Second image of the matching object (different viewpoint).
        img_B: Image of the oddity (different object).
        trial_info: Dict with trial metadata.
    """
    images = trial["images"]
    oddity_idx = trial["oddity_index"]

    img_B = images[oddity_idx]
    same_indices = [i for i in range(len(images)) if i != oddity_idx]
    img_A = images[same_indices[0]]
    img_A_prime = images[same_indices[1]] if len(same_indices) > 1 else images[same_indices[0]]

    trial_info = {
        "trial_name": trial["trial"],
        "dataset": trial["dataset"],
        "condition": trial.get("condition", trial["dataset"]),
        "oddity_index": oddity_idx,
        "n_images": len(images),
        "human_accuracy": trial.get("human_avg", None),
        "human_rt": trial.get("RT_avg", None),
    }

    return img_A, img_A_prime, img_B, trial_info


def _aggregate_selected_views(selection_dir, view_type_indices, start_epoch, end_epoch):
    """Aggregate selected-view counts from RL selection.json files.

    Walks every *_selection.json in `selection_dir`, sums how often each
    image filename appears as selected across epochs in [start_epoch, end_epoch]
    and across the requested view types. Also records which bucket each
    filename belongs to (a filename has one canonical view-type bucket).

    Returns:
        totals: {class_idx (int): {instance_id (str): {filename (str): count (int)}}}
        file_to_bucket: {class_idx (int): {filename (str): bucket_name (str)}}
    """
    view_types = [VIEW_TYPE_LIST[i] for i in view_type_indices]
    totals = {}
    file_to_bucket = {}
    for fn in sorted(os.listdir(selection_dir)):
        if not fn.endswith("selection.json"):
            continue
        with open(os.path.join(selection_dir, fn), "r") as f:
            whole = json.load(f)
        for ep in range(start_epoch, end_epoch + 1):
            if str(ep) not in whole:
                continue
            epoch_sel = whole[str(ep)]
            for cls_idx in range(len(MODELNET40_CLASSES)):
                cls_key = str(cls_idx)
                if cls_key not in epoch_sel:
                    continue
                totals.setdefault(cls_idx, {})
                file_to_bucket.setdefault(cls_idx, {})
                for vt in view_types:
                    for fpath in epoch_sel[cls_key].get(vt, []):
                        ins = fpath.split('_')[0]
                        totals[cls_idx].setdefault(ins, defaultdict(int))
                        totals[cls_idx][ins][fpath] += 1
                        # First-seen wins; bucket assignment is a property of
                        # the view filename, not of when it was selected.
                        file_to_bucket[cls_idx].setdefault(fpath, vt)
    return totals, file_to_bucket


def load_modelnet_selected(
    data_root,
    selection_dir,
    selected_view_type="01234",
    start_epoch=80,
    end_epoch=100,
    num_cam=4,
    split="test",
    per_cls_instances=5,
    random_seed=42,
):
    """Build a list of trials from RL-selected views with paired random baselines.

    For each ModelNet instance that has selected views, emits ONE trial dict
    containing both the agent-selected subset and a same-size random subset
    drawn from the same instance's available views. Image files are opened
    lazily — the dict stores paths; the caller can open via `open_trial_images`.

    Args:
        data_root: Root of `modelnet_32_60_1_23` — expects
            `{data_root}/{classname}/{split}/{instance}_<view>.png`.
        selection_dir: Folder containing the RL agent's `*_selection.json` files.
        selected_view_type: String of digits 0-4 indicating which view types
            from VIEW_TYPE_LIST to keep (e.g. "01234" = all).
        start_epoch, end_epoch: Inclusive epoch range to aggregate counts over.
        num_cam: How many views per subset.
        split: Subfolder under each class (e.g. "test", "test-down").
        per_cls_instances: Cap on instances per class (0 = no cap).
        random_seed: Seed for the random-baseline view sampling.

    Returns:
        List of dicts, each with:
            - trial: f"{classname}_{instance_id}"
            - dataset: classname (e.g. "chair")
            - class_idx: int
            - instance_id: str
            - agent_images_paths: list[str] of length num_cam
            - random_images_paths: list[str] of length num_cam
            - n_cam: int
    """
    view_type_idx = [int(c) for c in selected_view_type]
    counts, file_to_bucket = _aggregate_selected_views(
        selection_dir, view_type_idx, start_epoch, end_epoch
    )

    trials = []
    for cls_idx, cls_name in enumerate(MODELNET40_CLASSES):
        split_dir = os.path.join(data_root, cls_name, split)
        if not os.path.isdir(split_dir):
            continue
        all_files = sorted(os.listdir(split_dir))
        files_by_instance = defaultdict(list)
        for fn in all_files:
            files_by_instance[fn.split('_')[0]].append(fn)

        instance_counts = counts.get(cls_idx, {})
        if not instance_counts:
            continue

        kept = 0
        for ins_id, ins_counts in instance_counts.items():
            if per_cls_instances and kept >= per_cls_instances:
                break

            ins_rng = _instance_rng(random_seed, cls_idx, ins_id)

            agent_top = sorted(ins_counts, key=ins_counts.get, reverse=True)[:num_cam]
            available = files_by_instance.get(ins_id, [])
            if len(agent_top) < num_cam:
                short = num_cam - len(agent_top)
                pad_pool = [f for f in available if f not in agent_top]
                if len(pad_pool) < short:
                    continue
                agent_top = agent_top + ins_rng.sample(pad_pool, short)

            if len(available) < num_cam:
                continue
            random_pick = ins_rng.sample(available, num_cam)

            # Count which bucket each agent-selected filename came from.
            cls_buckets = file_to_bucket.get(cls_idx, {})
            agent_bucket_counts = {vt: 0 for vt in VIEW_TYPE_LIST}
            for f in agent_top:
                vt = cls_buckets.get(f)
                if vt in agent_bucket_counts:
                    agent_bucket_counts[vt] += 1

            trials.append({
                "trial": f"{cls_name}_{ins_id}",
                "dataset": cls_name,
                "class_idx": cls_idx,
                "instance_id": ins_id,
                "agent_images_paths": [os.path.join(split_dir, f) for f in agent_top],
                "random_images_paths": [os.path.join(split_dir, f) for f in random_pick],
                "n_cam": num_cam,
                "agent_bucket_counts": agent_bucket_counts,
            })
            kept += 1

    return trials


def _classify_view_by_filename(fname):
    """Bucket a single ModelNet view filename.

    Follows the assignment in MVSelect-main/src/datasets/modelnet40.py:189-193
    and the elif precedence used in trainer_mvcnn.py:253-266
    (Foreshortened > expanded > Foreshortened-like > Expanded-like > Remainder).
    Returns the bucket key used in selection.json (note 'expanded' is lower-case).
    """
    is_long = 'planar' in fname and 'short' not in fname
    is_short = 'short' in fname and 'like' not in fname
    is_long_like = 'like' in fname and 'short' not in fname
    is_short_like = 'like' in fname and 'short' in fname

    if is_short:
        return 'Foreshortened'
    if is_long:
        return 'expanded'
    if is_short_like:
        return 'Foreshortened-like'
    if is_long_like:
        return 'Expanded-like'
    return 'Remainder'


def build_view_type_index(
    data_root,
    split="test",
    per_cls_instances=5,
):
    """Build a per-instance, per-bucket index of ALL views on disk.

    Enumerates every PNG in `{data_root}/{class}/{split}/`, groups by instance
    id (filename prefix before the first underscore), and buckets each filename
    by its substring tags (planar/like/short, see `_classify_view_by_filename`).

    Independent of any selection.json — every available view of every type is
    indexed, so a probe that samples uniformly from each bucket is not biased
    by the agent's preferences.

    Returns:
        List of dicts, one per instance:
            {
                "class_idx": int,
                "dataset": str (class name),
                "instance_id": str,
                "by_bucket": {bucket_name: [absolute_path, ...]}
            }
    """
    index = []
    for cls_idx, cls_name in enumerate(MODELNET40_CLASSES):
        split_dir = os.path.join(data_root, cls_name, split)
        if not os.path.isdir(split_dir):
            continue

        by_instance = defaultdict(lambda: defaultdict(list))
        for fname in sorted(os.listdir(split_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            ins_id = fname.split('_')[0]
            bucket = _classify_view_by_filename(fname)
            by_instance[ins_id][bucket].append(os.path.join(split_dir, fname))

        # Deterministic instance ordering for reproducible per_cls_instances cap.
        ins_ids = sorted(by_instance.keys())
        if per_cls_instances:
            ins_ids = ins_ids[:per_cls_instances]
        for ins_id in ins_ids:
            buckets = by_instance[ins_id]
            if not buckets:
                continue
            index.append({
                "class_idx": cls_idx,
                "dataset": cls_name,
                "instance_id": ins_id,
                "by_bucket": dict(buckets),
            })
    return index


def open_trial_images(paths):
    """Open a list of image paths as RGB PIL images."""
    return [Image.open(p).convert("RGB") for p in paths]


def trials_by_condition(dataset):
    """Group trial indices by condition.

    Args:
        dataset: MOCHI HuggingFace dataset.

    Returns:
        Dict mapping condition name to list of trial indices.
    """
    groups = {}
    for i in range(len(dataset)):
        condition = dataset[i]["dataset"]
        if condition not in groups:
            groups[condition] = []
        groups[condition].append(i)
    return groups
