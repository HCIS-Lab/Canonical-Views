"""Datasets for controlled recognition training under replayed view policies."""

import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import VisionDataset


VIEW_TYPES = [
    "Expanded",
    "Expanded-like",
    "Foreshortened",
    "Foreshortened-like",
    "Remainder",
]


def classify_view_family(filename):
    """Apply the same filename rules used by ModelNet40/MVSelect."""
    is_expanded = "planar" in filename and "short" not in filename
    is_foreshortened = "short" in filename and "like" not in filename
    is_expanded_like = "like" in filename and "short" not in filename
    is_foreshortened_like = "like" in filename and "short" in filename
    if is_foreshortened:
        return "Foreshortened"
    if is_expanded:
        return "Expanded"
    if is_foreshortened_like:
        return "Foreshortened-like"
    if is_expanded_like:
        return "Expanded-like"
    return "Remainder"


def _stable_order(values, token):
    def key(value):
        digest = hashlib.sha256(f"{token}|{value}".encode("utf-8")).hexdigest()
        return digest, value
    return sorted(values, key=key)


def load_policy_counts(selection_dir, selected_view_types="01234"):
    """Aggregate action frequencies by epoch/class/instance/filename."""
    if not os.path.isdir(selection_dir):
        raise FileNotFoundError(f"Selection directory not found: {selection_dir}")
    selected_buckets = {VIEW_TYPES[int(index)] for index in selected_view_types}
    counts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter)))
    selection_files = sorted(
        os.path.join(selection_dir, name)
        for name in os.listdir(selection_dir)
        if name.endswith("_selection.json")
    )
    if not selection_files:
        raise FileNotFoundError(
            f"No *_selection.json files in {selection_dir}")

    for path in selection_files:
        with open(path) as handle:
            payload = json.load(handle)
        for epoch_text, epoch_data in payload.items():
            epoch = int(epoch_text)
            for class_text, class_data in epoch_data.items():
                class_idx = int(class_text)
                for bucket, filenames in class_data.items():
                    if bucket not in selected_buckets:
                        continue
                    for value in filenames:
                        filename = os.path.basename(value)
                        instance_id = filename.split("_", 1)[0]
                        counts[epoch][class_idx][instance_id][filename] += 1
    if not counts:
        raise ValueError(
            "Selection files contained no actions for --selected_view_types "
            f"{selected_view_types}.")
    return counts, selection_files


def nearest_epoch(available_epochs, requested):
    return min(available_epochs, key=lambda epoch: (abs(epoch - requested), epoch))


def make_policy_schedule(protocol, available_epochs, recognition_epochs,
                         policy_epoch=None, final_policy_epoch=None,
                         shuffle_seed=1729, random_warmup_epochs=None):
    """Map recognition-training epoch to source-policy epoch."""
    available = sorted(set(int(epoch) for epoch in available_epochs))
    if not available:
        raise ValueError("No policy epochs are available")
    if recognition_epochs < 1:
        raise ValueError("recognition_epochs must be positive")

    if recognition_epochs == 1:
        evolving = [available[-1]]
    else:
        targets = np.linspace(available[0], available[-1], recognition_epochs)
        evolving = [nearest_epoch(available, int(round(value))) for value in targets]

    if protocol == "frozen":
        if policy_epoch is None:
            raise ValueError("frozen replay requires --policy_epoch")
        source = nearest_epoch(available, int(policy_epoch))
        sequence = [source] * recognition_epochs
    elif protocol == "final":
        requested = available[-1] if final_policy_epoch is None else final_policy_epoch
        source = nearest_epoch(available, int(requested))
        sequence = [source] * recognition_epochs
    elif protocol in {"evolving", "family_matched_random"}:
        sequence = evolving
    elif protocol == "random":
        sequence = [None] * recognition_epochs
    elif protocol == "random_then_evolving":
        if random_warmup_epochs is None:
            raise ValueError(
                "random_then_evolving replay requires "
                "--random_warmup_epochs")
        warmup = int(random_warmup_epochs)
        if not 0 <= warmup < recognition_epochs:
            raise ValueError(
                "random_warmup_epochs must be in [0, recognition_epochs); "
                f"got {warmup} for {recognition_epochs} epochs")
        sequence = [None] * warmup + evolving[warmup:]
    elif protocol == "shuffled":
        sequence = list(evolving)
        rng = np.random.RandomState(shuffle_seed)
        rng.shuffle(sequence)
    else:
        raise ValueError(f"Unsupported replay protocol: {protocol}")
    return {
        epoch + 1: (None if source is None else int(source))
        for epoch, source in enumerate(sequence)
    }


class PolicyReplayModelNet40(VisionDataset):
    """Return K views selected by a source policy epoch for each object."""

    def __init__(self, root, classnames, selection_dir, num_views=5,
                 recognition_epochs=100, protocol="evolving", policy_epoch=None,
                 final_policy_epoch=None, shuffle_seed=1729, seed=0,
                 split="test", per_cls_instances=0,
                 selected_view_types="01234", random_warmup_epochs=None):
        super().__init__(root)
        self.root = root
        self.classnames = list(classnames)
        self.num_class = len(self.classnames)
        self.num_cam = int(num_views)
        self.protocol = protocol
        self.seed = int(seed)
        self.split = split
        self.transform = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),
        ])
        self.policy_counts, self.selection_files = load_policy_counts(
            selection_dir, selected_view_types)
        self.available_policy_epochs = sorted(self.policy_counts)
        self.policy_schedule = make_policy_schedule(
            protocol,
            self.available_policy_epochs,
            recognition_epochs,
            policy_epoch=policy_epoch,
            final_policy_epoch=final_policy_epoch,
            shuffle_seed=shuffle_seed,
            random_warmup_epochs=random_warmup_epochs,
        )
        self.current_recognition_epoch = 1
        self.instances = []
        self._selected_cache = {}
        self.fallback_instances = set()

        policy_ids = defaultdict(set)
        for epoch_data in self.policy_counts.values():
            for class_idx, class_data in epoch_data.items():
                policy_ids[class_idx].update(class_data)

        for class_idx, class_name in enumerate(self.classnames):
            split_dir = os.path.join(root, class_name, split)
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"Dataset split not found: {split_dir}")
            grouped = defaultdict(list)
            for filename in sorted(os.listdir(split_dir)):
                if filename.lower().endswith(".png"):
                    grouped[filename.split("_", 1)[0]].append(filename)
            matched_ids = sorted(set(grouped).intersection(policy_ids[class_idx]))
            if per_cls_instances:
                matched_ids = matched_ids[:per_cls_instances]
            for instance_id in matched_ids:
                self.instances.append({
                    "class_idx": class_idx,
                    "class_name": class_name,
                    "instance_id": instance_id,
                    "filenames": grouped[instance_id],
                    "split_dir": split_dir,
                })

        if not self.instances:
            raise ValueError(
                "No dataset instances matched the instance IDs in selection.json")
        per_class = Counter(item["class_idx"] for item in self.instances)
        missing_classes = [
            index for index in range(self.num_class) if per_class[index] == 0]
        if missing_classes:
            raise ValueError(
                f"Policy replay has no matched instances for classes {missing_classes}")

    def set_epoch(self, recognition_epoch):
        recognition_epoch = int(recognition_epoch)
        if recognition_epoch not in self.policy_schedule:
            raise ValueError(
                f"Recognition epoch {recognition_epoch} is outside schedule "
                f"1..{len(self.policy_schedule)}")
        self.current_recognition_epoch = recognition_epoch

    @property
    def source_policy_epoch(self):
        return self.policy_schedule[self.current_recognition_epoch]

    def _top_policy_views(self, item, source_epoch):
        class_idx = item["class_idx"]
        instance_id = item["instance_id"]
        available = set(item["filenames"])
        counter = self.policy_counts[source_epoch][class_idx].get(
            instance_id, Counter())
        ranked = sorted(
            (filename for filename in counter if filename in available),
            key=lambda filename: (-counter[filename], filename),
        )
        selected = ranked[:self.num_cam]
        if len(selected) < self.num_cam:
            self.fallback_instances.add((source_epoch, class_idx, instance_id))
            remaining = [
                filename for filename in item["filenames"]
                if filename not in selected
            ]
            token = (
                f"fallback|{self.seed}|{source_epoch}|{class_idx}|{instance_id}")
            selected.extend(
                _stable_order(remaining, token)[:self.num_cam - len(selected)])
        if len(selected) != self.num_cam:
            raise ValueError(
                f"{item['class_name']}/{instance_id} has only {len(selected)} "
                f"usable views; need {self.num_cam}")
        return selected

    def _family_matched_views(self, item, source_epoch, selected):
        by_family = defaultdict(list)
        for filename in item["filenames"]:
            by_family[classify_view_family(filename)].append(filename)
        selected_set = set(selected)
        output = []
        family_occurrence = Counter()
        for selected_filename in selected:
            family = classify_view_family(selected_filename)
            occurrence = family_occurrence[family]
            family_occurrence[family] += 1
            alternatives = [
                filename for filename in by_family[family]
                if filename not in selected_set and filename not in output
            ]
            if not alternatives:
                alternatives = [
                    filename for filename in by_family[family]
                    if filename not in output
                ]
            token = (
                f"family|{self.seed}|{source_epoch}|{item['class_idx']}|"
                f"{item['instance_id']}|{family}|{occurrence}")
            ordered = _stable_order(alternatives, token)
            output.append(ordered[0] if ordered else selected_filename)
        return output

    def selected_filenames(self, index):
        item = self.instances[index]
        source_epoch = self.source_policy_epoch
        use_random_views = source_epoch is None
        cache_source = (
            ("random", self.current_recognition_epoch)
            if use_random_views else ("policy", source_epoch)
        )
        cache_key = (cache_source, index, self.protocol)
        if cache_key not in self._selected_cache:
            if use_random_views:
                token = (
                    f"random|{self.seed}|{self.current_recognition_epoch}|"
                    f"{item['class_idx']}|{item['instance_id']}")
                selected = _stable_order(item["filenames"], token)[:self.num_cam]
                if len(selected) != self.num_cam:
                    raise ValueError(
                        f"{item['class_name']}/{item['instance_id']} has only "
                        f"{len(selected)} views; need {self.num_cam}")
            else:
                selected = self._top_policy_views(item, source_epoch)
            if self.protocol == "family_matched_random":
                selected = self._family_matched_views(
                    item, source_epoch, selected)
            self._selected_cache[cache_key] = tuple(selected)
        return self._selected_cache[cache_key]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        item = self.instances[index]
        filenames = self.selected_filenames(index)
        images = [
            self.transform(Image.open(
                os.path.join(item["split_dir"], filename)).convert("RGB"))
            for filename in filenames
        ]
        metadata = {
            "instance_id": item["instance_id"],
            "class_idx": item["class_idx"],
            "source_policy_epoch": (
                "random" if self.source_policy_epoch is None
                else self.source_policy_epoch
            ),
            "filenames": list(filenames),
        }
        return (
            torch.stack(images),
            item["class_idx"],
            torch.ones(self.num_cam, dtype=torch.bool),
            metadata,
        )


class FixedRandomViewModelNet40(VisionDataset):
    """Common held-out view sets, identical for every replay condition."""

    def __init__(self, root, classnames, num_views=5, split="test-down",
                 per_cls_instances=5, seed=2027):
        super().__init__(root)
        self.root = root
        self.classnames = list(classnames)
        self.num_class = len(self.classnames)
        self.num_cam = int(num_views)
        self.transform = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),
        ])
        self.instances = []
        for class_idx, class_name in enumerate(self.classnames):
            split_dir = os.path.join(root, class_name, split)
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"Dataset split not found: {split_dir}")
            grouped = defaultdict(list)
            for filename in sorted(os.listdir(split_dir)):
                if filename.lower().endswith(".png"):
                    grouped[filename.split("_", 1)[0]].append(filename)
            instance_ids = sorted(grouped)
            if per_cls_instances:
                instance_ids = instance_ids[:per_cls_instances]
            for instance_id in instance_ids:
                token = f"eval|{seed}|{class_idx}|{instance_id}"
                selected = _stable_order(grouped[instance_id], token)[:self.num_cam]
                if len(selected) != self.num_cam:
                    raise ValueError(
                        f"{class_name}/{instance_id} has fewer than "
                        f"{self.num_cam} views")
                self.instances.append({
                    "class_idx": class_idx,
                    "class_name": class_name,
                    "instance_id": instance_id,
                    "split_dir": split_dir,
                    "filenames": tuple(selected),
                })

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        item = self.instances[index]
        images = [
            self.transform(Image.open(
                os.path.join(item["split_dir"], filename)).convert("RGB"))
            for filename in item["filenames"]
        ]
        metadata = {
            "instance_id": item["instance_id"],
            "class_idx": item["class_idx"],
            "filenames": list(item["filenames"]),
        }
        return (
            torch.stack(images),
            item["class_idx"],
            torch.ones(self.num_cam, dtype=torch.bool),
            metadata,
        )
