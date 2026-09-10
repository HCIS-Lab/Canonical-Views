#!/usr/bin/env python3
"""Train matched recognition networks under replayed MVSelect policies.

The selector itself is not trained here. Recorded ``*_selection.json`` action
frequencies define a K-view training set for each object and source-policy
epoch. Recognition networks start from one shared state and receive the same
number of optimizer updates under one of these temporal protocols:

    frozen                 one early policy epoch for all recognition epochs
    final                  final policy epoch from recognition epoch 1
    evolving               policy epoch t at recognition epoch t
    shuffled               evolving epochs in a fixed random temporal order
    family_matched_random  evolving family counts, different view identities
    random                 random K views per object, resampled each epoch
    random_then_evolving   random warm-up, then source policy epoch t at epoch t

Evaluation always uses the same deterministic random K-view sets from a
disjoint split, so condition differences cannot come from test-time inputs.
"""

import argparse
import hashlib
import json
import math
import os
import random
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mvselect_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from filelock import FileLock
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.downstream_dataset import Downstream_ModelNet40
from src.datasets.policy_replay_dataset import (
    FixedRandomViewModelNet40,
    PolicyReplayModelNet40,
)
from src.models.architectures import ARCHITECTURE_CHOICES, resolve_architecture
from src.models.mvcnn import MVCNN
from src.models.mvselect import aggregate_feat


DEFAULT_DATA_ROOT = (
    "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/"
    "modelnet_32_60_1_23"
)
PROTOCOLS = [
    "frozen",
    "final",
    "evolving",
    "shuffled",
    "family_matched_random",
    "random",
    "random_then_evolving",
]
CONDITION_DISPLAY_NAMES = {
    "frozen_10": "Epoch-10 selections throughout",
    "frozen_20": "Epoch-20 selections throughout",
    "frozen_30": "Epoch-30 selections throughout",
    "final_from_start": "Final policy from start",
    "evolving": "Naturally evolving",
    "shuffled": "Temporally shuffled",
    "family_matched_random": "Family-matched replacement views",
    "random_views": "Random views (fresh recognizer)",
    "random_warmup_10": "Random first 10, then evolving",
    "random_warmup_20": "Random first 20, then evolving",
    "random_warmup_30": "Random first 30, then evolving",
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--selection_dir", required=True)
    parser.add_argument("--protocol", required=True, choices=PROTOCOLS)
    parser.add_argument("--policy_epoch", type=int, default=None,
                        help="Required by --protocol frozen.")
    parser.add_argument(
        "--random_warmup_epochs", type=int, default=None,
        help="Required by --protocol random_then_evolving.")
    parser.add_argument("--final_policy_epoch", type=int, default=None,
                        help="Default: latest epoch in selection JSON.")
    parser.add_argument("--shuffle_seed", type=int, default=1729)
    parser.add_argument("--selected_view_types", default="01234")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train_split", default="test",
                        help="Must contain the instance IDs in selection JSON.")
    parser.add_argument("--test_split", default="test-down",
                        help="Disjoint split used for common-input evaluation.")
    parser.add_argument("--train_instances_per_class", type=int, default=0,
                        help="0 uses every instance represented in selection JSON.")
    parser.add_argument("--test_instances_per_class", type=int, default=5)
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--test_num_views", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--arch", default="auto", choices=ARCHITECTURE_CHOICES)
    parser.add_argument("--aggregation", default="max", choices=["mean", "max"])
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--base_lr_ratio", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_view_seed", type=int, default=2027)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--non_deterministic", action="store_true",
                        help="Allow nondeterministic cuDNN kernels.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shared_init", default=None,
                        help="Shared state file. Created atomically if absent, "
                             "then loaded strictly for every condition.")
    parser.add_argument("--initialize_only", action="store_true",
                        help="Create/validate --shared_init and exit.")
    parser.add_argument("--no_save_model", action="store_true")
    args = parser.parse_args()
    if args.protocol == "frozen" and args.policy_epoch is None:
        parser.error("--protocol frozen requires --policy_epoch")
    if (args.protocol == "random_then_evolving"
            and args.random_warmup_epochs is None):
        parser.error(
            "--protocol random_then_evolving requires "
            "--random_warmup_epochs")
    if args.num_views < 1 or args.test_num_views < 1:
        parser.error("--num_views and --test_num_views must be positive")
    if args.epochs < 1 or args.eval_every < 1:
        parser.error("--epochs and --eval_every must be positive")
    if args.initialize_only and not args.shared_init:
        parser.error("--initialize_only requires --shared_init")
    return args


def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def fixed_viewset_hash(dataset):
    digest = hashlib.sha256()
    for item in dataset.instances:
        digest.update(str(item["class_idx"]).encode("utf-8"))
        digest.update(item["instance_id"].encode("utf-8"))
        for filename in item["filenames"]:
            digest.update(filename.encode("utf-8"))
    return digest.hexdigest()


def load_or_create_shared_init(model, path, metadata):
    if path is None:
        return state_hash(model.state_dict()), "process_seed_initialization"
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = f"{path}.lock"
    with FileLock(lock_path):
        if not os.path.exists(path):
            temporary = f"{path}.tmp.{os.getpid()}"
            torch.save({
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "metadata": metadata,
            }, temporary)
            os.replace(temporary, path)
            source = "created_shared_init"
        else:
            source = "loaded_shared_init"
        payload = torch.load(path, map_location="cpu")
        state = payload.get("state_dict", payload)
        model.load_state_dict(state, strict=True)
    return state_hash(model.state_dict()), source


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def evaluate(model, loader, device, save_arrays=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_margin = 0.0
    total = 0
    features, logits_all, labels, instance_ids = [], [], [], []
    for images, targets, keep_cams, metadata in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        keep_cams = keep_cams.to(device, non_blocking=True)
        per_view, _ = model.get_feat(images, None, 1)
        pooled = aggregate_feat(per_view, keep_cams, model.aggregation)
        logits = model.get_output(pooled)
        batch_size = targets.shape[0]
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        top_two = logits.topk(2, dim=1).values
        total_margin += (top_two[:, 0] - top_two[:, 1]).sum().item()
        total += batch_size
        if save_arrays:
            features.append(pooled.flatten(1).cpu().numpy())
            logits_all.append(logits.cpu().numpy())
            labels.append(targets.cpu().numpy())
            instance_ids.extend(list(metadata["instance_id"]))
    result = {
        "loss": total_loss / total,
        "accuracy_pct": 100.0 * total_correct / total,
        "prediction_margin": total_margin / total,
        "n_test_instances": total,
    }
    if save_arrays:
        arrays = {
            "features": np.concatenate(features, axis=0),
            "logits": np.concatenate(logits_all, axis=0),
            "labels": np.concatenate(labels, axis=0),
            "instance_ids": np.asarray(instance_ids),
        }
        return result, arrays
    return result, None


def class_silhouette(features, labels):
    if len(np.unique(labels)) < 2 or len(labels) <= len(np.unique(labels)):
        return float("nan")
    return float(silhouette_score(
        normalize(features, axis=1), labels, metric="cosine"))


def condition_label(args):
    if args.protocol == "frozen":
        return f"frozen_{args.policy_epoch}"
    if args.protocol == "final":
        return "final_from_start"
    if args.protocol == "family_matched_random":
        return "family_matched_random"
    if args.protocol == "random":
        return "random_views"
    if args.protocol == "random_then_evolving":
        return f"random_warmup_{args.random_warmup_epochs}"
    return args.protocol


def save_curve(metrics, output_dir, title):
    frame = pd.DataFrame(metrics)
    evaluated = frame[frame["phase"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].plot(evaluated["epoch"], evaluated["accuracy_pct"], marker="o")
    axes[0].set(xlabel="Recognition epoch", ylabel="Held-out accuracy (%)")
    axes[1].plot(evaluated["epoch"], evaluated["prediction_margin"], marker="o")
    axes[1].set(xlabel="Recognition epoch", ylabel="Mean prediction margin")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curve.png"), dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.arch = resolve_architecture(args.arch, args.selection_dir)
    set_seed(args.seed, not args.non_deterministic)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    classnames = Downstream_ModelNet40.classnames

    train_set = PolicyReplayModelNet40(
        args.data_root,
        classnames,
        args.selection_dir,
        num_views=args.num_views,
        recognition_epochs=args.epochs,
        protocol=args.protocol,
        policy_epoch=args.policy_epoch,
        final_policy_epoch=args.final_policy_epoch,
        shuffle_seed=args.shuffle_seed,
        seed=args.seed,
        split=args.train_split,
        per_cls_instances=args.train_instances_per_class,
        selected_view_types=args.selected_view_types,
        random_warmup_epochs=args.random_warmup_epochs,
    )
    model = MVCNN(
        train_set, args.arch, args.aggregation, dataset_name="rgb").to(device)
    init_hash, init_source = load_or_create_shared_init(
        model,
        args.shared_init,
        {
            "arch": args.arch,
            "aggregation": args.aggregation,
            "num_views": args.num_views,
            "seed": args.seed,
        },
    )
    if args.initialize_only:
        print(f"Shared initialization: {args.shared_init}")
        print(f"SHA256: {init_hash}")
        return

    test_set = FixedRandomViewModelNet40(
        args.data_root,
        classnames,
        num_views=args.test_num_views,
        split=args.test_split,
        per_cls_instances=args.test_instances_per_class,
        seed=args.eval_view_seed,
    )
    eval_input_hash = fixed_viewset_hash(test_set)
    train_generator = torch.Generator().manual_seed(args.seed + 1000)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_seed,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_seed,
    )
    parameters = [
        {
            "params": [parameter for name, parameter in model.named_parameters()
                       if "base" in name and parameter.requires_grad],
            "lr": args.lr * args.base_lr_ratio,
        },
        {
            "params": [parameter for name, parameter in model.named_parameters()
                       if "base" not in name and "select" not in name
                       and parameter.requires_grad],
            "lr": args.lr,
        },
    ]
    optimizer = torch.optim.Adam(
        parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_updates = args.epochs * len(train_loader)
    update_index = 0

    def lr_factor(_):
        progress = min(update_index / max(total_updates, 1), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    metrics = []
    initial_result, initial_arrays = evaluate(
        model, test_loader, device, save_arrays=True)
    metrics.append({
        **initial_result,
        "epoch": 0,
        "phase": "test",
        "source_policy_epoch": None,
        "train_loss": None,
    })
    np.savez_compressed(
        os.path.join(output_dir, "initial_test_features.npz"), **initial_arrays)

    for epoch in range(1, args.epochs + 1):
        train_set.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for images, targets, keep_cams, _ in tqdm(
                train_loader, desc=f"train e{epoch}", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            keep_cams = keep_cams.to(device, non_blocking=True)
            per_view, _ = model.get_feat(images, None, 1)
            pooled = aggregate_feat(per_view, keep_cams, model.aggregation)
            logits = model.get_output(pooled)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_index += 1
            scheduler.step()
            batch_size = targets.shape[0]
            loss_sum += loss.item() * batch_size
            sample_count += batch_size

        should_evaluate = (
            epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs)
        if should_evaluate:
            result, _ = evaluate(model, test_loader, device)
            recorded_source_epoch = train_set.source_policy_epoch
            metrics.append({
                **result,
                "epoch": epoch,
                "phase": "test",
                "source_policy_epoch": recorded_source_epoch,
                "train_loss": loss_sum / sample_count,
            })
            source_description = (
                "random views" if train_set.source_policy_epoch is None
                else f"policy_epoch={train_set.source_policy_epoch:03d}"
            )
            print(
                f"epoch={epoch:03d} {source_description} "
                f"train_loss={loss_sum/sample_count:.4f} "
                f"test_acc={result['accuracy_pct']:.2f}%")

    final_result, final_arrays = evaluate(
        model, test_loader, device, save_arrays=True)
    final_silhouette = class_silhouette(
        final_arrays["features"], final_arrays["labels"])
    final_path = os.path.join(output_dir, "final_test_features.npz")
    np.savez_compressed(final_path, **final_arrays)
    if not args.no_save_model:
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))

    label = condition_label(args)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.insert(0, "condition", label)
    metrics_frame.insert(1, "protocol", args.protocol)
    metrics_frame.insert(2, "seed", args.seed)
    metrics_frame.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    schedule = {
        str(epoch): ("random" if source is None else source)
        for epoch, source in train_set.policy_schedule.items()
    }
    with open(os.path.join(output_dir, "policy_epoch_schedule.json"), "w") as handle:
        json.dump(schedule, handle, indent=2)
    summary = {
        "condition": label,
        "protocol": args.protocol,
        "comparison_role": "matched_policy_replay",
        "policy_epoch": args.policy_epoch,
        "random_warmup_epochs": args.random_warmup_epochs,
        "seed": args.seed,
        "arch": args.arch,
        "selection_dir": os.path.abspath(args.selection_dir),
        "selection_files": [os.path.abspath(path) for path in train_set.selection_files],
        "shared_init": os.path.abspath(args.shared_init) if args.shared_init else None,
        "initialization_sha256": init_hash,
        "initialization_source": init_source,
        "num_train_instances": len(train_set),
        "num_test_instances": len(test_set),
        "num_views_train": args.num_views,
        "num_views_test": args.test_num_views,
        "num_optimizer_updates": update_index,
        "expected_optimizer_updates": total_updates,
        "evaluation_input_sha256": eval_input_hash,
        "fallback_instance_epoch_count": len(train_set.fallback_instances),
        "final_accuracy_pct": final_result["accuracy_pct"],
        "final_prediction_margin": final_result["prediction_margin"],
        "final_class_silhouette": final_silhouette,
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    save_curve(metrics, output_dir, CONDITION_DISPLAY_NAMES.get(label, label))
    print(f"Saved controlled replay run to {output_dir}")
    print(f"Initialization SHA256: {init_hash}")
    print(f"Optimizer updates: {update_index}")


if __name__ == "__main__":
    main()
