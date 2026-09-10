#!/usr/bin/env python3
"""Evaluate an original jointly trained MVSelect run on replay-control inputs.

The classifier and selector are not retrained here. Each saved checkpoint from
the original no-freeze run is loaded and its recognition backbone/classifier is
evaluated on the same deterministic held-out N-view inputs used by
``train_policy_replay.py``. The selector head is present in the checkpoint but
is not executed during this evaluation.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from types import SimpleNamespace

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mvselect_matplotlib"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.downstream_dataset import Downstream_ModelNet40
from src.datasets.policy_replay_dataset import FixedRandomViewModelNet40
from src.models.architectures import ARCHITECTURE_CHOICES, resolve_architecture
from src.models.mvcnn import MVCNN
from train_policy_replay import (
    class_silhouette,
    evaluate,
    fixed_viewset_hash,
    save_curve,
    set_seed,
    worker_seed,
)


DEFAULT_DATA_ROOT = (
    "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/"
    "modelnet_32_60_1_23"
)
CHECKPOINT_RE = re.compile(r"model_e(?P<epoch>\d+)\.pth$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", required=True,
                        help="Timestamped original no-freeze logs/<dataset>/ run.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset", default="rgb")
    parser.add_argument("--test_split", default="test-down")
    parser.add_argument("--test_instances_per_class", type=int, default=5)
    parser.add_argument("--test_num_views", type=int, default=5)
    parser.add_argument("--eval_view_seed", type=int, default=2027)
    parser.add_argument("--epoch_stride", type=int, default=10)
    parser.add_argument("--final_epoch", type=int, default=100)
    parser.add_argument("--epochs", nargs="+", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--arch", default="auto", choices=ARCHITECTURE_CHOICES)
    parser.add_argument("--aggregation", default="max", choices=["mean", "max"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--reference_seed", type=int, default=0,
                        help="Plot/aggregation identifier; not the unknown original seed.")
    return parser.parse_args()


def load_torch_state(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {path}")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload.items()
    }


def state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_index(logdir, final_epoch):
    if not os.path.isdir(logdir):
        raise FileNotFoundError(f"Joint-training logdir not found: {logdir}")
    checkpoints = {}
    for name in os.listdir(logdir):
        match = CHECKPOINT_RE.fullmatch(name)
        if match:
            checkpoints[int(match.group("epoch"))] = os.path.join(logdir, name)
    final_path = os.path.join(logdir, "model.pth")
    if os.path.exists(final_path):
        checkpoints[int(final_epoch)] = final_path
    if not checkpoints:
        raise FileNotFoundError(
            f"No model_e<E>.pth or model.pth checkpoints in {logdir}")
    return checkpoints


def select_epochs(checkpoints, explicit, stride, final_epoch):
    available = sorted(checkpoints)
    if explicit:
        missing = sorted(set(explicit).difference(available))
        if missing:
            raise FileNotFoundError(f"Requested checkpoint epochs missing: {missing}")
        selected = sorted(set(explicit))
    else:
        selected = [
            epoch for epoch in available
            if epoch == 1 or epoch % stride == 0 or epoch == final_epoch
        ]
    if final_epoch not in selected and final_epoch in checkpoints:
        selected.append(final_epoch)
    return sorted(set(selected))


def create_model(state, checkpoint_path, args, device):
    arch = resolve_architecture(args.arch, checkpoint_path, state)
    classifier_weight = state.get("classifier.weight")
    if classifier_weight is None:
        raise KeyError(f"Checkpoint lacks classifier.weight: {checkpoint_path}")
    cam_embedding = state.get("select_module.cam_emb")
    num_cam = int(cam_embedding.shape[0]) if cam_embedding is not None else 114
    stub = SimpleNamespace(
        num_cam=num_cam,
        num_class=int(classifier_weight.shape[0]),
    )
    model = MVCNN(stub, arch, args.aggregation, args.dataset).to(device)
    expected = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in expected and tuple(value.shape) == tuple(expected[key].shape)
    }
    required = {"classifier.weight", "classifier.bias"}
    if not required.issubset(compatible):
        raise RuntimeError(
            f"Checkpoint is incompatible with {arch}: {checkpoint_path}")
    model.load_state_dict(compatible, strict=False)
    return model, arch


def load_state_into_model(model, path):
    state = load_torch_state(path)
    expected = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in expected and tuple(value.shape) == tuple(expected[key].shape)
    }
    model.load_state_dict(compatible, strict=False)
    model.eval()
    return state


def main():
    args = parse_args()
    set_seed(args.reference_seed, True)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    checkpoints = checkpoint_index(args.logdir, args.final_epoch)
    epochs = select_epochs(
        checkpoints, args.epochs, args.epoch_stride, args.final_epoch)
    print(f"Joint no-freeze checkpoint epochs: {epochs}")

    test_set = FixedRandomViewModelNet40(
        args.data_root,
        Downstream_ModelNet40.classnames,
        num_views=args.test_num_views,
        split=args.test_split,
        per_cls_instances=args.test_instances_per_class,
        seed=args.eval_view_seed,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_seed,
    )
    eval_input_hash = fixed_viewset_hash(test_set)

    first_state = load_torch_state(checkpoints[epochs[0]])
    model, arch = create_model(
        first_state, checkpoints[epochs[0]], args, device)
    metrics = []
    final_arrays = None
    final_result = None
    final_state = None
    for epoch in epochs:
        state = load_state_into_model(model, checkpoints[epoch])
        save_arrays = epoch == args.final_epoch
        result, arrays = evaluate(
            model, test_loader, device, save_arrays=save_arrays)
        metrics.append({
            **result,
            "condition": "joint_no_freeze_reference",
            "protocol": "joint_selector_classifier_training",
            "seed": args.reference_seed,
            "epoch": epoch,
            "phase": "test",
            "source_policy_epoch": epoch,
            "train_loss": None,
        })
        print(
            f"joint epoch={epoch:03d} test_acc={result['accuracy_pct']:.2f}% "
            f"margin={result['prediction_margin']:.4f}")
        if save_arrays:
            final_arrays = arrays
            final_result = result
            final_state = state

    if final_arrays is None:
        final_epoch = epochs[-1]
        final_state = load_state_into_model(model, checkpoints[final_epoch])
        final_result, final_arrays = evaluate(
            model, test_loader, device, save_arrays=True)
        args.final_epoch = final_epoch

    pd.DataFrame(metrics).to_csv(
        os.path.join(output_dir, "metrics.csv"), index=False)
    np.savez_compressed(
        os.path.join(output_dir, "final_test_features.npz"), **final_arrays)
    summary = {
        "condition": "joint_no_freeze_reference",
        "protocol": "joint_selector_classifier_training",
        "comparison_role": "external_joint_reference",
        "seed": args.reference_seed,
        "source_training_seed": "unknown",
        "arch": arch,
        "joint_logdir": os.path.abspath(args.logdir),
        "evaluated_checkpoints": {
            str(epoch): os.path.abspath(checkpoints[epoch]) for epoch in epochs
        },
        "final_checkpoint_sha256": state_hash(final_state),
        "initialization_sha256": None,
        "num_optimizer_updates": None,
        "evaluation_input_sha256": eval_input_hash,
        "num_test_instances": len(test_set),
        "num_views_test": args.test_num_views,
        "final_accuracy_pct": final_result["accuracy_pct"],
        "final_prediction_margin": final_result["prediction_margin"],
        "final_class_silhouette": class_silhouette(
            final_arrays["features"], final_arrays["labels"]),
        "final_vs_initial_cka_available": False,
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    save_curve(metrics, output_dir, "External reference: joint no-freeze training")
    print(f"Saved joint-training reference to {output_dir}")


if __name__ == "__main__":
    main()
