#!/usr/bin/env python3
"""Zero-shot MOCHI oddity test using MVSelect/MVCNN features.

This is the MVSelect analogue of the pairwise-feature oddity rule:

    1. extract one feature vector for each image in a MOCHI trial
    2. compute pairwise cosine similarities between all images
    3. assign each image its mean similarity to the other images
    4. predict the oddity as the image with the lowest mean similarity

No MOCHI labels are used for training or fitting a probe. The classifier head
is ignored; only the trained image backbone feature is used.
"""

import argparse
import os
import random
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.mvcnn import MVCNN
from src.models.architectures import ARCHITECTURE_CHOICES, resolve_architecture


MOCHI_DATASET = "tzler/MOCHI"
KNOWN_CHANCE = {
    "barense": 1.0 / 4.0,
    "shapegen": 1.0 / 3.0,
    "shapenet": 1.0 / 3.0,
}


class _DummyDataset:
    def __init__(self, num_class=32, num_cam=114):
        self.num_class = num_class
        self.num_cam = num_cam


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--exp", action="append", default=[],
                   help="Experiment folder/logdir/checkpoint to evaluate. "
                        "Accepts PATH or PATH:LABEL. Bare experiment names "
                        "resolve to logs/<dataset>/<name>_<timestamp>/. "
                        "Repeatable.")
    p.add_argument("--checkpoint", action="append", default=[],
                   help="Direct checkpoint path, optionally PATH:LABEL. "
                        "Repeatable. Usually model.pth or model_e<E>.pth.")
    p.add_argument("--dataset", default="rgb",
                   help="MVSelect representation name used for log resolution "
                        "and MVCNN input-channel setup.")
    p.add_argument("--arch", default="auto", choices=ARCHITECTURE_CHOICES)
    p.add_argument("--aggregation", default="max")
    p.add_argument("--num_cam", type=int, default=114,
                   help="Number of cameras used by the checkpoint's selector "
                        "module. 114 for non-roll experiments.")
    p.add_argument("--num_class", type=int, default=32)
    p.add_argument("--down", type=int, default=1)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--n_trials", type=int, default=None,
                   help="Optional smoke-test cap. Samples trials after dataset "
                        "filtering with --random_seed.")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--per_epoch_checkpoint", action="store_true",
                   help="Use model_e<E>.pth snapshots from each resolved logdir. "
                        "Requires training with --save_every_epoch > 0.")
    p.add_argument("--epochs", default=None,
                   help="Optional epoch filter for --per_epoch_checkpoint, e.g. "
                        "'1-100', '10,20,30', or '1-20,30,40'.")
    p.add_argument("--include_unknown_datasets", action="store_true",
                   help="By default, keep only MOCHI datasets with known chance "
                        "levels used by the manuscript normalization: barense, "
                        "shapegen, shapenet.")
    return p.parse_args()


def parse_path_label(spec):
    path, label = spec, None
    if ":" in spec:
        prefix, _, suffix = spec.rpartition(":")
        if prefix and suffix and "/" not in suffix:
            path, label = prefix, suffix
    return path, label


def parse_epoch_filter(s):
    if not s:
        return None
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _checkpoint_epoch(path):
    m = re.search(r"model_e(\d+)\.pth$", os.path.basename(path))
    if m:
        return int(m.group(1))
    return None


def _has_final_checkpoint(d):
    return os.path.exists(os.path.join(d, "model.pth"))


def resolve_logdir_from_exp(path, dataset):
    """Resolve a checkpoint/log/meta experiment spec to a concrete logdir."""
    candidates = []

    if os.path.isfile(path) and path.endswith(".pth"):
        return os.path.dirname(os.path.abspath(path))

    if os.path.isdir(path):
        if _has_final_checkpoint(path) or glob_model_epochs(path):
            return os.path.abspath(path)
        basename = os.path.basename(os.path.normpath(path))
    else:
        basename = os.path.basename(os.path.normpath(path))

    logs_dir = ROOT_DIR / "logs" / dataset
    if logs_dir.is_dir():
        for d in logs_dir.iterdir():
            if d.is_dir() and d.name.startswith(basename + "_"):
                if _has_final_checkpoint(str(d)) or glob_model_epochs(str(d)):
                    candidates.append(str(d))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot resolve experiment '{path}' to a logdir with model.pth "
            f"or model_e<E>.pth. Searched logs/{dataset}/{basename}_<timestamp>/"
        )
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def glob_model_epochs(logdir):
    if not os.path.isdir(logdir):
        return []
    paths = []
    for name in os.listdir(logdir):
        if re.match(r"model_e\d+\.pth$", name):
            paths.append(os.path.join(logdir, name))
    return sorted(paths, key=lambda p: _checkpoint_epoch(p) or -1)


def resolve_checkpoint_specs(args):
    epoch_filter = parse_epoch_filter(args.epochs)
    specs = []

    for spec in args.checkpoint:
        path, label = parse_path_label(spec)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        label = label or Path(path).stem
        epoch = _checkpoint_epoch(path)
        specs.append({
            "experiment": label,
            "checkpoint": os.path.abspath(path),
            "epoch": epoch if epoch is not None else -1,
            "epoch_label": str(epoch) if epoch is not None else "final",
        })

    for spec in args.exp:
        path, label = parse_path_label(spec)
        logdir = resolve_logdir_from_exp(path, args.dataset)
        label = label or os.path.basename(os.path.normpath(logdir))

        if args.per_epoch_checkpoint:
            ckpts = glob_model_epochs(logdir)
            if epoch_filter is not None:
                ckpts = [p for p in ckpts if _checkpoint_epoch(p) in epoch_filter]
            if not ckpts:
                raise FileNotFoundError(
                    f"--per_epoch_checkpoint requested for {label}, but no "
                    f"matching model_e<E>.pth exists in {logdir}. Re-train "
                    f"with --save_every_epoch <N> to evaluate model-at-epoch."
                )
            for ckpt in ckpts:
                epoch = _checkpoint_epoch(ckpt)
                specs.append({
                    "experiment": label,
                    "checkpoint": os.path.abspath(ckpt),
                    "epoch": epoch,
                    "epoch_label": str(epoch),
                })
        else:
            ckpt = os.path.join(logdir, "model.pth")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(
                    f"No final model.pth for {label} at {ckpt}."
                )
            specs.append({
                "experiment": label,
                "checkpoint": os.path.abspath(ckpt),
                "epoch": -1,
                "epoch_label": "final",
            })

    if not specs:
        raise SystemExit("No --exp or --checkpoint given.")
    return specs


def load_mvcNN_feature_model(args, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    arch = resolve_architecture(args.arch, ckpt_path, state)
    dummy = _DummyDataset(num_class=args.num_class, num_cam=args.num_cam)
    model = MVCNN(dummy, arch, args.aggregation, args.dataset).to(device)
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape):
            filtered[k] = v
        else:
            skipped.append(k)
    model_state.update(filtered)
    model.load_state_dict(model_state)
    model.eval()
    return model, len(filtered), skipped


def pil_image(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    return Image.fromarray(np.array(x)).convert("RGB")


def build_transform():
    return T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def extract_trial_features(model, images, transform, device, down):
    tensors = [transform(pil_image(im)) for im in images]
    imgs = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
    feat, _ = model.get_feat(imgs, None, down)
    feat = feat[0].flatten(1).float()
    return F.normalize(feat, p=2, dim=1).cpu().numpy()


def predict_oddity_from_features(features):
    sim = features @ features.T
    n = sim.shape[0]
    scores = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        scores.append(float(np.mean(sim[i, others])))
    pred = int(np.argmin(scores))
    sorted_scores = sorted(scores)
    margin = float(sorted_scores[1] - sorted_scores[0]) if len(scores) > 1 else 0.0
    return pred, scores, margin


def filter_trials(dataset, include_unknown, n_trials, seed):
    indices = []
    for i, t in enumerate(dataset):
        ds = t["dataset"]
        # The manuscript Fig. 3 normalization covers these three datasets.
        if not include_unknown and ds not in KNOWN_CHANCE:
            continue
        indices.append(i)

    if n_trials is not None and n_trials < len(indices):
        rng = random.Random(seed)
        indices = rng.sample(indices, n_trials)
    return indices


def summarize(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()

    summaries = []
    group_cols = ["experiment", "epoch", "epoch_label", "checkpoint"]
    for key, block in df.groupby(group_cols, dropna=False):
        exp, epoch, epoch_label, checkpoint = key
        cond_rows = []
        for (dataset, condition), cond in block.groupby(["dataset", "condition"]):
            acc = float(cond["correct"].mean())
            chance = KNOWN_CHANCE.get(dataset)
            norm = np.nan if chance is None else (acc - chance) / (1.0 - chance)
            cond_rows.append(norm)
        summaries.append({
            "experiment": exp,
            "epoch": epoch,
            "epoch_label": epoch_label,
            "checkpoint": checkpoint,
            "n_trials": int(len(block)),
            "n_conditions": int(block.groupby(["dataset", "condition"]).ngroups),
            "trial_accuracy": float(block["correct"].mean()),
            "condition_macro_norm_accuracy": float(np.nanmean(cond_rows)),
            "mean_margin": float(block["margin"].mean()),
        })
    return df, pd.DataFrame(summaries).sort_values(["experiment", "epoch"])


def plot_summary(summary, output_dir):
    if summary.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    has_epochs = (summary["epoch"] >= 0).any()

    if has_epochs:
        pivot = summary.pivot_table(
            index="experiment",
            columns="epoch",
            values="condition_macro_norm_accuracy",
        )
        fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(pivot.columns) + 2),
                                        max(3, 0.45 * len(pivot.index) + 1)))
        finite = pivot.to_numpy()[np.isfinite(pivot.to_numpy())]
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        epochs = list(pivot.columns)
        n_ticks = min(len(epochs), 10)
        tick_idx = np.linspace(0, len(epochs) - 1, n_ticks).astype(int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(int(epochs[i])) for i in tick_idx], fontsize=8)
        ax.set_xlabel("Checkpoint epoch")
        ax.set_title("MOCHI zero-shot oddity accuracy from MVSelect features")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Condition-macro normalized accuracy")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "mochi_feature_oddity_heatmap.png"),
                    dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for exp, block in summary.groupby("experiment"):
            sub = block[block["epoch"] >= 0].sort_values("epoch")
            ax.plot(sub["epoch"], sub["condition_macro_norm_accuracy"],
                    "o-", lw=1.5, ms=4, label=exp)
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Checkpoint epoch")
        ax.set_ylabel("Condition-macro normalized accuracy")
        ax.set_title("MOCHI zero-shot oddity accuracy from MVSelect features")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "mochi_feature_oddity_over_epochs.png"),
                    dpi=150)
        plt.close(fig)
    else:
        sub = summary.sort_values("condition_macro_norm_accuracy", ascending=False)
        fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(sub)), 4.5))
        ax.bar(sub["experiment"], sub["condition_macro_norm_accuracy"],
               color="#4C78A8", edgecolor="black", linewidth=0.8)
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_ylabel("Condition-macro normalized accuracy")
        ax.set_title("MOCHI zero-shot oddity accuracy from MVSelect features")
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "mochi_feature_oddity_bar.png"),
                    dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if "cuda" in device:
        torch.cuda.set_device(args.gpu_id)

    output_dir = args.output_dir or str(ROOT_DIR / "compare" / "mochi_feature_oddity")
    os.makedirs(output_dir, exist_ok=True)

    specs = resolve_checkpoint_specs(args)
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print("Checkpoints:")
    for s in specs:
        print(f"  - {s['experiment']} epoch={s['epoch_label']} -> {s['checkpoint']}")

    print("\nLoading MOCHI...")
    dataset = load_dataset(MOCHI_DATASET)["train"]
    indices = filter_trials(dataset, args.include_unknown_datasets,
                            args.n_trials, args.random_seed)
    print(f"Using {len(indices)} trials")

    transform = build_transform()
    rows = []

    for spec in specs:
        print(f"\n{'=' * 72}")
        print(f"{spec['experiment']} epoch={spec['epoch_label']}")
        print(f"{'=' * 72}")
        model, n_loaded, skipped = load_mvcNN_feature_model(args, spec["checkpoint"], device)
        print(f"Loaded {n_loaded} tensors from checkpoint; skipped {len(skipped)} shape/name mismatches")

        for trial_idx in tqdm(indices, desc=f"{spec['experiment']}:{spec['epoch_label']}"):
            trial = dataset[int(trial_idx)]
            try:
                features = extract_trial_features(model, trial["images"],
                                                  transform, device, args.down)
                pred, scores, margin = predict_oddity_from_features(features)
                rows.append({
                    "experiment": spec["experiment"],
                    "epoch": spec["epoch"],
                    "epoch_label": spec["epoch_label"],
                    "checkpoint": spec["checkpoint"],
                    "trial_idx": int(trial_idx),
                    "trial_name": trial["trial"],
                    "dataset": trial["dataset"],
                    "condition": "shapegen" if trial.get("dataset") == "shapegen"
                                 else trial.get("condition", trial["dataset"]),
                    "oddity_index": int(trial["oddity_index"]),
                    "predicted_oddity": pred,
                    "correct": pred == int(trial["oddity_index"]),
                    "margin": margin,
                    "image_scores": ";".join(f"{x:.6f}" for x in scores),
                    "n_images": len(trial["images"]),
                })
            except Exception as e:
                print(f"  Error on {trial.get('trial', trial_idx)}: {e}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trials_df, summary_df = summarize(rows)
    trials_path = os.path.join(output_dir, "mochi_feature_oddity_trials.csv")
    summary_path = os.path.join(output_dir, "mochi_feature_oddity_summary.csv")
    trials_df.to_csv(trials_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_summary(summary_df, output_dir)

    print(f"\nSaved: {trials_path}")
    print(f"Saved: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
