"""Replay an experiment's per-epoch agent selections through the FINAL trained
classifier, with optional image manipulations, and track two things over epochs:

  1) Manipulation robustness — accuracy drop (`clean − manipulated`) for three
     manipulation conditions: rotation ±10°, color jitter, both.
  2) Prediction margin — top-1 minus top-2 logit on the unperturbed selections.

Both use the SAME final classifier and the same per-epoch selection lists from
`*_selection.json`. Holding the classifier fixed isolates the *selection-quality*
signal from the *model-improvement* signal: any movement of the curves over
epochs reflects how the agent's selections evolved.

Outputs (default `<selection_dir>/temporal_test/`):
    deviation_rotate.png
    deviation_jitter.png
    deviation_rotate_jitter.png
    margin_over_time.png
    temporal_test.csv

Usage:
    python temporal_selection_test.py \\
        --selection_dir meta_logs/rgb/resnet18steps5_train_ins25_..._e100 \\
        --checkpoint logs/rgb/<final_stage2_run>/model.pth \\
        --dataset rgb --non_roll --num_train_instances 25

    # Or let the script find the checkpoint via main.py's performance.txt lookup:
    python temporal_selection_test.py \\
        --selection_dir meta_logs/rgb/.../e100 \\
        --dataset rgb --non_roll --num_train_instances 25
"""

import argparse
import json
import os
import random
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets import ModelNet40
from src.models.mvcnn import MVCNN
from src.trainer_mvcnn import aggregate_feat


CONDITIONS = ["none", "rotate", "jitter", "rotate_jitter"]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--selection_dir", required=True,
                   help="meta_logs/<rep>/<exp>/ folder containing *_selection.json.")
    p.add_argument("--checkpoint", default=None,
                   help="Final classifier .pth. If omitted, looks up via "
                        "logs/<dataset>/<arch>[_freeze<N>]_performance.txt.")
    p.add_argument("--dataset", default="rgb")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--aggregation", default="max", choices=["mean", "max"])
    p.add_argument("--freeze_epoch", type=int, default=100,
                   help="Used only for performance.txt lookup. Ignored if --checkpoint set.")
    p.add_argument("--data_root", type=str,
                   default="/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23")
    p.add_argument("--split", default="test")
    p.add_argument("--num_train_instances", type=int, default=25)
    p.add_argument("--non_roll", action="store_true")
    p.add_argument("--non_like", action="store_true")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--rotation_deg", type=float, default=10.0,
                   help="Max absolute rotation in degrees (sampled from U(-D, +D)).")
    p.add_argument("--jitter_brightness", type=float, default=0.4)
    p.add_argument("--jitter_contrast", type=float, default=0.4)
    p.add_argument("--jitter_saturation", type=float, default=0.4)
    p.add_argument("--jitter_hue", type=float, default=0.1)
    p.add_argument("--max_epochs", type=int, default=None,
                   help="If set, evaluate only the first N epochs found (smoke test).")
    p.add_argument("--per_epoch_checkpoint", action="store_true",
                   help="Use the classifier-AT-epoch-E (loaded from model_e<E>.pth) "
                        "instead of the FINAL classifier for each epoch's evaluation. "
                        "Requires training to have been run with --save_every_epoch >0. "
                        "Curves answer 'what reward signal was the agent getting at "
                        "epoch t' instead of 'how does the fixed final classifier see "
                        "past selections'.")
    p.add_argument("--output_dir", default=None,
                   help="Defaults to <selection_dir>/temporal_test/.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def resolve_selection_dir(selection_dir, dataset):
    """Auto-resolve a --selection_dir argument.

    Expected layout is `meta_logs/<dataset>/<exp_folder>/`. If the user passes
    only the exp folder name (or any path that doesn't contain *_selection.json
    files), try prepending `meta_logs/<dataset>/` before failing.
    """
    candidates = [
        selection_dir,
        os.path.join("meta_logs", dataset, selection_dir),
    ]
    for cand in candidates:
        if not os.path.isdir(cand):
            continue
        for fn in os.listdir(cand):
            if fn.endswith("_selection.json"):
                return cand
    return None


def load_selections(selection_dir):
    """Walk every *_selection.json in selection_dir and merge per epoch.

    Returns: dict[int epoch] -> dict[cls_str] -> dict[view_type] -> list[fname]
    Duplicates across multiple selection.json files are kept; deduplication
    happens at the (epoch, instance) level in build_trials_for_epoch.
    """
    merged = {}
    for fn in sorted(os.listdir(selection_dir)):
        if not fn.endswith("_selection.json"):
            continue
        with open(os.path.join(selection_dir, fn)) as f:
            d = json.load(f)
        for ep, content in d.items():
            ep_int = int(ep)
            merged.setdefault(ep_int, {})
            for cls_str, view_dict in content.items():
                merged[ep_int].setdefault(cls_str, {})
                for vt, filenames in view_dict.items():
                    merged[ep_int][cls_str].setdefault(vt, []).extend(filenames)
    return merged


def build_trials_for_epoch(epoch_sel, data_root, classnames, split):
    """For an epoch's selections, return list of {cls_idx, instance_id, paths, target}.

    Aggregates filenames across all view-types per (class, instance) and dedupes.
    """
    by_instance = {}
    for cls_str, view_dict in epoch_sel.items():
        cls_idx = int(cls_str)
        cls_name = classnames[cls_idx]
        split_dir = os.path.join(data_root, cls_name, split)
        for filenames in view_dict.values():
            for fn in filenames:
                ins_id = fn.split('_')[0]
                full_path = os.path.join(split_dir, fn)
                by_instance.setdefault((cls_idx, ins_id), set()).add(full_path)

    trials = []
    for (cls_idx, ins_id), paths in by_instance.items():
        trials.append({
            "cls_idx": cls_idx,
            "instance_id": ins_id,
            "paths": sorted(paths),
            "target": cls_idx,
        })
    return trials


def find_stage2_logdirs(selection_dir, dataset):
    """Find logs/<dataset>/<exp_basename>_<timestamp>/ directories that match
    the given selection_dir.

    `meta_logs/<dataset>/<exp_basename>/` (no timestamp) corresponds to
    `logs/<dataset>/<exp_basename>_<timestamp>/` (with timestamp); main.py
    writes both. Returns a list of matching logdirs, most recent first.
    """
    exp_basename = os.path.basename(os.path.normpath(selection_dir))
    logs_root = os.path.join("logs", dataset)
    if not os.path.isdir(logs_root):
        return []
    candidates = []
    for d in sorted(os.listdir(logs_root)):
        full = os.path.join(logs_root, d)
        if not os.path.isdir(full):
            continue
        if d.startswith(exp_basename + "_") and os.path.exists(os.path.join(full, "model.pth")):
            candidates.append(full)
    # Most recent first by directory mtime
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates


def locate_final_checkpoint(args):
    """Locate the final stage-2 classifier checkpoint.

    Order:
      1. Explicit --checkpoint.
      2. logs/<dataset>/<exp_basename>_<timestamp>/model.pth that matches the
         selection_dir basename (most-recent timestamp).
      3. Hard error with the exact paths searched.
    """
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"--checkpoint not found: {args.checkpoint}")
        return args.checkpoint

    logdirs = find_stage2_logdirs(args.selection_dir, args.dataset)
    if logdirs:
        chosen = logdirs[0]
        if len(logdirs) > 1:
            print(f"Found {len(logdirs)} stage-2 logdirs matching this experiment; "
                  f"using most-recent:\n  {chosen}")
            print("  Other matches (override with --checkpoint to use a different run):")
            for d in logdirs[1:]:
                print(f"    {d}")
        else:
            print(f"Auto-located stage-2 checkpoint: {chosen}/model.pth")
        return os.path.join(chosen, "model.pth")

    exp_basename = os.path.basename(os.path.normpath(args.selection_dir))
    logs_root = os.path.join("logs", args.dataset)

    # Build a diagnostic listing so the user can see what IS in logs/<rep>/.
    if os.path.isdir(logs_root):
        entries = sorted(os.listdir(logs_root))
        listing_lines = []
        for d in entries:
            full = os.path.join(logs_root, d)
            if not os.path.isdir(full):
                continue
            has_model = os.path.exists(os.path.join(full, "model.pth"))
            marker = "  [model.pth ✓]" if has_model else "  [no model.pth]"
            # Highlight near-matches.
            close = d.startswith(exp_basename[:40])
            prefix = "  ➜ " if close else "    "
            listing_lines.append(f"{prefix}{d}{marker}")
        listing = "\n".join(listing_lines) if listing_lines else "  (empty)"
    else:
        listing = f"  ({logs_root}/ does not exist)"

    raise FileNotFoundError(
        f"Cannot find a stage-2 checkpoint for this experiment.\n"
        f"  Searched pattern: logs/{args.dataset}/{exp_basename}_<timestamp>/model.pth\n"
        f"\n"
        f"Note: logs/<dataset>/<arch>_performance.txt points to the STAGE-1\n"
        f"backbone, not the final stage-2 classifier you need here.\n"
        f"\n"
        f"Contents of {logs_root}/:\n"
        f"{listing}\n"
        f"\n"
        f"To proceed:\n"
        f"  - If one of the ➜ marked dirs above has model.pth, pass:\n"
        f"      --checkpoint logs/{args.dataset}/<matching_dir>/model.pth\n"
        f"  - If the checkpoint lives elsewhere (e.g. a different machine):\n"
        f"      --checkpoint <full path to model.pth>\n"
        f"  - If no stage-2 training has finished for this experiment yet,\n"
        f"    run it first (`main.py --steps 5 ...` matching the meta_logs config)."
    )


# ---------------------------------------------------------------------------
# Manipulations
# ---------------------------------------------------------------------------

def apply_condition(img_pil, condition, view_idx, epoch, args):
    """Apply the named manipulation deterministically per (epoch, view_idx)."""
    if condition == "none":
        return img_pil

    seed_str = f"{epoch}|{view_idx}"
    rng = random.Random(seed_str)

    if condition == "rotate":
        angle = rng.uniform(-args.rotation_deg, args.rotation_deg)
        return TF.rotate(img_pil, angle, fill=255)

    if condition in ("jitter", "rotate_jitter"):
        b = rng.uniform(max(0, 1 - args.jitter_brightness), 1 + args.jitter_brightness)
        c = rng.uniform(max(0, 1 - args.jitter_contrast), 1 + args.jitter_contrast)
        s = rng.uniform(max(0, 1 - args.jitter_saturation), 1 + args.jitter_saturation)
        h = rng.uniform(-args.jitter_hue, args.jitter_hue)
        if condition == "rotate_jitter":
            angle = rng.uniform(-args.rotation_deg, args.rotation_deg)
            img_pil = TF.rotate(img_pil, angle, fill=255)
        img_pil = TF.adjust_brightness(img_pil, b)
        img_pil = TF.adjust_contrast(img_pil, c)
        img_pil = TF.adjust_saturation(img_pil, s)
        img_pil = TF.adjust_hue(img_pil, h)
        return img_pil

    raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_trial(model, paths, condition, epoch, transform, device, args):
    """Run one trial: list of paths -> (margin, correct)."""
    imgs_pil = [Image.open(p).convert("RGB") for p in paths]
    imgs_pil = [apply_condition(im, condition, i, epoch, args)
                for i, im in enumerate(imgs_pil)]
    imgs = torch.stack([transform(im) for im in imgs_pil]).unsqueeze(0).to(device)
    keep_cams = torch.ones(1, imgs.shape[1], dtype=torch.bool, device=device)

    feat, _ = model.get_feat(imgs, None, 1)
    overall_feat = aggregate_feat(feat, keep_cams, model.aggregation)
    logits = model.get_output(overall_feat)[0]                # [num_class]
    sorted_logits, _ = torch.sort(logits, descending=True)
    margin = (sorted_logits[0] - sorted_logits[1]).item()
    pred = int(torch.argmax(logits).item())
    return margin, pred


def evaluate_epoch(model, trials, condition, epoch, transform, device, args):
    margins, corrects = [], []
    for trial in trials:
        try:
            margin, pred = forward_trial(model, trial["paths"], condition, epoch,
                                          transform, device, args)
        except Exception as e:
            print(f"  e={epoch} {condition} skip {trial['instance_id']}: {e}")
            continue
        margins.append(margin)
        corrects.append(pred == trial["target"])
    if not margins:
        return None
    return {
        "n_trials": len(margins),
        "accuracy": float(np.mean(corrects)),
        "mean_margin": float(np.mean(margins)),
        "std_margin": float(np.std(margins, ddof=1)) if len(margins) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_deviations(df, output_dir, per_epoch_mode=False):
    pivot = df.pivot_table(index="epoch", columns="condition", values="accuracy")
    epochs = pivot.index.to_numpy()
    label_map = {
        "rotate": "Rotation ±10°",
        "jitter": "Color jitter",
        "rotate_jitter": "Rotation + color jitter",
    }
    color_map = {"rotate": "#d62728", "jitter": "#1f77b4", "rotate_jitter": "#9467bd"}
    for cond, label in label_map.items():
        if cond not in pivot.columns or "none" not in pivot.columns:
            continue
        dev = (pivot["none"] - pivot[cond]).to_numpy()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, dev * 100.0, "o-", color=color_map[cond], lw=1.5, ms=4)
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Training epoch (selections from)")
        ax.set_ylabel("Accuracy drop (%) — clean minus perturbed")
        regime = "classifier-at-epoch-t" if per_epoch_mode else "final classifier held fixed"
        ax.set_title(f"Manipulation robustness over epochs: {label}\n({regime})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, f"deviation_{cond}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


def plot_margin(df, output_dir, per_epoch_mode=False):
    sub = df[df["condition"] == "none"].sort_values("epoch")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(sub["epoch"], sub["mean_margin"], yerr=sub["std_margin"],
                fmt="o-", color="#2ca02c", capsize=3, lw=1.5, ms=4,
                ecolor="#a6d854", alpha=0.95)
    ax.set_xlabel("Training epoch (selections from)")
    ax.set_ylabel("Mean prediction margin (top-1 − top-2 logit)")
    regime = "classifier-at-epoch-t" if per_epoch_mode else "final classifier held fixed"
    ax.set_title(f"Prediction-margin stability on agent-selected views\n({regime})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "margin_over_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve a possibly bare exp-folder name to the full meta_logs/<rep>/<exp>/ path.
    resolved = resolve_selection_dir(args.selection_dir, args.dataset)
    if resolved is None:
        raise SystemExit(
            f"No *_selection.json files found.\n"
            f"  Tried: {args.selection_dir}\n"
            f"  And:   meta_logs/{args.dataset}/{args.selection_dir}\n"
            f"\n"
            f"Expected layout: meta_logs/<rep>/<exp_folder>/<timestamp>_selection.json\n"
            f"Pass the FULL relative path, e.g.:\n"
            f"  --selection_dir meta_logs/{args.dataset}/<exp_folder>"
        )
    if resolved != args.selection_dir:
        print(f"Auto-resolved --selection_dir → {resolved}")
    args.selection_dir = resolved

    output_dir = args.output_dir or os.path.join(args.selection_dir, "temporal_test")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")

    # --- Load selections ---
    print(f"Loading selections from {args.selection_dir}")
    all_sel = load_selections(args.selection_dir)
    if not all_sel:
        raise SystemExit("No *_selection.json files found in resolved selection_dir.")
    epochs_all = sorted(all_sel.keys())
    if args.max_epochs is not None:
        epochs_all = epochs_all[:args.max_epochs]
    print(f"  {len(epochs_all)} epochs (range {epochs_all[0]}–{epochs_all[-1]})")

    # --- Build dataset (just for MVCNN init + class-name lookup) ---
    num_cam = 4096 if not args.non_roll else (192 if args.non_like else 114)
    test_set = ModelNet40(args.data_root, args.dataset, num_cam, split=args.split,
                          per_cls_instances=5, non_roll=args.non_roll,
                          non_like=args.non_like)
    classnames = test_set.classnames

    # --- Locate the matching stage-2 logdir (always needed: final checkpoint
    # in default mode, per-epoch checkpoints in --per_epoch_checkpoint mode).
    final_ckpt = locate_final_checkpoint(args)
    logdir_for_checkpoints = os.path.dirname(final_ckpt)
    print(f"Logdir: {logdir_for_checkpoints}")

    model = MVCNN(test_set, args.arch, args.aggregation, args.dataset).to(device)

    def load_state(path):
        state = torch.load(path, map_location=device)
        state = {k: v for k, v in state.items() if k in model.state_dict()}
        model.load_state_dict(state, strict=False)
        model.eval()

    if not args.per_epoch_checkpoint:
        print(f"Loading final checkpoint (held fixed across epochs): {final_ckpt}")
        load_state(final_ckpt)
    else:
        # Sanity check: must have at least one model_e<E>.pth in the logdir.
        per_epoch_files = sorted(
            f for f in os.listdir(logdir_for_checkpoints)
            if f.startswith("model_e") and f.endswith(".pth")
        )
        if not per_epoch_files:
            raise SystemExit(
                f"--per_epoch_checkpoint set, but no model_e<E>.pth files exist in\n"
                f"  {logdir_for_checkpoints}\n"
                f"Re-run training with --save_every_epoch <N> to write them, e.g.\n"
                f"  python main.py ... --steps 5 --save_every_epoch 1"
            )
        print(f"--per_epoch_checkpoint mode: found {len(per_epoch_files)} epoch "
              f"snapshots under {logdir_for_checkpoints}")

    # --- Image transform (matches downstream + main pipeline preprocessing) ---
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # --- Evaluate ---
    rows = []
    skipped_no_epoch_ckpt = []
    checkpoint_protocol = (
        "classifier_at_epoch_t" if args.per_epoch_checkpoint
        else "final_classifier_fixed"
    )
    for epoch in tqdm(epochs_all, desc="epochs"):
        active_ckpt = final_ckpt
        # Per-epoch model swap if requested
        if args.per_epoch_checkpoint:
            epoch_ckpt = os.path.join(logdir_for_checkpoints, f"model_e{epoch}.pth")
            if not os.path.exists(epoch_ckpt):
                skipped_no_epoch_ckpt.append(epoch)
                continue
            load_state(epoch_ckpt)
            active_ckpt = epoch_ckpt

        trials = build_trials_for_epoch(all_sel[epoch], args.data_root,
                                         classnames, args.split)
        if not trials:
            continue
        for cond in CONDITIONS:
            res = evaluate_epoch(model, trials, cond, epoch, transform, device, args)
            if res is None:
                continue
            rows.append({
                "epoch": epoch,
                "condition": cond,
                "checkpoint_protocol": checkpoint_protocol,
                "checkpoint_path": active_ckpt,
                **res,
            })

    if skipped_no_epoch_ckpt:
        print(f"Skipped {len(skipped_no_epoch_ckpt)} epoch(s) — no matching "
              f"model_e<E>.pth: {skipped_no_epoch_ckpt}")

    if not rows:
        raise SystemExit("No results produced.")
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "temporal_test.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} rows)")

    # --- Plots ---
    plot_deviations(df, output_dir, per_epoch_mode=args.per_epoch_checkpoint)
    plot_margin(df, output_dir, per_epoch_mode=args.per_epoch_checkpoint)
    print("\nDone.")


if __name__ == "__main__":
    main()
