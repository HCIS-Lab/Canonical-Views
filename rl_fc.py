# rl_view_selector_seq.py
import os
import random
import argparse
from glob import glob
from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import copy
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader


# 0: e+00_a000_principal_planar.png.pt
# 1: e+00_a030_planar.png.pt
# 2: e+00_a060_planar.png.pt
# 3: e+00_a090_principal_planar.png.pt
# 4: e+00_a120_planar.png.pt
# 5: e+00_a150_planar.png.pt
# 6: e+00_a180_principal_planar.png.pt
# 7: e+00_a210_planar.png.pt
# 8: e+00_a240_planar.png.pt
# 9: e+00_a270_principal_planar.png.pt
# 10: e+00_a300_planar.png.pt
# 11: e+00_a330_planar.png.pt
# 12: e+30_a000.png.pt
# 13: e+30_a030.png.pt
# 14: e+30_a060.png.pt
# 15: e+30_a090.png.pt
# 16: e+30_a120.png.pt
# 17: e+30_a150.png.pt
# 18: e+30_a180.png.pt
# 19: e+30_a210.png.pt
# 20: e+30_a240.png.pt
# 21: e+30_a270.png.pt
# 22: e+30_a300.png.pt
# 23: e+30_a330.png.pt
# 24: e+60_a000.png.pt
# 25: e+60_a030.png.pt
# 26: e+60_a060.png.pt
# 27: e+60_a090.png.pt
# 28: e+60_a120.png.pt
# 29: e+60_a150.png.pt
# 30: e+60_a180.png.pt
# 31: e+60_a210.png.pt
# 32: e+60_a240.png.pt
# 33: e+60_a270.png.pt
# 34: e+60_a300.png.pt
# 35: e+60_a330.png.pt
# 36: e+90_a000_principal_planar.png.pt
# 37: e-30_a000.png.pt
# 38: e-30_a030.png.pt
# 39: e-30_a060.png.pt
# 40: e-30_a090.png.pt
# 41: e-30_a120.png.pt
# 42: e-30_a150.png.pt
# 43: e-30_a180.png.pt
# 44: e-30_a210.png.pt
# 45: e-30_a240.png.pt
# 46: e-30_a270.png.pt
# 47: e-30_a300.png.pt
# 48: e-30_a330.png.pt
# 49: e-60_a000.png.pt
# 50: e-60_a030.png.pt
# 51: e-60_a060.png.pt
# 52: e-60_a090.png.pt
# 53: e-60_a120.png.pt
# 54: e-60_a150.png.pt
# 55: e-60_a180.png.pt
# 56: e-60_a210.png.pt
# 57: e-60_a240.png.pt
# 58: e-60_a270.png.pt
# 59: e-60_a300.png.pt
# 60: e-60_a330.png.pt
# 61: e-90_a000_principal_planar.png.pt

# ---------------------------
# Utilities and data scanning
# ---------------------------

Instance = namedtuple("Instance", ["class_idx", "class_name", "instance_id", "view_paths"])

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def find_instances(features_root):
    """Scan dataset/<features_subdir>/<class>/<instance>/<view>.pt and return Instance records."""
    class_dirs = sorted([d for d in glob(os.path.join(features_root, "*")) if os.path.isdir(d)])
    instances = []
    for ci, cdir in enumerate(class_dirs):
        class_name = os.path.basename(cdir)
        inst_dirs = sorted([d for d in glob(os.path.join(cdir, "*")) if os.path.isdir(d)])
        for idir in inst_dirs:
            instance_id = os.path.basename(idir)
            vpaths = sorted(glob(os.path.join(idir, "*.pt")), key=natural_key)
            for i, name in enumerate(vpaths):
                print(str(i)+ ': ' + str(name))
            if len(vpaths) == 0:
                continue
            instances.append(Instance(ci, class_name, instance_id, vpaths))
    return instances, [os.path.basename(d) for d in class_dirs]

def split_train_test(instances, m_per_class, t_per_class=None, seed=0):
    class_to_insts = defaultdict(list)
    for inst in instances:
        class_to_insts[inst.class_idx].append(inst)
    train, test = [], []
    rng = random.Random(seed)
    for cls, insts in class_to_insts.items():
        rng.shuffle(insts)
        train += insts[:m_per_class]
        if t_per_class is None:
            test += insts[m_per_class:]
        else:
            test += insts[m_per_class:m_per_class + t_per_class]
    return train, test

def load_instance_features(instance: Instance, device):
    feats = []
    for p in instance.view_paths:
        t = torch.load(p, map_location="cpu")
        if isinstance(t, dict) and "feat" in t:
            t = t["feat"]
        t = t.float().view(-1)
        feats.append(t)
    feats = torch.stack(feats, dim=0).to(device)  # [V, D]
    return feats

# ---------------------------
# Sequential RL Selector (62-way classifier policy)
# ---------------------------

class SeqSelector(nn.Module):
    """
    A sequential policy that, given:
      - all view features for an instance [V, D]
      - a 62-dim one-hot/multi-hot 'selected mask' (poses chosen so far)
    outputs a 62-way distribution (logits) over the next pose to pick.

    Implementation:
      per-view encoder: feat -> H
      state encoder: mean over selected per-view encodings  (masked mean)
      mask encoder: 62-dim (multi-hot) -> H
      policy head: concat(state, mask_enc) -> logits[62]
    """
    def __init__(self, feat_dim, num_poses=62, hidden=512):
        super().__init__()
        self.num_poses = num_poses
        self.view_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.mask_encoder = nn.Sequential(
            nn.Linear(num_poses, hidden),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(2*hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_poses),
        )

    def _encode_views(self, views):  # [V, D] -> [V, H]
        return self.view_mlp(views)

    def logits(self, view_encs, selected_mask):
        """view_encs: [V, H]; selected_mask: [V] 0/1; returns [V] logits"""
        V, H = view_encs.shape
        assert V == self.num_poses, f"Expected {self.num_poses} poses, got {V}"
        sm = selected_mask.float().unsqueeze(-1)  # [V,1]
        denom = torch.clamp(sm.sum(), min=1.0)
        state_mean = (view_encs * sm).sum(dim=0) / denom  # [H]
        mask_h = self.mask_encoder(selected_mask.float().unsqueeze(0)).squeeze(0)  # [H]
        fused = torch.cat([state_mean, mask_h], dim=-1)    # [2H]
        return self.policy_head(fused)  # [V]

def sample_seq(selector, view_encs, K, start_idx, device, greedy=False):
    """Pick K poses after a starting pose. Returns (chosen, logp_sum, final_mask)."""
    V = view_encs.shape[0]
    selected_mask = F.one_hot(torch.tensor(start_idx, device=device), num_classes=V).to(torch.float32)
    chosen = []
    logp_sum = torch.tensor(0.0, device=device)
    for _ in range(K):
        mask_t = selected_mask.clone()
        logits = selector.logits(view_encs, mask_t)
        masked_logits = logits.masked_fill(mask_t.bool(), -float('inf'))
        if greedy:
            idx = masked_logits.argmax(dim=-1)
        else:
            probs = F.softmax(masked_logits, dim=-1)
            dist = Categorical(probs=probs)
            idx = dist.sample()
            logp_sum = logp_sum + dist.log_prob(idx)
        chosen.append(int(idx.item()))
        selected_mask = mask_t + F.one_hot(idx, num_classes=V).to(mask_t.dtype)
        selected_mask = (selected_mask > 0).to(mask_t.dtype)
    return chosen, logp_sum, selected_mask

def aggregate_views(views, idxs, method="mean"):
    sel = views[idxs]
    if method == "mean":
        return sel.mean(dim=0)
    elif method == "max":
        return sel.max(dim=0).values
    elif method == "concat":
        return sel.reshape(-1)
    else:
        raise ValueError(f"Unknown aggregate method: {method}")

# ---------------------------
# Classifier (on K (or K+1) selected views)
# ---------------------------

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=512, depth=1, dropout=0.0):
        super().__init__()
        layers, dim = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_classifier(model, feats, labels, epochs=5, lr=1e-3, weight_decay=1e-4, device="cpu", progress_desc=None, return_accs=False, batch_size=256):
    model.to(device)
    x = torch.stack(feats, dim=0).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    accs = []
    for _ in trange(epochs, desc=progress_desc or "clf-train", leave=False):
        model.train()
        for bx, by in dl:
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        with torch.no_grad():
            model.eval()
            logits = model(x)
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean().item()
            accs.append(acc)
    if return_accs:
        return model, accs
    return model

@torch.no_grad()
def evaluate_classifier(model, feats, labels, device="cpu", batch_size=1024):
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(feats), batch_size):
        bx = torch.stack(feats[i:i+batch_size], dim=0).to(device)
        by = torch.tensor(labels[i:i+batch_size], dtype=torch.long, device=device)
        logits = model(bx)
        pred = logits.argmax(dim=-1)
        correct += (pred == by).sum().item()
        total += by.numel()
    return correct / max(1, total)

# ---------------------------
# Main training loop
# ---------------------------

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")

    features_root = os.path.join(args.dataset_root, args.features_subdir)
    instances, class_names = find_instances(features_root)
    num_classes = len(class_names)
    assert num_classes > 0, "No classes found under features subdir"
    print(f"Found {len(instances)} instances across {num_classes} classes.")

    train_list, test_list = split_train_test(instances, m_per_class=args.m, t_per_class=args.t, seed=args.seed)
    print(f"Train instances: {len(train_list)} | Test instances: {len(test_list)}")

    # Probe dims & num poses
    probe = load_instance_features(train_list[0], device="cpu")
    V, feat_dim = probe.shape
    if args.num_poses is None:
        args.num_poses = V
    assert V == args.num_poses, f"Expected {args.num_poses} poses per instance, got {V}"
    print(f"Detected: {V} poses per instance | feature dim {feat_dim}")

    # --- Parse pose metadata and principal/planar tags from filenames ---
    sample_paths = train_list[0].view_paths
    pose_elev = [None] * V
    pose_azim = [None] * V
    pose_is_principal = [False] * V
    pose_is_planar = [False] * V
    for i, p in enumerate(sample_paths):
        base = os.path.basename(p)
        m = re.search(r"e([+\-]?\d+).*?a(\d+)", base)
        if m:
            try:
                pose_elev[i] = int(m.group(1))
                pose_azim[i] = int(m.group(2))
            except ValueError:
                pass
        if "_principal" in base:
            pose_is_principal[i] = True
        if "_planar" in base:
            pose_is_planar[i] = True

    # Merge with CLI-provided index tags, if any
    principal_ids_set = set(i for i, f in enumerate(pose_is_principal) if f)
    planar_ids_set = set(i for i, f in enumerate(pose_is_planar) if f)
    if args.principal_ids:
        principal_ids_set |= {int(s) for s in args.principal_ids.split(',') if s.strip() != ''}
    if args.planar_ids:
        planar_ids_set |= {int(s) for s in args.planar_ids.split(',') if s.strip() != ''}

    def is_principal_idx(idx: int) -> bool:
        return idx in principal_ids_set
    def is_planar_idx(idx: int) -> bool:
        return idx in planar_ids_set

    pose_meta = {
        "elev": pose_elev,
        "azim": pose_azim,
        "principal": [bool(x) for x in pose_is_principal],
        "planar": [bool(x) for x in pose_is_planar],
    }

    # Selector and optimizer
    selector = SeqSelector(feat_dim, num_poses=args.num_poses, hidden=args.selector_hidden).to(device)
    sel_opt = torch.optim.AdamW(selector.parameters(), lr=args.selector_lr, weight_decay=args.selector_wd)

    # --- Fixed classifier init (same weights every episode) ---
    clf_in_dim = feat_dim if args.agg != "concat" else feat_dim * (args.K + (1 if args.include_start_in_clf else 0))
    clf_init = SimpleClassifier(clf_in_dim, num_classes, hidden=args.clf_hidden, depth=args.clf_depth, dropout=args.clf_dropout)
    clf_init_state = copy.deepcopy(clf_init.state_dict())

    baseline = 0.0
    beta = args.baseline_beta

    # Stats accumulators & output dir
    os.makedirs(args.out_dir, exist_ok=True)
    acc_history = []
    principal_ratio_hist = []
    planar_ratio_hist = []
    principal_count_hist = []
    planar_count_hist = []
    total_selected_hist = []
    sel_counts_total = np.zeros(V, dtype=np.int64)
    start_counts_total = np.zeros(V, dtype=np.int64)
    sel_counts_hist = []  # per-episode selection vectors [V]

    for ep in trange(1, args.episodes + 1, desc="Episodes"):
        selector.train()
        train_feats, train_labels = [], []
        test_feats,  test_labels  = [], []
        logp_sum = torch.tensor(0.0, device=device)

        # Episode-level stats
        start_counts_ep = np.zeros(V, dtype=np.int64)
        sel_counts_ep = np.zeros(V, dtype=np.int64)
        principal_sel_ep = 0
        planar_sel_ep = 0
        total_sel_ep = 0

        # ---- TRAIN selection ----
        for inst in tqdm(train_list, desc=f"Ep {ep} train-select", leave=False):
            views = load_instance_features(inst, device)
            view_encs = selector._encode_views(views)
            start_idx = torch.randint(low=0, high=V, size=(1,), device=device).item()
            chosen, lp, _ = sample_seq(selector, view_encs, args.K, start_idx, device, greedy=False)
            
            start_counts_ep[start_idx] += 1
            for idx in chosen:
                sel_counts_ep[idx] += 1
                total_sel_ep += 1
                if is_principal_idx(idx):
                    principal_sel_ep += 1
                if is_planar_idx(idx):
                    planar_sel_ep += 1

            clf_indices = chosen.copy()
            if args.include_start_in_clf:
                clf_indices = [start_idx] + clf_indices
            agg = aggregate_views(views, clf_indices, method=args.agg)
            train_feats.append(agg.detach().cpu())
            train_labels.append(inst.class_idx)
            logp_sum = logp_sum + lp

        # ---- Train classifier on TRAIN selections ----
        clf = SimpleClassifier(clf_in_dim, num_classes, hidden=args.clf_hidden, depth=args.clf_depth, dropout=args.clf_dropout)
        clf.load_state_dict(clf_init_state)
        clf, train_accs = train_classifier(
            clf, train_feats, train_labels,
            epochs=args.clf_epochs, lr=args.clf_lr, weight_decay=args.clf_wd,
            device=device, progress_desc=f"Ep {ep} clf", return_accs=True, batch_size=args.clf_batch_size)
        best_train_acc = max(train_accs) if len(train_accs) > 0 else float("nan")

        # ---- TEST selection ----
        for inst in tqdm(test_list, desc=f"Ep {ep} test-select", leave=False):
            views = load_instance_features(inst, device)
            view_encs = selector._encode_views(views)
            start_idx = torch.randint(low=0, high=V, size=(1,), device=device).item()
            chosen, lp, _ = sample_seq(selector, view_encs, args.K, start_idx, device, greedy=args.greedy_eval)
            
            clf_indices = chosen.copy()
            if args.include_start_in_clf:
                clf_indices = [start_idx] + clf_indices
            agg = aggregate_views(views, clf_indices, method=args.agg)
            test_feats.append(agg.detach().cpu())
            test_labels.append(inst.class_idx)
            if not args.greedy_eval:
                logp_sum = logp_sum + lp

        # ---- Reward & policy update ----
        acc = evaluate_classifier(clf, test_feats, test_labels, device=device)
        reward = float(acc)
        baseline = (1 - beta) * baseline + beta * reward if ep > 1 else reward
        advantage = reward - baseline
        loss = -(advantage) * logp_sum
        sel_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), args.grad_clip)
        sel_opt.step()

        # ---- Log stats ----
        acc_history.append(float(acc))
        principal_ratio = (principal_sel_ep / max(1, total_sel_ep))
        planar_ratio = (planar_sel_ep / max(1, total_sel_ep))
        principal_ratio_hist.append(float(principal_ratio))
        planar_ratio_hist.append(float(planar_ratio))
        principal_count_hist.append(int(principal_sel_ep))
        planar_count_hist.append(int(planar_sel_ep))
        total_selected_hist.append(int(total_sel_ep))
        sel_counts_total += sel_counts_ep
        start_counts_total += start_counts_ep
        sel_counts_hist.append(sel_counts_ep.copy())

        ep_stat = {
            "episode": ep,
            "test_accuracy": float(acc),
            "start_counts": start_counts_ep.tolist(),
            "selected_counts": sel_counts_ep.tolist(),
            "principal_selected": int(principal_sel_ep),
            "planar_selected": int(planar_sel_ep),
            "total_selected": int(total_sel_ep),
            "principal_ratio": float(principal_ratio),
            "planar_ratio": float(planar_ratio),
            "pose_meta": pose_meta,
            "train_accs": train_accs,
        }
        with open(os.path.join(args.out_dir, f"stats_episode_{ep:04d}.json"), "w") as f:
            json.dump(ep_stat, f, indent=2)
        with open(os.path.join(args.out_dir, "stats_history.jsonl"), "a") as f:
            f.write(json.dumps(ep_stat) + "\\n")

        print(f"[Episode {ep:03d}] test_acc={acc:.4f} | best_train_acc={best_train_acc:.4f} | last_train_acc={(train_accs[-1] if len(train_accs)>0 else float('nan')):.4f}")

    # ---- Save summary stats & plots ----
    summary = {
        "acc_history": acc_history,
        "principal_ratio_history": principal_ratio_hist,
        "planar_ratio_history": planar_ratio_hist,
        "sel_counts_total": sel_counts_total.tolist(),
        "start_counts_total": start_counts_total.tolist(),
    }
    with open(os.path.join(args.out_dir, "stats_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if len(acc_history) > 0:
        # Accuracy over episodes
        plt.figure(); plt.plot(range(1, len(acc_history)+1), acc_history, marker='o')
        plt.xlabel("Episode"); plt.ylabel("Test accuracy"); plt.title("Accuracy over episodes")
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "accuracy_over_episodes.png"), dpi=200); plt.close()

        # Mean selection frequency across episodes
        plt.figure(); mean_sel = (sel_counts_total / max(1, len(acc_history)))
        plt.bar(np.arange(len(mean_sel)), mean_sel)
        plt.xlabel("Pose index"); plt.ylabel("Mean selections/episode"); plt.title("Selection frequency (mean per episode)")
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "selection_frequency.png"), dpi=200); plt.close()

        # Principal/Planar ratios over episodes
        plt.figure();
        plt.plot(range(1, len(principal_ratio_hist)+1), principal_ratio_hist, marker='o', label='Principal ratio')
        plt.plot(range(1, len(planar_ratio_hist)+1), planar_ratio_hist, marker='o', label='Planar ratio')
        plt.xlabel("Episode"); plt.ylabel("Ratio of selections"); plt.title("Principal & Planar ratios over episodes")
        plt.legend(); plt.tight_layout();
        plt.savefig(os.path.join(args.out_dir, "principal_planar_ratios_over_episodes.png"), dpi=200); plt.close()

        # Principal/Planar counts per episode
        plt.figure();
        plt.plot(range(1, len(principal_count_hist)+1), principal_count_hist, marker='o', label='Principal count/ep')
        plt.plot(range(1, len(planar_count_hist)+1), planar_count_hist, marker='o', label='Planar count/ep')
        plt.xlabel("Episode"); plt.ylabel("Count of selected views"); plt.title("Principal & Planar counts per episode")
        plt.legend(); plt.tight_layout();
        plt.savefig(os.path.join(args.out_dir, "principal_planar_counts_over_episodes.png"), dpi=200); plt.close()

        # Heatmaps across episodes
        if len(sel_counts_hist) > 0:
            arr = np.stack(sel_counts_hist, axis=0)  # [E, V]
            plt.figure(); plt.imshow(arr.T, aspect='auto', origin='lower'); plt.colorbar(label="Selections per episode")
            plt.xlabel("Episode"); plt.ylabel("Pose index"); plt.title("Per-view selection counts over episodes")
            plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "selection_counts_heatmap.png"), dpi=200); plt.close()

            denom = np.maximum(1, np.array(total_selected_hist, dtype=np.float32))[:, None]
            share = arr / denom
            plt.figure(); plt.imshow(share.T, aspect='auto', origin='lower', vmin=0.0, vmax=share.max() if share.size else 1.0); plt.colorbar(label="Share per episode")
            plt.xlabel("Episode"); plt.ylabel("Pose index"); plt.title("Per-view selection share over episodes")
            plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "selection_share_heatmap.png"), dpi=200); plt.close()

    if args.save_selector:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "rl_selector_seq.pt")
        torch.save({
            "model": selector.state_dict(),
            "feat_dim": feat_dim,
            "num_poses": args.num_poses,
            "K": args.K,
            "agg": args.agg,
            "include_start_in_clf": args.include_start_in_clf,
            "class_names": class_names
        }, path)
        print(f"Saved selector to: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential RL view selector with filename-tagged stats and test-accuracy reward")
    parser.add_argument("--dataset_root", type=str, default='/nfs/wattrel/data/md0/kung/shapenet_dataset', help="Path to dataset root")
    parser.add_argument("--features_subdir", type=str, default='/nfs/wattrel/data/md0/kung/shapenet_dataset/feature_r50', help="Subdir with features (.pt)")
    parser.add_argument("--m", type=int, default=10, help="Train instances per class")
    parser.add_argument("--t", type=int, default=5, help="Test instances per class after the m train instances (None=all remaining)")
    parser.add_argument("--K", type=int, default=3, help="Number of views to select per instance (after the random start)")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # Poses / policy
    parser.add_argument("--num_poses", type=int, default=None, help="Number of canonical views (default: infer)")
    parser.add_argument("--selector_hidden", type=int, default=512)
    parser.add_argument("--selector_lr", type=float, default=5e-4)
    parser.add_argument("--selector_wd", type=float, default=1e-3)
    parser.add_argument("--baseline_beta", type=float, default=0.3)
    parser.add_argument("--grad_clip", type=float, default=2.0)

    # Aggregation & classifier
    parser.add_argument("--agg", type=str, default="mean", choices=["mean", "max", "concat"], help="Aggregate for classifier input")
    parser.add_argument("--include_start_in_clf", action="store_true", help="Include random start view in classifier input")
    parser.add_argument("--clf_hidden", type=int, default=512)
    parser.add_argument("--clf_depth", type=int, default=1)
    parser.add_argument("--clf_dropout", type=float, default=0.0)
    parser.add_argument("--clf_epochs", type=int, default=10)
    parser.add_argument("--clf_lr", type=float, default=2e-3)
    parser.add_argument("--clf_wd", type=float, default=1e-2)
    parser.add_argument("--clf_batch_size", type=int, default=64)

    # Eval behavior
    parser.add_argument("--greedy_eval", action="store_true", help="Use greedy selection at test time")

    # Optional manual index tags (will be UNIONed with filename tags)
    parser.add_argument("--principal_ids", type=str, default="", help="CSV of pose indices to mark as principal")
    parser.add_argument("--planar_ids", type=str, default="", help="CSV of pose indices to mark as planar")

    # Saving
    parser.add_argument("--save_selector", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs")

    args = parser.parse_args()

    # Auto-name run directory (optional)
    out_dir = 'm: ' + str(args.m) + '_t: ' + str(args.t) + '_K: ' + str(args.K) + '_epi: ' + str(args.episodes)
    out_dir = out_dir + '_num_poses: ' + str(args.num_poses) + '_slr: ' + str(args.selector_lr) + '_swd: ' + str(args.selector_wd)
    out_dir = out_dir + '_epoch: ' + str(args.clf_epochs) + '_clr: ' + str(args.clf_lr) + '_cwd: ' + str(args.clf_wd)
    args.out_dir = os.path.join(args.out_dir, out_dir)
    run(args)
