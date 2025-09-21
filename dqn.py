# rl_view_selector_seq.py
import os
import random
import argparse
from glob import glob
from collections import defaultdict, namedtuple, deque

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
import time

# ---------------------------
# Utilities and data scanning
# ---------------------------

Instance = namedtuple("Instance", ["class_idx", "class_name", "instance_id", "view_paths"])

def set_seed(seed=42):

    if seed == None:
        seed = time.time()

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def find_instances(features_root, num_classes=None):
    """Scan <features_root>/<class>/<instance>/<view>.pt and return Instance records."""
    class_dirs = sorted([d for d in glob(os.path.join(features_root, "*")) if os.path.isdir(d)])
    if num_classes != None:
        class_dirs = class_dirs[:num_classes]
    instances = []
    for ci, cdir in enumerate(class_dirs):
        class_name = os.path.basename(cdir)
        inst_dirs = sorted([d for d in glob(os.path.join(cdir, "*")) if os.path.isdir(d)])
        for idir in inst_dirs:
            instance_id = os.path.basename(idir)
            vpaths = sorted(glob(os.path.join(idir, "*.pt")), key=natural_key)
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

def load_instance_features(instance: Instance, device, selector_feature='r50', train=True, test_idx=None):
    feats = []
    edges = []
    for i, p in enumerate(instance.view_paths):
        t = torch.load(p, map_location="cpu")
        if isinstance(t, dict) and "feat" in t:
            t = t["feat"]
        t = t.float().view(-1)
        feats.append(t)

        # if selector_feature == 'edge':

    feats = torch.stack(feats, dim=0).to(device)  # [V, D]


    return feats

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

def sample_seq(selector, view_encs, K, start_idx, device, greedy=False, temperature=1.0):
    """Pick K poses after a starting pose. Returns (chosen, logp_sum, final_mask)."""
    
    entropy_sum = torch.tensor(0.0, device=device)

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
            # probs = F.softmax(masked_logits, dim=-1)
            probs = F.softmax(masked_logits / max(1e-6, temperature), dim=-1)
            dist = Categorical(probs=probs)
            idx = dist.sample()
            logp_sum = logp_sum + dist.log_prob(idx)
            entropy_sum = entropy_sum + dist.entropy()
        chosen.append(int(idx.item()))
        selected_mask = mask_t + F.one_hot(idx, num_classes=V).to(mask_t.dtype)
        selected_mask = (selected_mask > 0).to(mask_t.dtype)
    return chosen, logp_sum, selected_mask, entropy_sum


def aggregate_views(views, idxs, method="mean", train=True):
    if train:
        sel = views[idxs]  # [K, D]
    else:
        sel = views

    if method == "mean":
        return sel.mean(dim=0)
    elif method == "max":
        return sel.max(dim=0).values
    elif method == "concat":
        return sel.reshape(-1)
    else:
        raise ValueError(f"Unknown aggregate method: {method}")


def sample_seq_pg(selector: SeqSelector, view_encs, K, start_idx, device, greedy=False):
    V = view_encs.shape[0]
    selected_mask = F.one_hot(torch.tensor(start_idx, device=device), num_classes=V).to(torch.float32)
    chosen = []
    logp_sum = torch.tensor(0.0, device=device)
    hx = selector.initial_state(device)
    for _ in range(K):
        mask_t = selected_mask.clone()
        logits, hx = selector(view_encs, mask_t, hx)
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


class DQNSelector(nn.Module):
    """
    Q-network that outputs Q(s, a) for ALL actions in one forward pass.
    State encodes the aggregated selected features + mask embedding.
    Actions are camera indices with learned embeddings from one-hot.
    Q(s,a) = (E[a] @ W) @ phi(s) + b[a] (vectorized across all a).
    """
    def __init__(self, feat_dim, num_poses=62, enc_hidden=512, state_hidden=512, action_dim=128):
        super().__init__()
        self.num_poses = num_poses
        # Encode per-view features (to be aggregated)
        self.view_mlp = nn.Sequential(
            nn.Linear(feat_dim, enc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(enc_hidden, enc_hidden),
            nn.ReLU(inplace=True),
        )
        # Encode selected-mask
        self.mask_encoder = nn.Sequential(
            nn.Linear(num_poses, enc_hidden),
            nn.ReLU(inplace=True),
        )
        # State projector
        self.state_proj = nn.Sequential(
            nn.Linear(2 * enc_hidden, state_hidden),
            nn.ReLU(inplace=True),
        )
        # Action embeddings (from one-hot camera index)
        self.action_emb = nn.Embedding(num_poses, action_dim)
        nn.init.normal_(self.action_emb.weight, mean=0.0, std=0.02)
        # Bilinear scoring
        self.W = nn.Parameter(torch.randn(action_dim, state_hidden) * 0.01)
        self.a_bias = nn.Parameter(torch.zeros(num_poses))


    def encode_views(self, views):
        return self.view_mlp(views) # [V, Henc]


    def state_vector(self, view_encs, selected_mask):
        sm = selected_mask.float().unsqueeze(-1) # [V,1]
        denom = torch.clamp(sm.sum(), min=1.0)
        state_mean = (view_encs * sm).sum(dim=0) / denom # [Henc]
        mask_h = self.mask_encoder(selected_mask.float().unsqueeze(0)).squeeze(0) # [Henc]
        s = self.state_proj(torch.cat([state_mean, mask_h], dim=-1)) # [Hs]
        return s


    def q_values(self, view_encs, selected_mask):
        """Return Q(s, a) for all a as [V]."""
        s = self.state_vector(view_encs, selected_mask) # [Hs]
        # [V, Adim] @ [Adim, Hs] -> [V, Hs]; then dot with s -> [V]
        Aw = self.action_emb.weight @ self.W # [V, Hs]
        q = (Aw @ s) + self.a_bias # [V]
        return q

# ---------------------------
# Replay buffer for DQN
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.buf = deque(maxlen=self.capacity)
        self.device = device
    
    def push(self, s, a, r, s2, done, mask2):
        # s, s2: torch.Tensor [Hs]
        # a: int, r: float, done: bool, mask2: torch.BoolTensor [V]
        self.buf.append((s.detach().cpu(), int(a), float(r), s2.detach().cpu() if s2 is not None else None, bool(done), mask2.cpu() if mask2 is not None else None))
    
    def __len__(self):
        return len(self.buf)
    
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), size=min(batch_size, len(self.buf)), replace=False)
        s, a, r, s2, done, mask2 = zip(*[self.buf[i] for i in idx])
        s = torch.stack([torch.tensor(x) if not torch.is_tensor(x) else x for x in s], dim=0).to(self.device).float()
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device)
        s2 = [None if x is None else (torch.tensor(x) if not torch.is_tensor(x) else x) for x in s2]
        mask2 = [None if x is None else (torch.tensor(x) if not torch.is_tensor(x) else x) for x in mask2]
        s2 = torch.stack([torch.zeros_like(s[0]) if x is None else x for x in s2], dim=0).to(self.device).float()
        # mask2: pack into tensor of shape [B, V] (False=invalid? we will use True for available here and invert later)
        if mask2[0] is None:
            mask2 = torch.ones((len(idx), 1), dtype=torch.bool, device=self.device) # placeholder
        else:
            mask2 = torch.stack(mask2, dim=0).to(self.device).bool()
        return s, a, r, s2, done, mask2

def _angle_delta(a1, a2):
    """Smallest absolute difference between two azimuth angles in degrees."""
    d = torch.abs(a1 - a2)
    return torch.minimum(d, 360.0 - d)

def _compute_pose_novelty(chosen_idx, selected_mask, pose_elev_t, pose_azim_t, pose_roll_t):
    """Return novelty in [0,1], higher means further from selected set by e/a/r (normalized)."""
    sel_ids = torch.where(selected_mask.bool())[0]
    if sel_ids.numel() == 0:
        return torch.tensor(1.0, device=pose_elev_t.device)  # max novelty when nothing selected
    e1 = pose_elev_t[chosen_idx].expand_as(sel_ids)
    a1 = pose_azim_t[chosen_idx].expand_as(sel_ids)
    r1 = pose_roll_t[chosen_idx].expand_as(sel_ids)
    e2 = pose_elev_t[sel_ids]; a2 = pose_azim_t[sel_ids]; r2 = pose_roll_t[sel_ids]
    de = torch.abs(e1 - e2) / 180.0              # in [0, ~1]
    da = _angle_delta(a1, a2) / 180.0            # wrap-around
    dr = torch.abs(r1 - r2) / 180.0
    d = torch.stack([de, da, dr], dim=-1)        # [|S|, 3]
    d_mean = d.mean(dim=-1)                      # [|S|]
    novelty = d_mean.min()                       # min distance to selected set (encourage covering space)
    return novelty.clamp(0.0, 1.0)

def _compute_feat_novelty(chosen_idx, selected_mask, view_encs):
    """Cosine distance to centroid of selected set in [0,2]. We'll map to [0,1]."""
    sel_ids = torch.where(selected_mask.bool())[0]
    if sel_ids.numel() == 0:
        return torch.tensor(1.0, device=view_encs.device)
    centroid = view_encs[sel_ids].mean(dim=0, keepdim=True)           # [1,H]
    v = view_encs[chosen_idx:chosen_idx+1]                             # [1,H]
    v = F.normalize(v, dim=-1)
    c = F.normalize(centroid, dim=-1)
    cos_sim = (v * c).sum(dim=-1).squeeze(0)                          # [-1,1]
    cos_dist = (1.0 - cos_sim)                                        # [0,2]
    return (cos_dist / 2.0).clamp(0.0, 1.0)

def compute_shaping_reward(kind, chosen_idx, selected_mask, view_encs, pose_elev_t, pose_azim_t, pose_roll_t,
                           is_principal, is_planar, principal_bonus, planar_bonus):
    reward = torch.tensor(0.0, device=view_encs.device)
    if kind in ("novelty_pose", "hybrid"):
        reward = reward + _compute_pose_novelty(chosen_idx, selected_mask, pose_elev_t, pose_azim_t, pose_roll_t)
    if kind in ("novelty_feat", "hybrid"):
        reward = reward + _compute_feat_novelty(chosen_idx, selected_mask, view_encs)
    if kind in ("principal_planar", "hybrid"):
        b = (principal_bonus if is_principal else 0.0) + (planar_bonus if is_planar else 0.0)
        reward = reward + torch.tensor(b, device=view_encs.device)
    return reward.item()


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


def train_classifier(model, feats, labels, epochs=5, lr=1e-3, weight_decay=1e-4, device="cuda", progress_desc=None, return_accs=False, batch_size=256):
    model.to(device)
    x = torch.stack(feats, dim=0).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    accs = []
    for i in trange(epochs, desc=progress_desc or "clf-train", leave=False):
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
    device = torch.device("cuda")
    set_seed(args.seed)
    print(f"Using device: {device}")

    features_root = os.path.join(args.dataset_root, args.features_subdir)
    instances, class_names = find_instances(features_root, num_classes=args.num_classes)
    num_classes = len(class_names)
    assert num_classes > 0, "No classes found under features_subdir"
    print(f"Found {len(instances)} instances across {num_classes} classes.")

    train_list, test_list = split_train_test(instances, m_per_class=args.m, t_per_class=args.t, seed=args.seed)
    print(f"Train instances: {len(train_list)} | Test instances: {len(test_list)}")

    # Probe dims & num poses
    probe = load_instance_features(train_list[0], device="cpu")
    V, feat_dim = probe.shape
    # V=5
    if args.num_poses is None:
        args.num_poses = V
    assert V == args.num_poses, f"Expected {args.num_poses} poses per instance, got {V}"
    print(f"Detected: {V} poses per instance | feature dim {feat_dim}")

    
    # Precompute fixed random test-view sets per instance for reproducible eval
    def precompute_test_sets(test_list, V, K, num_sets, seed):
        rng = random.Random(seed)
        per_instance_sets = {}
        for inst in test_list:
            sets = []
            for _ in range(num_sets):
                perm = list(range(V))
                rng.shuffle(perm)
                sets.append(perm[:K])
            per_instance_sets[inst.instance_id] = sets
        return per_instance_sets

    test_sets_map = precompute_test_sets(test_list, V, args.K, args.test_eval_sets, args.test_eval_seed)
    test_feats, test_labels = [], []
    for inst in test_list:
        views = load_instance_features(inst, device)
        for s in range(args.test_eval_sets):
            idxs_full = test_sets_map[inst.instance_id][s]
            agg = aggregate_views(views, idxs_full, method=args.agg)
            test_feats.append(agg.cpu())
            test_labels.append(inst.class_idx)


    # --- Parse pose metadata and principal/planar tags from feature filenames ---
    sample_paths = train_list[0].view_paths
    pose_elev = [None] * V
    pose_azim = [None] * V
    pose_roll = [None] * V
    pose_is_principal = [False] * V
    pose_is_planar = [False] * V
    pose_is_short = [False] * V
    pose_is_long = [False] * V
    for i, p in enumerate(sample_paths):
        base = os.path.basename(p)
        m_ea = re.search("e([+-]?[0-9]+).*?a([0-9]+)", base)
        if m_ea:
            try:
                pose_elev[i] = int(m_ea.group(1))
                pose_azim[i] = int(m_ea.group(2))
            except ValueError:
                pass
        m_r = re.search("r([+-]?[0-9]+)", base)
        if m_r:
            try:
                pose_roll[i] = int(m_r.group(1))
            except ValueError:
                pass

        if "_principal" in base:
            pose_is_principal[i] = True
        if "_planar" in base:
            pose_is_planar[i] = True
            if "_short" in base:
                pose_is_short[i] = True
            else:
                pose_is_long[i] = True

    # Merge with CLI-provided index tags, if any
    principal_ids_set = set(i for i, f in enumerate(pose_is_principal) if f)
    planar_ids_set = set(i for i, f in enumerate(pose_is_planar) if f)
    short_ids_set = set(i for i, f in enumerate(pose_is_short) if f)
    long_ids_set = set(i for i, f in enumerate(pose_is_long) if f)
    if getattr(args, 'principal_ids', None):
        principal_ids_set |= {int(s) for s in args.principal_ids.split(',') if s.strip() != ''}
    if getattr(args, 'planar_ids', None):
        planar_ids_set |= {int(s) for s in args.planar_ids.split(',') if s.strip() != ''}
    if getattr(args, 'short_ids', None):
        short_ids_set |= {int(s) for s in args.short_ids.split(',') if s.strip() != ''}
    if getattr(args, 'long_ids', None):
        long_ids_set |= {int(s) for s in args.long_ids.split(',') if s.strip() != ''}

    def is_principal_idx(idx: int) -> bool:
        return idx in principal_ids_set

    def is_planar_idx(idx: int) -> bool:
        return idx in planar_ids_set

    def is_short_idx(idx: int) -> bool:
        return idx in short_ids_set

    def is_long_idx(idx: int) -> bool:
        return idx in long_ids_set

    # Pose tensors for shaping (fallback to zeros when None)
    def _to_tensor(lst, fill):
        out = torch.tensor([fill if (v is None) else float(v) for v in lst], dtype=torch.float32)
        return out
    pose_elev_t_cpu = _to_tensor(pose_elev, 0.0)
    pose_azim_t_cpu = _to_tensor(pose_azim, 0.0)
    pose_roll_t_cpu = _to_tensor(pose_roll, 0.0)

    pose_meta = {
        "elev": pose_elev,
        "azim": pose_azim,
        "roll": pose_roll,
        "principal": [bool(x) for x in pose_is_principal],
        "planar": [bool(x) for x in pose_is_planar],
        "short": [bool(x) for x in pose_is_short],
        "long": [bool(x) for x in pose_is_long],
    }

    # # --- Selector setup (both variants available) ---
    # if args.algo == "reinforce":
    #     selector = LSTMSelector(feat_dim, num_poses=V, enc_hidden=args.selector_hidden, lstm_hidden=args.lstm_hidden).to(device)
    #     sel_opt = torch.optim.AdamW(selector.parameters(), lr=args.selector_lr, weight_decay=args.selector_wd)
    # else:
    selector = DQNSelector(feat_dim, num_poses=V, enc_hidden=args.selector_hidden,
                           state_hidden=args.dqn_state_hidden, action_dim=args.dqn_action_dim).to(device)
    target_selector = copy.deepcopy(selector).to(device)
    dqn_opt = torch.optim.AdamW(selector.parameters(), lr=args.dqn_lr, weight_decay=args.dqn_wd)
    replay = ReplayBuffer(args.replay_size, device)


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
    short_ratio_hist = []
    long_ratio_hist = []
    principal_count_hist = []

    planar_count_hist = []
    short_count_hist = []
    long_count_hist = []
    total_selected_hist = []

    sel_counts_total = np.zeros(V, dtype=np.int64)
    start_counts_total = np.zeros(V, dtype=np.int64)
    sel_counts_hist = []

    for ep in trange(1, args.episodes + 1, desc="Episodes"):
        
        if args.algo == "reinforce":
            selector.train()
            logp_sum = torch.tensor(0.0, device=device)
            entropy_sum = torch.tensor(0.0, device=device)

        train_feats, train_labels = [], []

        # Episode-level stats
        start_counts_ep = np.zeros(V, dtype=np.int64)
        sel_counts_ep = np.zeros(V, dtype=np.int64)
        principal_sel_ep = 0
        planar_sel_ep = 0
        short_sel_ep = 0
        long_sel_ep = 0
        total_sel_ep = 0
        
        selector.train()
        train_masks = {}
        train_starts = {}
        pending_round_transitions = []  # list of lists, one sublist per round


        for inst in train_list:
            start_idx = torch.randint(low=0, high=V, size=(1,), device=device).item()
            train_starts[inst.instance_id] = start_idx
            m = F.one_hot(torch.tensor(start_idx, device=device), num_classes=V).to(torch.float32)
            train_masks[inst.instance_id] = m
            start_counts_ep[start_idx] += 1

        # For K rounds
        for r in range(1, args.K + 1):
            round_transitions = []  # accum transitions for the round
            # Select one new view for every train instance
            for inst in tqdm(train_list, desc=f"Ep {ep} DQN round {r}/{args.K} select", leave=False):
                views = load_instance_features(inst, device)
                view_encs = selector.encode_views(views)
                mask = train_masks[inst.instance_id]

                with torch.no_grad():
                    s_vec = selector.state_vector(view_encs, mask)

                # epsilon-greedy on available actions
                q_all = selector.q_values(view_encs, mask).masked_fill(mask.bool(), -float('inf'))
                # if random.random() < eps:
                #     avail = (~mask.bool()).nonzero(as_tuple=False).view(-1)
                #     a = int(avail[torch.randint(len(avail), (1,), device=device)].item())
                # else:
                a = int(torch.argmax(q_all).item())

                # shaping reward before updating mask
                pose_elev_t = pose_elev_t_cpu.to(device)
                pose_azim_t = pose_azim_t_cpu.to(device)
                pose_roll_t = pose_roll_t_cpu.to(device)
                is_pr = bool(a in principal_ids_set)
                is_pl = bool(a in planar_ids_set)
                is_long = bool(a in long_ids_set)
                is_short = bool(a in short_ids_set)

                r_shape = 0.0
                # if args.shaping != 'none':
                #     r_shape = compute_shaping_reward(
                #         args.shaping, a, mask, view_encs,
                #         pose_elev_t, pose_azim_t, pose_roll_t,
                #         is_pr, is_pl, args.pp_principal_bonus, args.pp_planar_bonus
                #     )

                # next state
                next_mask = (mask + F.one_hot(torch.tensor(a, device=device), num_classes=V).to(mask.dtype))
                next_mask = (next_mask > 0).to(mask.dtype)
                with torch.no_grad():
                    s2_vec = selector.state_vector(view_encs, next_mask)

                done_i = (r == args.K)
                round_transitions.append((inst.instance_id, s_vec.detach(), a, float(r_shape), s2_vec.detach(), done_i, (~next_mask.bool()).detach()))

                # stats
                sel_counts_ep[a] += 1
                total_sel_ep += 1
                if is_pr: principal_sel_ep += 1
                if is_pl: planar_sel_ep += 1
                if is_long: long_sel_ep += 1
                if is_short: short_sel_ep += 1

                # commit mask
                train_masks[inst.instance_id] = next_mask


            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # start_event.record()
            # ---
            # After the round: build datasets, (re)init+train classifier, eval on test with r views, compute acc_r
            # Build train features from current masks
            train_feats, train_labels = [], []
            for inst in train_list:
                views = load_instance_features(inst, device)  # CPU tensor okay
                m = train_masks[inst.instance_id]
                idxs = torch.where(m.bool())[0].tolist()
                if not args.include_start_in_clf and train_starts[inst.instance_id] in idxs:
                    idxs.remove(train_starts[inst.instance_id])
                agg = aggregate_views(views, idxs, method=args.agg)
                train_feats.append(agg.cpu())
                train_labels.append(inst.class_idx)

            # Train classifier from fixed init
            clf = SimpleClassifier(clf_in_dim, num_classes, hidden=args.clf_hidden, depth=args.clf_depth, dropout=args.clf_dropout)
            clf.load_state_dict(clf_init_state)
            clf, train_accs = train_classifier(
                clf, train_feats, train_labels,
                epochs=args.clf_epochs, lr=args.clf_lr, weight_decay=args.clf_wd,
                device=device, progress_desc=f"Ep {ep} clf r{r}", return_accs=True, batch_size=args.clf_batch_size)
            best_train_acc = max(train_accs) if len(train_accs) > 0 else float("nan")

            # ---
            # end_event.record()
            # elapsed_time = start_event.elapsed_time(end_event) / 1000 # Convert ms to seconds
            # print(f"Training took: {elapsed_time:.4f} seconds")



            acc_r = evaluate_classifier(clf, test_feats, test_labels, device=device)
            acc_history.append(float(acc_r)) # keep full curve; last one is ep score # keep full curve; last one is ep score


            # Push transitions from this round
            # if args.algo == "dqn":
            for (_iid, s_vec, a, r_shape, s2_vec, done_i, mask2) in round_transitions:
                r_tot = args.shaping_coef * float(r_shape)
                if done_i:
                    r_tot += args.terminal_coef * float(acc_r)
                replay.push(s_vec, a, r_tot, (None if done_i and args.terminal_no_s2 else s2_vec), done_i, mask2)

                # DQN updates after each round
                if len(replay) >= args.min_replay_to_train:
                    selector.train()
                    for _ in range(args.dqn_updates_per_ep):  # reuse per-ep count per round for now
                        s, a, r_return, s2, done, mask2 = replay.sample(args.dqn_batch_size)
                        with torch.no_grad():
                            Aw = selector.action_emb.weight @ selector.W  # [V, Hs]

                        q_all = (Aw @ s.T).T + selector.a_bias  # [B, V]
                        q_sa = q_all.gather(1, a.view(-1,1)).squeeze(1)
                        with torch.no_grad():
                            Aw_t = target_selector.action_emb.weight @ target_selector.W
                            q2_all = (Aw_t @ s2.T).T + target_selector.a_bias
                            invalid = ~mask2  # mask2=True means available
                            q2_all = q2_all.masked_fill(invalid, -1e9)
                            q2_max = q2_all.max(dim=1).values
                            target = r_return + (~done).float() * (args.gamma * q2_max)
                        loss = F.mse_loss(q_sa, target)
                        dqn_opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(selector.parameters(), args.grad_clip)
                        dqn_opt.step()

                # Epsilon decay per round
                # eps = max(args.eps_end, eps * args.eps_decay)

                # Target network update per round
                if (r % args.target_update_interval) == 0:
                    target_selector.load_state_dict(selector.state_dict())

        # ---- Log stats ----
        acc_history.append(float(acc_r))
        principal_ratio = (principal_sel_ep / max(1, total_sel_ep))
        planar_ratio = (planar_sel_ep / max(1, total_sel_ep))
        short_ratio = (short_sel_ep / max(1, total_sel_ep))
        long_ratio = (long_sel_ep / max(1, total_sel_ep))

        principal_ratio_hist.append(float(principal_ratio))
        planar_ratio_hist.append(float(planar_ratio))
        short_ratio_hist.append(float(short_ratio))
        long_ratio_hist.append(float(long_ratio))

        principal_count_hist.append(int(principal_sel_ep))
        planar_count_hist.append(int(planar_sel_ep))
        short_count_hist.append(int(short_sel_ep))
        long_count_hist.append(int(long_sel_ep))
        total_selected_hist.append(int(total_sel_ep))

        sel_counts_total += sel_counts_ep
        start_counts_total += start_counts_ep
        sel_counts_hist.append(sel_counts_ep.copy())

        ep_stat = {
            "episode": ep,
            "test_accuracy": float(acc_r),
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
        # with open(os.path.join(args.out_dir, f"stats_episode_{ep:04d}.json"), "w") as f:
        #     json.dump(ep_stat, f, indent=2)
        # with open(os.path.join(args.out_dir, "stats_history.jsonl"), "a") as f:
        #     f.write(json.dumps(ep_stat) + "\n")

        print(f"[Episode {ep:03d}] test_acc={acc_r:.4f} | best_train_acc={best_train_acc:.4f} | last_train_acc={(train_accs[-1] if len(train_accs)>0 else float('nan')):.4f}")

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
        # plt.plot(range(1, len(principal_ratio_hist)+1), principal_ratio_hist, marker='o', label='Principal ratio')
        plt.plot(range(1, len(planar_ratio_hist)+1), planar_ratio_hist, marker='o', label='Planar ratio')
        plt.plot(range(1, len(short_ratio_hist)+1), short_ratio_hist, marker='o', label='Short ratio')
        plt.plot(range(1, len(long_ratio_hist)+1), long_ratio_hist, marker='o', label='Longest ratio')
        plt.xlabel("Episode"); plt.ylabel("Ratio of selections"); plt.title("Principal & Planar ratios over episodes")
        plt.legend(); plt.tight_layout();
        plt.savefig(os.path.join(args.out_dir, "planar_ratios_over_episodes.png"), dpi=200); plt.close()

        # Principal/Planar counts per episode
        plt.figure();
        # plt.plot(range(1, len(principal_count_hist)+1), principal_count_hist, marker='o', label='Principal count/ep')
        plt.plot(range(1, len(planar_count_hist)+1), planar_count_hist, marker='o', label='Planar count/ep')
        plt.plot(range(1, len(short_count_hist)+1), short_count_hist, marker='o', label='Short count/ep')
        plt.plot(range(1, len(long_count_hist)+1), long_count_hist, marker='o', label='Longest count/ep')
        plt.xlabel("Episode"); plt.ylabel("Count of selected views"); plt.title("Planar/Short/Longest counts per episode")
        plt.legend(); plt.tight_layout();
        plt.savefig(os.path.join(args.out_dir, "planar_counts_over_episodes.png"), dpi=200); plt.close()

        # # Heatmaps across episodes
        # if len(sel_counts_hist) > 0:
        #     arr = np.stack(sel_counts_hist, axis=0)  # [E, V]
        #     plt.figure(); plt.imshow(arr.T, aspect='auto', origin='lower'); plt.colorbar(label="Selections per episode")
        #     plt.xlabel("Episode"); plt.ylabel("Pose index"); plt.title("Per-view selection counts over episodes")
        #     plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "selection_counts_heatmap.png"), dpi=200); plt.close()

        #     denom = np.maximum(1, np.array(total_selected_hist, dtype=np.float32))[:, None]
        #     share = arr / denom
        #     plt.figure(); plt.imshow(share.T, aspect='auto', origin='lower', vmin=0.0, vmax=share.max() if share.size else 1.0); plt.colorbar(label="Share per episode")
        #     plt.xlabel("Episode"); plt.ylabel("Pose index"); plt.title("Per-view selection share over episodes")
        #     plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "selection_share_heatmap.png"), dpi=200); plt.close()

        # Heatmaps across episodes (with tagged high-frequency rows)
        if len(sel_counts_hist) > 0:
            arr = np.stack(sel_counts_hist, axis=0) # [E, V]

        # Helper to build compact labels
        def _pose_label(i):
            lbl = f"{i}"
            try:
                e = pose_meta.get('elev', [None]*arr.shape[1])[i]
                a = pose_meta.get('azim', [None]*arr.shape[1])[i]
                r = pose_meta.get('roll', [None]*arr.shape[1])[i]
                if e is not None and a is not None:
                    lbl += f" e{int(e):+d} a{int(a):03d} r{int(r):03d}"
                tags = []
                # if pose_meta.get('principal', [False]*arr.shape[1])[i]: tags.append('P')
                if pose_meta.get('planar', [False]*arr.shape[1])[i]: tags.append('Pl')
                if pose_meta.get('short', [False]*arr.shape[1])[i]: tags.append('S')
                if pose_meta.get('long', [False]*arr.shape[1])[i]: tags.append('L')
                if tags: lbl += " " + "/".join(tags)
            except Exception:
                pass
            return lbl


        # Counts heatmap + tagged rows
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        im = ax.imshow(arr.T, aspect='auto', origin='lower')
        cbar = fig.colorbar(im, ax=ax, label="Selections per episode")
        ax.set_xlabel("Episode"); ax.set_ylabel("Pose index"); ax.set_title("Per-view selection counts over episodes")
        mean_counts = arr.mean(axis=0) # [V]
        topk = getattr(args, 'heatmap_label_topk', 12)
        topk = int(min(max(1, topk), mean_counts.shape[0]))
        top_ids = np.argsort(mean_counts)[-topk:]
        ax.set_yticks(top_ids)
        ax.set_yticklabels([_pose_label(i) for i in top_ids], fontsize=7)
        for y in top_ids:
            ax.axhline(y=y, lw=0.5, alpha=0.4, color='w')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "selection_counts_heatmap.png"), dpi=getattr(args, 'heatmap_dpi', 300))
        plt.close(fig)


        # Share heatmap + tagged rows
        denom = np.maximum(1, np.array(total_selected_hist, dtype=np.float32))[:, None]
        share = arr / denom
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        im = ax.imshow(share.T, aspect='auto', origin='lower', vmin=0.0, vmax=share.max() if share.size else 1.0)
        cbar = fig.colorbar(im, ax=ax, label="Share per episode")
        ax.set_xlabel("Episode"); ax.set_ylabel("Pose index"); ax.set_title("Per-view selection share over episodes")
        mean_share = share.mean(axis=0)
        top_ids_share = np.argsort(mean_share)[-topk:]
        ax.set_yticks(top_ids_share)
        ax.set_yticklabels([_pose_label(i) for i in top_ids_share], fontsize=7)
        for y in top_ids_share:
            ax.axhline(y=y, lw=0.5, alpha=0.4, color='w')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "selection_share_heatmap.png"), dpi=getattr(args, 'heatmap_dpi', 300))
        plt.close(fig)


        # Trend lines for top views (share)
        episodes = np.arange(1, arr.shape[0]+1)
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for i in top_ids:
            ax.plot(episodes, share[:, i], label=_pose_label(i), linewidth=1.0)
        ax.set_xlabel("Episode"); ax.set_ylabel("Selection share")
        ax.set_title("Top views selection share trend")
        ax.legend(fontsize=7, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "top_views_share_trend.png"), dpi=getattr(args, 'heatmap_dpi', 300))
        plt.close(fig)


    if args.save_selector:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "rl_selector_lstm.pt")
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
    parser = argparse.ArgumentParser(description="LSTM-based RL view selector with filename-tagged stats and test-accuracy reward")

    parser.add_argument("--selector_features", type=str, default="r50", choices=["r50", "edge"])
    parser.add_argument("--dataset_root", type=str, default='/nfs/wattrel/data/md0/kung/shapenet_dataset_short', help="Path to dataset root")
    parser.add_argument("--features_subdir", type=str, default='/nfs/wattrel/data/md0/kung/shapenet_dataset_short/feature_r50', help="Subdir with features (.pt)")
    parser.add_argument("--m", type=int, default=3, help="Train instances per class")
    parser.add_argument("--t", type=int, default=3, help="Test instances per class after the m train instances (None=all remaining)")
    parser.add_argument("--K", type=int, default=3, help="Number of views to select per instance (after the random start)")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of views to select per instance (after the random start)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--algo", type=str, choices=["reinforce", "dqn"], default="dqn")
    
    # Poses / policy
    parser.add_argument("--num_poses", type=int, default=None, help="Number of canonical views (default: infer)")
    parser.add_argument("--selector_hidden", type=int, default=512, help="Per-view encoder hidden (Henc)")
    parser.add_argument("--lstm_hidden", type=int, default=512, help="LSTM hidden size (Hpol)")
    parser.add_argument("--selector_lr", type=float, default=5e-4)
    parser.add_argument("--selector_wd", type=float, default=1e-3)
    parser.add_argument("--baseline_beta", type=float, default=0.3)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--temp", type=float, default=1.0)

    # DQN specifics
    parser.add_argument("--dqn_state_hidden", type=int, default=512)
    parser.add_argument("--dqn_action_dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--dqn_lr", type=float, default=1e-4)
    parser.add_argument("--dqn_wd", type=float, default=1e-4)
    parser.add_argument("--replay_size", type=int, default=20000)
    parser.add_argument("--min_replay_to_train", type=int, default=512)
    parser.add_argument("--dqn_batch_size", type=int, default=256)
    parser.add_argument("--dqn_updates_per_ep", type=int, default=200)
    parser.add_argument("--target_update_interval", type=int, default=1)
    parser.add_argument("--eps_start", type=float, default=0.2)
    parser.add_argument("--eps_end", type=float, default=0.02)
    parser.add_argument("--eps_decay", type=float, default=0.99)
    parser.add_argument("--terminal_no_s2", action="store_true", help="If set, terminal transitions omit s2")
    parser.add_argument("--shaping", type=str, default=None)
    parser.add_argument("--shaping_coef", type=float, default=0.5, help="Scale for per-step shaping reward")
    parser.add_argument("--terminal_coef", type=float, default=1.0, help="Scale for terminal reward")

    # Aggregation & classifier
    parser.add_argument("--agg", type=str, default="max", choices=["mean", "max", "concat"], help="Aggregate for classifier input")
    parser.add_argument("--include_start_in_clf", action="store_true", help="Include random start view in classifier input")
    parser.add_argument("--clf_hidden", type=int, default=512)
    parser.add_argument("--clf_depth", type=int, default=1)
    parser.add_argument("--clf_dropout", type=float, default=0.0)
    parser.add_argument("--clf_epochs", type=int, default=5)
    parser.add_argument("--clf_lr", type=float, default=1e-3)
    parser.add_argument("--clf_wd", type=float, default=1e-2)
    parser.add_argument("--clf_batch_size", type=int, default=32)

    # Eval behavior
    parser.add_argument("--greedy_eval", action="store_true", help="Use greedy selection at test time")
    parser.add_argument("--num_test_sample_per_instance", type=int, default=10)
    parser.add_argument("--principal_ids", type=str, default="", help="CSV of pose indices to mark as principal")
    parser.add_argument("--planar_ids", type=str, default="", help="CSV of pose indices to mark as planar")
    parser.add_argument("--heatmap_label_topk", type=int, default=15, help="Number of top poses to label on heatmaps")
    parser.add_argument("--heatmap_dpi", type=int, default=300, help="DPI for heatmap figures")

    # Test evaluation configuration
    parser.add_argument("--test_eval_sets", type=int, default=2, help="Number of random K-view sets per test instance for evaluation")
    parser.add_argument("--test_eval_seed", type=int, default=42, help="Fixed RNG seed for generating test-view sets (independent of args.seed)")

    # Optional manual index tags (unioned with filename tags)
    parser.add_argument("--short_ids", type=str, default="", help="CSV of pose indices to mark as planar")
    parser.add_argument("--long_ids", type=str, default="", help="CSV of pose indices to mark as planar")

    # Saving
    parser.add_argument("--save_selector", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs")

    args = parser.parse_args()

    # Auto-name run directory
    out_dir = args.selector_features
    out_dir = out_dir + '_m: ' + str(args.m) + '_t: ' + str(args.t) + '_K: ' + str(args.K) + '_epi: ' + str(args.episodes) + '_num_classes: ' + str(args.num_classes)
    out_dir = out_dir + '_num_poses: ' + str(args.num_poses) + '_slr: ' + str(args.selector_lr) + '_swd: ' + str(args.selector_wd) + '_entropy: ' + str(args.entropy_coef) + '_temp: ' + str(args.temp)
    out_dir = out_dir + '_epoch: ' + str(args.clf_epochs) + '_clr: ' + str(args.clf_lr) + '_cwd: ' + str(args.clf_wd)
    args.out_dir = os.path.join(args.out_dir, out_dir)
    run(args)
