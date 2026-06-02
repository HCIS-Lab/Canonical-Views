#!/usr/bin/env python3
"""Aggregate `run_evaluation_views.py` outputs into per-class and overall summaries.

Reads one or more `*_views.csv` files produced by `run_evaluation_views.py` and,
for the metric columns you pick, computes:

  - Per-class: agent_mean, random_mean, delta (=agent - random), paired Wilcoxon p,
               and n_instances. One row per ModelNet class.
  - Overall:   instance-level pooled mean (mean over all rows) and macro-mean
               (unweighted mean over per-class means), plus paired Wilcoxon p
               over the full pooled set.

Default metric pairs (each = one `agent_*` column and its `random_*` twin):

  vggt:     pair_mean    (primary, paper-aligned)
            image_mean   (set-level, only if VGGT/Pi3 columns present)
  pi3:      pair_mean
            image_mean
  dust3r:   pair_mean
  mast3r:   pair_mean
  dinov2:   sim_mean     (cosine similarity)

Override with --metrics (space-separated suffixes — the part AFTER `<model>_agent_`
and `<model>_random_`).

Usage:
    python scripts/aggregate_views.py \\
        --csv results/views/vggt_views.csv results/views/dinov2_views.csv \\
        --output_dir results/views/summary/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_METRICS = {
    "vggt":   ["pair_mean", "image_mean"],
    "pi3":    ["pair_mean", "image_mean"],
    "dust3r": ["pair_mean"],
    "mast3r": ["pair_mean"],
    "dinov2": ["sim_mean"],
}


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate agent-vs-random view CSVs.")
    p.add_argument("--csv", nargs="+", required=True,
                   help="One or more *_views.csv files from run_evaluation_views.py.")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="Metric suffixes to aggregate (override per-model defaults).")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def _wilcoxon_p(agent, random):
    a = np.asarray(agent, dtype=float)
    r = np.asarray(random, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(r))
    a, r = a[mask], r[mask]
    if len(a) < 2 or np.allclose(a, r):
        return float("nan")
    try:
        return float(stats.wilcoxon(a, r, zero_method="zsplit").pvalue)
    except ValueError:
        return float("nan")


BUCKET_COUNT_COLS = [
    "agent_n_expanded", "agent_n_expanded_like",
    "agent_n_foreshortened", "agent_n_foreshortened_like",
    "agent_n_remainder",
]


def _per_class_row(cls, grp, model, metric, agent_col, random_col, extra=None):
    agent = grp[agent_col].astype(float)
    rand = grp[random_col].astype(float)
    row = {
        "dataset": cls,
        "model": model,
        "metric": metric,
        "n_instances": len(grp),
        "agent_mean": float(agent.mean()),
        "agent_std": float(agent.std(ddof=1)) if len(agent) > 1 else 0.0,
        "random_mean": float(rand.mean()),
        "random_std": float(rand.std(ddof=1)) if len(rand) > 1 else 0.0,
        "delta_mean": float((agent - rand).mean()),
        "delta_std": float((agent - rand).std(ddof=1)) if len(grp) > 1 else 0.0,
        "wilcoxon_p": _wilcoxon_p(agent, rand),
    }
    for col in BUCKET_COUNT_COLS:
        if col in grp.columns:
            row[col] = int(grp[col].sum())
    if extra:
        row.update(extra)
    return row


def _summarize(df, model, metric):
    """Per-class and overall summaries. Splits by (epoch_start, epoch_end) too if present."""
    agent_col = f"{model}_agent_{metric}"
    random_col = f"{model}_random_{metric}"
    if agent_col not in df.columns or random_col not in df.columns:
        return None, None

    # Carry selected_view_type forward if present (constant per CSV).
    svt = (df["selected_view_type"].iloc[0]
           if "selected_view_type" in df.columns and len(df) else None)
    has_epochs = "epoch_start" in df.columns and "epoch_end" in df.columns

    per_class_rows = []
    overall_rows = []

    if has_epochs:
        for (es, ee), block in df.groupby(["epoch_start", "epoch_end"]):
            extra = {"epoch_start": int(es), "epoch_end": int(ee)}
            if svt is not None:
                extra["selected_view_type"] = svt
            block_rows = [
                _per_class_row(cls, grp, model, metric, agent_col, random_col, extra)
                for cls, grp in block.groupby("dataset")
            ]
            per_class_rows.extend(block_rows)
            class_means = pd.DataFrame(block_rows)
            ov = {
                "model": model,
                "metric": metric,
                "epoch_start": int(es),
                "epoch_end": int(ee),
                "n_classes": int(block["dataset"].nunique()),
                "n_instances_total": int(block.shape[0]),
                "instance_pooled_agent_mean": float(block[agent_col].astype(float).mean()),
                "instance_pooled_random_mean": float(block[random_col].astype(float).mean()),
                "instance_pooled_delta_mean": float((block[agent_col].astype(float) - block[random_col].astype(float)).mean()),
                "macro_mean_agent": float(class_means["agent_mean"].mean()),
                "macro_mean_random": float(class_means["random_mean"].mean()),
                "macro_mean_delta": float(class_means["delta_mean"].mean()),
                "wilcoxon_p_pooled": _wilcoxon_p(block[agent_col].values, block[random_col].values),
            }
            if svt is not None:
                ov["selected_view_type"] = svt
            for col in BUCKET_COUNT_COLS:
                if col in (block.columns if has_epochs else df.columns):
                    ov[col] = int((block if has_epochs else df)[col].sum())
            overall_rows.append(ov)
    else:
        for cls, grp in df.groupby("dataset"):
            extra = {"selected_view_type": svt} if svt is not None else None
            per_class_rows.append(_per_class_row(cls, grp, model, metric, agent_col, random_col, extra))
        class_means = pd.DataFrame(per_class_rows)
        ov = {
            "model": model,
            "metric": metric,
            "n_classes": int(df["dataset"].nunique()),
            "n_instances_total": int(df.shape[0]),
            "instance_pooled_agent_mean": float(df[agent_col].astype(float).mean()),
            "instance_pooled_random_mean": float(df[random_col].astype(float).mean()),
            "instance_pooled_delta_mean": float((df[agent_col].astype(float) - df[random_col].astype(float)).mean()),
            "macro_mean_agent": float(class_means["agent_mean"].mean()),
            "macro_mean_random": float(class_means["random_mean"].mean()),
            "macro_mean_delta": float(class_means["delta_mean"].mean()),
            "wilcoxon_p_pooled": _wilcoxon_p(df[agent_col].values, df[random_col].values),
        }
        if svt is not None:
            ov["selected_view_type"] = svt
        overall_rows.append(ov)

    sort_cols = ["epoch_start", "epoch_end", "dataset"] if has_epochs else ["dataset"]
    per_class = pd.DataFrame(per_class_rows).sort_values(sort_cols).reset_index(drop=True)
    return per_class, overall_rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Concat ALL input CSVs into one frame first, then summarise per
    # (selected_view_type, model). This lets shard CSVs merge correctly while
    # keeping different --selected_view_type runs separate.
    dfs = []
    for csv_path in args.csv:
        df = pd.read_csv(csv_path)
        if "model" not in df.columns or "dataset" not in df.columns:
            print(f"Skipping {csv_path}: missing 'model'/'dataset' columns", file=sys.stderr)
            continue
        df["_source_csv"] = os.path.basename(csv_path)
        dfs.append(df)
    if not dfs:
        print("No usable CSVs.", file=sys.stderr)
        sys.exit(1)
    all_df = pd.concat(dfs, ignore_index=True)
    # Deduplicate rows that may overlap if the same shard CSV was passed twice.
    dedup_cols = [c for c in
                  ("trial", "model", "epoch_start", "epoch_end", "selected_view_type")
                  if c in all_df.columns]
    if dedup_cols:
        before = len(all_df)
        all_df = all_df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
        if before != len(all_df):
            print(f"Dropped {before - len(all_df)} duplicate rows during merge.")

    has_svt = "selected_view_type" in all_df.columns
    svts = sorted(all_df["selected_view_type"].dropna().unique().tolist()) if has_svt else [None]

    all_per_class = []
    all_overall = []
    for svt in svts:
        block = all_df if svt is None else all_df[all_df["selected_view_type"] == svt]
        models = block["model"].dropna().unique().tolist()
        for model in models:
            sub = block[block["model"] == model]
            metrics = args.metrics or DEFAULT_METRICS.get(model, [])
            if not metrics:
                print(f"  No default metrics for model={model}; pass --metrics.", file=sys.stderr)
                continue
            for metric in metrics:
                per_class, overall_rows = _summarize(sub, model, metric)
                if per_class is None:
                    print(f"  Skip {model}/{metric}: columns missing.", file=sys.stderr)
                    continue
                all_per_class.append(per_class)
                all_overall.extend(overall_rows)
                for ov in overall_rows:
                    tag = (f"epochs {ov['epoch_start']}-{ov['epoch_end']}: "
                           if "epoch_start" in ov else "")
                    svt_tag = f"v{svt} " if svt else ""
                    print(f"  {svt_tag}{model}/{metric} {tag}"
                          f"{ov['n_instances_total']} instances, "
                          f"Δ={ov['instance_pooled_delta_mean']:+.4f}, "
                          f"p={ov['wilcoxon_p_pooled']:.3g}")

    if not all_per_class:
        print("Nothing aggregated.", file=sys.stderr)
        sys.exit(1)

    per_class_out = pd.concat(all_per_class, ignore_index=True)
    overall_out = pd.DataFrame(all_overall)

    per_class_path = os.path.join(args.output_dir, "per_class_summary.csv")
    overall_path = os.path.join(args.output_dir, "overall_summary.csv")
    per_class_out.to_csv(per_class_path, index=False)
    overall_out.to_csv(overall_path, index=False)
    print(f"\nSaved: {per_class_path} ({len(per_class_out)} rows)")
    print(f"Saved: {overall_path} ({len(overall_out)} rows)")


if __name__ == "__main__":
    main()
