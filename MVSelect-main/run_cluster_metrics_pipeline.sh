#!/usr/bin/env bash
# Two-step cluster-metrics pipeline:
#   1) compute_cluster_metrics.py — walks meta_logs/ and writes
#      cluster_metrics.csv into every experiment folder with feature dumps.
#   2) aggregate_cluster_metrics.py — overlays the experiments listed in
#      the EXPS array below into shared comparison plots.
#
# Both steps respect cluster_metrics.csv → re-running step 1 is a no-op
# unless OVERWRITE=1.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Comparison configuration — edit these
# ---------------------------------------------------------------------------
COMPARISON_NAME="${COMPARISON_NAME:-cluster_metrics_freeze_sweep}"
DATASET="${DATASET:-rgb}"

# Mirror the layout used by other aggregators: each entry resolves to
# meta_logs/${DATASET}/<exp_folder>/.
EXPS=(
    "resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:no_freeze"
    "freeze_10_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_10"
    "freeze_20_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_20"
    "freeze_30_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_30"
    "freeze_40_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_40"
    "freeze_50_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_50"
)

# Compute-step knobs
NUM_RUNS="${NUM_RUNS:-}"             # cap on runs aggregated per exp; empty = all
MAX_SAMPLES="${MAX_SAMPLES:-2000}"    # downsample N for silhouette (O(n²) cost)
FEATURE_DIM="${FEATURE_DIM:-512}"
OVERWRITE="${OVERWRITE:-0}"          # 1 = recompute even if cluster_metrics.csv exists
EVERY_N_EPOCHS="${EVERY_N_EPOCHS:-10}"  # only use epochs where epoch %% N == 0
                                        #   default 10 = matches the "I dumped every
                                        #   epoch before 20 then every 10 afterwards"
                                        #   cadence; set to 1 (or 0) to disable.

# Aggregate-step knobs
METRICS="${METRICS:-separability silhouette_class silhouette_view}"
STYLE="${STYLE:-heatmap}"            # line | heatmap | sorted_bars | rank_stacked | both | all
BIN_EPOCHS="${BIN_EPOCHS:-0}"
SMOOTH="${SMOOTH:-1}"
TITLE_SUFFIX="${TITLE_SUFFIX:-}"

OUTPUT_DIR="${ROOT_DIR}/compare/${COMPARISON_NAME}"

# ---------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"
echo "=========================================="
echo "Cluster-metrics pipeline:"
echo "  COMPARISON  = ${COMPARISON_NAME}"
echo "  DATASET     = ${DATASET}"
echo "  MAX_SAMPLES = ${MAX_SAMPLES}"
echo "  METRICS     = ${METRICS}"
echo "  STYLE       = ${STYLE}"
echo "  EVERY_N_EP  = ${EVERY_N_EPOCHS}"
echo "  OUTPUT      = ${OUTPUT_DIR}"
echo "  EXPS (${#EXPS[@]}):"
for e in "${EXPS[@]}"; do echo "    - ${e}"; done
echo "=========================================="

# --- Step 1: compute per-experiment metrics (idempotent; uses --overwrite if set) ---
echo "[1/2] Computing per-experiment cluster_metrics.csv ..."
compute_args=(
    --root "${ROOT_DIR}/meta_logs"
    --rep_list "${DATASET}"
    --max_samples "${MAX_SAMPLES}"
    --feature_dim "${FEATURE_DIM}"
)
[ -n "${NUM_RUNS}" ] && compute_args+=(--num_runs "${NUM_RUNS}")
[ "${OVERWRITE}" = "1" ] && compute_args+=(--overwrite)
[ "${EVERY_N_EPOCHS}" != "0" ] && [ "${EVERY_N_EPOCHS}" != "1" ] && \
    compute_args+=(--every_n_epochs "${EVERY_N_EPOCHS}")
python3 "${ROOT_DIR}/compute_cluster_metrics.py" "${compute_args[@]}"

# --- Step 2: overlay across the EXPS list ---
echo
echo "[2/2] Aggregating across experiments ..."
exp_args=()
for spec in "${EXPS[@]}"; do
    if [[ "${spec}" == *:* ]]; then
        folder="${spec%%:*}"
        label="${spec##*:}"
    else
        folder="${spec}"
        label="${spec}"
    fi
    exp_path="${ROOT_DIR}/meta_logs/${DATASET}/${folder}"
    if [ ! -f "${exp_path}/cluster_metrics.csv" ]; then
        echo "  WARNING: no cluster_metrics.csv for ${label} (expected ${exp_path}/cluster_metrics.csv)"
    fi
    exp_args+=(--exp "${exp_path}:${label}")
done

agg_extra=()
[ -n "${TITLE_SUFFIX}" ] && agg_extra+=(--title_suffix "${TITLE_SUFFIX}")
agg_extra+=(--style "${STYLE}")
[ "${BIN_EPOCHS}" != "0" ] && agg_extra+=(--bin_epochs "${BIN_EPOCHS}")
[ "${SMOOTH}" != "1" ] && agg_extra+=(--smooth "${SMOOTH}")
[ "${EVERY_N_EPOCHS}" != "0" ] && [ "${EVERY_N_EPOCHS}" != "1" ] && \
    agg_extra+=(--every_n_epochs "${EVERY_N_EPOCHS}")
agg_extra+=(--metrics ${METRICS})

python3 "${ROOT_DIR}/aggregate_cluster_metrics.py" \
    --output_dir "${OUTPUT_DIR}" \
    "${exp_args[@]}" \
    "${agg_extra[@]}"

echo "=========================================="
echo "Done. Plots + aggregated.csv in ${OUTPUT_DIR}"
echo "=========================================="
