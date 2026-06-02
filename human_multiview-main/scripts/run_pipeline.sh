#!/usr/bin/env bash
# Full pipeline: parallel multi-GPU evaluation -> aggregate -> plot.
#
# Override any setting via env vars, e.g.:
#   GPUS=4 VIEW_TYPE=01 ./scripts/run_pipeline.sh
#   LIMIT=5 ./scripts/run_pipeline.sh                 # smoke test
#   MODELS="vggt" GPUS=2 ./scripts/run_pipeline.sh
#
# Defaults reproduce the 10-epoch-group VGGT+DINOv2 run across all 5 view buckets.

set -euo pipefail

# --- Config (override via env vars) ---
GPUS="${GPUS:-4}"
MODELS="${MODELS:-vggt dinov2}"
VIEW_TYPE="${VIEW_TYPE:-01234}"
EPOCH_GROUPS="${EPOCH_GROUPS:-1-10,11-20,21-30,31-40,41-50,51-60,61-70,71-80,81-90,91-100}"
NUM_CAM="${NUM_CAM:-5}"
SPLIT="${SPLIT:-test}"
PER_CLS="${PER_CLS:-5}"
LIMIT="${LIMIT:-}"

# --- Derived paths ---
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/results/views/v${VIEW_TYPE}"
SUMMARY_DIR="${OUTPUT_DIR}/summary"
PLOTS_DIR="${OUTPUT_DIR}/plots"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${SUMMARY_DIR}" "${PLOTS_DIR}" "${LOG_DIR}"

LIMIT_FLAG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

echo "=========================================="
echo "Pipeline config:"
echo "  GPUS         = ${GPUS}"
echo "  MODELS       = ${MODELS}"
echo "  VIEW_TYPE    = ${VIEW_TYPE}"
echo "  EPOCH_GROUPS = ${EPOCH_GROUPS}"
echo "  NUM_CAM      = ${NUM_CAM}"
echo "  SPLIT        = ${SPLIT}"
echo "  PER_CLS      = ${PER_CLS}"
echo "  LIMIT        = ${LIMIT:-(none)}"
echo "  Output dir   = ${OUTPUT_DIR}"
echo "=========================================="

# --- Step 1: Launch N sharded evaluations in parallel ---
echo "[1/3] Launching ${GPUS} shard(s)..."
pids=()
for i in $(seq 0 $((GPUS - 1))); do
    log_file="${LOG_DIR}/shard${i}.log"
    echo "  shard ${i}/${GPUS} -> GPU ${i}, log: ${log_file}"
    CUDA_VISIBLE_DEVICES=${i} python3 "${ROOT_DIR}/scripts/run_evaluation_views.py" \
        --models ${MODELS} \
        --gpu_id 0 \
        --selected_view_type "${VIEW_TYPE}" \
        --epoch_groups "${EPOCH_GROUPS}" \
        --num_cam "${NUM_CAM}" \
        --split "${SPLIT}" \
        --per_cls_instances "${PER_CLS}" \
        --shard "${i}/${GPUS}" \
        ${LIMIT_FLAG} \
        > "${log_file}" 2>&1 &
    pids+=($!)
done

echo "  Waiting for shards (tail -f ${LOG_DIR}/shard*.log to watch progress)..."
fail=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    if ! wait "${pid}"; then
        echo "  ERROR: shard ${idx} (PID ${pid}) failed. See ${LOG_DIR}/shard${idx}.log"
        fail=1
    else
        echo "  shard ${idx} done."
    fi
done
if [ "${fail}" -ne 0 ]; then
    echo "One or more shards failed. Aborting pipeline."
    exit 1
fi

# --- Step 2: Aggregate shard CSVs ---
echo "[2/3] Aggregating shard CSVs..."
shopt -s nullglob
if [ "${GPUS}" -eq 1 ]; then
    csvs=( "${OUTPUT_DIR}"/*_views.csv )
else
    csvs=( "${OUTPUT_DIR}"/*_views_shard*.csv )
fi
shopt -u nullglob
if [ ${#csvs[@]} -eq 0 ]; then
    echo "ERROR: no shard CSVs found in ${OUTPUT_DIR}"
    exit 1
fi
echo "  Merging ${#csvs[@]} CSV(s)..."
python3 "${ROOT_DIR}/scripts/aggregate_views.py" \
    --csv "${csvs[@]}" \
    --output_dir "${SUMMARY_DIR}"

# --- Step 3: Plot epoch trends ---
echo "[3/3] Generating plots..."
python3 "${ROOT_DIR}/scripts/plot_epoch_trends.py" \
    --summary "${SUMMARY_DIR}/per_class_summary.csv" \
    --output_dir "${PLOTS_DIR}"

echo "=========================================="
echo "Done."
echo "  CSVs:     ${OUTPUT_DIR}/"
echo "  Summary:  ${SUMMARY_DIR}/"
echo "  Plots:    ${PLOTS_DIR}/"
echo "=========================================="
