#!/usr/bin/env bash
# Parallel multi-GPU launcher for view_type_single_view_confidence.py.
#
# Defaults: 4 GPUs, VGGT only, 5 runs, 25 instances per class.
# Override via env vars:
#   GPUS=2 MODELS="vggt pi3" N_RUNS=10 ./scripts/run_view_type_single_view.sh
#   PER_CLS=5 LIMIT=10 ./scripts/run_view_type_single_view.sh         # smoke test

set -euo pipefail

GPUS="${GPUS:-4}"
MODELS="${MODELS:-vggt}"
N_RUNS="${N_RUNS:-5}"
PER_CLS="${PER_CLS:-25}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/single_view}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

LIMIT_FLAG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

echo "=========================================="
echo "Single-view confidence pipeline:"
echo "  GPUS     = ${GPUS}"
echo "  MODELS   = ${MODELS}"
echo "  N_RUNS   = ${N_RUNS}"
echo "  PER_CLS  = ${PER_CLS}"
echo "  SPLIT    = ${SPLIT}"
echo "  LIMIT    = ${LIMIT:-(none)}"
echo "  OUTPUT   = ${OUTPUT_DIR}"
echo "=========================================="

# --- Step 1: launch N sharded evals in parallel ---
echo "[1/2] Launching ${GPUS} shard(s)..."
pids=()
for i in $(seq 0 $((GPUS - 1))); do
    log_file="${LOG_DIR}/shard${i}.log"
    echo "  shard ${i}/${GPUS} -> GPU ${i}, log: ${log_file}"
    CUDA_VISIBLE_DEVICES=${i} python3 "${ROOT_DIR}/scripts/view_type_single_view_confidence.py" \
        --models ${MODELS} \
        --gpu_id 0 \
        --n_runs "${N_RUNS}" \
        --per_cls_instances "${PER_CLS}" \
        --split "${SPLIT}" \
        --output_dir "${OUTPUT_DIR}" \
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
    echo "One or more shards failed. Aborting."
    exit 1
fi

# --- Step 2: merge shard CSVs and plot ---
echo "[2/2] Merging shard CSVs and plotting..."
shopt -s nullglob
if [ "${GPUS}" -eq 1 ]; then
    csvs=( "${OUTPUT_DIR}"/*_single_view_confidence.csv )
else
    csvs=( "${OUTPUT_DIR}"/*_single_view_confidence_shard*.csv )
fi
shopt -u nullglob
if [ ${#csvs[@]} -eq 0 ]; then
    echo "ERROR: no shard CSVs found in ${OUTPUT_DIR}"
    exit 1
fi
python3 "${ROOT_DIR}/scripts/view_type_single_view_confidence.py" \
    --from_csv "${csvs[@]}" \
    --n_runs "${N_RUNS}" \
    --output_dir "${OUTPUT_DIR}"

echo "=========================================="
echo "Done."
echo "  CSVs + plots: ${OUTPUT_DIR}/"
echo "=========================================="
