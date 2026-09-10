#!/usr/bin/env bash
# Plot active single-view bias; optionally include a matched active-pair control.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${ROOT_DIR}/experiment_arch.sh"
cd "${ROOT_DIR}"

DATASET="${DATASET:-rgb}"
PAIR_EXPERIMENT="${PAIR_EXPERIMENT:-${MVSELECT_ACTIVE_PAIR_EXPERIMENT}}"
SINGLE_EXPERIMENT="${SINGLE_EXPERIMENT:-${MVSELECT_ACTIVE_SINGLE_EXPERIMENT}}"
PAIR_DIR="${PAIR_DIR:-${ROOT_DIR}/meta_logs/${DATASET}/${PAIR_EXPERIMENT}}"
SINGLE_DIR="${SINGLE_DIR:-${ROOT_DIR}/meta_logs/${DATASET}/${SINGLE_EXPERIMENT}}"
INCLUDE_PAIR="${INCLUDE_PAIR:-0}"
COMPARISON_NAME="${COMPARISON_NAME:-$(mvselect_arch_scoped_name active_single_view_bias)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/compare/${COMPARISON_NAME}}"
EPOCH_BINS="${EPOCH_BINS:-10}"

if [ ! -d "${SINGLE_DIR}" ]; then
    echo "ERROR: active single-view experiment not found: ${SINGLE_DIR}"
    echo "Run ./run_active_single_view_experiment.sh first, or override SINGLE_DIR."
    exit 1
fi
if [ "${INCLUDE_PAIR}" = "1" ] && [ ! -d "${PAIR_DIR}" ]; then
    echo "ERROR: optional active-pair experiment not found: ${PAIR_DIR}"
    exit 1
fi

echo "=========================================="
echo "Active single-view bias comparison"
echo "  ARCH        = ${ARCH}"
echo "  ACTIVE N=1  = ${SINGLE_DIR}"
echo "  INCLUDE PAIR = ${INCLUDE_PAIR}"
if [ "${INCLUDE_PAIR}" = "1" ]; then
    echo "  ACTIVE PAIR = ${PAIR_DIR}"
fi
echo "  EPOCH BINS  = ${EPOCH_BINS}"
echo "  OUTPUT      = ${OUTPUT_DIR}"
echo "=========================================="

args=(
    --single_dir "${SINGLE_DIR}"
    --epoch_bins "${EPOCH_BINS}"
    --output_dir "${OUTPUT_DIR}"
)
if [ "${INCLUDE_PAIR}" = "1" ]; then
    args+=(--pair_dir "${PAIR_DIR}")
fi

python3 "${ROOT_DIR}/compare_active_single_view_bias.py" "${args[@]}"

echo "Done. Plots and CSV files are in ${OUTPUT_DIR}"
