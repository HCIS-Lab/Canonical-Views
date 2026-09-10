#!/usr/bin/env bash
# Replay per-epoch selector scores for one or more exact experiment settings.
# Timestamped training runs sharing the exact experiment name are aggregated;
# different freeze/selector-limit names are never mixed.
#
# Examples:
#   GPU_ID=0 ./run_selector_score_analysis.sh
#   EXPERIMENTS="freeze_10_resnet18steps5_..._e100" GPU_ID=1 \
#       ./run_selector_score_analysis.sh
#   EXPERIMENTS="resnet18steps5_..._e100 resnet18steps5_selview_remainder_..._e100" \
#       GPU_ID=2 ./run_selector_score_analysis.sh
#   EPOCHS="10 50 100" NUM_INITIAL_CAMS=10 LIMIT_TRIALS=40 \
#       ./run_selector_score_analysis.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${ROOT_DIR}/experiment_arch.sh"
cd "${ROOT_DIR}"

DEFAULT_EXPERIMENT="${MVSELECT_BASE_EXPERIMENT}"
EXPERIMENTS="${EXPERIMENTS:-${DEFAULT_EXPERIMENT}}"
GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-rgb}"
DATA_ROOT="${DATA_ROOT:-/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23}"
EPOCH_STRIDE="${EPOCH_STRIDE:-10}"
EPOCHS="${EPOCHS:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
RUN_LIMIT="${RUN_LIMIT:-}"
NUM_INITIAL_CAMS="${NUM_INITIAL_CAMS:-10}"
INITIAL_CAMS="${INITIAL_CAMS:-}"
LIMIT_TRIALS="${LIMIT_TRIALS:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
METRICS="${METRICS:-}"
NO_SAVE_RAW="${NO_SAVE_RAW:-0}"
ALLOW_REPLAY_MISMATCH="${ALLOW_REPLAY_MISMATCH:-0}"
FORCE_RECOMPUTE_CACHE="${FORCE_RECOMPUTE_CACHE:-0}"

if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU_ID must be a non-negative integer."
    exit 1
fi

read -r -a experiment_list <<< "${EXPERIMENTS}"
if [ "${#experiment_list[@]}" -eq 0 ]; then
    echo "ERROR: EXPERIMENTS is empty."
    exit 1
fi

args=(
    --dataset "${DATASET}"
    --arch "${ARCH}"
    --data_root "${DATA_ROOT}"
    --epoch_stride "${EPOCH_STRIDE}"
    --num_initial_cams "${NUM_INITIAL_CAMS}"
    --batch_size "${BATCH_SIZE}"
    --gpu_id 0
)
for experiment in "${experiment_list[@]}"; do
    args+=(--experiment "${experiment}")
done
if [ -n "${EPOCHS}" ]; then
    read -r -a epoch_list <<< "${EPOCHS}"
    args+=(--epochs "${epoch_list[@]}")
fi
if [ -n "${MAX_EPOCHS}" ]; then
    args+=(--max_epochs "${MAX_EPOCHS}")
fi
if [ -n "${RUN_LIMIT}" ]; then
    args+=(--run_limit "${RUN_LIMIT}")
fi
if [ -n "${INITIAL_CAMS}" ]; then
    read -r -a initial_cam_list <<< "${INITIAL_CAMS}"
    args+=(--initial_cams "${initial_cam_list[@]}")
fi
if [ -n "${LIMIT_TRIALS}" ]; then
    args+=(--limit_trials "${LIMIT_TRIALS}")
fi
if [ -n "${METRICS}" ]; then
    read -r -a metric_list <<< "${METRICS}"
    args+=(--metrics "${metric_list[@]}")
fi
if [ "${NO_SAVE_RAW}" = "1" ]; then
    args+=(--no_save_raw)
fi
if [ "${ALLOW_REPLAY_MISMATCH}" = "1" ]; then
    args+=(--allow_replay_mismatch)
fi
if [ "${FORCE_RECOMPUTE_CACHE}" = "1" ]; then
    args+=(--force_recompute_cache)
fi

echo "=========================================="
echo "Selector-score analysis:"
echo "  DATASET       = ${DATASET}"
echo "  ARCH          = ${ARCH}"
echo "  GPU_ID        = ${GPU_ID}"
echo "  EPOCHS        = ${EPOCHS:-every ${EPOCH_STRIDE}}"
echo "  INIT_CAMS     = ${INITIAL_CAMS:-${NUM_INITIAL_CAMS} evenly spaced}"
echo "  RUN_LIMIT     = ${RUN_LIMIT:-(all matching timestamped runs)}"
echo "  EXPERIMENTS:"
for experiment in "${experiment_list[@]}"; do
    echo "    - ${experiment}"
done
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
    python3 "${ROOT_DIR}/selector_score_analysis.py" "${args[@]}"

echo "Done. Each experiment was saved to:"
for experiment in "${experiment_list[@]}"; do
    echo "  meta_logs/${DATASET}/${experiment}/selector_score"
done
