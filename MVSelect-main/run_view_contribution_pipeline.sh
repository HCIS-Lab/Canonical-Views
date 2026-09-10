#!/usr/bin/env bash
# Run leave-one-out and selected-to-unselected replacement analysis for the
# no-freeze experiment.
#
# Defaults:
#   GPU_ID=0
#   EPOCH_STRIDE=10             evaluate epochs 10,20,...,100
#   NUM_INITIAL_CAMS=10         evenly spaced initial-camera rollouts
#   CONTEXT=agent_only          omit the initial input from S
#   REPLACEMENT_SAMPLES=5       shared unselected samples per rollout
#   PER_EPOCH_CHECKPOINT=0      hold the final classifier fixed
#
# Examples:
#   GPU_ID=2 ./run_view_contribution_pipeline.sh
#   EPOCHS="20 100" LIMIT_TRIALS=20 ./run_view_contribution_pipeline.sh
#   REPLACEMENT_SAMPLES=10 NUM_INITIAL_CAMS=0 GPU_ID=1 \
#       ./run_view_contribution_pipeline.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${ROOT_DIR}/experiment_arch.sh"
cd "${ROOT_DIR}"

DATASET="${DATASET:-rgb}"
SPLIT="${SPLIT:-test}"
EXPERIMENT="${EXPERIMENT:-${MVSELECT_BASE_EXPERIMENT}}"
DATA_ROOT="${DATA_ROOT:-/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23}"
CACHE_CSV="${CACHE_CSV:-${ROOT_DIR}/cache/midlevel_features_v2_${SPLIT}.csv}"
GPU_ID="${GPU_ID:-0}"
EPOCH_STRIDE="${EPOCH_STRIDE:-10}"
EPOCHS="${EPOCHS:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
LIMIT_TRIALS="${LIMIT_TRIALS:-}"
NUM_INITIAL_CAMS="${NUM_INITIAL_CAMS:-10}"
INITIAL_CAMS="${INITIAL_CAMS:-}"
CONTEXT="${CONTEXT:-agent_only}"
CORRELATION="${CORRELATION:-spearman}"
REPLACEMENT_SAMPLES="${REPLACEMENT_SAMPLES:-5}"
REPLACEMENT_SEED="${REPLACEMENT_SEED:-42}"
PER_EPOCH_CHECKPOINT="${PER_EPOCH_CHECKPOINT:-0}"
FORCE_RECOMPUTE_CACHE="${FORCE_RECOMPUTE_CACHE:-0}"
NON_ROLL="${NON_ROLL:-1}"
NON_LIKE="${NON_LIKE:-0}"
TEST_PER_CLS_INSTANCES="${TEST_PER_CLS_INSTANCES:-5}"
AGGREGATION="${AGGREGATION:-max}"
CHECKPOINT="${CHECKPOINT:-}"

if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU_ID must be a non-negative integer."
    exit 1
fi
if ! [[ "${REPLACEMENT_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: REPLACEMENT_SAMPLES must be a positive integer."
    exit 1
fi

if [[ "${EXPERIMENT}" = /* ]]; then
    EXP_DIR="${EXPERIMENT}"
else
    EXP_DIR="${ROOT_DIR}/meta_logs/${DATASET}/${EXPERIMENT}"
fi
if [ "${PER_EPOCH_CHECKPOINT}" = "1" ]; then
    PROTOCOL_TAG="classifier_at_t"
else
    PROTOCOL_TAG="final_classifier"
fi
if [ -n "${INITIAL_CAMS}" ]; then
    CAMERA_TAG="cams_${INITIAL_CAMS// /-}"
elif [ "${NUM_INITIAL_CAMS}" = "0" ]; then
    CAMERA_TAG="cams_all"
else
    CAMERA_TAG="cams_${NUM_INITIAL_CAMS}"
fi
ANALYSIS_TAG="${ANALYSIS_TAG:-${CONTEXT}_${PROTOCOL_TAG}_${CAMERA_TAG}_repl${REPLACEMENT_SAMPLES}_seed${REPLACEMENT_SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXP_DIR}/view_contribution/${ANALYSIS_TAG}}"
RUN_LOG="${OUTPUT_DIR}/run.log"

if [ ! -d "${EXP_DIR}" ]; then
    echo "ERROR: no no-freeze experiment directory at ${EXP_DIR}"
    exit 1
fi
if ! compgen -G "${EXP_DIR}/*_selection.json" >/dev/null; then
    echo "ERROR: no *_selection.json in ${EXP_DIR}"
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "View-contribution analysis (no_freeze):"
echo "  EXPERIMENT    = ${EXP_DIR}"
echo "  GPU_ID        = ${GPU_ID}"
echo "  EPOCHS        = ${EPOCHS:-every ${EPOCH_STRIDE}}"
echo "  INIT_CAMS     = ${INITIAL_CAMS:-${NUM_INITIAL_CAMS} evenly spaced}"
echo "  CONTEXT       = ${CONTEXT}"
echo "  REPLACEMENTS  = ${REPLACEMENT_SAMPLES} per rollout"
echo "  REPLACE_SEED  = ${REPLACEMENT_SEED}"
echo "  ANALYSIS_TAG  = ${ANALYSIS_TAG}"
echo "  CORRELATION   = ${CORRELATION}"
echo "  PROTOCOL      = $([ "${PER_EPOCH_CHECKPOINT}" = "1" ] && echo 'classifier-at-epoch-t' || echo 'final classifier fixed')"
echo "  OUTPUT        = ${OUTPUT_DIR}"
echo "=========================================="

if [ ! -f "${CACHE_CSV}" ] || [ "${FORCE_RECOMPUTE_CACHE}" = "1" ]; then
    echo "[1/2] Building shared mid-level feature cache..."
    cache_extra=()
    [ "${FORCE_RECOMPUTE_CACHE}" = "1" ] && cache_extra+=(--force_recompute)
    python3 "${ROOT_DIR}/midlevel_shape_features.py" \
        --data_root "${DATA_ROOT}" \
        --split "${SPLIT}" \
        --cache_csv "${CACHE_CSV}" \
        --output_dir "${ROOT_DIR}/cache/midlevel_cache_build" \
        "${cache_extra[@]}"
else
    echo "[1/2] Using shared mid-level feature cache: ${CACHE_CSV}"
fi

extra=()
if [ -n "${EPOCHS}" ]; then
    read -r -a explicit_epochs <<< "${EPOCHS}"
    extra+=(--epochs "${explicit_epochs[@]}")
else
    extra+=(--epoch_stride "${EPOCH_STRIDE}")
fi
[ -n "${MAX_EPOCHS}" ] && extra+=(--max_epochs "${MAX_EPOCHS}")
[ -n "${LIMIT_TRIALS}" ] && extra+=(--limit_trials "${LIMIT_TRIALS}")
[ "${PER_EPOCH_CHECKPOINT}" = "1" ] && extra+=(--per_epoch_checkpoint)
[ "${NON_ROLL}" = "1" ] && extra+=(--non_roll)
[ "${NON_LIKE}" = "1" ] && extra+=(--non_like)
[ -n "${CHECKPOINT}" ] && extra+=(--checkpoint "${CHECKPOINT}")
extra+=(--test_per_cls_instances "${TEST_PER_CLS_INSTANCES}")
extra+=(--context "${CONTEXT}")
if [ -n "${INITIAL_CAMS}" ]; then
    read -r -a explicit_initial_cams <<< "${INITIAL_CAMS}"
    extra+=(--initial_cams "${explicit_initial_cams[@]}")
else
    extra+=(--num_initial_cams "${NUM_INITIAL_CAMS}")
fi

echo "[2/2] Running leave-one-out and replacement interventions..."
if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
    python3 "${ROOT_DIR}/view_contribution_analysis.py" \
        --selection_dir "${EXP_DIR}" \
        --dataset "${DATASET}" \
        --arch "${ARCH}" \
        --aggregation "${AGGREGATION}" \
        --data_root "${DATA_ROOT}" \
        --split "${SPLIT}" \
        --cache_csv "${CACHE_CSV}" \
        --gpu_id 0 \
        --correlation "${CORRELATION}" \
        --replacement_samples "${REPLACEMENT_SAMPLES}" \
        --replacement_seed "${REPLACEMENT_SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        "${extra[@]}" \
        > "${RUN_LOG}" 2>&1; then
    echo "ERROR: analysis failed; see ${RUN_LOG}"
    exit 1
fi

echo "Done. Outputs: ${OUTPUT_DIR}"
echo "Log: ${RUN_LOG}"
