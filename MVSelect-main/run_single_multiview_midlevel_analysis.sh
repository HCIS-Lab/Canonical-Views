#!/usr/bin/env bash
# Compare no-freeze N=1, random N=5, and agent-selected N=5 inputs.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${ROOT_DIR}/experiment_arch.sh"
cd "${ROOT_DIR}"

DATASET="${DATASET:-rgb}"
EXPERIMENT="${EXPERIMENT:-${MVSELECT_BASE_EXPERIMENT}}"
DATA_ROOT="${DATA_ROOT:-/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23}"
CACHE_CSV="${CACHE_CSV:-${ROOT_DIR}/cache/midlevel_features_v2_test.csv}"
GPU_ID="${GPU_ID:-0}"
SELECTION_EPOCH="${SELECTION_EPOCH:-100}"
RANDOM_SETS="${RANDOM_SETS:-100}"
RANDOM_SEED="${RANDOM_SEED:-42}"
NUM_INITIAL_CAMS="${NUM_INITIAL_CAMS:-10}"
INITIAL_CAMS="${INITIAL_CAMS:-}"
LIMIT_INSTANCES="${LIMIT_INSTANCES:-}"
CV_FOLDS="${CV_FOLDS:-5}"
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PLOT_ONLY="${PLOT_ONLY:-0}"
FAMILY_MEANS_ONLY="${FAMILY_MEANS_ONLY:-0}"

if [[ "${EXPERIMENT}" = /* ]]; then
    EXP_DIR="${EXPERIMENT}"
else
    EXP_DIR="${ROOT_DIR}/meta_logs/${DATASET}/${EXPERIMENT}"
fi
if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${EXP_DIR}/single_multiview_midlevel"
fi
mkdir -p "${OUTPUT_DIR}"
RUN_LOG="${OUTPUT_DIR}/run.log"

extra=(--non_roll)
[ -n "${CHECKPOINT}" ] && extra+=(--checkpoint "${CHECKPOINT}")
[ -n "${LIMIT_INSTANCES}" ] && extra+=(--limit_instances "${LIMIT_INSTANCES}")
[ "${PLOT_ONLY}" = "1" ] && extra+=(--plot_only)
[ "${FAMILY_MEANS_ONLY}" = "1" ] && extra+=(--family_means_only)
if [ -n "${INITIAL_CAMS}" ]; then
    read -r -a camera_values <<< "${INITIAL_CAMS}"
    extra+=(--initial_cams "${camera_values[@]}")
else
    extra+=(--num_initial_cams "${NUM_INITIAL_CAMS}")
fi

echo "=========================================="
echo "Single/multiview mid-level analysis (no_freeze):"
echo "  EXPERIMENT    = ${EXP_DIR}"
echo "  GPU_ID        = ${GPU_ID}"
echo "  EPOCH         = ${SELECTION_EPOCH}"
echo "  RANDOM N=5    = ${RANDOM_SETS} sets per instance"
echo "  INITIAL CAMS  = ${INITIAL_CAMS:-${NUM_INITIAL_CAMS} evenly spaced}"
echo "  OUTPUT        = ${OUTPUT_DIR}"
echo "  PLOT ONLY     = ${PLOT_ONLY}"
echo "  FAMILY MEANS  = ${FAMILY_MEANS_ONLY}"
echo "=========================================="

if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
    python3 "${ROOT_DIR}/single_multiview_midlevel_analysis.py" \
        --selection_dir "${EXP_DIR}" \
        --selection_epoch "${SELECTION_EPOCH}" \
        --dataset "${DATASET}" \
        --arch "${ARCH}" \
        --data_root "${DATA_ROOT}" \
        --cache_csv "${CACHE_CSV}" \
        --gpu_id 0 \
        --random_sets_per_instance "${RANDOM_SETS}" \
        --random_seed "${RANDOM_SEED}" \
        --cv_folds "${CV_FOLDS}" \
        --output_dir "${OUTPUT_DIR}" \
        "${extra[@]}" \
        > "${RUN_LOG}" 2>&1; then
    echo "ERROR: analysis failed; see ${RUN_LOG}"
    exit 1
fi

echo "Done. Outputs: ${OUTPUT_DIR}"
echo "Log: ${RUN_LOG}"
