#!/usr/bin/env bash
# Train active single-view recognizers; optionally train a matched pair control.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${ROOT_DIR}/experiment_arch.sh"
cd "${ROOT_DIR}"

DATASET="${DATASET:-rgb}"
SEEDS="${SEEDS:-0 1 2 3 4}"
PAIR_GPU="${PAIR_GPU:-1}"
SINGLE_GPU="${SINGLE_GPU:-0}"
EPOCHS="${EPOCHS:-100}"
NUM_TRAIN_INSTANCES="${NUM_TRAIN_INSTANCES:-25}"
BATCH_SIZE="${BATCH_SIZE:-6}"
LR="${LR:-0.0005}"
SELECT_LR="${SELECT_LR:-0.0001}"
SELECT_WD="${SELECT_WD:-0.0001}"
BASE_LR_RATIO="${BASE_LR_RATIO:-1.0}"
OTHER_LR_RATIO="${OTHER_LR_RATIO:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
AGGREGATION="${AGGREGATION:-max}"
DETERMINISTIC="${DETERMINISTIC:-true}"
SAVE_FEATURE="${SAVE_FEATURE:-0}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-10}"
TRAIN_PAIR="${TRAIN_PAIR:-0}"
TRAIN_SINGLE="${TRAIN_SINGLE:-1}"
INCLUDE_PAIR="${INCLUDE_PAIR:-${TRAIN_PAIR}}"

PAIR_EXPERIMENT="${PAIR_EXPERIMENT:-${ARCH}steps1_train_ins${NUM_TRAIN_INSTANCES}_lr${LR}base${BASE_LR_RATIO}other${OTHER_LR_RATIO}select_wd${SELECT_WD}select${SELECT_LR}_e${EPOCHS}}"
SINGLE_EXPERIMENT="${SINGLE_EXPERIMENT:-${ARCH}steps1_active_single_train_ins${NUM_TRAIN_INSTANCES}_lr${LR}base${BASE_LR_RATIO}other${OTHER_LR_RATIO}select_wd${SELECT_WD}select${SELECT_LR}_e${EPOCHS}}"
PAIR_DIR="${ROOT_DIR}/meta_logs/${DATASET}/${PAIR_EXPERIMENT}"
SINGLE_DIR="${ROOT_DIR}/meta_logs/${DATASET}/${SINGLE_EXPERIMENT}"

common_args=(
    --arch "${ARCH}"
    --dataset "${DATASET}"
    --epochs "${EPOCHS}"
    --steps 1
    --non_roll
    --num_train_instances "${NUM_TRAIN_INSTANCES}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --select_lr "${SELECT_LR}"
    --select_wd "${SELECT_WD}"
    --base_lr_ratio "${BASE_LR_RATIO}"
    --other_lr_ratio "${OTHER_LR_RATIO}"
    --weight_decay "${WEIGHT_DECAY}"
    --aggregation "${AGGREGATION}"
    --deterministic "${DETERMINISTIC}"
    --save_every_epoch "${SAVE_EVERY_EPOCH}"
    --skip_stage1
)
[ "${SAVE_FEATURE}" = "1" ] && common_args+=(--save_feature)

echo "=========================================="
echo "Active single-view experiment"
echo "  ARCH/REP       = ${ARCH}/${DATASET}"
echo "  SEEDS          = ${SEEDS}"
echo "  EPOCHS         = ${EPOCHS}"
echo "  SINGLE GPU     = ${SINGLE_GPU}"
echo "  TRAIN PAIR CTRL = ${TRAIN_PAIR}"
if [ "${TRAIN_PAIR}" = "1" ]; then
    echo "  PAIR GPU       = ${PAIR_GPU}"
fi
echo "  INITIALIZATION  = matched ImageNet backbone; classifier + selector joint"
echo "  SAVE FEATURE   = ${SAVE_FEATURE}"
echo "  PAIR META      = ${PAIR_DIR}"
echo "  SINGLE META    = ${SINGLE_DIR}"
echo "=========================================="

run_pair() {
    local seed="$1"
    echo "[GPU ${PAIR_GPU}] seed ${seed}: active pair (initial + selected)"
    CUDA_VISIBLE_DEVICES="${PAIR_GPU}" PYTHONUNBUFFERED=1 \
        python3 "${ROOT_DIR}/main.py" "${common_args[@]}" --seed "${seed}"
}

run_single() {
    local seed="$1"
    echo "[GPU ${SINGLE_GPU}] seed ${seed}: active single (selected only)"
    CUDA_VISIBLE_DEVICES="${SINGLE_GPU}" PYTHONUNBUFFERED=1 \
        python3 "${ROOT_DIR}/main.py" "${common_args[@]}" \
        --seed "${seed}" --active_single_view
}

for seed in ${SEEDS}; do
    if [ "${TRAIN_PAIR}" = "1" ] && [ "${TRAIN_SINGLE}" = "1" ] && \
       [ "${PAIR_GPU}" != "${SINGLE_GPU}" ]; then
        run_pair "${seed}" &
        pair_pid=$!
        run_single "${seed}" &
        single_pid=$!
        failed=0
        wait "${pair_pid}" || failed=1
        wait "${single_pid}" || failed=1
        if [ "${failed}" -ne 0 ]; then
            echo "ERROR: at least one condition failed for seed ${seed}."
            exit 1
        fi
    else
        if [ "${TRAIN_PAIR}" = "1" ]; then
            run_pair "${seed}"
        fi
        if [ "${TRAIN_SINGLE}" = "1" ]; then
            run_single "${seed}"
        fi
    fi
done

ARCH="${ARCH}" DATASET="${DATASET}" \
INCLUDE_PAIR="${INCLUDE_PAIR}" \
PAIR_EXPERIMENT="${PAIR_EXPERIMENT}" \
SINGLE_EXPERIMENT="${SINGLE_EXPERIMENT}" \
"${ROOT_DIR}/run_active_single_view_comparison.sh"
