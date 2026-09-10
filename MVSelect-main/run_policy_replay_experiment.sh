#!/usr/bin/env bash
# Train matched recognizers under early/final/evolving/shuffled replay policies.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${ROOT_DIR}/experiment_arch.sh"
cd "${ROOT_DIR}"

DATASET="${DATASET:-rgb}"
EXPERIMENT="${EXPERIMENT:-${MVSELECT_BASE_EXPERIMENT}}"
if [[ "${EXPERIMENT}" = /* ]]; then
    SELECTION_DIR="${SELECTION_DIR:-${EXPERIMENT}}"
    EXPERIMENT_NAME="$(basename "${EXPERIMENT}")"
else
    SELECTION_DIR="${SELECTION_DIR:-${ROOT_DIR}/meta_logs/${DATASET}/${EXPERIMENT}}"
    EXPERIMENT_NAME="${EXPERIMENT}"
fi
DATA_ROOT="${DATA_ROOT:-/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/compare/policy_replay_${ARCH}/${EXPERIMENT_NAME}}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-100}"
EVAL_EVERY="${EVAL_EVERY:-10}"
NUM_VIEWS="${NUM_VIEWS:-5}"
TEST_NUM_VIEWS="${TEST_NUM_VIEWS:-5}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-5e-5}"
BASE_LR_RATIO="${BASE_LR_RATIO:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
AGGREGATION="${AGGREGATION:-max}"
TRAIN_SPLIT="${TRAIN_SPLIT:-test}"
TEST_SPLIT="${TEST_SPLIT:-test-down}"
TRAIN_PER_CLS="${TRAIN_PER_CLS:-0}"
TEST_PER_CLS="${TEST_PER_CLS:-5}"
EVAL_VIEW_SEED="${EVAL_VIEW_SEED:-2027}"
SHUFFLE_SEED="${SHUFFLE_SEED:-1729}"
INCLUDE_FAMILY_MATCHED="${INCLUDE_FAMILY_MATCHED:-1}"
INCLUDE_RANDOM="${INCLUDE_RANDOM:-1}"
INCLUDE_RANDOM_WARMUP="${INCLUDE_RANDOM_WARMUP:-1}"
RANDOM_WARMUP_EPOCHS="${RANDOM_WARMUP_EPOCHS:-10 20 30}"
INCLUDE_JOINT_REFERENCE="${INCLUDE_JOINT_REFERENCE:-1}"
JOINT_LOGDIR="${JOINT_LOGDIR:-}"
JOINT_FINAL_EPOCH="${JOINT_FINAL_EPOCH:-100}"
OVERWRITE="${OVERWRITE:-0}"
GPUS="${GPUS:-1}"
GPU_IDS="${GPU_IDS:-}"

if [ "${INCLUDE_RANDOM_WARMUP}" = "1" ]; then
    for warmup_epoch in ${RANDOM_WARMUP_EPOCHS}; do
        if ! [[ "${warmup_epoch}" =~ ^[0-9]+$ ]]; then
            echo "ERROR: RANDOM_WARMUP_EPOCHS contains a non-integer: ${warmup_epoch}"
            exit 1
        fi
        if [ "${warmup_epoch}" -ge "${EPOCHS}" ]; then
            echo "ERROR: random warm-up ${warmup_epoch} must be smaller than EPOCHS=${EPOCHS}."
            echo "Set INCLUDE_RANDOM_WARMUP=0 for a short smoke test, or override RANDOM_WARMUP_EPOCHS."
            exit 1
        fi
    done
fi

if [ -z "${GPU_IDS}" ]; then
    GPU_IDS="$(seq 0 $((GPUS - 1)) | tr '\n' ' ')"
fi
read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [ "${#GPU_ARRAY[@]}" -lt 1 ]; then
    echo "ERROR: no GPU IDs were provided."
    exit 1
fi

if [ ! -d "${SELECTION_DIR}" ]; then
    echo "ERROR: selection directory not found: ${SELECTION_DIR}"
    exit 1
fi
if ! compgen -G "${SELECTION_DIR}/*_selection.json" >/dev/null; then
    echo "ERROR: no *_selection.json files in ${SELECTION_DIR}"
    exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/shared_initializations"

common_args=(
    --selection_dir "${SELECTION_DIR}"
    --data_root "${DATA_ROOT}"
    --train_split "${TRAIN_SPLIT}"
    --test_split "${TEST_SPLIT}"
    --train_instances_per_class "${TRAIN_PER_CLS}"
    --test_instances_per_class "${TEST_PER_CLS}"
    --num_views "${NUM_VIEWS}"
    --test_num_views "${TEST_NUM_VIEWS}"
    --epochs "${EPOCHS}"
    --eval_every "${EVAL_EVERY}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --arch "${ARCH}"
    --aggregation "${AGGREGATION}"
    --lr "${LR}"
    --base_lr_ratio "${BASE_LR_RATIO}"
    --weight_decay "${WEIGHT_DECAY}"
    --eval_view_seed "${EVAL_VIEW_SEED}"
    --shuffle_seed "${SHUFFLE_SEED}"
    --gpu_id 0
)

echo "=========================================="
echo "Controlled policy-replay experiment"
echo "  ARCH          = ${ARCH}"
echo "  SELECTION_DIR = ${SELECTION_DIR}"
echo "  OUTPUT_ROOT   = ${OUTPUT_ROOT}"
echo "  SEEDS         = ${SEEDS}"
echo "  EPOCHS        = ${EPOCHS}"
echo "  TRAIN/TEST K  = ${NUM_VIEWS}/${TEST_NUM_VIEWS}"
echo "  GPU_IDS       = ${GPU_IDS}"
echo "  FAMILY CTRL   = ${INCLUDE_FAMILY_MATCHED}"
echo "  RANDOM CTRL   = ${INCLUDE_RANDOM}"
echo "  RANDOM WARMUP = ${INCLUDE_RANDOM_WARMUP} (${RANDOM_WARMUP_EPOCHS})"
echo "  JOINT REF     = ${INCLUDE_JOINT_REFERENCE}"
echo "=========================================="

# Create one exact initialization per seed before parallel condition runs.
for seed in ${SEEDS}; do
    init_path="${OUTPUT_ROOT}/shared_initializations/seed_${seed}.pth"
    init_log="${OUTPUT_ROOT}/logs/initialize_seed_${seed}.log"
    echo "[init] seed ${seed} -> ${init_path}"
    CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" PYTHONUNBUFFERED=1 \
        python3 "${ROOT_DIR}/train_policy_replay.py" \
        "${common_args[@]}" \
        --protocol evolving \
        --seed "${seed}" \
        --shared_init "${init_path}" \
        --initialize_only \
        --output_dir "${OUTPUT_ROOT}/_initialize/seed_${seed}" \
        > "${init_log}" 2>&1
done

conditions=(
    "frozen_10|frozen|10|"
    "frozen_20|frozen|20|"
    "frozen_30|frozen|30|"
    "final_from_start|final||"
    "evolving|evolving||"
    "shuffled|shuffled||"
)
if [ "${INCLUDE_FAMILY_MATCHED}" = "1" ]; then
    conditions+=("family_matched_random|family_matched_random||")
fi
if [ "${INCLUDE_RANDOM}" = "1" ]; then
    conditions+=("random_views|random||")
fi
if [ "${INCLUDE_RANDOM_WARMUP}" = "1" ]; then
    for warmup_epoch in ${RANDOM_WARMUP_EPOCHS}; do
        conditions+=(
            "random_warmup_${warmup_epoch}|random_then_evolving||${warmup_epoch}")
    done
fi

required_conditions=()
for specification in "${conditions[@]}"; do
    IFS='|' read -r label _ _ _ <<< "${specification}"
    required_conditions+=("${label}")
done

pids=()
names=()
wait_batch() {
    local failed=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "  done: ${names[$index]}"
        else
            echo "  FAILED: ${names[$index]} (see its log)"
            failed=1
        fi
    done
    pids=()
    names=()
    if [ "${failed}" -ne 0 ]; then
        exit 1
    fi
}

task_index=0
for seed in ${SEEDS}; do
    init_path="${OUTPUT_ROOT}/shared_initializations/seed_${seed}.pth"
    for specification in "${conditions[@]}"; do
        IFS='|' read -r label protocol policy_epoch random_warmup <<< "${specification}"
        run_dir="${OUTPUT_ROOT}/runs/${label}/seed_${seed}"
        run_log="${OUTPUT_ROOT}/logs/${label}_seed_${seed}.log"
        if [ "${OVERWRITE}" != "1" ] && [ -f "${run_dir}/run_summary.json" ]; then
            echo "SKIP existing ${label}/seed_${seed}"
            continue
        fi
        gpu="${GPU_ARRAY[$((task_index % ${#GPU_ARRAY[@]}))]}"
        extra=()
        [ -n "${policy_epoch}" ] && extra+=(--policy_epoch "${policy_epoch}")
        [ -n "${random_warmup}" ] && extra+=(
            --random_warmup_epochs "${random_warmup}")
        echo "[run] ${label}/seed_${seed} -> GPU ${gpu}"
        CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
            python3 "${ROOT_DIR}/train_policy_replay.py" \
            "${common_args[@]}" \
            --protocol "${protocol}" \
            --seed "${seed}" \
            --shared_init "${init_path}" \
            --output_dir "${run_dir}" \
            "${extra[@]}" \
            > "${run_log}" 2>&1 &
        pids+=("$!")
        names+=("${label}/seed_${seed}")
        task_index=$((task_index + 1))
        if [ "${#pids[@]}" -ge "${#GPU_ARRAY[@]}" ]; then
            wait_batch
        fi
    done
done
[ "${#pids[@]}" -gt 0 ] && wait_batch

# Evaluate the original jointly trained no-freeze checkpoint series on the
# exact same fixed held-out views. This is an external reference, not one of
# the shared-initialization replay controls.
if [ "${INCLUDE_JOINT_REFERENCE}" = "1" ]; then
    if [ -z "${JOINT_LOGDIR}" ]; then
        shopt -s nullglob
        joint_candidates=(
            "${ROOT_DIR}/logs/${DATASET}/${EXPERIMENT_NAME}_"????-??-??_??-??-??
        )
        shopt -u nullglob
        valid_candidates=()
        for candidate in "${joint_candidates[@]}"; do
            [ -f "${candidate}/model.pth" ] && valid_candidates+=("${candidate}")
        done
        if [ "${#valid_candidates[@]}" -eq 0 ]; then
            echo "ERROR: no original joint no-freeze logdir matched:"
            echo "  logs/${DATASET}/${EXPERIMENT_NAME}_<timestamp>/model.pth"
            echo "Set JOINT_LOGDIR=<timestamped-logdir>, or INCLUDE_JOINT_REFERENCE=0."
            exit 1
        fi
        JOINT_LOGDIR="${valid_candidates[$((${#valid_candidates[@]} - 1))]}"
        if [ "${#valid_candidates[@]}" -gt 1 ]; then
            echo "Found ${#valid_candidates[@]} joint runs; using lexically latest:"
            echo "  ${JOINT_LOGDIR}"
            echo "Override with JOINT_LOGDIR=<path>."
        fi
    fi
    if [ ! -f "${JOINT_LOGDIR}/model.pth" ]; then
        echo "ERROR: JOINT_LOGDIR has no model.pth: ${JOINT_LOGDIR}"
        exit 1
    fi
    reference_seed="${SEEDS%% *}"
    reference_dir="${OUTPUT_ROOT}/runs/joint_no_freeze_reference/seed_${reference_seed}"
    reference_log="${OUTPUT_ROOT}/logs/joint_no_freeze_reference.log"
    if [ "${OVERWRITE}" != "1" ] && [ -f "${reference_dir}/run_summary.json" ]; then
        echo "SKIP existing joint_no_freeze_reference/seed_${reference_seed}"
    else
        echo "[reference] joint no-freeze -> GPU ${GPU_ARRAY[0]}"
        CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" PYTHONUNBUFFERED=1 \
            python3 "${ROOT_DIR}/evaluate_joint_policy_reference.py" \
            --logdir "${JOINT_LOGDIR}" \
            --output_dir "${reference_dir}" \
            --data_root "${DATA_ROOT}" \
            --dataset "${DATASET}" \
            --test_split "${TEST_SPLIT}" \
            --test_instances_per_class "${TEST_PER_CLS}" \
            --test_num_views "${TEST_NUM_VIEWS}" \
            --eval_view_seed "${EVAL_VIEW_SEED}" \
            --epoch_stride "${EVAL_EVERY}" \
            --final_epoch "${JOINT_FINAL_EPOCH}" \
            --batch_size "${BATCH_SIZE}" \
            --num_workers "${NUM_WORKERS}" \
            --arch "${ARCH}" \
            --aggregation "${AGGREGATION}" \
            --reference_seed "${reference_seed}" \
            --gpu_id 0 \
            > "${reference_log}" 2>&1
        echo "  done: joint_no_freeze_reference/seed_${reference_seed}"
    fi
fi

python3 "${ROOT_DIR}/aggregate_policy_replay.py" \
    --input_root "${OUTPUT_ROOT}" \
    --output_dir "${OUTPUT_ROOT}" \
    --required_conditions "${required_conditions[@]}"

echo "=========================================="
echo "Done: ${OUTPUT_ROOT}"
echo "Check control_audit.csv before interpreting plots."
echo "=========================================="
