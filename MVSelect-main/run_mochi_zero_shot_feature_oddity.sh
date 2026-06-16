#!/usr/bin/env bash
# Run zero-shot MOCHI oddity evaluation using MVSelect/MVCNN features.
#
# Comparison sets:
#   COMPARISON_SET=freeze          no_freeze vs freeze_10..freeze_50
#   COMPARISON_SET=selector_limit  select_all vs selector-view-limit runs
#
# By default this evaluates final model.pth only. Set PER_EPOCH_CHECKPOINT=1
# to evaluate model_e<E>.pth snapshots, which requires training with
# --save_every_epoch >0.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

COMPARISON_SET="${COMPARISON_SET:-freeze}"  # freeze | selector_limit
if [ -z "${COMPARISON_NAME:-}" ]; then
    case "${COMPARISON_SET}" in
        freeze)
            COMPARISON_NAME="mochi_feature_oddity_freeze_sweep"
            ;;
        selector_limit)
            COMPARISON_NAME="mochi_feature_oddity_selector_limit_sweep"
            ;;
        *)
            echo "ERROR: unknown COMPARISON_SET=${COMPARISON_SET}. Use freeze or selector_limit."
            exit 1
            ;;
    esac
fi

DATASET="${DATASET:-rgb}"
ARCH="${ARCH:-resnet18}"
NUM_CAM="${NUM_CAM:-114}"
GPU_ID="${GPU_ID:-0}"
N_TRIALS="${N_TRIALS:-}"
PER_EPOCH_CHECKPOINT="${PER_EPOCH_CHECKPOINT:-0}"
EPOCHS="${EPOCHS:-}"

FREEZE_EXPS=(
    "resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:no_freeze"
    "freeze_10_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_10"
    "freeze_20_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_20"
    "freeze_30_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_30"
    "freeze_40_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_40"
    "freeze_50_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_50"
)
SELECTOR_LIMIT_EXPS=(
    "resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:select_all"
    "resnet18steps5_selview_expanded_family_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:select_expanded"
    "resnet18steps5_selview_foreshortened_family_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:select_foreshortened"
    "resnet18steps5_selview_foreshortened_family_remainder_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:select_foreshortened_remainder"
    "resnet18steps5_selview_remainder_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:select_remainder"
)
case "${COMPARISON_SET}" in
    freeze)
        EXPS=("${FREEZE_EXPS[@]}")
        ;;
    selector_limit)
        EXPS=("${SELECTOR_LIMIT_EXPS[@]}")
        ;;
esac

OUTPUT_DIR="${ROOT_DIR}/compare/${COMPARISON_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "MOCHI zero-shot feature oddity:"
echo "  SET                  = ${COMPARISON_SET}"
echo "  COMPARISON           = ${COMPARISON_NAME}"
echo "  DATASET              = ${DATASET}"
echo "  ARCH                 = ${ARCH}"
echo "  NUM_CAM              = ${NUM_CAM}"
echo "  GPU_ID               = ${GPU_ID}"
echo "  N_TRIALS             = ${N_TRIALS:-(all)}"
echo "  PER_EPOCH_CHECKPOINT = ${PER_EPOCH_CHECKPOINT}"
echo "  EPOCHS               = ${EPOCHS:-(all saved)}"
echo "  OUTPUT               = ${OUTPUT_DIR}"
echo "  EXPS:"
for e in "${EXPS[@]}"; do echo "    - ${e}"; done
echo "=========================================="

args=(
    --dataset "${DATASET}"
    --arch "${ARCH}"
    --num_cam "${NUM_CAM}"
    --gpu_id "${GPU_ID}"
    --output_dir "${OUTPUT_DIR}"
)
for spec in "${EXPS[@]}"; do
    args+=(--exp "${spec}")
done
[ -n "${N_TRIALS}" ] && args+=(--n_trials "${N_TRIALS}")
[ "${PER_EPOCH_CHECKPOINT}" = "1" ] && args+=(--per_epoch_checkpoint)
[ -n "${EPOCHS}" ] && args+=(--epochs "${EPOCHS}")

python3 "${ROOT_DIR}/mochi_zero_shot_feature_oddity.py" "${args[@]}"

echo "Done. Outputs are in ${OUTPUT_DIR}"
