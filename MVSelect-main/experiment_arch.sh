#!/usr/bin/env bash
# Shared architecture-aware experiment names for analysis launchers.

ARCH="${ARCH:-resnet18}"
case "${ARCH}" in
    resnet18|vit|tinyvit) ;;
    *)
        echo "ERROR: ARCH=${ARCH} is unsupported. Use resnet18, vit, or tinyvit."
        return 1 2>/dev/null || exit 1
        ;;
esac

MVSELECT_BASE_EXPERIMENT="${ARCH}steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100"
MVSELECT_ACTIVE_PAIR_EXPERIMENT="${ARCH}steps1_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100"
MVSELECT_ACTIVE_SINGLE_EXPERIMENT="${ARCH}steps1_active_single_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100"

MVSELECT_FREEZE_EXPS=(
    "${MVSELECT_BASE_EXPERIMENT}:no_freeze"
    "freeze_10_${MVSELECT_BASE_EXPERIMENT}:freeze_10"
    "freeze_20_${MVSELECT_BASE_EXPERIMENT}:freeze_20"
    "freeze_30_${MVSELECT_BASE_EXPERIMENT}:freeze_30"
    "freeze_40_${MVSELECT_BASE_EXPERIMENT}:freeze_40"
    "freeze_50_${MVSELECT_BASE_EXPERIMENT}:freeze_50"
)

MVSELECT_SELECTOR_LIMIT_EXPS=(
    "${MVSELECT_BASE_EXPERIMENT}:select_all"
    "${ARCH}steps5_selview_expanded_family_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:select_expanded"
    "${ARCH}steps5_selview_foreshortened_family_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:select_foreshortened"
    "${ARCH}steps5_selview_foreshortened_family_remainder_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:select_foreshortened_remainder"
    "${ARCH}steps5_selview_remainder_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:select_remainder"
)

mvselect_arch_scoped_name() {
    local base="$1"
    if [ "${ARCH}" = "resnet18" ]; then
        echo "${base}"
    elif [[ "${base}" == *"_${ARCH}" ]]; then
        echo "${base}"
    else
        echo "${base}_${ARCH}"
    fi
}
