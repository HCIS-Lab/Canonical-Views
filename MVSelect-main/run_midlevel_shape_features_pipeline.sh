#!/usr/bin/env bash
# Compute and aggregate image-derived mid-level shape features for selected
# MVSelect views.
#
# Defaults:
#   COMPARISON_SET=freeze       no_freeze vs freeze_10..freeze_50
#   VALUE=lift                  selected mean - all-candidate-view baseline
#   STYLE=heatmap
#   BIN_EPOCHS=10
#
# Examples:
#   ./run_midlevel_shape_features_pipeline.sh
#   COMPARISON_SET=selector_limit ./run_midlevel_shape_features_pipeline.sh
#   VALUE=selected STYLE=both ./run_midlevel_shape_features_pipeline.sh

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

COMPARISON_SET="${COMPARISON_SET:-freeze}"  # freeze | selector_limit
if [ -z "${COMPARISON_NAME:-}" ]; then
    case "${COMPARISON_SET}" in
        freeze)
            COMPARISON_NAME="midlevel_shape_freeze_sweep"
            ;;
        selector_limit)
            COMPARISON_NAME="midlevel_shape_selector_limit_sweep"
            ;;
        *)
            echo "ERROR: unknown COMPARISON_SET=${COMPARISON_SET}. Use freeze or selector_limit."
            exit 1
            ;;
    esac
fi

DATASET="${DATASET:-rgb}"
SPLIT="${SPLIT:-test}"
DATA_ROOT="${DATA_ROOT:-/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23}"
CACHE_CSV="${CACHE_CSV:-${ROOT_DIR}/cache/midlevel_features_${SPLIT}.csv}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"
LIMIT_IMAGES="${LIMIT_IMAGES:-}"

VALUE="${VALUE:-lift}"          # selected | lift | baseline
STYLE="${STYLE:-heatmap}"       # heatmap | line | both
BIN_EPOCHS="${BIN_EPOCHS:-10}"
TITLE_SUFFIX="${TITLE_SUFFIX:-}"
METRICS="${METRICS:-ellipse_aspect_ratio skeleton_elongation skeleton_length_norm bilateral_symmetry medial_axis_symmetry skeleton_branch_density edge_anisotropy edge_entropy}"

FREEZE_EXPS=(
    "resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:no_freeze"
    "freeze_10_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_10"
    "freeze_20_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_20"
    "freeze_30_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_30"
    "freeze_40_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_40"
    "freeze_50_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_50"
)

pick_exp() {
    local label="$1"
    local exact="$2"
    shift 2
    local meta_dir="${ROOT_DIR}/meta_logs/${DATASET}"
    local p

    for p in "${meta_dir}/${exact}" "$@"; do
        for d in ${p}; do
            if [ -f "${d}/midlevel_features/selected_midlevel_summary.csv" ]; then
                echo "$(basename "${d}"):${label}"
                return 0
            fi
        done
    done
    for p in "${meta_dir}/${exact}" "$@"; do
        for d in ${p}; do
            if compgen -G "${d}/*_selection.json" >/dev/null; then
                echo "$(basename "${d}"):${label}"
                return 0
            fi
        done
    done
    for p in "${meta_dir}/${exact}" "$@"; do
        for d in ${p}; do
            if [ -d "${d}" ]; then
                echo "$(basename "${d}"):${label}"
                return 0
            fi
        done
    done
    echo "${exact}:${label}"
}

SELECTOR_LIMIT_EXPS=(
    "$(pick_exp select_all \
        resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100)"
    "$(pick_exp select_expanded \
        resnet18steps5_selview_expanded_family_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100 \
        "${ROOT_DIR}/meta_logs/${DATASET}/"*selview_expanded_family_train*)"
    "$(pick_exp select_foreshortened \
        resnet18steps5_selview_foreshortened_family_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100 \
        "${ROOT_DIR}/meta_logs/${DATASET}/"*selview_foreshortened_family_train*)"
    "$(pick_exp select_foreshortened_remainder \
        resnet18steps5_selview_foreshortened_family_remainder_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100 \
        "${ROOT_DIR}/meta_logs/${DATASET}/"*selview_foreshortened_family_remainder_train*)"
    "$(pick_exp select_remainder \
        resnet18steps5_selview_remainder_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100 \
        "${ROOT_DIR}/meta_logs/${DATASET}/"*selview_remainder_train*)"
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
echo "Mid-level shape feature pipeline:"
echo "  SET        = ${COMPARISON_SET}"
echo "  COMPARISON = ${COMPARISON_NAME}"
echo "  DATASET    = ${DATASET}"
echo "  SPLIT      = ${SPLIT}"
echo "  VALUE      = ${VALUE}"
echo "  STYLE      = ${STYLE}"
echo "  BIN_EPOCHS = ${BIN_EPOCHS}"
echo "  CACHE_CSV  = ${CACHE_CSV}"
echo "  OUTPUT     = ${OUTPUT_DIR}"
echo "  EXPS:"
for e in "${EXPS[@]}"; do echo "    - ${e}"; done
echo "=========================================="

compute_extra=()
[ "${FORCE_RECOMPUTE}" = "1" ] && compute_extra+=(--force_recompute)
[ -n "${LIMIT_IMAGES}" ] && compute_extra+=(--limit_images "${LIMIT_IMAGES}")

echo "[1/2] Computing per-experiment selected mid-level summaries..."
for spec in "${EXPS[@]}"; do
    folder="${spec%%:*}"
    label="${spec##*:}"
    exp_dir="${ROOT_DIR}/meta_logs/${DATASET}/${folder}"
    if [ ! -d "${exp_dir}" ]; then
        echo "SKIP ${label}: no experiment folder at ${exp_dir}"
        continue
    fi
    if ! compgen -G "${exp_dir}/*_selection.json" >/dev/null; then
        echo "SKIP ${label}: no *_selection.json in ${exp_dir}"
        continue
    fi
    echo "  ${label} -> ${exp_dir}/midlevel_features"
    python3 "${ROOT_DIR}/midlevel_shape_features.py" \
        --data_root "${DATA_ROOT}" \
        --split "${SPLIT}" \
        --cache_csv "${CACHE_CSV}" \
        --selection_dir "${exp_dir}" \
        --bin_epochs "${BIN_EPOCHS}" \
        "${compute_extra[@]}"
done

echo
echo "[2/2] Aggregating across experiments..."
exp_args=()
for spec in "${EXPS[@]}"; do
    exp_args+=(--exp "${spec}")
done

agg_extra=()
[ -n "${TITLE_SUFFIX}" ] && agg_extra+=(--title_suffix "${TITLE_SUFFIX}")

python3 "${ROOT_DIR}/aggregate_midlevel_shape_features.py" \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --value "${VALUE}" \
    --style "${STYLE}" \
    --bin_epochs "${BIN_EPOCHS}" \
    --metrics ${METRICS} \
    "${exp_args[@]}" \
    "${agg_extra[@]}"

echo "=========================================="
echo "Done. Plots + aggregated.csv in ${OUTPUT_DIR}"
echo "=========================================="
