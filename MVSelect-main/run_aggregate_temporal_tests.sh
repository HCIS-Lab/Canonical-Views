#!/usr/bin/env bash
# Aggregate temporal_selection_test.py CSVs from a hard-coded list of
# experiments into overlaid comparison plots. Edit the EXPS array below
# to define a comparison set.
#
# Each entry is "EXPERIMENT_FOLDER[:LEGEND_LABEL]". The folder is
# resolved against meta_logs/<DATASET>/ when bare. If LEGEND_LABEL is
# omitted, the folder basename is used.
#
# Outputs go under compare/<COMPARISON_NAME>/.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Comparison configuration — edit these
# ---------------------------------------------------------------------------
COMPARISON_SET="${COMPARISON_SET:-freeze}"  # freeze | selector_limit
if [ -z "${COMPARISON_NAME:-}" ]; then
    case "${COMPARISON_SET}" in
        freeze)
            COMPARISON_NAME="steps5_freeze_sweep"
            ;;
        selector_limit)
            COMPARISON_NAME="steps5_selector_limit_sweep"
            ;;
        *)
            echo "ERROR: unknown COMPARISON_SET=${COMPARISON_SET}. Use freeze or selector_limit."
            exit 1
            ;;
    esac
fi
DATASET="${DATASET:-rgb}"

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

# Optional plot tuning
SMOOTH="${SMOOTH:-1}"               # rolling mean window (1 = off)
YMAX_DEV="${YMAX_DEV:-}"             # cap deviation plots at this % (empty = auto)
YMAX_MARGIN="${YMAX_MARGIN:-}"       # cap margin plot (empty = auto)
TITLE_SUFFIX="${TITLE_SUFFIX:-}"     # extra title line
STYLE="${STYLE:-heatmap}"            # line | heatmap | sorted_bars | rank_stacked | both | all
                                     #   default heatmap: easier than overlaid lines for many experiments
                                     #   sorted_bars: grouped bars per epoch sorted left→right by value
                                     #                (real y-axis; rank flips show as colour reshuffles)
                                     #   rank_stacked: same idea but stacked (y-axis is a sum)
BIN_EPOCHS="${BIN_EPOCHS:-10}"       # bin epochs into N columns (heatmap/bars; 0 = no binning)

OUTPUT_DIR="${ROOT_DIR}/compare/${COMPARISON_NAME}"

# ---------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"
echo "=========================================="
echo "Aggregating temporal_test runs:"
echo "  SET        = ${COMPARISON_SET}"
echo "  COMPARISON = ${COMPARISON_NAME}"
echo "  DATASET    = ${DATASET}"
echo "  OUTPUT     = ${OUTPUT_DIR}"
echo "  EXPS:"
for e in "${EXPS[@]}"; do echo "    - ${e}"; done
echo "=========================================="

exp_args=()
for spec in "${EXPS[@]}"; do
    exp_args+=(--exp "${spec}")
done

extra=()
[ -n "${TITLE_SUFFIX}" ] && extra+=(--title_suffix "${TITLE_SUFFIX}")
[ -n "${YMAX_DEV}" ] && extra+=(--ymax_dev "${YMAX_DEV}")
[ -n "${YMAX_MARGIN}" ] && extra+=(--ymax_margin "${YMAX_MARGIN}")
[ "${SMOOTH}" != "1" ] && extra+=(--smooth "${SMOOTH}")
extra+=(--style "${STYLE}")
[ "${BIN_EPOCHS}" != "0" ] && extra+=(--bin_epochs "${BIN_EPOCHS}")

python3 "${ROOT_DIR}/aggregate_temporal_tests.py" \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    "${exp_args[@]}" \
    "${extra[@]}"

echo "Done. Plots and aggregated.csv are in ${OUTPUT_DIR}"
