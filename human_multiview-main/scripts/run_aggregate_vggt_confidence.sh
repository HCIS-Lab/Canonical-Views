#!/usr/bin/env bash
# Aggregate VGGT confidence-on-selections CSVs from a hard-coded list of
# experiments into overlaid comparison plots. Edit the SUMMARIES array
# below to define a comparison set.
#
# Each entry is "SUMMARY_PATH[:LEGEND_LABEL]". SUMMARY_PATH should be the
# directory containing overall_summary.csv (output of aggregate_views.py),
# or the CSV path directly. If LEGEND_LABEL is omitted, the parent folder
# name is used.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# Comparison configuration — edit these
# ---------------------------------------------------------------------------
COMPARISON_SET="${COMPARISON_SET:-freeze}"  # freeze | selector_limit
if [ -z "${COMPARISON_NAME:-}" ]; then
    case "${COMPARISON_SET}" in
        freeze)
            COMPARISON_NAME="vggt_confidence_freeze_sweep"
            ;;
        selector_limit)
            COMPARISON_NAME="vggt_confidence_selector_limit_sweep"
            ;;
        *)
            echo "ERROR: unknown COMPARISON_SET=${COMPARISON_SET}. Use freeze or selector_limit."
            exit 1
            ;;
    esac
fi
MODEL="${MODEL:-vggt}"
VALUE="${VALUE:-agent_mean}"          # agent_mean | delta_mean | macro_agent | ...

# Each entry: "<path_to_summary_dir>[:LABEL]"
#
# Paths below match the per-experiment layout produced by
# run_views_pipeline_sweep.sh — one summary folder per experiment under
# results/views/v<VIEW_TYPE>/<exp_label>/summary/.
# If you ran the sweep with a different layout, edit accordingly.
FREEZE_SUMMARIES=(
    "results/views/v01234/no_freeze/summary:no_freeze"
    "results/views/v01234/freeze_10/summary:freeze_10"
    "results/views/v01234/freeze_20/summary:freeze_20"
    "results/views/v01234/freeze_30/summary:freeze_30"
    "results/views/v01234/freeze_40/summary:freeze_40"
    "results/views/v01234/freeze_50/summary:freeze_50"
)
SELECTOR_LIMIT_SUMMARIES=(
    "results/views/v01234/no_freeze/summary:select_all"
    "results/views/v01234/select_expanded/summary:select_expanded"
    "results/views/v01234/select_foreshortened/summary:select_foreshortened"
    "results/views/v01234/select_foreshortened_remainder/summary:select_foreshortened_remainder"
    "results/views/v01234/select_remainder/summary:select_remainder"
)
case "${COMPARISON_SET}" in
    freeze)
        SUMMARIES=("${FREEZE_SUMMARIES[@]}")
        ;;
    selector_limit)
        SUMMARIES=("${SELECTOR_LIMIT_SUMMARIES[@]}")
        ;;
esac

STYLE="${STYLE:-heatmap}"            # line | heatmap | sorted_bars | rank_stacked | both | all
                                     #   sorted_bars: grouped bars per epoch sorted left→right by value
                                     #                (real y-axis; rank flips show as colour reshuffles)
                                     #   rank_stacked: same idea but stacked (y-axis is a sum)
BIN_EPOCHS="${BIN_EPOCHS:-0}"
SMOOTH="${SMOOTH:-1}"
TITLE_SUFFIX="${TITLE_SUFFIX:-}"

OUTPUT_DIR="${ROOT_DIR}/compare/${COMPARISON_NAME}"

# ---------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"
echo "=========================================="
echo "Aggregating VGGT confidence:"
echo "  SET        = ${COMPARISON_SET}"
echo "  COMPARISON = ${COMPARISON_NAME}"
echo "  MODEL      = ${MODEL}"
echo "  VALUE      = ${VALUE}"
echo "  STYLE      = ${STYLE}"
echo "  OUTPUT     = ${OUTPUT_DIR}"
echo "  SUMMARIES:"
for s in "${SUMMARIES[@]}"; do echo "    - ${s}"; done
echo "=========================================="

summary_args=()
for spec in "${SUMMARIES[@]}"; do
    summary_args+=(--summary "${spec}")
done

extra=()
[ -n "${TITLE_SUFFIX}" ] && extra+=(--title_suffix "${TITLE_SUFFIX}")
extra+=(--style "${STYLE}")
[ "${BIN_EPOCHS}" != "0" ] && extra+=(--bin_epochs "${BIN_EPOCHS}")
[ "${SMOOTH}" != "1" ] && extra+=(--smooth "${SMOOTH}")

python3 "${ROOT_DIR}/scripts/aggregate_vggt_confidence.py" \
    --model "${MODEL}" \
    --value "${VALUE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${summary_args[@]}" \
    "${extra[@]}"

echo "Done. Plots and aggregated.csv are in ${OUTPUT_DIR}"
