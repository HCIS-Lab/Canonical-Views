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
COMPARISON_NAME="${COMPARISON_NAME:-vggt_confidence_freeze_sweep}"
MODEL="${MODEL:-vggt}"
VALUE="${VALUE:-agent_mean}"          # agent_mean | delta_mean | macro_agent | ...

# Each SUMMARY entry points at <results>/views/v<vt>/summary/ for one experiment
SUMMARIES=(
    "results/views/v01234/summary_no_freeze:no_freeze"
    "results/views/v01234/summary_freeze_10:freeze_10"
    "results/views/v01234/summary_freeze_20:freeze_20"
    "results/views/v01234/summary_freeze_30:freeze_30"
    "results/views/v01234/summary_freeze_40:freeze_40"
    "results/views/v01234/summary_freeze_50:freeze_50"
    # Add more lines as needed.
)

STYLE="${STYLE:-line}"               # line | heatmap | both
BIN_EPOCHS="${BIN_EPOCHS:-0}"
SMOOTH="${SMOOTH:-1}"
TITLE_SUFFIX="${TITLE_SUFFIX:-}"

OUTPUT_DIR="${ROOT_DIR}/compare/${COMPARISON_NAME}"

# ---------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"
echo "=========================================="
echo "Aggregating VGGT confidence:"
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
