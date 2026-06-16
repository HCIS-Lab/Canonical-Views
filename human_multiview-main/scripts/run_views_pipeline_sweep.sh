#!/usr/bin/env bash
# Run the full agent-vs-random VGGT pipeline (run_pipeline.sh) once per
# MVSelect experiment in the EXPS array below, sequentially. Each experiment
# uses all GPUs internally via sharding.
#
# Output layout:
#   results/views/v<VIEW_TYPE>/<exp_label>/                  ← per-exp eval CSVs
#   results/views/v<VIEW_TYPE>/<exp_label>/summary/          ← overall_summary.csv (used by aggregate_vggt_confidence)
#   results/views/v<VIEW_TYPE>/<exp_label>/plots/            ← per-experiment plots
#   results/views/v<VIEW_TYPE>/<exp_label>/logs/             ← per-shard logs
#
# After this finishes, run:
#   ./scripts/run_aggregate_vggt_confidence.sh
# to overlay the experiments' curves on shared plots.
#
# Edit EXPS to define the comparison set. Each entry is
# "MVSELECT_EXP_FOLDER_NAME[:LABEL]". The folder is resolved against
# ../MVSelect-main/meta_logs/<dataset>/, and LABEL controls the per-experiment
# output subfolder name (defaults to the folder basename).

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# --- Config (override via env vars) ---
DATASET="${DATASET:-rgb}"
GPUS="${GPUS:-4}"
MODELS="${MODELS:-vggt dinov2}"
VIEW_TYPE="${VIEW_TYPE:-01234}"
EPOCH_GROUPS="${EPOCH_GROUPS:-1-10,11-20,21-30,31-40,41-50,51-60,61-70,71-80,81-90,91-100}"
NUM_CAM="${NUM_CAM:-5}"
SPLIT="${SPLIT:-test}"
PER_CLS="${PER_CLS:-5}"
LIMIT="${LIMIT:-}"

# Where to look up MVSelect experiment folders by bare name
MVSELECT_META_LOGS="${MVSELECT_META_LOGS:-${ROOT_DIR}/../MVSelect-main/meta_logs/${DATASET}}"
COMPARISON_SET="${COMPARISON_SET:-freeze}"  # freeze | selector_limit

# --- Experiment list (EDIT ME) ---
# Each entry: "<exp_folder_basename>[:<label>]"
FREEZE_EXPS=(
    "resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:no_freeze"
    "freeze_10_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_10"
    "freeze_20_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_20"
    "freeze_30_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_30"
    "freeze_40_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_40"
    "freeze_50_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:freeze_50"
)
SELECTOR_LIMIT_EXPS=(
    "resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100:no_freeze"
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
    *)
        echo "ERROR: unknown COMPARISON_SET=${COMPARISON_SET}. Use freeze or selector_limit."
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------

echo "=========================================="
echo "Per-experiment VGGT pipeline sweep"
echo "  COMPARISON_SET     = ${COMPARISON_SET}"
echo "  DATASET            = ${DATASET}"
echo "  GPUS per exp       = ${GPUS}"
echo "  MODELS             = ${MODELS}"
echo "  VIEW_TYPE          = ${VIEW_TYPE}"
echo "  EPOCH_GROUPS       = ${EPOCH_GROUPS}"
echo "  NUM_CAM            = ${NUM_CAM}"
echo "  PER_CLS            = ${PER_CLS}"
echo "  LIMIT              = ${LIMIT:-(none)}"
echo "  MVSELECT_META_LOGS = ${MVSELECT_META_LOGS}"
echo "  EXPS (${#EXPS[@]}):"
for e in "${EXPS[@]}"; do echo "    - ${e}"; done
echo "=========================================="

if [ ! -d "${MVSELECT_META_LOGS}" ]; then
    echo "ERROR: MVSELECT_META_LOGS does not exist: ${MVSELECT_META_LOGS}"
    echo "       Override with MVSELECT_META_LOGS=<absolute path>"
    exit 1
fi

n=${#EXPS[@]}
idx=0
fail=0
for spec in "${EXPS[@]}"; do
    idx=$((idx + 1))
    if [[ "${spec}" == *:* ]]; then
        folder="${spec%%:*}"
        label="${spec##*:}"
    else
        folder="${spec}"
        label="${spec}"
    fi
    selection_dir="${MVSELECT_META_LOGS}/${folder}"
    output_dir="${ROOT_DIR}/results/views/v${VIEW_TYPE}/${label}"

    echo
    echo "------------------------------------------------------------------"
    echo "[${idx}/${n}] ${label}"
    echo "  selection_dir = ${selection_dir}"
    echo "  output_dir    = ${output_dir}"
    echo "------------------------------------------------------------------"

    if [ ! -d "${selection_dir}" ]; then
        echo "  SKIP: selection_dir not found."
        fail=1
        continue
    fi
    if ! ls "${selection_dir}"/*_selection.json > /dev/null 2>&1; then
        echo "  SKIP: no *_selection.json files in selection_dir."
        fail=1
        continue
    fi

    SELECTION_DIR="${selection_dir}" \
    OUTPUT_DIR="${output_dir}" \
    GPUS="${GPUS}" \
    MODELS="${MODELS}" \
    VIEW_TYPE="${VIEW_TYPE}" \
    EPOCH_GROUPS="${EPOCH_GROUPS}" \
    NUM_CAM="${NUM_CAM}" \
    SPLIT="${SPLIT}" \
    PER_CLS="${PER_CLS}" \
    LIMIT="${LIMIT}" \
        "${ROOT_DIR}/scripts/run_pipeline.sh" || {
            echo "  ERROR: pipeline failed for ${label}. Continuing to next exp."
            fail=1
            continue
        }
    echo "  done: ${label}"
done

echo "=========================================="
echo "Sweep finished."
echo "Per-experiment artifacts under: ${ROOT_DIR}/results/views/v${VIEW_TYPE}/<label>/"
echo
echo "Next step — overlay the experiments' curves:"
echo "    ./scripts/run_aggregate_vggt_confidence.sh"
echo "(make sure that script's SUMMARIES list matches the labels used here.)"
if [ "${fail}" -ne 0 ]; then
    echo
    echo "Note: at least one experiment failed or was skipped. See above."
fi
echo "=========================================="
