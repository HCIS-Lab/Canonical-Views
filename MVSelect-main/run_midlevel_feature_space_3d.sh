#!/usr/bin/env bash
# Plot every rendered view in the test split in the three primary cue axes.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${ROOT_DIR}/experiment_arch.sh"
SPLIT="${SPLIT:-test}"
DATA_ROOT="${DATA_ROOT:-/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_23}"
CACHE_CSV="${CACHE_CSV:-${ROOT_DIR}/cache/midlevel_features_v2_${SPLIT}.csv}"
HIGHLIGHT_SELECTED="${HIGHLIGHT_SELECTED:-1}"
SELECTION_DIR="${SELECTION_DIR:-${ROOT_DIR}/meta_logs/rgb/${MVSELECT_BASE_EXPERIMENT}}"
SELECTION_EPOCH="${SELECTION_EPOCH:-100}"
SELECTION_RUN="${SELECTION_RUN:-}"
INITIAL_CAMERA="${INITIAL_CAMERA:-0}"
EXPECTED_SELECTED_VIEWS="${EXPECTED_SELECTED_VIEWS:-5}"
MARKER_SIZE="${MARKER_SIZE:-9}"
ALPHA="${ALPHA:-0.50}"
SELECTED_MARKER_SIZE="${SELECTED_MARKER_SIZE:-24}"
CLASS_NAME="${CLASS_NAME:-}"
PER_INSTANCE="${PER_INSTANCE:-0}"
PER_INSTANCE_COUNT="${PER_INSTANCE_COUNT:-5}"
ONE_INSTANCE_PER_CLASS="${ONE_INSTANCE_PER_CLASS:-0}"
if [ -n "${CLASS_NAME}" ]; then
    CLASS_SLUG="$(printf '%s' "${CLASS_NAME}" | tr '[:upper:] /' '[:lower:]__')"
    DEFAULT_OUTPUT_DIR="${ROOT_DIR}/compare/midlevel_feature_space_3d/${SPLIT}/class_${CLASS_SLUG}"
    EXPECTED_CLASSES="${EXPECTED_CLASSES:-1}"
elif [ "${ONE_INSTANCE_PER_CLASS}" = "1" ]; then
    DEFAULT_OUTPUT_DIR="${ROOT_DIR}/compare/midlevel_feature_space_3d/${SPLIT}/one_instance_per_class"
    EXPECTED_CLASSES="${EXPECTED_CLASSES:-32}"
else
    DEFAULT_OUTPUT_DIR="${ROOT_DIR}/compare/midlevel_feature_space_3d/${SPLIT}"
    EXPECTED_CLASSES="${EXPECTED_CLASSES:-32}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"
ALLOW_MISSING="${ALLOW_MISSING:-1}"
LIMIT_IMAGES="${LIMIT_IMAGES:-}"
EXPECTED_OBJECTS_PER_CLASS="${EXPECTED_OBJECTS_PER_CLASS:-0}"
EXPECTED_VIEWS_PER_OBJECT="${EXPECTED_VIEWS_PER_OBJECT:-114}"
SKIP_DISK_COVERAGE_CHECK="${SKIP_DISK_COVERAGE_CHECK:-0}"

extra=()
[ "${FORCE_RECOMPUTE}" = "1" ] && extra+=(--force_recompute)
[ "${ALLOW_MISSING}" = "1" ] && extra+=(--allow_missing)
[ -n "${LIMIT_IMAGES}" ] && extra+=(--limit_images "${LIMIT_IMAGES}")
[ "${SKIP_DISK_COVERAGE_CHECK}" = "1" ] && extra+=(--skip_disk_coverage_check)
[ -n "${CLASS_NAME}" ] && extra+=(--classes "${CLASS_NAME}")
[ "${PER_INSTANCE}" = "1" ] && extra+=(--per_instance)
[ "${PER_INSTANCE}" = "1" ] && extra+=(--per_instance_count "${PER_INSTANCE_COUNT}")
[ "${ONE_INSTANCE_PER_CLASS}" = "1" ] && extra+=(--one_instance_per_class)
if [ "${HIGHLIGHT_SELECTED}" = "1" ]; then
    extra+=(--selection_dir "${SELECTION_DIR}")
    extra+=(--selection_epoch "${SELECTION_EPOCH}")
    extra+=(--initial_camera "${INITIAL_CAMERA}")
    extra+=(--expected_selected_views "${EXPECTED_SELECTED_VIEWS}")
    [ -n "${SELECTION_RUN}" ] && extra+=(--selection_run "${SELECTION_RUN}")
fi

echo "=========================================="
echo "All-view mid-level 3D feature space"
echo "  SPLIT      = ${SPLIT}"
echo "  DATA_ROOT  = ${DATA_ROOT}"
echo "  CACHE_CSV  = ${CACHE_CSV}"
echo "  OUTPUT_DIR = ${OUTPUT_DIR}"
echo "  CLASS      = ${CLASS_NAME:-(all)}"
echo "  PER INSTANCE = ${PER_INSTANCE}"
[ "${PER_INSTANCE}" = "1" ] && echo "  INSTANCE COUNT = ${PER_INSTANCE_COUNT}"
echo "  ONE INSTANCE PER CLASS = ${ONE_INSTANCE_PER_CLASS}"
echo "  HIGHLIGHT  = ${HIGHLIGHT_SELECTED}"
if [ "${HIGHLIGHT_SELECTED}" = "1" ]; then
    echo "  SELECTION  = ${SELECTION_DIR}"
    echo "  EPOCH      = ${SELECTION_EPOCH}"
    echo "  RUN        = ${SELECTION_RUN:-(first with exact feature dump)}"
    echo "  INITIAL CAM = ${INITIAL_CAMERA}"
    echo "  AGENT VIEWS = ${EXPECTED_SELECTED_VIEWS}"
fi
echo "  OMIT UNDEFINED COORDINATES = ${ALLOW_MISSING}"
echo "=========================================="

python3 "${ROOT_DIR}/plot_midlevel_feature_space_3d.py" \
    --data_root "${DATA_ROOT}" \
    --split "${SPLIT}" \
    --cache_csv "${CACHE_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --expected_classes "${EXPECTED_CLASSES}" \
    --expected_objects_per_class "${EXPECTED_OBJECTS_PER_CLASS}" \
    --expected_views_per_object "${EXPECTED_VIEWS_PER_OBJECT}" \
    --marker_size "${MARKER_SIZE}" \
    --alpha "${ALPHA}" \
    --selected_marker_size "${SELECTED_MARKER_SIZE}" \
    "${extra[@]}"
