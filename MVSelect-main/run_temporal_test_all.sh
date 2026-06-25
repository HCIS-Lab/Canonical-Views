#!/usr/bin/env bash
# Run temporal_selection_test.py on every selector experiment under meta_logs/.
#
# For each meta_logs/<rep>/<exp>/ folder that contains *_selection.json files,
# parses --freeze_epoch and --arch from the folder name, then invokes the test.
# Experiments without _selection.json and folders missing checkpoints are skipped
# with a warning so the loop keeps going.
#
# Env-var overrides (with defaults):
#   GPUS=1                    parallel workers (round-robin GPU assignment)
#   REP_LIST="rgb depth edge" which representation folders to walk
#   NON_ROLL=1                pass --non_roll (matches your selector training)
#   NON_LIKE=0                pass --non_like
#   NUM_TRAIN_INSTANCES=25
#   MAX_EPOCHS=               optional smoke-test cap
#   COMPARISON_SET=all        all | freeze | selector_limit
#   PER_EPOCH_CHECKPOINT=0    final-classifier protocol (default). Set to 1
#                             for model-at-t, which requires training with
#                             --save_every_epoch >0.
#
# Examples:
#   ./run_temporal_test_all.sh                            # final classifier (default)
#   GPUS=4 ./run_temporal_test_all.sh                     # 4-GPU round-robin
#   COMPARISON_SET=selector_limit GPUS=4 ./run_temporal_test_all.sh
#   PER_EPOCH_CHECKPOINT=1 ./run_temporal_test_all.sh     # classifier-at-epoch-t
#   REP_LIST=rgb MAX_EPOCHS=5 ./run_temporal_test_all.sh  # smoke test

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
META_LOGS="${ROOT_DIR}/meta_logs"

GPUS="${GPUS:-1}"
REP_LIST="${REP_LIST:-rgb depth edge}"
NON_ROLL="${NON_ROLL:-1}"
NON_LIKE="${NON_LIKE:-0}"
NUM_TRAIN_INSTANCES="${NUM_TRAIN_INSTANCES:-25}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
COMPARISON_SET="${COMPARISON_SET:-all}"  # all | freeze | selector_limit
PER_EPOCH_CHECKPOINT="${PER_EPOCH_CHECKPOINT:-0}"

# Pass-through flags
extra_flags=""
[ "${NON_ROLL}" = "1" ] && extra_flags="${extra_flags} --non_roll"
[ "${NON_LIKE}" = "1" ] && extra_flags="${extra_flags} --non_like"
[ -n "${MAX_EPOCHS}" ] && extra_flags="${extra_flags} --max_epochs ${MAX_EPOCHS}"
[ "${PER_EPOCH_CHECKPOINT}" = "1" ] && extra_flags="${extra_flags} --per_epoch_checkpoint"

echo "=========================================="
echo "Temporal selection test — sweep"
echo "  ROOT      = ${ROOT_DIR}"
echo "  META_LOGS = ${META_LOGS}"
echo "  GPUS      = ${GPUS}"
echo "  REP_LIST  = ${REP_LIST}"
echo "  NON_ROLL  = ${NON_ROLL}"
echo "  NON_LIKE  = ${NON_LIKE}"
echo "  NUM_INS   = ${NUM_TRAIN_INSTANCES}"
echo "  MAX_EPOCH = ${MAX_EPOCHS:-(all)}"
echo "  SET       = ${COMPARISON_SET}"
echo "  PROTOCOL  = $([ "${PER_EPOCH_CHECKPOINT}" = "1" ] && echo 'classifier-at-epoch-t (--per_epoch_checkpoint)' || echo 'final classifier (fixed)')"
echo "=========================================="

if [ ! -d "${META_LOGS}" ]; then
    echo "ERROR: ${META_LOGS} does not exist."
    exit 1
fi

# Detect freeze_epoch (default 100) and arch (default resnet18) from folder name.
parse_freeze_epoch() {
    local name="$1"
    if [[ "${name}" =~ ^freeze_([0-9]+)_ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "100"
    fi
}
parse_arch() {
    local name="$1"
    local stripped="${name#freeze_*_}"           # strip optional freeze_<N>_ prefix
    # arch is the lowercase token before "steps" or "train"
    if [[ "${stripped}" =~ ^([a-z0-9]+)(steps|train) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "resnet18"
    fi
}

should_include_exp() {
    local name="$1"
    case "${COMPARISON_SET}" in
        all)
            return 0
            ;;
        freeze)
            case "${name}" in
                resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100|\
                freeze_10_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100|\
                freeze_20_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100|\
                freeze_30_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100|\
                freeze_40_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100|\
                freeze_50_resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100)
                    return 0
                    ;;
            esac
            return 1
            ;;
        selector_limit)
            case "${name}" in
                resnet18steps5_train_ins25_lr0.0005base1.0other1.0select_wd0.0001select0.0001_e100|\
                *selview_expanded_family_train*|\
                *selview_foreshortened_family_train*|\
                *selview_foreshortened_family_remainder_train*|\
                *selview_remainder_train*)
                    return 0
                    ;;
            esac
            return 1
            ;;
        *)
            echo "ERROR: unknown COMPARISON_SET=${COMPARISON_SET}. Use all, freeze, or selector_limit."
            exit 1
            ;;
    esac
}

# Collect work items: (rep, exp_dir, freeze_epoch, arch) tuples
WORK=()
for rep in ${REP_LIST}; do
    rep_dir="${META_LOGS}/${rep}"
    [ -d "${rep_dir}" ] || { echo "Skip ${rep_dir}: not a directory."; continue; }

    for exp_dir in "${rep_dir}"/*/; do
        [ -d "${exp_dir}" ] || continue
        exp_name=$(basename "${exp_dir}")

        if ! should_include_exp "${exp_name}"; then
            continue
        fi

        # Skip if no *_selection.json
        shopt -s nullglob
        sel_files=( "${exp_dir}"/*_selection.json )
        shopt -u nullglob
        if [ ${#sel_files[@]} -eq 0 ]; then
            echo "[skip] ${rep}/${exp_name} — no _selection.json"
            continue
        fi

        freeze_epoch=$(parse_freeze_epoch "${exp_name}")
        arch=$(parse_arch "${exp_name}")
        WORK+=("${rep}|${exp_dir}|${freeze_epoch}|${arch}")
    done
done

if [ ${#WORK[@]} -eq 0 ]; then
    echo "No stage-2 experiments found. Nothing to do."
    exit 0
fi

echo "Found ${#WORK[@]} stage-2 experiment(s) to evaluate."
echo

# Worker function
run_one() {
    local rep="$1" exp_dir="$2" freeze_epoch="$3" arch="$4" gpu="$5"
    local exp_name; exp_name=$(basename "${exp_dir}")
    local log_dir="${exp_dir}/temporal_test"
    mkdir -p "${log_dir}"
    local log_file="${log_dir}/run.log"
    echo "[gpu ${gpu}] ${rep}/${exp_name} (freeze=${freeze_epoch} arch=${arch}) → log ${log_file}"
    CUDA_VISIBLE_DEVICES=${gpu} python3 "${ROOT_DIR}/temporal_selection_test.py" \
        --selection_dir "${exp_dir}" \
        --dataset "${rep}" \
        --arch "${arch}" \
        --freeze_epoch "${freeze_epoch}" \
        --num_train_instances "${NUM_TRAIN_INSTANCES}" \
        --gpu_id 0 \
        ${extra_flags} \
        > "${log_file}" 2>&1
    local rc=$?
    if [ ${rc} -ne 0 ]; then
        echo "  [gpu ${gpu}] FAILED ${rep}/${exp_name} (rc=${rc}); see ${log_file}"
    else
        echo "  [gpu ${gpu}] done ${rep}/${exp_name}"
    fi
}
export -f run_one
export ROOT_DIR NUM_TRAIN_INSTANCES extra_flags

# Round-robin GPU dispatch
pids=()
idx=0
for item in "${WORK[@]}"; do
    IFS='|' read -r rep exp_dir freeze_epoch arch <<< "${item}"
    gpu=$((idx % GPUS))
    run_one "${rep}" "${exp_dir}" "${freeze_epoch}" "${arch}" "${gpu}" &
    pids+=($!)
    idx=$((idx + 1))
    # When all GPU slots are full, wait for them to clear before the next batch
    if [ $((idx % GPUS)) -eq 0 ]; then
        for pid in "${pids[@]}"; do wait "${pid}" || true; done
        pids=()
    fi
done

# Wait for any trailing partial batch
for pid in "${pids[@]}"; do wait "${pid}" || true; done

echo "=========================================="
echo "All experiments processed."
echo "Per-experiment outputs are in <exp_dir>/temporal_test/."
echo "=========================================="
