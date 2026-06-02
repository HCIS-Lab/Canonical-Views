#!/usr/bin/env bash

if [ "$#" -lt 10 ]; then
  echo "Usage: $0 N GPU_ID script.py [script_args...]"
  exit 1
fi

N=$1
GPU_ID=$2
SCRIPT=$3
shift 3   # remove N, GPU_ID, script from args

for ((i=1; i<=N; i++)); do
  echo "Run $i / $N on GPU $GPU_ID"
  CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$SCRIPT" "$@"
done

# steps, epochs, dataset, --non_roll
# ./run_downstream.sh 10 1 train_downstream.py --selected_view_type 4 --selected_rep depth --start 80 --end 100 --num_cam 5 --non_roll --num_train_instances 25 --epochs 100
./run_downstream.sh 10 3 train_downstream.py --selected_view_type 01234 --selected_rep rgb_depth_edge --start 80 --end 100 --num_cam 1 --non_roll --num_train_instances 10 --epochs 100
