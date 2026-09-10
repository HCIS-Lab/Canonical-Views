#!/usr/bin/env bash

if [ "$#" -lt 20 ]; then
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
# ./run_main.sh 10 7 main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 --dataset rgb --save_feature --freeze_epoch 10
# ./run_main.sh 10 5 main.py --epochs 100 --non_roll --steps 5 --num_train_instances 1 --dataset rgb --save_feature --freeze_epoch 30

# ./run_main.sh 10 1 main.py --epochs 3 --non_roll --steps 5 --num_train_instances 1 --dataset rgb --save_feature --freeze_epoch 1 --name test
# ./run_main.sh 5 4 main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 --dataset rgb --save_feature --freeze_epoch 50


# ./run_main.sh 5 5 main.py --epochs 100 --non_roll --steps 5 --num_train_instances 5 --dataset rgb --save_feature --freeze_epoch 50
# ./run_main.sh 7 5 main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 --dataset rgb --arch vit --batch_size 2 --lr 1e-5 --save_feature

# CUDA_VISIBLE_DEVICES=3 python main.py --epochs 10 --non_roll --num_train_instances 25 --dataset rgb --arch vit


# ./run_main.sh 10 0 main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 --dataset rgb --arch vit --batch_size 6 --lr 1e-5 --save_feature --freeze_backbone

# ./run_main.sh 10 2 main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 --dataset rgb --batch_size 6 --save_feature --selector_view_limit expanded_family
# ./run_main.sh 3 3 main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 --dataset rgb --batch_size 6 --lr 1e-5 --arch vit --freeze_backbone

# main.py --epochs 100 --non_roll --num_train_instances 25 --dataset rgb --weight_decay 5e-2 --lr 1e-4 --train_num_views 5
# ./run_main.sh 5 6 main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 --dataset rgb --batch_size 12 --lr 5e-5 --arch vit --freeze_backbone
# ./run_main.sh 5 8 main.py --non_roll --steps 3 --num_train_instances 25 --dataset rgb --batch_size 6 --lr 5e-4 --weight_decay 1e-3 --epochs 100 --save_feature --skip_stage1 
# python main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 --dataset rgb --save_feature --selector_view_limit expanded_family


# ./run_main.sh 5 0 main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 --dataset rgb --batch_size 6 --save_feature --skip_stage1 --lr 5e-4 --weight_decay 1e-3

# ./run_main.sh 5 1 main.py --epochs 100 --non_roll --steps 5 --num_train_instances 25 --dataset rgb --batch_size 6 --save_feature --selector_view_limit foreshortened_family_remainder --skip_stage1 --lr 5e-4 --weight_decay 1e-3
# ./run_main.sh 5 1 main.py --epochs 100 --non_roll --steps 1 --num_train_instances 25 --dataset rgb --batch_size 6 --save_feature --selector_view_limit foreshortened_family_remainder --skip_stage1 --lr 5e-4 --weight_decay 1e-3
