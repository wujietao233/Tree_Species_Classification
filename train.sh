#!/bin/bash

for LR in 0.001
do
  for dataset_dir in 'Trunk12'
  do
    for model_dir in 'mobilenetv2' 'mobilenetv2_elu' 'mobilenetv2_scse' 'mobilenetv2_scse_elu'
    do
      echo $dataset_dir $model_dir
      python train.py --dataset_dir=$dataset_dir --model_dir=$model_dir --LR=$LR --split_ratio=[6,2,2] --MAX_EPOCH=200 --BATCH_SIZE=64 --scheduler_factor=0.4 --seed=1 --scheduler_patience=5 --scheduler_cooldown=5 --scheduler_min_lr=1e-10
    done
  done
done
