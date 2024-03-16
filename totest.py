"""
批量评估模型准确率。
"""
import os
import json

dataset_list = ['BarkVN-50', 'Trunk12']
model_dir = 'mobilenetv2'

for dataset_dir in dataset_list:
    time_list = os.listdir(f"weights/{model_dir}/{dataset_dir}")
    for time in time_list:
        log_path = f"weights/{model_dir}/{dataset_dir}/{time}"
        log_dir = os.listdir(log_path)
        if 'best.pkl' in log_dir and 'last.pkl' in log_dir:
            print(log_path)
            with open(f"{log_path}/params.json") as fp:
                params_json = json.load(fp)
            LR = params_json["LR"]
            BATCH_SIZE = params_json["BATCH_SIZE"]
            MAX_EPOCH = 1
            seed = params_json["seed"]
            scheduler_factor = params_json["scheduler_factor"]
            split_ratio = params_json["split_ratio"]
            pkl_path = f'{log_path}/best.pkl'
            print(pkl_path)
            os.system(
                f"python train.py --BATCH_SIZE={BATCH_SIZE} --dataset_dir={dataset_dir} --model_dir={model_dir} --LR={LR} --split_ratio={split_ratio} --MAX_EPOCH={MAX_EPOCH} --scheduler_factor={scheduler_factor} --seed={seed} --pkl_path={pkl_path}")
