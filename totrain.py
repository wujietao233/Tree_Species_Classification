import os

dataset_dir = 'BarkVN-50'
model_dir = 'mobilenetv3_large'
LR = 0.001
BATCH_SIZE = 16
MAX_EPOCH = 200
seed = 0
scheduler_factor = 0.5
split_ratio = "[6,2,2]"

os.system(
    f"python train.py --BATCH_SIZE={BATCH_SIZE} --dataset_dir={dataset_dir} --model_dir={model_dir} --LR={LR} --split_ratio={split_ratio} --MAX_EPOCH={MAX_EPOCH} --scheduler_factor={scheduler_factor} --seed={seed}")
