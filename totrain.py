import os

# tensorboard可视化命令：
# tensorboard --logdir 'I:\Python program\mobilenet_tree\Tsc\ten_log'
# tensorboard --logdir 'I:\Python program\mobilenet_tree\Tsc\ten_log\lr0.01_min1e-06_fac_0.25_2024-03-09_23.19.07' --port 6007
# pkl_path = "weights/mobilenetv3_large/BarkVN-50/2024-03-08_19.50.47/last.pkl"

dataset_dir = 'Bark-Combination-88'
model_dir = 'vision_transformer'
LR = 0.001
BATCH_SIZE = 16
MAX_EPOCH = 100
seed = 0
scheduler_factor = 0.5
scheduler_min_lr = 1e-06
split_ratio = "[6,2,2]"

os.system(f"python train.py --BATCH_SIZE={BATCH_SIZE} --dataset_dir={dataset_dir} --model_dir={model_dir} --LR={LR} --split_ratio={split_ratio} --MAX_EPOCH={MAX_EPOCH} --scheduler_factor={scheduler_factor} --scheduler_min_lr={scheduler_min_lr} --seed={seed}")
