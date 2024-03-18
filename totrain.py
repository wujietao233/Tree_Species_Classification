import os

'''
tensorboard可视化命令：
tensorboard --logdir 'I:\Python program\mobilenet_tree\Tsc\ten_log'
tensorboard --logdir 'I:\Python program\mobilenet_tree\Tsc\ten_log\lr0.01_min1e-06_fac_0.25_2024-03-09_23.19.07' --port 6007
'''

dataset_dir = 'BarkVN-50'
model_dir = 'mobilenetv3_large'
LR = 0.0008
BATCH_SIZE = 64
MAX_EPOCH = 200
seed = 0
scheduler_factor = 0.5
scheduler_min_lr = 1e-06
split_ratio = "[6,2,2]"
# pkl_path = "weights/mobilenetv3_large/BarkVN-50/2024-03-08_19.50.47/last.pkl"

os.system(
    f"python train.py --BATCH_SIZE={BATCH_SIZE} --dataset_dir={dataset_dir} --model_dir={model_dir} --LR={LR} --split_ratio={split_ratio} --MAX_EPOCH={MAX_EPOCH} --scheduler_factor={scheduler_factor} --scheduler_min_lr={scheduler_min_lr} --seed={seed}")

# os.system(
#     f"python train.py --BATCH_SIZE={BATCH_SIZE} --dataset_dir={dataset_dir} --model_dir={model_dir} --LR={LR} --split_ratio={split_ratio} --MAX_EPOCH={MAX_EPOCH} --scheduler_factor={scheduler_factor} --seed={seed} --pkl_path={pkl_path}")