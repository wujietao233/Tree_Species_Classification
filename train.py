import json
import pandas as pd
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

# 导入自定义数据集类和模型类
from data.dataset import ClassifierDataset
from model.mobilenetv2_elu import MobileNetV2_ELU
from model.mobilenetv2_scse import MobileNetV2_SCSE
from model.mobilenetv2_scse_elu import MobileNetV2_SCSE_ELU
from model.mobilenetv2 import MobileNetV2
from model.mobilenetv3 import MobileNetV3_large
from model.vggnet import vgg
from model.googlenet import GoogLeNet
from model.densenet import densenet121
from model.resnet import resnet50

# 导入自定义工具包
from utils import plot_curve, make_directory, split_dataset, eval_model, Logger

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# 宏定义一些数据，如epoch数，batchsize等
parser = argparse.ArgumentParser()
# 最大训练轮数
parser.add_argument('--MAX_EPOCH', type=int, default=100)
# 每个批次的样本数
parser.add_argument('--BATCH_SIZE', type=int, default=128)
# 学习率
parser.add_argument('--LR', type=float, default=0.001)
# 预训练模型参数
parser.add_argument('--pkl_path', type=str, default="")
# 训练参数json
parser.add_argument('--json_path', type=str, default="")
# 未知参数
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
# 是否使用学习率调度器？
parser.add_argument('--use_scheduler', type=int, default=1)
# 学习率调度器衰减倍率
parser.add_argument('--scheduler_factor', type=float, default=0.9)
# 学习率调度器衰减次数
parser.add_argument('--scheduler_patience', type=float, default=5)
# 学习率调度器最小值
parser.add_argument('--scheduler_min_lr', type=float, default=1e-5)
# 学习率调度器冷静次数
parser.add_argument('--scheduler_cooldown', type=float, default=5)
# 每隔多少个批次打印一次训练日志
parser.add_argument('--log_interval', type=int, default=3)
# 每隔多少个epoch进行一次验证
parser.add_argument('--val_interval', type=int, default=1)
# 训练的数据集名
parser.add_argument('--dataset_dir', type=str, default="Trunk")
# 训练的模型名
parser.add_argument('--model_dir', type=str, default="mobilenetv2")
# 数据形状
parser.add_argument('--split_ratio', type=str, default="(8, 1, 1)")
# 随机数种子
parser.add_argument('--seed', type=int, default=random.randint(0, 2 ** 32 - 1))
# 获取参数
args = parser.parse_args().__dict__

# 根据字典获取参数
MAX_EPOCH = args['MAX_EPOCH']
BATCH_SIZE = args['BATCH_SIZE']
LR = args['LR']
pkl_path = args['pkl_path']
betas = args['betas']
use_scheduler = args['use_scheduler']
scheduler_factor = args['scheduler_factor']
scheduler_patience = args['scheduler_patience']
scheduler_min_lr = args['scheduler_min_lr']
scheduler_cooldown = args['scheduler_cooldown']
log_interval = args['log_interval']
val_interval = args['val_interval']
dataset_dir = args['dataset_dir']
model_dir = args['model_dir']
train_ratio, valid_ratio, test_ratio = eval(args['split_ratio'])
seed = args['seed']


# 设置随机数种子
random.seed(seed)

# ============================ step 1/7 训练前准备 ============================

# 检查文件夹
make_directory(f"data/{dataset_dir}")

# 获取当前时间
strftime = time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime())

# 初始化TensorBoard
logger = Logger(f"ten_log\\lr{LR}_min{scheduler_min_lr}_fac_{scheduler_factor}_{strftime}")

# 创建文件夹
make_directory(f"weights/{model_dir}/{dataset_dir}/{strftime}")

# ============================ step 2/7 数据 ============================
# 随机划分数据集
print(f'根据随机数种子{seed}划分数据集{dataset_dir}..')
# split_dataset(f'data/{dataset_dir}', seed=seed, train_ratio=train_ratio, val_ratio=valid_ratio, test_ratio=test_ratio)
split_dataset(f'data/{dataset_dir}', seed=seed, train_ratio=train_ratio, val_ratio=valid_ratio, test_ratio=test_ratio)

# 定义数据分割目录和训练/验证数据目录
split_dir = os.path.join(".", f"data/{dataset_dir}", "splitData")  # 数据分割目录
train_dir = os.path.join(split_dir, "train")  # 训练数据目录
valid_dir = os.path.join(split_dir, "valid")  # 验证数据目录
test_dir = os.path.join(split_dir, "test")  # 测试数据目录

# 定义数据转换的字典，包含训练和验证两种模式的数据转换
data_transform = {
    "train": transforms.Compose([  # 训练模式的数据转换，使用Compose将多个数据转换操作组合在一起
        transforms.RandomResizedCrop(224),  # 随机裁剪图像并调整大小为224x224像素
        transforms.RandomHorizontalFlip(),  # 以50%的概率进行随机水平翻转
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 对张量进行归一化操作
    ]),
    "valid": transforms.Compose([  # 验证模式的数据转换，使用Compose将多个数据转换操作组合在一起
        transforms.Resize(256),  # 调整图像大小为256x256像素
        transforms.CenterCrop(224),  # 在中心位置裁剪图像为224x224像素
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 对张量进行归一化操作
    ]),
    "test": transforms.Compose([
        transforms.Resize(256),  # 调整图像大小为256x256像素
        transforms.CenterCrop(224),  # 在中心位置裁剪图像为224x224像素
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 对张量进行归一化操作
    ])
}

# 构建MyDataset实例
train_data = ClassifierDataset(
    data_dir=train_dir,
    transform=data_transform['train']
)
valid_data = ClassifierDataset(
    data_dir=valid_dir,
    transform=data_transform['valid']
)
test_data = ClassifierDataset(
    data_dir=test_dir,
    transform=data_transform['test']
)

# 构建DataLoader
# 训练集数据最好打乱
# DataLoader的实质就是把数据集加上一个索引号，再返回
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

# ============================ step 3/7 模型 ============================

# 根据参数确定模型
num_classes = len(os.listdir(train_dir))
dict_net = {
    'mobilenetv2': MobileNetV2(num_classes=num_classes),
    'mobilenetv2_elu': MobileNetV2_ELU(num_classes=num_classes),
    'mobilenetv2_scse': MobileNetV2_SCSE(num_classes=num_classes),
    'mobilenetv2_scse_elu': MobileNetV2_SCSE_ELU(num_classes=num_classes),
    'mobilenetv3_large': MobileNetV3_large(num_classes=num_classes),
    'vggnet16': vgg(model_name='vgg16', num_classes=num_classes, init_weights=False),
    'googlenet': GoogLeNet(num_classes=num_classes, aux_logits=False),
    'densenet121': densenet121(num_classes=num_classes),
    'resnet50': resnet50(num_classes=num_classes)
}
# 选择模型
net = dict_net[model_dir.lower()]
# 记录每一次的数据，方便绘图
train_curve = list()
valid_curve = list()
lr_curve = list()
train_accuracy_global = 0.0
valid_accuracy_global = 0.0
# 加载到cuda
if torch.cuda.is_available():
    net.cuda()
# 加载模型参数
if os.path.exists(pkl_path):
    net.load_state_dict(torch.load(pkl_path))
    print(f"成功加载{pkl_path}参数")
    # 保存学习率
    lr_curve.append(LR)
    # 评估训练集
    net, train_accuracy_global = eval_model(net, train_loader, device)
    train_curve.append(train_accuracy_global)
    # 评估验证集
    net, valid_accuracy_global = eval_model(net, valid_loader, device)
    valid_curve.append(valid_accuracy_global)
    # 评估测试集
    net, test_accuracy = eval_model(net, test_loader, device)
    print(f'train_acc={train_accuracy_global},valid_acc={valid_accuracy_global},test_acc={test_accuracy}')

# ============================ step 4/7 损失函数 ============================
criterion = nn.CrossEntropyLoss()
# ============================ step 5/7 优化器 ============================
optimizer = optim.AdamW(
    net.parameters(),
    lr=LR,
    betas=betas
)
# 设置一个默认值防止报错
scheduler = None
if use_scheduler:
    print("使用学习率调度器")
    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        # 检测loss是否不再提高
        mode='max',
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True,
        min_lr=scheduler_min_lr,
        cooldown=scheduler_cooldown
    )
# ============================ step 6/7 训练 ============================

# 最终的学习率
learning_rate = LR
# 训练
for epoch in range(MAX_EPOCH):
    # 误差参数
    train_correct = torch.tensor(0., device=device)
    train_total = 0.

    # 将网络模式设置为训练模型
    net.train()

    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    # 遍历训练集的每一个元素
    for i, data in loop:
        img, label = data
        # 设置允许自动求导
        img.requires_grad = True

        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        # 前向传播
        out = net(img)
        optimizer.zero_grad()  # 归0梯度
        loss = criterion(out, label)  # 得到损失函数

        # 获取loss的值
        print_loss = loss.data.item()
        # 反向传播
        loss.backward()
        # 优化
        optimizer.step()

        # 预测
        _, predicted = torch.max(out.data, 1)

        # 获取当前学习率
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        # 正确的个数
        correct = sum(predicted == label)
        # 样本的个数
        total = label.size(0)
        # 准确率
        acc = (correct / total).item()

        # 累加结果
        train_total += total
        train_correct += correct

        # 打印
        loop.set_description(f'{model_dir.capitalize()} {dataset_dir.capitalize()} Epoch [{epoch}/{MAX_EPOCH}]')
        loop.set_postfix(loss=print_loss, acc=acc, lr=learning_rate)

        logger.write(epoch, stage='Train', loss=print_loss, acc=acc, lr=learning_rate)

    # 计算训练集的准确率
    train_accuracy = train_correct.item() / train_total
    # 添加准确率
    train_curve.append(train_accuracy)
    # 记录最优的准确率
    if train_accuracy > train_accuracy_global:
        torch.save(net.state_dict(), f'weights/{model_dir}/{dataset_dir}/{strftime}/best.pkl')
        # 打印提示信息
        print(f"训练集准确率由：{100 * train_accuracy_global:.2f}%上升至：{100 * train_accuracy:.2f}%")
        print(f"已更新并保存权值为weights/{model_dir}/{dataset_dir}/{strftime}/best.pkl")
        train_accuracy_global = train_accuracy

    # 添加功能计算验证集的准确率
    net, valid_accuracy = eval_model(net, valid_loader, device)
    valid_curve.append(valid_accuracy)
    # 记录最优的准确率
    if valid_accuracy > valid_accuracy_global:
        # torch.save(net.state_dict(), f'weights/{model_dir}/{dataset_dir}/{strftime}/best.pkl')
        # 打印提示信息
        print(f"验证集准确率由：{100 * valid_accuracy_global:.2f}%上升至：{100 * valid_accuracy:.2f}%")
        # print(f"已更新并保存权值为weights/{model_dir}/{dataset_dir}/{strftime}/best.pkl")
        valid_accuracy_global = valid_accuracy

    # 添加功能记录学习率
    lr_curve.append(learning_rate)

    # 更新学习率
    if use_scheduler:
        # 更新学习率并监测验证集上的性能
        scheduler.step(train_accuracy)

    # 设置进度条右边的内容
    print(f'train_acc={train_accuracy}, valid_acc={valid_accuracy}')

    logger.write(epoch, stage='Valid', loss=print_loss, acc=valid_accuracy, lr=learning_rate)

# ============================ step 7/7 保存 ============================
# 添加功能计算测试集的准确率
net, test_accuracy = eval_model(net, test_loader, device)

# 保存最终学习率
args['final_learning_rate'] = learning_rate

# 保存设置的参数
params_path = f'weights/{model_dir}/{dataset_dir}/{strftime}/params.json'
with open(params_path, 'w') as f:
    print(f"模型参数为：")
    for k, v in args.items():
        print(f"{k}: {v}")
    f.write(json.dumps(args))

# 设置进度条右边的内容
print(f'test_acc={test_accuracy}')

# 添加csv功能
pd.DataFrame(
    data={
        "train_accuracy": train_curve,
        "valid_accuracy": valid_curve,
        'learning_rate': lr_curve,
    },
    index=range(1, len(train_curve) + 1)
).to_csv(f"weights/{model_dir}/{dataset_dir}/{strftime}/log.csv")

# 保存模型参数
torch.save(net.state_dict(), f'weights/{model_dir}/{dataset_dir}/{strftime}/last.pkl')
print(f"训练完毕，权重已保存为：weights/{model_dir}/{dataset_dir}/{strftime}/last.pkl")

# 绘制训练曲线
plot_curve(train_curve, valid_curve, test_accuracy, xlabel='epoch', ylabel='accuracy',
           title=f'{model_dir} {dataset_dir}',
           savepath=f'weights/{model_dir}/{dataset_dir}/{strftime}/train_{max(train_curve)}_valid_{max(valid_curve)}_test_{test_accuracy}.png')
# plt.show()

logger.close()
