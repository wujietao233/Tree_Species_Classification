from matplotlib import pyplot as plt
import random
import shutil
from tqdm import tqdm
import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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


def plot_curve(train_curve, valid_curve, test_accuracy, xlabel="", ylabel="", title="", savepath=""):
    """
    绘制训练曲线
    """
    plt.figure(dpi=200)
    plt.plot(range(1, len(train_curve) + 1), train_curve, 'o-', label='train')
    plt.plot(range(1, len(valid_curve) + 1), valid_curve, 'o-', label='valid')
    plt.axhline(y=test_accuracy, xmin=0, xmax=1, label='test', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(savepath)
    # plt.show()


def make_new_directory(directory_path):
    """
    检查目录是否存在，若存在则清空，若不存在则创建
    """
    if os.path.exists(directory_path):
        # 遍历目录中的文件和子文件夹，然后删除
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    # 如果是文件，删除文件
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    # 如果是文件夹，递归删除文件夹及其内容
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"无法删除 {file_path}: {e}")

    else:
        os.makedirs(directory_path)


def make_directory(directory_path):
    """
    检查目录是否存在，若不存在则创建
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def eval_model(net, data_loader, num_classes, device):
    """
    评估模型，包括分类的准确率Accuracy、精确率Precision、召回率Recall、F1分数F1 Score
    """
    # 创建一个metrics的计算对象
    # metrics_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    # metrics_recall = torchmetrics.Recall(task="multiclass", average='none', num_classes=num_classes).to(device)
    # metrics_precision = torchmetrics.Precision(task="multiclass", average='none', num_classes=num_classes).to(device)

    correct = torch.tensor(0., device=device)
    total = 0.
    # 将网络模式设置为评估
    net.eval()
    # 不计算梯度
    with torch.no_grad():
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
        loop.set_description(f'Eval Model')
        for i, data in loop:
            # 获取数据
            img, label = data
            # img, label = Variable(img), Variable(label)
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
                # 预测标签
            out = net(img)
            # 获取预测的样本的标签
            _, predicted = torch.max(out.data, 1)
            # 计算样本总个数
            total += label.size(0)
            # 正确的个数
            correct += sum(predicted == label)
            # 每个batch进行计算迭代
            # metrics_acc(predicted, label)
            # metrics_recall(predicted, label)
            # metrics_precision(predicted, label)

        # 这是验证集的准确率
        accuracy = correct.item() / total
        # 添加精确率、召回率、F1分数
        # acc = metrics_acc.compute()
        # recall = metrics_recall.compute()
        # precision = metrics_precision.compute()

        return net, accuracy


def split_dataset(rootPath, seed=0, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    根据随机数种子，按一定比例随机划分训练集、验证集和测试集
    """

    # 设置随机数种子
    random.seed(seed)

    # 训练集、验证集、测试集比例
    sum_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= sum_ratio
    val_ratio /= sum_ratio
    test_ratio /= sum_ratio
    print(f'{train_ratio}:{val_ratio}:{test_ratio}')

    # 原始数据路径
    rawData = f'{rootPath}/rawData'
    # 划分后数据路径
    splitData = f'{rootPath}/splitData'

    # 检查文件夹
    make_new_directory(f'{splitData}/train')
    make_new_directory(f'{splitData}/valid')
    make_new_directory(f'{splitData}/test')

    # 遍历每一个文件夹
    for sub_dir in tqdm(os.listdir(rawData)):

        # 获取当前文件夹下的所有文件名
        sub_listdir = os.listdir(f"{rawData}/{sub_dir}")

        # 计算各部分的样本数量
        total_samples = len(sub_listdir)
        train_samples = int(total_samples * train_ratio)
        val_samples = int(total_samples * val_ratio)
        test_samples = total_samples - train_samples - val_samples

        # 随机打乱
        random.shuffle(sub_listdir)

        # 获取训练集、验证集、测试集
        train_set = sub_listdir[:train_samples]
        val_set = sub_listdir[train_samples:train_samples + val_samples]
        test_set = sub_listdir[train_samples + val_samples:]

        # 确保划分比例正确
        assert len(train_set) == train_samples
        assert len(val_set) == val_samples
        assert len(test_set) == test_samples

        # 保存到目标文件夹
        for set, split_dir in zip([train_set, val_set, test_set], ['train', 'valid', 'test']):
            # 检查文件夹
            make_new_directory(f'{splitData}/{split_dir}/{sub_dir}')
            for img in set:
                # 保存文件
                shutil.copy2(f'{rawData}/{sub_dir}/{img}', f'{splitData}/{split_dir}/{sub_dir}/{img}')


def test_model(model_path, best=True, last=True):
    """
    model_path表示模型参数所在的文件夹的路径（不是模型参数文件的路径）
    best参数表示是否评估best.pkl
    last参数表示是否评估last.pkl
    """
    model_listdir = os.listdir(model_path)
    if "params.json" not in model_listdir:
        print("params.json不存在")
    with open(f"{model_path}/params.json") as fp:
        params_json = json.load(fp)

    if not best and "best.pkl" not in model_listdir:
        print("best.pkl不存在")
        return
    if not last and "last.pkl" not in model_listdir:
        print("last.pkl不存在")
        return

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    seed = params_json["seed"]
    dataset_dir = params_json["dataset_dir"]
    train_ratio, valid_ratio, test_ratio = eval(params_json["split_ratio"])
    BATCH_SIZE = params_json["BATCH_SIZE"]
    model_dir = params_json["model_dir"]

    # 检查文件夹
    make_directory(f"data/{dataset_dir}")

    print(f'根据随机数种子{seed}划分数据集{dataset_dir}..')
    split_dataset(f'data/{dataset_dir}', seed=seed, train_ratio=train_ratio, val_ratio=valid_ratio,
                  test_ratio=test_ratio)

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

    # 加载到cuda
    if torch.cuda.is_available():
        net.cuda()

    pkl_list = [f"{x}.pkl" for x, y in {"best": best, "last": last}.items() if y]
    result = {}
    for pkl in pkl_list:
        pkl_path = f"{model_path}/{pkl}"
        net.load_state_dict(torch.load(pkl_path))
        print(f"成功加载{pkl_path}参数")

        # 评估训练集
        net, train_accuracy_global = eval_model(net, train_loader, num_classes, device)
        # 评估验证集
        net, valid_accuracy_global = eval_model(net, valid_loader, num_classes, device)
        # 评估测试集
        net, test_accuracy = eval_model(net, test_loader, num_classes, device)
        print(f'train_acc={train_accuracy_global},valid_acc={valid_accuracy_global},test_acc={test_accuracy}')

        result[pkl] = [train_accuracy_global, valid_accuracy_global, test_accuracy]
    return result
