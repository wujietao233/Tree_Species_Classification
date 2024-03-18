from matplotlib import pyplot as plt
import os
import random
import shutil
from tqdm import tqdm
import torch

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, path):
        self.path = Path(path)
        # create an instance of SummaryWriter
        self.writer = SummaryWriter(self.path)

    def write(self, epoch, stage='Train', **kwargs):
    # record various of loss and distance

        if stage == 'Train':
            for key, value in kwargs.items():
                self.writer.add_scalar('Train/'+str(key), value, epoch)

        elif stage == 'Valid':
            for key, value in kwargs.items():
                self.writer.add_scalar('Valid/' + str(key), value, epoch)

        else:
            raise ValueError
        return


    def close(self):
        self.writer.close()
        return

def plot_curve(train_curve, valid_curve, test_accuracy, xlabel="", ylabel="", title="",savepath=""):
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


def eval_model(net, data_loader, device):
    """
    评估模型
    """
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
        # 这是验证集的准确率
        accuracy = correct.item() / total
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

