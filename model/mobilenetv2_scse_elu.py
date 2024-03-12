"""激活函数为el包括scse"""

from torch import nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    确保所有层的通道数是8的倍数的函数。

    参数:
    - ch (int): 输入通道数。
    - divisor (int): 用于确保通道数是8的倍数的除数，默认为8。
    - min_ch (int): 最小通道数，如果未指定，则默认为除数的值。

    返回:
    int: 最接近输入通道数的8的倍数的通道数。
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)

    # 确保向下取整不低于原通道数的10%。
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class SCSEBlock(nn.Module):
    """
    实现了SCSE（Spatial and Channel-wise Squeeze & Excitation）块的模块。

    参数:
    - channel (int): 输入特征图的通道数。
    - reduction (int): 通道数减少的比例，默认为2。

    属性:
    - cSE (nn.Module): 通道注意力机制（Channel-wise Squeeze & Excitation）模块。
    - sSE (nn.Module): 空间注意力机制（Spatial Squeeze & Excitation）模块。

    方法:
    - forward(x): 前向传播函数，接收输入张量 x，并返回处理后的张量。
    """

    def __init__(self, channel, reduction=2):
        super(SCSEBlock, self).__init__()
        squeeze_c = _make_divisible(channel // reduction, 8)

        # 通道注意力机制
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, squeeze_c, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(squeeze_c, channel, 1),
            nn.Sigmoid()
        )

        # 空间注意力机制
        self.sSE = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        SCSE 块的前向传播函数。

        参数:
        - x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 处理后的张量。
        """
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBNPReLU(nn.Sequential):
    """
    包含卷积、批归一化和PReLU激活的序列模块。

    参数:
    - in_channel (int): 输入通道数。
    - out_channel (int): 输出通道数。
    - kernel_size (int): 卷积核大小，默认为3。
    - stride (int): 步幅大小，默认为1。
    - groups (int): 分组卷积参数，默认为1。当 groups=1 时，是普通的卷积操作。

    属性:
    - Conv2d (nn.Conv2d): 卷积层。
    - BatchNorm2d (nn.BatchNorm2d): 批归一化层。
    - PReLU (nn.PReLU): Parametric Rectified Linear Unit（PReLU）激活层。

    方法:
    该类继承自 nn.Sequential，因此包含了序列模块的基本功能。

    使用示例:
    ```python
    conv_block = ConvBNPReLU(in_channel=3, out_channel=64, kernel_size=3, stride=1, groups=1)
    output = conv_block(input_tensor)
    ```
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        """
        初始化 ConvBNPReLU 模块。

        参数:
        - in_channel (int): 输入通道数。
        - out_channel (int): 输出通道数。
        - kernel_size (int): 卷积核大小，默认为3。
        - stride (int): 步幅大小，默认为1。
        - groups (int): 分组卷积参数，默认为1。当 groups=1 时，是普通的卷积操作。
        """
        padding = (kernel_size - 1) // 2  # 计算卷积的padding值
        super(ConvBNPReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),  # 批归一化层
            nn.ELU(inplace=True)  # 使用ELU激活函数
        )


# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):   #groups=1时是普通的卷积操作
#         padding = (kernel_size - 1) // 2  #若kernel_size是7*7，5*5，3*3，1*1常见的则padding是    3，2 ，1 ，0
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True)
#         )
# class scse_layer(nn.Module):
#     def __init__(self, input_c: int, squeeze_factor: int = 4):
#         super(SqueezeExcitation, self).__init__()
#         squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
#         self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
#         self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)
#
#     def forward(self, x: Tensor) -> Tensor:
#         scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # 改变长和宽为1x1，不改变channel
#         scale = self.fc1(scale)
#         scale = F.relu(scale, inplace=True)
#         scale = self.fc2(scale)
#         scale = F.hardsigmoid(scale, inplace=True)
#         return scale * x

class InvertedResidual(nn.Module):
    """
    实现了倒残差模块（Inverted Residual Block）。

    参数:
    - in_channel (int): 输入通道数。
    - out_channel (int): 输出通道数。
    - stride (int): 步幅大小。
    - expand_ratio (int): 扩展比例，用于控制中间隐藏通道的扩展。

    属性:
    - use_shortcut (bool): 是否使用快捷连接（shortcut connection）。
    - conv (nn.Sequential): 倒残差模块的主要卷积操作序列。

    方法:
    - forward(x): 前向传播函数，接收输入张量 x，并返回处理后的张量。

    使用示例:
    ```python
    inverted_residual_block = InvertedResidual(in_channel=64, out_channel=128, stride=2, expand_ratio=6)
    output = inverted_residual_block(input_tensor)
    ```

    注意:
    - 当 `stride` 为1且 `in_channel` 等于 `out_channel` 时，将使用快捷连接。
    - 该类中包含了一系列卷积操作，其中包括深度可分离卷积、通道注意力机制等。
    """

    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        """
        初始化 InvertedResidual 模块。

        参数:
        - in_channel (int): 输入通道数。
        - out_channel (int): 输出通道数。
        - stride (int): 步幅大小。
        - expand_ratio (int): 扩展比例，用于控制中间隐藏通道的扩展。
        """
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNPReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNPReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # SCSEBlock
            SCSEBlock(hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """
        InvertedResidual 模块的前向传播函数。

        参数:
        - x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 处理后的张量。
        """
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_SCSE_ELU(nn.Module):
    """
    实现了 MobileNetV2 模型。

    参数:
    - num_classes (int): 分类任务的类别数，默认为1000。
    - alpha (float): 控制模型的宽度倍数，默认为1.0。
    - round_nearest (int): 用于计算通道数的最近倍数，默认为8。

    属性:
    - features (nn.Sequential): MobileNetV2 的主要特征提取部分。
    - avgpool (nn.AdaptiveAvgPool2d): 自适应平均池化层。
    - classifier (nn.Sequential): 分类器部分，包括全局平均池化和线性层。

    方法:
    - forward(x): 前向传播函数，接收输入张量 x，并返回模型的输出。

    使用示例:
    ```python
    mobilenet_v2 = MobileNetV2(num_classes=10, alpha=1.0, round_nearest=8)
    output = mobilenet_v2(input_tensor)
    ```

    注意:
    - MobileNetV2 模型结构包含了一系列的倒残差模块（Inverted Residual Blocks）和卷积层。
    - 模型的输入通道数为3（RGB图像）。
    - 分类器部分包括全局平均池化和线性层，输出为类别数。
    - 模型的权重采用了特定的初始化方式。
    """

    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        """
        初始化 MobileNetV2 模型。

        参数:
        - num_classes (int): 分类任务的类别数，默认为1000。
        - alpha (float): 控制模型的宽度倍数，默认为1.0。
        - round_nearest (int): 用于计算通道数的最近倍数，默认为8。
        """

        super(MobileNetV2_SCSE_ELU, self).__init__()
        block = InvertedResidual  # 定义一个名为block的类别别名，用于构建倒转残差块
        input_channel = _make_divisible(32 * alpha, round_nearest)  # 计算输入通道数，确保可被round_nearest整除
        last_channel = _make_divisible(1280 * alpha, round_nearest)  # 计算输出通道数，确保可被round_nearest整除

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNPReLU(3, input_channel, stride=2))  # 添加第一层卷积-BN-ReLU层
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1  # 针对bottleneck，每层的第一次步距为s，其他层（n-1）步距为1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))  # 添加倒转残差块
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNPReLU(input_channel, last_channel, 1))  # 添加最后一层卷积-BN-ReLU层
        # combine feature layers
        self.features = nn.Sequential(*features)  # 将所有特征层组合成一个Sequential模块

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 添加自适应平均池化层
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )  # 添加分类器，包含Dropout和全连接层

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')  # 卷积层参数初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)  # BN层参数初始化
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 全连接层参数初始化
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        MobileNetV2 模型的前向传播函数。

        参数:
        - x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 模型的输出张量。
        """

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
