"""
将ReLu激活函数全部改为ELU激活函数
ELU激活函数是对ReLU激活函数的改进版本，解决了ReLU在负数区域出现的梯度截断问题。
$$
ELU(x) = \begin{cases}
    x,if x\geq 0
    \alpha(e^x-1),if x<0
\end{cases}
$$
"""
from torch import nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


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


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
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
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_ELU(nn.Module):
    """
    在原先MobileNetV2的基础上将激活函数由ReLu6改为ELU
    """

    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2_ELU, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

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
        features.append(ConvBNPReLU(3, input_channel, stride=2))
        # building inverted residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNPReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
