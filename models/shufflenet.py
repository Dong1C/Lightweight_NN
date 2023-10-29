import torch
import torch.nn as nn


# 完成通道混洗
class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, num_channels, height, width)
        return x

# 嵌入通道混洗的新residual block
class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleUnit, self).__init__()

        mid_channels = in_channels // 4

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=groups),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        # 添加一个Identity操作以保证通道数量不变
        if in_channels != out_channels:
            self.re = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.re = lambda x: x
        self.shuffle = ShuffleBlock(groups)

    def forward(self, x):
        residual = self.re(x)
        x = self.residual(x)
        x += residual
        x = self.shuffle(x)
        return x

class ShuffleNet(nn.Module):
    def __init__(self, num_classes=1000, groups=3):
        super(ShuffleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(24, 144, 4, groups)
        self.stage3 = self._make_stage(144, 288, 8, groups)
        self.stage4 = self._make_stage(288, 576, 4, groups)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(576, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, groups):
        layers = [ShuffleUnit(in_channels, out_channels, groups)]
        for _ in range(1, num_blocks):
            layers.append(ShuffleUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Unit test
if __name__ == '__main__':
    import torch.nn.functional as F

    # 创建 MobileNet 模型实例 类型数量100 输入一个[1, 3, 32, 32]的张量来模拟
    x_in = torch.randn(1, 3, 32, 32)
    model = ShuffleNet(num_classes=100)
    x_out = model(x_in)
    print("Input X: ", x_in.shape)
    print("Output X: ", x_out.shape)
    # print(model)
