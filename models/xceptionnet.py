import torch
import torch.nn as nn

class DWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DWConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(XceptionBlock, self).__init__()
        self.sep_conv1 = DWConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sep_conv2 = DWConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        # print("----in the XceptionBlock----")
        residual = self.skip(x)
        x = self.sep_conv1(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.sep_conv2(x)
        x = self.pool(x)
        # print(x.shape)
        x = x + residual
        # print("----out Block----")
        return x

class XceptionNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(XceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.block1 = XceptionBlock(64, 128)
        self.block2 = XceptionBlock(128, 256)
        self.block3 = XceptionBlock(256, 728)
        self.block_seq = nn.Sequential(*[XceptionBlock(728, 728) for i in range(3)])
        self.block4 = XceptionBlock(728, 1024)
        self.sep_conv3 = DWConv2d(1024, 1536, kernel_size=3, padding=1)
        self.sep_conv4 = DWConv2d(1536, 2048, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block_seq(x)
        x = self.block4(x)
        x = self.sep_conv3(x)
        x = self.relu(x)
        x = self.sep_conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test1():
    # test1 of the DWConv2d
    x= torch.randn(1, 10, 100, 100)
    DWconv = DWConv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
    if DWconv(x).shape == torch.Size([1, 20, 100, 100]):
        return True
    else:
        return False


def test2():
    try:
        model = XceptionNet(num_classes=100)
        x_in = torch.randn(1, 3, 224, 224)
        if model(x_in).shape == torch.Size([1, 100]):
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def test3():
    # test3 of the XceptionBlock
    for c in range(32, 64):
        x = torch.randn(1, c, 30, 30)
        XB = XceptionBlock(c, 64)
        if XB(x).shape == torch.Size([1, 64, 15, 15]):
            continue
        else:
            return False
    return True


if __name__ == '__main__':
    print("test1 passed!" if test1() else "test1 failed...")
    print("test2 passed!" if test2() else "test2 failed...")
    print("test3 passed!" if test3() else "test3 failed...")


