import torch
import torch.nn as nn
import torch.nn.functional as F

class FireModule(nn.Module):
    def __init__(self, in_channels, s1x1, e1x1, e3x3):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, s1x1, kernel_size=1)
        self.expand1x1 = nn.Conv2d(s1x1, e1x1, kernel_size=1)
        self.expand3x3 = nn.Conv2d(s1x1, e3x3, kernel_size=3, padding=1)

    def forward(self, x):
        squeeze_out = self.squeeze(x)
        expand1x1_out = self.expand1x1(squeeze_out)
        expand3x3_out = self.expand3x3(squeeze_out)
        output = torch.cat([expand1x1_out, expand3x3_out], 1)
        return output

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

# Unit test
if __name__ == '__main__':
    x_in = torch.randn(1, 3, 32, 32)
    model = SqueezeNet(num_classes=100)
    x_in = F.interpolate(x_in, size=(224, 224), mode='bilinear', align_corners=False)
    x_out = model(x_in)
    print("Input X: ", x_in.shape)
    print("Output X: ", x_out.shape)
    # print(model)

