import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WiderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0)

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x_res = self.residual(x)
        return x_shortcut + x_res


class WRN(nn.Module):
    def __init__(self, depth = 10, k = 5, num_classes = 60, init_weights = True):
        super().__init__()
        N = int((depth-4)/6)
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = self._make_layer(16*k, N, 1)
        self.conv3 = self._make_layer(32*k, N, 2)
        self.conv4 = self._make_layer(64*k, N, 2)
        self.bn = nn.BatchNorm2d(64*k)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64*k, num_classes)

        # weight initialization
        if init_weights:
            self._weights_initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.sigmoid(self.fc(x))

        return x

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(WiderBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    # weight initialization
    def _weights_initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)