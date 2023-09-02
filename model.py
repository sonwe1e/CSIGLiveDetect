from torch import nn
import torch as t
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    實現子Module: Residual Block
    """

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, 1, 0, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 1, 1, 0, bias=True),
            nn.BatchNorm2d(outchannel),
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        resisdual = x if self.right is None else self.right(x)
        out += resisdual
        return F.relu(out)


class ResNet(nn.Module):
    """
    實現主Module: ResNet34
    ResNet34包含多個layer，每個layer又包含多個Residual block
    用子Module來實現Residual Block，用make_layer函數來實現layer
    """

    def __init__(self, num_classes=1):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 4, 4, 0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 分類的Layer，分別有3, 4, 6個Residual Block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        # 分類用的Fully Connection
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        構建Layer，包含多個Residual Block
        """
        shortcut = (
            nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                nn.BatchNorm2d(outchannel),
            )
            if stride == 1
            else nn.Sequential(
                nn.AvgPool2d(2, 2),
                nn.Conv2d(inchannel, outchannel, 1, 1, bias=False),
                nn.BatchNorm2d(outchannel),
            )
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = ResNet()
input = t.autograd.Variable(t.randn(1, 3, 224, 224))
o = model(input)
print(o)
