"""
Code created by JunYan Wang
Submission date 2023-4-13
GitHub:https://github.com/OnlyForWW/MobileNet-Pro
"""

import torch
import torch.nn as nn

class DSConv(nn.Module):
    def __init__(self, in_chs, out_chs, stride, ratio, k):
        super(DSConv, self).__init__()
        self.stride = stride
        self.in_chs = int(ratio * in_chs)
        self.out_chs = int(ratio * out_chs)
        self.dw_conv = nn.Conv2d(in_channels=self.in_chs,
                                 out_channels=self.in_chs,
                                 kernel_size=k,
                                 stride=stride,
                                 groups=self.in_chs,
                                 bias=False,
                                 padding=(k // 2))
        self.pw_conv = nn.Conv2d(in_channels=self.in_chs,
                                 out_channels=self.out_chs,
                                 kernel_size=1,
                                 stride=1,
                                 bias=False,
                                 padding=0)
        self.bn_dw = nn.BatchNorm2d(self.in_chs)
        self.bn_pw = nn.BatchNorm2d(self.out_chs)
        self.act = nn.ReLU(inplace=True)

        if stride == 2 or in_chs != out_chs:
            self.downsample = nn.Sequential(*[
                nn.Conv2d(in_channels=in_chs,
                          out_channels=out_chs,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(out_chs)
            ])
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        res = x
        x = self.dw_conv(x)
        x = self.bn_dw(x)
        x = self.act(x)
        x = self.pw_conv(x)
        x = self.bn_pw(x)

        identity = self.downsample(res)
        x = x + identity
        x = self.act(x)

        return x

class stem(nn.Module):
    def __init__(self, ratio):
        super(stem, self).__init__()
        self.out_chs = int(32 * ratio)
        self.conv = nn.Conv2d(in_channels=3,
                              out_channels=self.out_chs,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(self.out_chs)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class MobilNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, ratio=1.0, dropout=0.2):
        super(MobilNet, self).__init__()
        self.num_classes = num_classes
        self.ratio = ratio
        self.cfgs = cfgs
        self.stem = stem(ratio=self.ratio)
        self.dropout = nn.Dropout(dropout)

        layers = []
        ds_conv = DSConv
        for in_chs, out_chs, stride, k in self.cfgs:
            layers.append(ds_conv(in_chs, out_chs, stride, ratio=self.ratio, k=k))

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(int(self.ratio * 1024), self.num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

cfgs = [
    [32, 64, 1, 3],

    [64, 128, 2, 7],
    [128, 128, 1, 7],
    [128, 128, 1, 7],

    [128, 256, 2, 7],
    [256, 256, 1, 7],
    [256, 256, 1, 7],

    [256, 512, 2, 7],
    [512, 512, 1, 7],
    [512, 512, 1, 7],
    [512, 512, 1, 7],
    [512, 512, 1, 7],
    [512, 512, 1, 7],
    [512, 512, 1, 7],
    [512, 512, 1, 7],
    [512, 512, 1, 7],

    [512, 1024, 2, 7],
    [1024, 1024, 1, 7],
    [1024, 1024, 1, 7]
]

def MobileNetPro_100(**kwargs):
    return MobilNet(cfgs=cfgs, **kwargs)

def MobileNetV1_075(**kwargs):
    return MobilNet(cfgs=cfgs, ratio=0.75, **kwargs)

def MobileNetPro_050(**kwargs):
    return MobilNet(cfgs=cfgs, ratio=0.5, **kwargs)

if __name__ == '__main__':
    net = MobileNetPro_100(num_classes=7)
    x = torch.randn(1, 3, 224, 224)
    net.eval()
    o = net(x)
    print(o.shape)
