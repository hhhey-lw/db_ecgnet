import math
from typing import Tuple, List

import torch
from torch import Tensor
from torch import nn
from thop import profile, clever_format
from torchprofile import profile_macs


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.excitation = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=round(in_channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(in_channels // reduction), out_features=in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        B, C = x.shape[0], x.shape[1]
        out = self.squeeze(x)
        out = out.reshape(B, C)
        out = self.excitation(out)
        out = out.reshape(B, C, 1, 1)
        return x * out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: Tuple, stride: Tuple, dropout=0.):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        padding = (math.ceil((kernel_size[0] - stride[0]) / 2), math.ceil((kernel_size[1] - stride[1]) / 2))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(self.dropout(self.relu(self.bn(x))))


class DKRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: List[Tuple], stride: List[Tuple], dropout=0.):
        super().__init__()
        self.conv3xN = ConvBlock(in_channels, out_channels, kernel_size=kernel_size[0], stride=stride[0])
        self.conv1xN = ConvBlock(out_channels, out_channels, kernel_size=kernel_size[1], stride=stride[1],
                                 dropout=dropout)

    def forward(self, x):
        x = self.conv3xN(x)
        x = self.conv1xN(x)
        return x


class block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 3x15
        self.convBlock1 = ConvBlock(in_channels, out_channels, kernel_size=(3, 15), stride=(1, 2))
        self.se1 = SEBlock(in_channels=out_channels)

        self.conv1_residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                        stride=(1, 2))
        self.bn1_residual = nn.BatchNorm2d(num_features=out_channels)
        # 1x15 [x2]
        self.convBlock2 = ConvBlock(out_channels, out_channels, kernel_size=(1, 15), stride=(1, 1))
        self.convBlock3 = ConvBlock(out_channels, out_channels, kernel_size=(1, 15), stride=(1, 1))
        self.se2 = SEBlock(in_channels=out_channels)

        self.conv2_residual = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1),
                                        stride=(1, 1))
        self.bn2_residual = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        residual = self.bn1_residual(self.conv1_residual(x))
        out = self.convBlock1(x)
        out = self.se1(out)
        out = out + residual

        residual = self.bn2_residual(self.conv2_residual(out))
        out = self.convBlock2(out)
        out = self.convBlock3(out)
        out = self.se2(out)
        out = out + residual

        return out


class block2(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        # 3x15 => 1x15
        self.DKR1 = DKRBlock(in_channels, out_channels, kernel_size=[(3, 15), (1, 15)], stride=[(1, 2), (1, 1)])
        self.convResidual1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                       stride=(1, 2))
        self.bnResidual1 = nn.BatchNorm2d(num_features=out_channels)
        self.se1 = SEBlock(in_channels=out_channels)
        # 1x15 => 1x15 [x3]
        self.DKR2 = nn.Sequential(
            DKRBlock(out_channels, out_channels, kernel_size=[(1, 15), (1, 15)], stride=[(1, 1), (1, 1)],
                     dropout=dropout),
            DKRBlock(out_channels, out_channels, kernel_size=[(1, 15), (1, 15)], stride=[(1, 1), (1, 1)],
                     dropout=dropout),
            DKRBlock(out_channels, out_channels, kernel_size=[(1, 15), (1, 15)], stride=[(1, 1), (1, 1)],
                     dropout=dropout)
        )
        self.se2 = SEBlock(in_channels=out_channels)
        self.convResidual2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1),
                                       stride=(1, 1))
        self.bnResidual2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        residual = self.bnResidual1(self.convResidual1(x))
        out = self.DKR1(x)
        out = out + residual

        out = self.se1(out)

        residual = self.bnResidual2(self.convResidual2(out))
        out = self.DKR2(out)
        out = self.se2(out)
        out = out + residual

        return out


class block3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int, dropout=0.):
        super().__init__()
        # 3xN -> 1xN
        self.DKR1 = DKRBlock(in_channels, out_channels, kernel_size=[(3, kernel_size), (1, kernel_size)],
                             stride=[(1, 2), (1, 1)], dropout=dropout)
        self.convResidual1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                       stride=(1, 2))
        self.bnResidual1 = nn.BatchNorm2d(num_features=out_channels)
        self.se1 = SEBlock(in_channels=out_channels)
        # 1xN -> 1xN  [x5]
        self.DKR2 = nn.Sequential(
            DKRBlock(out_channels, out_channels, kernel_size=[(1, kernel_size), (1, kernel_size)],
                     stride=[(1, 1), (1, 1)], dropout=dropout),
            DKRBlock(out_channels, out_channels, kernel_size=[(1, kernel_size), (1, kernel_size)],
                     stride=[(1, 1), (1, 1)], dropout=dropout),
            DKRBlock(out_channels, out_channels, kernel_size=[(1, kernel_size), (1, kernel_size)],
                     stride=[(1, 1), (1, 1)], dropout=dropout),
            DKRBlock(out_channels, out_channels, kernel_size=[(1, kernel_size), (1, kernel_size)],
                     stride=[(1, 1), (1, 1)], dropout=dropout),
            DKRBlock(out_channels, out_channels, kernel_size=[(1, kernel_size), (1, kernel_size)],
                     stride=[(1, 1), (1, 1)], dropout=dropout)
        )
        self.se2 = SEBlock(in_channels=out_channels)

        self.convResidual2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1),
                                       stride=(1, 1))
        self.bnResidual2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        residual = self.bnResidual1(self.convResidual1(x))
        out = self.DKR1(x)
        out = out + residual
        out = self.se1(out)

        residual = self.bnResidual2(self.convResidual2(out))
        out = self.DKR2(out)
        out = self.se2(out)
        out = out + residual

        return out


'''
    paper: IM-ECG: An interpretable framework for arrhythmia detection using multi-lead ECG
    unofficial realization
'''


class IM_ECGNet(nn.Module):
    def __init__(self, num_classes=9, input_channels=12):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 50), stride=(1, 2), padding=(0, 24))
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()

        self.block1 = block1(in_channels=32, out_channels=64)
        self.block2 = block2(in_channels=64, out_channels=128, dropout=0.5)
        self.block3_1 = block3(in_channels=128, out_channels=256, kernel_size=5, dropout=0.5)
        self.block3_2 = block3(in_channels=128, out_channels=256, kernel_size=7, dropout=0.5)
        self.block3_3 = block3(in_channels=128, out_channels=256, kernel_size=15, dropout=0.5)
        self.pw_conv = nn.Conv2d(768, 768, kernel_size=1)

        self.avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        self.head = nn.Linear(in_features=768, out_features=num_classes)

    def forward(self, x):
        out = x.unsqueeze(dim=1)  # => shape: B, 1, C, L
        out = self.relu1(self.bn1(self.conv1(out)))

        out = self.block1(out)
        out = self.block2(out)

        out1 = self.block3_1(out)
        out2 = self.block3_2(out)
        out3 = self.block3_3(out)

        out = self.pw_conv(torch.cat((out1, out2, out3), dim=1))

        out = self.avg(out)
        out = self.head(out)
        return out


if __name__ == '__main__':
    model = IM_ECGNet()

    input_size = (1, 12, 1000)
    x = torch.randn(size=input_size)
    output = model(x)
    print(output.shape)

    # FLOPs : 21.25G
    # Params: 25.44M
    _, params = profile(model, inputs=(torch.randn(*(1, 12, 1000)),))
    # print(f"FLOPs : {clever_format(flops)}")
    print(f"Params: {clever_format(params)}")

    flops = profile_macs(model, torch.randn((1, 12, 1000)))
    print(f"FLOPs : {clever_format(flops)}")
