import math

import torch
import torch.nn as nn
from thop import profile, clever_format
from torchprofile import profile_macs


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample=None):
        super(ResBlock, self).__init__()
        padding = math.ceil((kernel_size-1)/2)
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.downsample = downsample
        if downsample is not None:
            self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            out = self.maxpool(out)
            identity = self.downsample(identity)

        out += identity

        return out


class ECGNet(nn.Module):

    def __init__(self, kernel_sizes=[15, 17, 19, 21], in_channels=12, fixed_kernel_size=17, num_classes=9):
        super(ECGNet, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.planes = 16
        self.parallel_conv = nn.ModuleList()

        for i, kernel_size in enumerate(kernel_sizes):
            padding = math.ceil((kernel_size - 1) / 2)
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=kernel_size, padding=padding)
            self.parallel_conv.append(sep_conv)

        self.out_channel = self.planes * 4
        self.bn1 = nn.BatchNorm1d(num_features=self.out_channel)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=fixed_kernel_size,
                               stride=2, padding=math.ceil((fixed_kernel_size - 1)/2))

        self.block = self._make_layer(kernel_size=fixed_kernel_size, stride=1)

        self.bn2 = nn.BatchNorm1d(num_features=self.out_channel)
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(160, 640)
        )
        self.rnn = nn.LSTM(input_size=12, hidden_size=40, num_layers=1, bidirectional=False)
        self.fc = nn.Linear(in_features=680, out_features=num_classes)

    def _make_layer(self, kernel_size, stride, num_blocks=15):
        layers = []
        base_channel = 32
        for i in range(num_blocks):
            if (i + 1) % 4 == 0:
                downsample = nn.Sequential(
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Conv1d(in_channels=self.out_channel, out_channels=self.out_channel+base_channel, kernel_size=1)
                )
                layers.append(ResBlock(in_channels=self.out_channel, out_channels=self.out_channel+base_channel, kernel_size=kernel_size,
                                       stride=stride, downsample=downsample))
                self.out_channel += base_channel
            else:
                downsample = None
                layers.append(ResBlock(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=kernel_size,
                                       stride=stride, downsample=downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        out_sep = []

        for i in range(len(self.kernel_sizes)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=1)  # => [b, 64, 1000]
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)  # out => [b, 64, 500]

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)  # out => [b, 640]

        rnn_out, (rnn_h, rnn_c) = self.rnn(x.permute(2, 0, 1))  # rnn input shape: [len_seq, batch, dim_feature]
        new_rnn_h = rnn_h[-1, :, :]  # rnn_h => [b, 40]

        new_out = torch.cat([out, new_rnn_h], dim=1)  # out => [b, 680]
        result = self.fc(new_out)  # out => [b, 20]

        return result


def getECGNet(input_channels=12, num_classes=9):
    return ECGNet(in_channels=input_channels, num_classes=num_classes)


"""
    paper: ECGNet: Deep Network for Arrhythmia Classification
    source code: https://github.com/Amadeuszhao/SE-ECGNet
    modification: 1. inception block result => torch.cat(out_sep, dim=1), in channel dimension
                  2. According to the original paper, in the self.block, the length of each four layers is halved,
                     To reduce the number of parameters, the number of channels is increased by only 32
    note: In RNN, following the code author, we treat the ECG signal length as the sequence length 
          and the number of leads as the input feature dimension.
"""


if __name__ == '__main__':
    input = torch.randn(1, 12, 1000)
    model = ECGNet()
    output = model(input)
    print(output.shape)

    _, params = profile(model, inputs=(torch.randn(*(1, 12, 1000)),))
    print(f"Params: {clever_format(params)}")
    flops = profile_macs(model, torch.randn((1, 12, 1000)))
    print(f"FLOPs : {clever_format(flops)}")