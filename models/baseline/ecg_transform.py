import torch
from thop import profile, clever_format
from torch import nn
from torchprofile import profile_macs


class model_config():
    def __init__(self, input_channels=12, num_classes=9):
        super().__init__()
        # data parameters
        self.num_classes = num_classes
        self.sequence_len = 1000

        # model configs
        self.input_channels = input_channels  # 15
        self.kernel_size = 8
        self.stride = 1
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 128

        # Transformer
        self.trans_dim = 128
        self.num_heads = 4


class ecgTransForm(nn.Module):
    def __init__(self, configs):
        super(ecgTransForm, self).__init__()

        filter_sizes = [5, 9, 11]
        self.conv1 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[0],
                               stride=configs.stride, bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[1],
                               stride=configs.stride, bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[2],
                               stride=configs.stride, bias=False, padding=(filter_sizes[2] // 2))

        self.bn = nn.BatchNorm1d(configs.mid_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2)  # , padding=1
        self.do = nn.Dropout(configs.dropout)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # , padding=1
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # , padding=1
        )

        self.inplanes = 128
        self.crm = self._make_layer(SEBasicBlock, 128, 3)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=configs.trans_dim, nhead=configs.num_heads,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.aap = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(configs.final_out_channels, configs.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_in):

        # Multi-scale Convolutions
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)

        x_concat = torch.mean(torch.stack([x1, x2, x3], 2), 2)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))

        x = self.conv_block2(x_concat)
        x = self.conv_block3(x)

        # Channel Recalibration Module
        x = self.crm(x)

        # batch, channel, length => batch, length, channel [batch, num_token, dim_token]
        x = x.transpose(-2, -1)

        # Bi-directional Transformer
        x1 = self.transformer_encoder(x)
        x2 = self.transformer_encoder(torch.flip(x, [2]))
        x = x1 + x2

        x = x.transpose(-2, -1)

        x = self.aap(x)
        x_flat = x.reshape(x.shape[0], -1)
        x_out = self.clf(x_flat)
        return x_out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def getModel(num_classes=9, input_channels=12):
    config = model_config(num_classes=num_classes, input_channels=input_channels)
    model = ecgTransForm(config)
    return model


"""
    The code comes from: https://github.com/emadeldeen24/ECGTransForm
    paper: ECGTransForm: Empowering Adaptive ECG Arrhythmia Classification Framework with Bidirectional Transformer
    note: 1. The necessary hyperparameter adjustment is made
          2. The Transformer Block is transposed front to back => [batch, length, channel]⭐ 
          3. Why implement Bidirectional Transformer ?
"""


if __name__ == '__main__':
    x = torch.randn((1, 12, 1000))
    config = model_config()
    model = ecgTransForm(config)
    out = model(x)
    print(out.shape)

    # Params: 218.31K
    # FLOPs : 489.40M
    _, params = profile(model, inputs=(torch.randn(*(1, 12, 1000)),))
    print(f"Params: {clever_format(params)}")
    flops = profile_macs(model, torch.randn((1, 12, 1000)))
    print(f"FLOPs : {clever_format(flops)}")