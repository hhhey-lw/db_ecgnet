import warnings

import torch
from torch import nn

warnings.filterwarnings("ignore", category=UserWarning)


# You can adjust reduction to 16
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels // reduction),
            nn.GELU(),
            nn.Linear(in_features=in_channels // reduction, out_features=in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.squeeze(x)
        out = out.view(out.size(0), -1)
        out = self.excitation(out)
        out = out.view(out.size(0), out.size(1), 1)
        out = out * x
        return out


# InceptionNext, or MixNet
class InceptionDWConv1d(nn.Module):
    def __init__(self, in_channels, kernel_size=[3, 7, 11, 19]):
        super().__init__()
        self.num_groups = len(kernel_size)
        gc = int(in_channels / self.num_groups)  # channel numbers of a convolution branch
        self.dwconv1 = nn.Conv1d(gc, gc, kernel_size=kernel_size[0], padding=kernel_size[0] // 2, groups=gc)
        self.dwconv2 = nn.Conv1d(gc, gc, kernel_size=kernel_size[1], padding=kernel_size[1] // 2, groups=gc)
        self.dwconv3 = nn.Conv1d(gc, gc, kernel_size=kernel_size[2], padding=kernel_size[2] // 2, groups=gc)
        self.dwconv4 = nn.Conv1d(gc, gc, kernel_size=kernel_size[3], padding=kernel_size[3] // 2, groups=gc)

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, self.num_groups, dim=1)
        return torch.cat((self.dwconv1(x1), self.dwconv2(x2), self.dwconv3(x3), self.dwconv4(x4)), dim=1)


# MobileNet-v3
class InvertedBottleneckBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pw_conv1 = nn.Conv1d(in_channels, in_channels * 2, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(in_channels * 2)
        self.act1 = nn.GELU()

        self.dw_conv = InceptionDWConv1d(in_channels * 2)
        self.norm2 = nn.BatchNorm1d(in_channels * 2)
        self.act2 = nn.GELU()

        self.se = SEBlock(in_channels * 2)

        self.pw_conv2 = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1)
        self.norm3 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        x = self.act1(self.norm1(self.pw_conv1(x)))
        x = self.act2(self.norm2(self.dw_conv(x)))
        x = self.se(x)
        x = self.norm3(self.pw_conv2(x))
        return x


# PVT-v2. take advantage of the redundancy of feature maps
class PoolAttention(nn.Module):
    def __init__(self, dim, num_heads=8, reduction=5, ratio=1., attn_drop=0.):
        super().__init__()
        self.reduction = reduction
        self.dim = int(dim * ratio)
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, self.dim)
        self.kv = nn.Linear(dim, self.dim * 2)
        if reduction > 1:
            self.pool = nn.Sequential(
                nn.AvgPool1d(kernel_size=reduction, stride=reduction),
                nn.Conv1d(dim, dim, kernel_size=1),
            )
            self.norm = nn.Sequential(
                nn.LayerNorm(dim),
                nn.GELU()
            )

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        b, l, c = x.shape

        q = self.q(x).reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        if self.reduction > 1:
            x = x.transpose(1, 2)
            x = self.pool(x)
            x = x.transpose(1, 2)
            x = self.norm(x)

        kv = self.kv(x).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, l, -1)
        return x


# MaxViT/MobileViT, another name stride-attention. Take advantage of the periodicity of the ECG
class GridAttention(nn.Module):
    def __init__(self, dim, window_size=5, num_heads=8, ratio=1., attn_drop=0.):
        super().__init__()
        self.window_size = window_size
        self.ratio = ratio
        self.num_heads = num_heads
        self.dim = int(dim * ratio)
        self.head_dim = self.dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, self.dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)

    # Such as => [1, 2, 3, 1, 2, 3, 1, 2, 3] => [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    def unfold(self, x, b, l, c):
        x = x.view(b, -1, self.window_size, c).transpose(1, 2).reshape(b * self.window_size, -1, c)
        return x

    def fold(self, x, b, l, c):
        x = x.view(b, self.window_size, -1, c).transpose(1, 2).reshape(b, l, c)
        return x

    def MHSA(self, x):
        b, l, c = x.shape

        qkv = self.qkv(x).reshape(b, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, l, -1)
        return x

    def forward(self, x):
        b, l, c = x.shape
        if self.window_size > 1:
            x = self.unfold(x, b, l, c)

        x = self.MHSA(x)

        if self.window_size > 1:
            x = self.fold(x, b, l, int(c * self.ratio))
        return x


# eg. Shunted-Attn/HiLo-Attn/P2T/CloFormer/Dilateformer...  => different attention heads are treated differently
class DualAttention(nn.Module):
    def __init__(self, dim, window_size=5, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        ratio = 0.5
        self.grid_attn = GridAttention(dim, window_size=window_size, num_heads=int(num_heads * ratio), ratio=ratio,
                                       attn_drop=attn_drop)
        self.pool_attn = PoolAttention(dim, reduction=window_size, num_heads=int(num_heads - num_heads * ratio),
                                       ratio=1 - ratio, attn_drop=attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # q, k, v = Conv1x1(x),   y=SA(q, k, v),   output = Conv1x1(y)
    def forward(self, x):
        x1 = self.grid_attn(x)
        x2 = self.pool_attn(x)

        x = torch.cat((x1, x2), dim=-1)
        x = self.proj_drop(self.proj(x))  # Fusion

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, expand_ratio=2, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = in_features
        hidden_features = in_features * expand_ratio
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channel, o_channel=None):
        super().__init__()
        if o_channel is None:
            o_channel = channel * 2
        self.downsample = nn.Conv1d(channel, o_channel, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.downsample(x)


# x = x + f(norm(x))
class Residual(nn.Module):
    def __init__(self, in_channel, f, norm, num_heads=None, window_size=None):
        super().__init__()
        if num_heads is None:
            self.f = f(in_channel)
        else:
            self.f = f(in_channel, num_heads=num_heads, window_size=window_size)
        self.norm = norm(in_channel)

    def forward(self, x):
        return x + self.f(self.norm(x))


# MetaFormer structure:  [token mixer]->[channel mixer]
class MetaBlock(nn.Module):
    def __init__(self, dim, num_heads=None, window_size=None):
        super().__init__()
        # Time Mixer
        self.local_perception = Residual(dim, InvertedBottleneckBlock, nn.BatchNorm1d)
        self.global_perception = Residual(dim, DualAttention, nn.LayerNorm, num_heads, window_size)
        # Channel Mixer
        self.ffn = Residual(dim, Mlp, nn.LayerNorm)

    def forward(self, x):
        # There are three steps: 1. Local aggregation 2. Global aggregation 3. Channel aggregation
        x = self.local_perception(x)
        x = x.transpose(-2, -1)  # pixel level token
        x = self.global_perception(x)
        x = self.ffn(x)
        x = x.transpose(-2, -1)
        return x


class ModelConfig:
    def __init__(self, dim: list, depth: list, num_heads: list, window_size: list):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size


class MetaECGNet(nn.Module):
    def __init__(self, input_channels=12, num_classes=9, config: ModelConfig = None):
        super().__init__()
        self.dim = config.dim
        self.depth = config.depth
        self.num_stage = len(self.depth)
        self.num_heads = config.num_heads
        self.window_size = config.window_size

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, self.dim[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.dim[0])
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.dim[-1], num_classes)
        )

        for i in range(self.num_stage):
            block = nn.Sequential()
            for _ in range(self.depth[i]):
                block.append(MetaBlock(self.dim[i], num_heads=self.num_heads[i], window_size=self.window_size[i]))
            if i < self.num_stage - 1:
                block.append(Downsample(self.dim[i], self.dim[i + 1]))
            setattr(self, f'block{i}', block)

    def forward(self, x):
        x = self.stem(x)

        for i in range(self.num_stage):
            block = getattr(self, f'block{i}')
            x = block(x)

        x = self.classifier(x)
        return x


# EfficientViT architecture. Because the network can perceive the global view, it does not stack four stages
def getMetaECGNet(input_channels=12, num_classes=9):
    dim = [144, 216, 288]
    depth = [2, 3, 4]
    num_heads = [4, 4, 4]
    window_size = [5, 5, 1]
    cfg = ModelConfig(dim, depth, num_heads, window_size)
    return MetaECGNet(input_channels, num_classes, cfg)


if __name__ == '__main__':
    size = (1, 12, 1000)
    x = torch.randn(size)
    model = getMetaECGNet()
    out = model(x)
    print(out.shape)
