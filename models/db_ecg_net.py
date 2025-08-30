import torch
from thop import profile
import thop.utils as u
from torch import nn
from torch.nn import functional as F
from torchprofile import profile_macs
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Downsample(nn.Module):
    def __init__(self, channel, o_channel=None):
        super().__init__()
        if o_channel is None:
            o_channel = channel * 2
        self.downsample = nn.Conv1d(channel, o_channel, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.downsample(x)


class TokenEmbed(nn.Module):
    def __init__(self, channel, o_channel=None):
        super().__init__()
        if o_channel is None:
            o_channel = channel * 2
        self.downsample = nn.Linear(channel, o_channel)

    def forward(self, x):
        return self.downsample(x)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(2, keepdim=True)
            s = (x - u).pow(2).mean(2, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[None, :, None] * x + self.bias[None, :, None]
            return x


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1), bn.bias + (
            conv_bias - bn.running_mean) * bn.weight / std


def merge_into_large_kernel(large_kernel, small_kernel):
    large_k = large_kernel.size(2)
    small_k = small_kernel.size(2)
    pad_size = (large_k - small_k) // 2
    merged_kernel = large_kernel + F.pad(small_kernel, (pad_size, pad_size))
    return merged_kernel


class MSDWConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.is_reparam = False
        kernel_size = 9
        self.lk_origin = nn.Conv1d(channels, channels, kernel_size, stride=1,
                                   padding=kernel_size // 2, dilation=1, groups=channels, bias=False)

        self.kernel_sizes = [3, 5, 7]
        self.dilates = [1, 1, 1]

        for k, r in zip(self.kernel_sizes, self.dilates):
            self.__setattr__('sk_conv_k{}'.format(k),
                             nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                       padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                       bias=False))

    def forward(self, x):
        out = self.lk_origin(x)
        if self.is_reparam is False:
            for k in self.kernel_sizes:
                conv = self.__getattr__('sk_conv_k{}'.format(k))
                out = out + conv(x)
        return out

    def merge_branches(self):
        origin_k = self.lk_origin.weight
        for k in self.kernel_sizes:
            branch_k = self.__getattr__('sk_conv_k{}'.format(k)).weight
            origin_k = merge_into_large_kernel(origin_k, branch_k)
        merged_conv = nn.Conv1d(in_channels=origin_k.size(0), out_channels=origin_k.size(0),
                                kernel_size=origin_k.size(2), stride=1,
                                padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=False)
        merged_conv.weight.data = origin_k
        self.lk_origin = merged_conv
        for k in self.kernel_sizes:
            self.__delattr__('sk_conv_k{}'.format(k))
        self.is_reparam = True


class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pw_conv1 = nn.Conv1d(in_channels, in_channels * 2, kernel_size=1, bias=False)
        self.pw_norm1 = nn.BatchNorm1d(in_channels * 2)
        self.pw_act1 = nn.GELU()

        self.dw_conv = MSDWConvBlock(in_channels * 2)
        self.dw_norm = nn.BatchNorm1d(in_channels * 2)
        self.dw_act = nn.GELU()

        self.pw_conv2 = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.pw_norm2 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        x = self.pw_act1(self.pw_norm1((self.pw_conv1(x))))
        x = self.dw_act(self.dw_norm(self.dw_conv(x)))
        x = self.pw_norm2(self.pw_conv2(x))
        return x

    def reparameterize_block(self):
        if hasattr(self.pw_norm1, 'running_var'):
            weight, bias = fuse_bn(self.pw_conv1, self.pw_norm1)
            conv = nn.Conv1d(self.pw_conv1.in_channels, self.pw_conv1.out_channels, self.pw_conv1.kernel_size,
                             padding=self.pw_conv1.padding, groups=self.pw_conv1.groups, bias=True)
            conv.weight.data = weight
            conv.bias.data = bias
            self.pw_conv1 = conv
            self.pw_norm1 = nn.Identity()
        if hasattr(self.dw_norm, 'running_var'):
            self.dw_conv.merge_branches()
            weight, bias = fuse_bn(self.dw_conv.lk_origin, self.dw_norm)
            conv = nn.Conv1d(self.dw_conv.lk_origin.in_channels, self.dw_conv.lk_origin.out_channels, self.dw_conv.lk_origin.kernel_size,
                             padding=self.dw_conv.lk_origin.padding, groups=self.dw_conv.lk_origin.groups, bias=True)
            conv.weight.data = weight
            conv.bias.data = bias
            self.dw_conv.lk_origin = conv
            self.dw_norm = nn.Identity()
        if hasattr(self.pw_norm2, 'running_var'):
            weight, bias = fuse_bn(self.pw_conv2, self.pw_norm2)
            conv = nn.Conv1d(self.pw_conv2.in_channels, self.pw_conv2.out_channels, self.pw_conv2.kernel_size,
                             padding=self.pw_conv2.padding, groups=self.pw_conv2.groups, bias=True)
            conv.weight.data = weight
            conv.bias.data = bias
            self.pw_conv2 = conv
            self.pw_norm2 = nn.Identity()


class MS_IRB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels)
        self.func = InvertedBottleneck(in_channels)

    def forward(self, x):
        return x + self.func(self.norm(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn = None

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, l, c = x.shape

        qkv = self.qkv(x).reshape(b, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn = attn

        x = (attn @ v).transpose(1, 2).reshape(b, l, -1)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim, expand_ratio=2, act_layer=nn.GELU):
        super().__init__()
        hidden_dim = dim * expand_ratio
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.sa = Attention(dim=dim, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.attn = None

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, kv = x
        b, l, c = q.shape

        q = self.q_norm(q)
        kv = self.kv_norm(kv)

        q = self.q(q).reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(kv).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn = attn

        o = (attn @ v).transpose(1, 2).reshape(b, l, -1)
        o = self.proj(o)
        return o


class DualBranchBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.local_branch = MS_IRB(dim)
        self.global_branch = TransformerEncoder(dim, num_heads)
        self.local2Global = CrossAttention(dim, num_heads)

    def forward(self, x):
        feature_map, tokens_set = x
        feature_map = self.local_branch(feature_map)  # extract multi-scale waveform features
        tokens_set = tokens_set + self.local2Global((tokens_set, feature_map.transpose(-2, -1)))  # inject waveform detail
        tokens_set = self.global_branch(tokens_set)
        return feature_map, tokens_set


class DB_ECGNet(nn.Module):
    def __init__(self, input_channels=12, num_classes=9, patch_size=100):
        super().__init__()
        self.dim = [72, 144, 288]
        self.num_heads = [2, 4, 8]

        self.depth = [2, 3, 4]
        self.num_stage = len(self.depth)

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, self.dim[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.dim[0])
        )
        self.classifier = nn.Linear(self.dim[-1], num_classes)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim[0]))
        self.proj = nn.Conv1d(input_channels, self.dim[0], kernel_size=patch_size, stride=patch_size)
        num_patch = int(1000/patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patch, self.dim[0]))
        self.norm = nn.LayerNorm(self.dim[0])

        for i in range(self.num_stage):
            block = nn.Sequential()
            for j in range(self.depth[i]):
                block.append(DualBranchBlock(self.dim[i], num_heads=self.num_heads[i]))

            if i < self.num_stage - 1:
                setattr(self, f'downsample{i}', Downsample(self.dim[i], self.dim[i + 1]))
                setattr(self, f'token_embed{i}', TokenEmbed(self.dim[i], self.dim[i + 1]))
            setattr(self, f'block{i}', block)

    def reparameterize_model(self):
        if type(self.stem) is nn.Sequential:
            conv, bn = self.stem[0], self.stem[1]
            weight, bias = fuse_bn(conv, bn)
            temp = nn.Conv1d(conv.in_channels, conv.out_channels, conv.kernel_size,
                             stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True)
            temp.weight.data = weight
            temp.bias.data = bias
            self.stem = temp

        for i in range(self.num_stage):
            blocks = getattr(self, f'block{i}')
            for block in blocks:
                block.local_branch.func.reparameterize_block()

    def patch_embedding(self, x):
        x = self.norm(self.proj(x).transpose(-1, -2) + self.pos_embed)
        cls_token = self.norm(self.cls_token.expand(x.shape[0], -1, -1))
        return torch.cat((cls_token, x), dim=1)

    def forward(self, x):
        t = self.patch_embedding(x)
        x = self.stem(x)
        for i in range(self.num_stage):
            block = getattr(self, f'block{i}')
            x, t = block((x, t))
            if i < self.num_stage - 1:
                downsample = getattr(self, f'downsample{i}')
                token_embed = getattr(self, f'token_embed{i}')
                x = downsample(x)
                t = token_embed(t)

        cls_token = t[:, 0]
        x = self.classifier(cls_token)
        return x


def getDB_ECGNet_Patch10(input_channels=12, num_classes=9):
    return DB_ECGNet(input_channels, num_classes, patch_size=10)


def getDB_ECGNet_Patch20(input_channels=12, num_classes=9):
    return DB_ECGNet(input_channels, num_classes, patch_size=20)


def getDB_ECGNet_Patch50(input_channels=12, num_classes=9):
    return DB_ECGNet(input_channels, num_classes, patch_size=50)


"""
    Inspired by:
    1. Mobile-Former: Bridging MobileNet and Transformer
    2. VISION TRANSFORMER ADAPTER FOR DENSE PREDICTIONS
    3. UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition
"""


if __name__ == '__main__':
    x = torch.randn((1, 12, 1000))
    f = DB_ECGNet().eval()

    flops, params = profile(f, (x,))
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    origin_y = f(x)
    _, params = profile(f, inputs=(x,))
    macs = profile_macs(f, x)
    print(u.clever_format(params), u.clever_format(macs))

    f.reparameterize_model()

    reparams_y = f(x)

    print(reparams_y - origin_y)
    _, params = profile(f, inputs=(x,))
    macs = profile_macs(f, x)
    print(u.clever_format(params), u.clever_format(macs))