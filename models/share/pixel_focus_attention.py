import torch
from torch import nn
import torch.nn.functional as F


# For more, check out "TransNeXt" - CVPR 2024
# AggregatedAttention
class PixelFocusAttention(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=5, sr_ratio=2):  # , shift_size=0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.sr_ratio = sr_ratio
        self.window_size = window_size  # * 2
        # self.shift_size = shift_size

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        if sr_ratio > 1:
            self.pool = nn.Sequential(
                nn.AvgPool1d(kernel_size=sr_ratio, stride=sr_ratio),
                nn.Conv1d(dim, dim, kernel_size=1),
            )
            self.norm = nn.Sequential(
                nn.LayerNorm(dim),
                nn.GELU()
            )

        self.proj = nn.Linear(dim, dim)

        # self.learnable_tokens = nn.Parameter(
        #     nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.window_size), mean=0, std=0.02))
    # def getMask(self, num_token, window_size, shift_size, device, dtype):
    #     mask = torch.zeros((num_token, window_size), dtype=dtype, device=device)
    #     pattern = torch.tensor([[0, 0, 0, float(-100.0), float(-100.0)]] * (window_size - shift_size) +
    #                            [[float(-100.0), float(-100.0), float(-100.0), 0, 0]] * shift_size
    #                            , dtype=dtype, device=device)
    #     mask[-window_size:] = pattern
    #     return mask

    def forward(self, x):
        B, N, C = x.shape

        # if self.shift_size > 0:
        #     x = torch.roll(x, shifts=-self.shift_size, dims=1)  # 往左移动
        #     mask = self.getMask(N, self.window_size, self.shift_size, x.device, x.dtype)
        # else:
        #     mask = None

        # q = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, N // self.window_size, self.window_size, 2, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        k_local, v_local = kv[0], kv[1]
        # k_local = F.normalize(k_local, dim=-1)
        # Simple implementation => The tokens in the window share the same local neighbor
        k_local = k_local.repeat_interleave(self.window_size, dim=-3)
        v_local = v_local.repeat_interleave(self.window_size, dim=-3)

        attn_local = (q.unsqueeze(-2) @ k_local.transpose(-2, -1)).squeeze(-2) * self.scale
        # if mask is not None:
        #     attn_local += mask.unsqueeze(0).unsqueeze(0)

        if self.sr_ratio > 1:
            x = x.transpose(1, 2)
            x = self.pool(x)
            x = x.transpose(1, 2)
            x = self.norm(x)

        kv_pool = self.kv(x).reshape(B, -1, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)
        # k_pool = F.normalize(k_pool, dim=-1)

        attn_pool = q @ k_pool.transpose(-2, -1) * self.scale

        # Key tokens are collected from local neighbors[Fine grained] and global pooling features[coarse-grained]
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)

        attn_local, attn_pool = torch.split(attn, [self.window_size, N // self.sr_ratio], dim=-1)
        x_local = (attn_local.unsqueeze(-2) @ v_local).squeeze(-2)
        # x_local = (((q @ self.learnable_tokens) + attn_local).unsqueeze(-2) @ v_local).squeeze(-2)

        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)

        # if self.shift_size > 0:
        #     x = torch.roll(x, shifts=self.shift_size, dims=(1))

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, l, c = x.shape

        qkv = self.qkv(x).reshape(b, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(b, l, -1)

        x = self.proj(x)
        return x

