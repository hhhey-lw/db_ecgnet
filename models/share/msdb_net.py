import torch
from torch import nn
from torch.nn import functional as F


# Based on the InceptionTime block  =>  multi-scale
class CNNBlock(nn.Module):
    def __init__(self, input_channels=12):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=39, padding=19)
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=19, padding=9)
        self.conv8 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, padding=4)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1)
        self.conv9 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)

        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.conv3(out)

        outs = [self.conv5(self.maxpool(out))]
        out1 = self.conv4(out)
        outs.append(self.conv6(out1))
        outs.append(self.conv7(out1))
        outs.append(self.conv8(out1))

        out1 = self.bn(torch.cat(outs, dim=1))
        out2 = self.bn(self.conv9(out))
        out = self.relu(out1 + out2)
        return out


# Multi-Scale Embedding Layer  -- generate token
class MSEmbeddingLayer(nn.Module):
    def __init__(self, in_c=128, embed_dim=212, kernel_size=[5, 100], stride=5, padding=95):
        super().__init__()
        self.padding_left = padding
        self.conv1 = nn.Conv1d(in_channels=in_c, out_channels=embed_dim, kernel_size=kernel_size[0], stride=stride)
        self.conv2 = nn.Conv1d(in_channels=in_c, out_channels=embed_dim, kernel_size=kernel_size[1], stride=stride)

    def forward(self, X):
        out1 = self.conv1(X).transpose(1, 2)
        out2 = self.conv2(F.pad(X, (0, self.padding_left), "constant", 0)).transpose(1, 2)
        return torch.cat((out1, out2), dim=2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, l, c = x.shape

        qkv = self.qkv(x).reshape(b, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, l, -1)
        x = self.proj_drop(self.proj(x))
        # return x, attn
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, expand_ratio=4, act_layer=nn.GELU, drop=0.):
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


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.sa = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)

    def forward(self, x):
        # out, attn = self.sa(self.norm1(x))
        # out = x + out
        # out = out + self.ffn(self.norm2(out))
        # return out, attn
        x = x + self.sa(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers=8, num_heads=8, target_layer=4):
        super().__init__()
        # self.beta = 0.99
        self.target_layer = target_layer-1  # index
        self.blocks = nn.ModuleList(EncoderBlock(dim, num_heads=num_heads) for _ in range(num_layers))

    def forward(self, x):
        # ref => https://github.com/ChenMnZ/CF-ViT/blob/main/lvvit/models/lvvit.py#L205
        # global_attention = 0
        for i, block in enumerate(self.blocks):
            # x, attn = block(x)
            x = block(x)
            # if i >= self.target_layer:
            #     global_attention = self.beta * global_attention + (1 - self.beta) * attn
        # return x, global_attention
        return x


# Token selection
class TSEncoderBlock(nn.Module):
    def __init__(self, num_layers=8, token_dim=424, eta=0.5, target_layers=4):
        super().__init__()
        # self.num_layers = num_layers
        self.token_dim = token_dim
        # self.eta = eta
        self.encoder = TransformerEncoder(dim=token_dim, num_layers=num_layers, target_layer=target_layers)

    def forward(self, x):
        out = x
        out = self.encoder(out)
        # out, global_attn = self.encoder(out)
        # cls_attn = global_attn.mean(dim=1)[:, 0, 1:].squeeze(dim=1)

        # here, to do token select
        # out [batch, num_token, dim_token]
        # cls = out[:, 0].unsqueeze(dim=1)  # [batch, 1, token_dim]
        # tokens = out[:, 1:]  # [batch, num_token, token_dim]

        # k = int(tokens.shape[1] * self.eta)
        # _, top_indices = torch.topk(cls_attn, k)
        # select top k tokens
        # tokens = torch.gather(tokens, dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[2]))
        # out = torch.cat((cls, tokens), dim=1)

        out = self.encoder(out)
        # return out[0]
        return out


'''
    Simple implementation, unofficial
    paper: A token selection-based multi-scale dual-branch CNN-transformer network for 12-lead ECG signal classification
'''


# Multi-Scale Dual-Branch CNN-Transformer
class MSDBNet(nn.Module):
    def __init__(self, input_channels=12, num_classes=9, drop_ratio=0.):
        super().__init__()
        self.cnn_block1 = CNNBlock(input_channels)
        self.cnn_block2 = CNNBlock(128)

        self.UpperMSEL = MSEmbeddingLayer(kernel_size=[20, 40], stride=20, padding=20)
        self.cls_token1 = nn.Parameter(torch.zeros(size=(1, 1, 424)))
        self.pos_embed1 = nn.Parameter(torch.empty(1, 51, 424).normal_(std=0.02))
        self.encoder1 = TSEncoderBlock(target_layers=4)

        self.LowerMSEL = MSEmbeddingLayer()
        self.cls_token2 = nn.Parameter(torch.zeros(size=(1, 1, 424)))
        self.pos_embed2 = nn.Parameter(torch.empty(1, 201, 424).normal_(std=0.02))
        self.encoder2 = TSEncoderBlock(num_layers=5, target_layers=3)

        self.fc = nn.Linear(424 * 2, num_classes)

    def forward(self, X):
        out = self.cnn_block1(X)
        out = self.cnn_block2(out)

        out1 = self.UpperMSEL(out)
        cls_token1 = self.cls_token1.expand(X.shape[0], -1, -1)
        out1 = torch.cat((cls_token1, out1), dim=1) + self.pos_embed1
        out1 = self.encoder1(out1)
        cls_upper = out1[:, 0]

        out2 = self.LowerMSEL(out)
        cls_token2 = self.cls_token2.expand(X.shape[0], -1, -1)
        out2 = torch.cat((cls_token2, out2), dim=1) + self.pos_embed2
        out2 = self.encoder2(out2)
        cls_lower = out2[:, 0]

        out = torch.cat((cls_upper, cls_lower), dim=1)
        return self.fc(out)


if __name__ == '__main__':
    input_size = (4, 12, 1000)
    data = torch.randn(size=input_size)

    # Training this model, the learning rate needs to be lowered

    net = MSDBNet()
    out = net(data)
    print(out.shape)
    