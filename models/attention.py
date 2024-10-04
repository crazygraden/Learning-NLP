import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class DilatedConvFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, kernel_size, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size, padding=20),  # Linear transformation
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, dim, kernel_size, dilation=2, padding=20),  # Linear transformation with dilation
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        before_net = x
        x = self.net(x.permute(0, 2, 1))  # Permute dimensions to [batch_size, input_dim, sequence_length]
        output_shape = x.shape[-1]
        padding_length = before_net.shape[-2] - output_shape
        x_padded = F.pad(x, (0, padding_length))
        return x_padded.permute(0, 2, 1)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # ================== my =============================
        distances = torch.cdist(q, k, p=2)  # 计算 Q 和 K 之间的欧氏距离
        distance_weights = torch.exp(-distances)  # 使用高斯核函数计算权重，距离越近权重越大
        # 通过乘以权重调整点积结果
        adjusted_dots = dots * distance_weights
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            adjusted_dots.masked_fill_(~mask, float('-inf'))
            del mask
        attn = adjusted_dots.softmax(dim=-1)

        # # # ====================================================
        # if mask is not None:
        #     mask = F.pad(mask.flatten(1), (1, 0), value=True)
        #     assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
        #     mask = mask[:, None, :] * mask[:, :, None]
        #     dots.masked_fill_(~mask, float('-inf'))
        #     del mask
        # attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, kernel_size, dropout, flag):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))) if flag == 0 else Residual(PreNorm(dim, DilatedConvFeedForward(dim, mlp_dim, kernel_size=kernel_size, dropout=dropout)))
                # Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        # for attn, conv, ff in self.layers:
        #     x = attn(x, mask=mask)
        #     x = conv(x)
        #     x = ff(x)
        return x

class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, kernel_size, flag, timesteps, channels=1, dropout=0.1):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, kernel_size, dropout, flag)
        self.to_c_token = nn.Identity()
        self.timesteps = timesteps

    def forward(self, forward_seq, windows):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, -windows:, :])  # my_ideal
        # c_t = self.to_c_token(x[:, 0])    # orig
        return c_t

