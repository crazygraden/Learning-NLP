import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # 注意力机制的具体实现可以根据需要进行修改

    def forward(self, query, key, value, attn_mask=None):
        # 实现注意力机制的前向传播
        return query, None  # 这里是一个占位符，返回一个模拟的输出和注意力权重


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        d_ff = d_ff or 4 * self.d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = x.view(x.shape[0], x.shape[2], self.d_model)
        x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = conv_layer(x.transpose(1, 2)).transpose(1, 2)
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        self.configs = configs

        attn_layers = [
            EncoderLayer(
                attention=MultiHeadAttention(d_model=configs.final_out_channels, num_heads=configs.num_heads),
                d_model=configs.final_out_channels
            ),
        ]
        conv_layers = [
            nn.Sequential(
                nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                          stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(configs.dropout)
            ),
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            ),
            nn.Sequential(
                nn.Conv1d(128, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=2),
                nn.BatchNorm1d(configs.final_out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=8, stride=2, padding=2),
            )
        ]

        # 创建 Encoder 实例
        self.encoder = Encoder(attn_layers=attn_layers, conv_layers=conv_layers)

        # 模型输出维度
        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x_in = x_in.to(torch.float32)
        x, attns = self.encoder(x_in)  # 获取编码后的输出和注意力权重
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x

import torch
import torch.nn as nn

# 假设输入
batch_size = 32
seq_len = 2160
input_channels = 1
final_out_channels = 256
# 示例输入数据
x_in = torch.randn(batch_size, input_channels, seq_len)
# 定义卷积层
conv_layers = nn.Sequential(
    nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
    nn.Dropout(0.5),

    nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=2),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=2, stride=2, padding=1),

    nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=2),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=4, stride=2, padding=1),

    nn.Conv1d(128, final_out_channels, kernel_size=8, stride=1, padding=2),
    nn.BatchNorm1d(final_out_channels),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=8, stride=2, padding=2),
)
# 应用卷积层
x_out = conv_layers(x_in)
# 打印形状
print("卷积后的形状:", x_out.shape)  # 形状应该是 [32, 256, 2160]
# 转换形状以适应 LayerNorm
x_out = x_out.permute(0, 2, 1)  # 将形状从 [32, 256, 2160] 转换为 [32, 2160, 256]
# 定义 LayerNorm
layer_norm = nn.LayerNorm(final_out_channels)
# 应用 LayerNorm
x_out = layer_norm(x_out)
# 打印最终形状
print("LayerNorm 之后的形状:", x_out.shape)  # 形状应该是 [32, 2160, 256]
