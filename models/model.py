from torch import nn
import torch
from models.attention import Seq_Transformer


class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        self.configs = configs
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(128, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=2),
        )
        model_output_dim = configs.features_len
        # print(model_output_dim)
        # print(configs.final_out_channels)
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)
    def forward(self, x_in):
        # print(x_in.shape)    # torch.Size([32, 1, 2160])
        x_in = x_in.to(torch.float32)
        x = self.conv_block1(x_in)
        # print(x.shape)    # torch.Size([32, 32, 1081])
        x = self.conv_block2(x)
        # print(x.shape)    # torch.Size([32, 64, 540])
        x = self.conv_block3(x)
        # print(x.shape)      # torch.Size([32, 128, 268])
        x = self.conv_block4(x)
        # print(x.shape)      # torch.Size([32, 256, 131])
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x