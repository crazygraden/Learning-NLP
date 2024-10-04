import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer


class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        self.flag = configs.TC.flag
        self.windows = configs.TC.windows
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels,
                                               dim=configs.TC.hidden_dim,
                                               depth=8,
                                               heads=4,
                                               mlp_dim=512,
                                               kernel_size=configs.kernel_size,
                                               flag=self.flag,
                                               timesteps=self.timestep)

    # # # ============================== orig ================================
    # def forward(self, z_aug1, z_aug2):
    #     seq_len = z_aug1.shape[2]
    #
    #     z_aug1 = z_aug1.transpose(1, 2)
    #     z_aug2 = z_aug2.transpose(1, 2)
    #
    #     batch = z_aug1.shape[0]
    #     t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(
    #         self.device)  # randomly pick time stamps
    #
    #     nce = 0  # average over timestep and batch
    #     encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
    #
    #     for i in np.arange(1, self.timestep + 1):
    #         encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
    #     forward_seq = z_aug1[:, :t_samples + 1, :]
    #
    #     c_t = self.seq_transformer(forward_seq, self.windows)
    #     # print(c_t.shape)
    #     pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
    #     for i in np.arange(0, self.timestep):
    #         linear = self.Wk[i]
    #         pred[i] = linear(c_t)
    #     for i in np.arange(0, self.timestep):
    #         total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
    #         nce += torch.sum(torch.diag(self.lsoftmax(total)))
    #     nce /= -1. * batch * self.timestep
    #     return nce, self.projection_head(c_t)
    # ======================== my ========================
    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        # print(seq_len)
        t_samples = torch.randint(self.windows, seq_len - self.timestep, size=(1,)).long().to(self.device)
        nce = 0  # average over timestep and batch
        # encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        encode_samples = torch.empty((self.timestep, batch, self.windows, self.num_channels)).float().to(self.device)
        for i in np.arange(1, self.timestep + 1):
            start_idx = t_samples + i
            end_idx = start_idx + self.windows
            if end_idx > seq_len:  # 如果窗口超过了序列长度，调整窗口
                end_idx = seq_len
                start_idx = end_idx - self.windows
            encode_samples[i - 1] = z_aug2[:, start_idx:end_idx, :]
        forward_seq = z_aug1[:, :t_samples + 1, :]
        c_t = self.seq_transformer(forward_seq, self.windows)  # batchsize, windows, configs.TC.hidden_dim
        # ================================== my ========================================
        # 重新定义 pred，适应新的 encode_samples 维度
        pred = torch.empty((self.timestep, batch, self.windows, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)

        for i in np.arange(0, self.timestep):
            for j in np.arange(0, self.windows):
                encode_sample = encode_samples[i][:, j, :]  # shape: (batch, num_channels)
                pred_sample = pred[i][:, j, :]  # shape: (batch, num_channels)
                total = torch.mm(encode_sample, torch.transpose(pred_sample, 0, 1))  # shape: (batch, batch)
                nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep * self.windows
        return nce, self.projection_head(c_t.reshape(-1, c_t.shape[-1]))
