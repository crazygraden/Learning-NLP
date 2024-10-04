# @File  : test.py
# @Author: fkx
# @Time: 2024/3/3 16:04 
# -*- coding: utf-8 -*-
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取CSV文件并转换为NumPy数组
# tensor_list = pd.read_csv('self_supervised_Fuzzy.csv').to_numpy()
#
# # 获取数据的行数和列数
# num_rows, num_cols = tensor_list.shape
#
# # 创建画布和子图
# fig, axs = plt.subplots(num_cols, 1, figsize=(10, 6*num_cols))
#
# # 绘制数据
# for i in range(num_cols):
#     axs[i].plot(range(num_rows), tensor_list[:, i])
#     axs[i].set_title('{}'.format(i+1))
#     axs[i].set_xlabel('Epoch')
#     axs[i].set_ylabel('Loss')
#
# # 显示图形
# plt.tight_layout()
# plt.show()


# import pandas as pd
# import numpy as np
# import re
#
# # 示例列表
# tensor_list = pd.read_csv('fine_tune.csv')
# # 获取列名
# column_names = tensor_list.columns.tolist()
# tensor_list = tensor_list.to_numpy()
# # 初始化 result_list
# result_list = np.zeros_like(tensor_list, dtype=float)
#
# # 遍历数组并提取数字
# for j in range(tensor_list.shape[0]):
#     for i in range(tensor_list.shape[1]):
#         data = tensor_list[j][i]
#         match = re.search(r'\d+\.\d+', str(data))  # 将数据转换为字符串以便正则表达式匹配
#         if match:
#             extracted_number = float(match.group())
#             result_list[j][i] = extracted_number
#
# # 自定义列名
# columns = ['{}'.format(i) for i in column_names]
#
# # 将 result_list 转换为 DataFrame，并指定列名
# result_df = pd.DataFrame(result_list, columns=columns)
#
# # 将结果写入 CSV 文件
# result_df.to_csv('fine_tune_original.csv', index=False)



# import re
# a_d = []
# # 打开.log文件
# with open('experiments_logs/Exp1/run_1/fine_tune_seed_123/logs_Aldata_11_03_2024_16_43_31.log', 'r') as file:
#     # 遍历文件的每一行
#     for line in file:
#         # 使用正则表达式匹配 "Train Accuracy" 后面的数字
#         match = re.search(r'Train Accuracy\s*:\s*(\d+\.\d+)', line)
#         if match:
#             train_accuracy = float(match.group(1))
#             a_d.append(train_accuracy)
# a_m = []
# with open('experiments_logs/Exp1/run_1/fine_tune_seed_123/logs_Aldata_11_03_2024_16_47_33.log', 'r') as file:
#     # 遍历文件的每一行
#     for line in file:
#         # 使用正则表达式匹配 "Train Accuracy" 后面的数字
#         match = re.search(r'Train Accuracy\s*:\s*(\d+\.\d+)', line)
#         if match:
#             train_accuracy = float(match.group(1))
#             a_m.append(train_accuracy)
# fuzzy = []
# with open('experiments_logs/Exp1/run_1/fine_tune_seed_123/logs_Aldata_20_03_2024_10_48_12.log', 'r') as file:
#     # 遍历文件的每一行
#     for line in file:
#         # 使用正则表达式匹配 "Train Accuracy" 后面的数字
#         match = re.search(r'Train Accuracy\s*:\s*(\d+\.\d+)', line)
#         if match:
#             train_accuracy = float(match.group(1))
#             fuzzy.append(train_accuracy)
# orig = []
# with open('experiments_logs/Exp1/run_1/fine_tune_seed_123/logs_Aldata_19_03_2024_20_24_53.log', 'r') as file:
#     # 遍历文件的每一行
#     for line in file:
#         # 使用正则表达式匹配 "Train Accuracy" 后面的数字
#         match = re.search(r'Train Accuracy\s*:\s*(\d+\.\d+)', line)
#         if match:
#             train_accuracy = float(match.group(1))
#             orig.append(train_accuracy)
#
# plt.plot(a_d, label='Attention + DilatedConv')
# plt.plot(a_m, label='Attention +  MLP')
# # plt.plot(fuzzy, label='Fuzzy True')
# # plt.plot(orig, label='Fuzzy False')
# plt.title('Train Accuracy Comparison')
# plt.xlabel('Epoch')
# plt.ylabel('Train Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()
# import os
# import shutil
#
# # 指定.py文件路径
# source_py_file = "D:\python\PycharmProjects\TS-TCC-main\config_files\HAR_Configs.py"
#
# # 指定目录路径
# directory_path = r"D:\python\PycharmProjects\TS-TCC-main\data_preprocessing"
#
# # 获取目标目录下的所有目录名，并排除指定的目录
# directories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))
#                and d not in ['AIdata', 'MixedShapesRegularTrain', 'StarLightCurves']]
#
#
# print(directories)
# for directory in directories:
#     # print(directories)
#     new_py_file_name = os.path.join("D:\python\PycharmProjects\TS-TCC-main\config_files",
#                                     directory + "_Configs.py")
#     shutil.copy(source_py_file, new_py_file_name)


# import pandas as pd
# from datetime import datetime
# file_path = 'result1.xlsx'
# fuzzy = 'True'
# block = 'DilatConv'  # DilatConv -> true ; mlp -> false
# instanceloss = 'True'
# data_type = 'Fishs'
# x = 0.8928
# current_time = datetime.now().strftime('%m-%d,%H:%M')
# df = pd.read_excel(file_path, header=None)
# print(df.shape)
# df.loc[df.shape[0]+1, 0] = '_'.join([fuzzy, block, instanceloss, data_type, current_time])
# df.loc[df.shape[0], 1] = x
# df.to_excel(file_path, index=False, header=False)

# import torch
# import numpy as np
# import torch.nn.functional as F
#
#
# class LDAMLoss(torch.nn.Module):
#     def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
#         '''
#         :param cls_num_list : 每个类别样本个数
#         :param max_m : LDAM中最大margin参数,default =0.5
#         :param weight :
#         :param s : 缩放因子,控制logits的范围
#         '''
#         super(LDAMLoss, self).__init__()
#         m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))  # n_j 四次方根的倒数
#         m_list = m_list * (max_m / np.max(m_list))  # 归一化，C相当于 max_m/ np.max(m_list)，确保没有大于max_m的
#         m_list = torch.FloatTensor(m_list)
#         self.m_list = m_list
#         assert s > 0
#         self.s = s
#         self.weight = weight
#
#     def forward(self, x, target):
#         index = torch.zeros_like(x, dtype=torch.uint8) # 创建一个跟X一样的tensor
#         index.scatter_(1, target.data.view(-1, 1), 1)  # 将每一行对应的target的序号设为1，其余保持为0
#         index_float = index.type(torch.FloatTensor)
#         batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))  # 矩阵乘法，不同类别有不同的margin
#         batch_m = batch_m.view((-1, 1))
#         x_m = x - batch_m
#         output = torch.where(index, x_m, x)
#         return F.cross_entropy(self.s * output, target,
#                                weight=self.weight)  # 通过一个缩放因子来放大logits，从而在使用softmax函数时增加计算结果的稳定性
#
#
# # Demo to create LDAMLoss and validate with x and target
# cls_num_list = [100, 10]  # Number of samples per class
# ldam_loss = LDAMLoss(cls_num_list)
# # logits output by the model for a batch of 2 samples
# x = torch.tensor([[-1.5, 0.5],
#                   [0.2, -0.8]])
#
# # true class labels for the batch
# target = torch.tensor([0, 1])
# # Calculate loss
# loss = ldam_loss(x, target)
# print(loss.item())


import torch
import torch.nn.functional as F
import numpy as np

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)   # 计算余弦相似度，dim=-1以size的最后一维
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)    # 对角线元素向下移动k个单位
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)    # 对角线元素向上移动k个单位
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v
    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
        logits = torch.cat((positives, negatives), dim=1)  # 沿着 列 拼接张量序列
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        return loss / (2 * self.batch_size)

# class NTXentLoss(torch.nn.Module):
#
#     def __init__(self, device, batch_size, temperature, use_cosine_similarity):
#         super(NTXentLoss, self).__init__()
#         self.batch_size = batch_size
#         self.temperature = temperature
#         self.device = device
#         self.softmax = torch.nn.Softmax(dim=-1)
#         # self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
#         self.similarity_function = self._get_similarity_function(use_cosine_similarity)
#         self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
#
#     def _get_similarity_function(self, use_cosine_similarity):
#         if use_cosine_similarity:
#             self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)   # 计算余弦相似度，dim=-1以size的最后一维
#             return self._cosine_simililarity
#         else:
#             return self._dot_simililarity
#
#     def _get_correlated_mask(self, time_steps):
#         diag = np.eye(2 * self.batch_size)
#         l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)    # 对角线元素向下移动k个单位
#         l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)    # 对角线元素向上移动k个单位
#         mask = torch.from_numpy((diag + l1 + l2))
#         mask = (1 - mask).type(torch.bool)
#         mask = mask.unsqueeze(0).expand(time_steps, -1, -1)
#         # print(mask.shape)   # T 2B 2B
#         return mask.to(self.device)
#
#     @staticmethod
#     def _dot_simililarity(x, y):
#         v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
#         # x shape: (N, 1, C)
#         # y shape: (1, C, 2N)
#         # v shape: (N, 2N)
#         return v
#
#     def _cosine_simililarity(self, x, y):
#         # x shape: (N, 1, C)
#         # y shape: (1, 2N, C)
#         # v shape: (N, 2N)
#         v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
#         return v
#     def forward(self, zis, zjs):
#         representations = torch.cat([zjs, zis], dim=0)   # B T C
#         # print(representations.shape)  # 2B T C
#         similarity_matrix = self.similarity_function(representations, representations)  # 2B 2B T
#         similarity_matrix = similarity_matrix.permute(2, 0, 1)  # T 2B 2B
#         l_pos = torch.stack([similarity_matrix[i].diagonal(self.batch_size) for i in range(similarity_matrix.shape[0])])
#         r_pos = torch.stack([similarity_matrix[i].diagonal(-self.batch_size) for i in range(similarity_matrix.shape[0])])
#         # l_pos = torch.diag(similarity_matrix, self.batch_size)
#         # r_pos = torch.diag(similarity_matrix, -self.batch_size)
#         # print(l_pos.shape)   # T B
#         positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, l_pos.shape[0]).permute(1, 0).unsqueeze(
#             -1)  # T 2B 1
#         negatives = similarity_matrix[self._get_correlated_mask(similarity_matrix.shape[0])].view(l_pos.shape[0],
#                                                                                                   2 * self.batch_size,
#                                                                                                   -1)  # [10, 64, 62]
#         logits = torch.cat((positives, negatives), dim=2)  # 沿着 列 拼接张量序列
#         logits /= self.temperature
#
#         # labels = torch.zeros(2 * self.batch_size).to(self.device).long()
#         # loss = self.criterion(logits, labels)
#         labels = torch.zeros((l_pos.shape[0], 2 * self.batch_size)).to(self.device).long()
#         loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
#
#         return loss / ((2 * self.batch_size)*l_pos.shape[0])


import os
import shutil
rootdir="D:\python\PycharmProjects\TS-TCC-main\data\Aldata"
filelist=os.listdir(rootdir)
for file in filelist:
    if '2160' not in file:
        del_file = rootdir + '\\' + file #当代码和要删除的文件不在同一个文件夹时，必须使用绝对路径
        shutil.rmtree(del_file)
        print("已经删除：",del_file)





