# @File  : my_test.py
# @Author: fkx
# @Time: 2024/6/11 15:58 
# -*- coding: utf-8 -*-


# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.preprocessing import RobustScaler
# # # 指定搜索模式
# data_file = "5228_2160_margtsc2_7.csv"
#
# data = pd.read_csv(data_file, header=None)
# data = data.iloc[:, 1:]
# scaler = RobustScaler()
# data_scaled = scaler.fit_transform(data)
#
# # 转置数据以使每一行为一条数据
# data_scaled_transposed = pd.DataFrame(data_scaled).T
# print(data_scaled_transposed)
# # 绘制核密度估计图
# plt.figure(figsize=(10, 6))
# for i in range(data_scaled_transposed.shape[1]):
#     sns.kdeplot(data=data_scaled_transposed[i], x=[-1, 1])
#
# plt.title('KDE Plot of Standardized Data (Rows)')
# plt.legend()
# plt.show()

