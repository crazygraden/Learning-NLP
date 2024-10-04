import pandas as pd
import os
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
# 指定搜索模式
data_file = "data_file/5228_2160_margtsc2_5.csv"

data = pd.read_csv(data_file, header=None)
print(data.shape)
output_dir = f"../../data/Aldata/5228_2160_5"
# 检查目录是否存在
if not os.path.exists(output_dir):
    # 如果目录不存在，则创建目录
    os.makedirs(output_dir)

y = data.iloc[:, 0]
x = data.iloc[:, 1:]
pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
print(x)
print(y)
x = x.to_numpy()
y = y.to_numpy()
num_counts_y = Counter(y)
print(num_counts_y)
# # y = y - 1
# scaler = RobustScaler()
scaler = StandardScaler()
x = scaler.fit_transform(x)
# 将数据从 [-1, 1] 映射到 [0, 2]
x = (x + 1) * 1
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.45, random_state=6)  # 666
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.01, random_state=6)  # 666
# 统计类别数量
num_counts_train = Counter(y_train)
print(num_counts_train)
num_counts_test = Counter(y_test)
print(num_counts_test)



dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train).unsqueeze(1)
dat_dict["labels"] = torch.from_numpy(y_train)
torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_val).unsqueeze(1)
dat_dict["labels"] = torch.from_numpy(y_val)
torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test).unsqueeze(1)
dat_dict["labels"] = torch.from_numpy(y_test)
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))
