
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import csv
from datetime import datetime
from readin_data import df_label_split_with_time_ranges

def accuracy_score(targets, predictions):
    # 确保目标和预测是NumPy数组
    targets = np.array(targets)
    predictions = np.array(predictions)

    # 确保目标和预测具有相同的形状
    assert targets.shape == predictions.shape, "targets and predictions must have the same shape"

    # 计算预测正确的样本数量
    correct_predictions = (targets == predictions).sum()

    # 计算准确率
    accuracy = correct_predictions / targets.shape[0]

    return accuracy

with open('4s_exp_res.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file,
                            fieldnames=['n_timestep', 'batch_size', 'num_layer', 'hidden_size',
                                        'seed',
                                        'exp_index', 'result'])
    writer.writeheader()

file_name = '4s停顿带回退.xlsx'  # 请确保这是您的文件路径
stop_time_ranges = [
    [(pd.Timestamp('1970-01-01 00:00:00.000000'), pd.Timestamp('1970-01-01 00:00:04.000000')),
     (pd.Timestamp('1970-01-01 00:00:09.000000'), pd.Timestamp('1970-01-01 00:00:13.000000')),
     (pd.Timestamp('1970-01-01 00:00:18.000000'), pd.Timestamp('1970-01-01 00:00:22.000000')),
     (pd.Timestamp('1970-01-01 00:00:27.000000'), pd.Timestamp('1970-01-01 00:00:31.000000')),
     (pd.Timestamp('1970-01-01 00:00:45.000000'), pd.Timestamp('1970-01-01 00:00:49.000000')),
     (pd.Timestamp('1970-01-01 00:00:54.000000'), pd.Timestamp('1970-01-01 00:00:58.000000')),
     (pd.Timestamp('1970-01-01 00:01:03.000000'), pd.Timestamp('1970-01-01 00:01:07.000000')),
     (pd.Timestamp('1970-01-01 00:01:21.000000'), pd.Timestamp('1970-01-01 00:01:25.000000')),
     (pd.Timestamp('1970-01-01 00:01:30.000000'), pd.Timestamp('1970-01-01 00:01:34.000000')),
     (pd.Timestamp('1970-01-01 00:01:39.000000'), pd.Timestamp('1970-01-01 00:01:43.000000')),
     (pd.Timestamp('1970-01-01 00:01:57.000000'), pd.Timestamp('1970-01-01 00:02:01.000000')),
     (pd.Timestamp('1970-01-01 00:02:06.000000'), pd.Timestamp('1970-01-01 00:02:10.000000')),
     (pd.Timestamp('1970-01-01 00:02:15.000000'), pd.Timestamp('1970-01-01 00:02:19.000000')),
     (pd.Timestamp('1970-01-01 00:02:24.000000'), pd.Timestamp('1970-01-01 00:02:28.000000'))],
    [(pd.Timestamp('1970-01-01 00:04:24.000000'), pd.Timestamp('1970-01-01 00:04:28.000000')),
     (pd.Timestamp('1970-01-01 00:04:33.000000'), pd.Timestamp('1970-01-01 00:04:37.000000')),
     (pd.Timestamp('1970-01-01 00:04:42.000000'), pd.Timestamp('1970-01-01 00:04:46.000000')),
     (pd.Timestamp('1970-01-01 00:04:51.000000'), pd.Timestamp('1970-01-01 00:04:55.000000')),
     (pd.Timestamp('1970-01-01 00:05:09.000000'), pd.Timestamp('1970-01-01 00:05:13.000000')),
     (pd.Timestamp('1970-01-01 00:05:18.000000'), pd.Timestamp('1970-01-01 00:05:22.000000')),
     (pd.Timestamp('1970-01-01 00:05:27.000000'), pd.Timestamp('1970-01-01 00:05:31.000000')),
     (pd.Timestamp('1970-01-01 00:05:45.000000'), pd.Timestamp('1970-01-01 00:05:49.000000')),
     (pd.Timestamp('1970-01-01 00:05:54.000000'), pd.Timestamp('1970-01-01 00:05:58.000000')),
     (pd.Timestamp('1970-01-01 00:06:03.000000'), pd.Timestamp('1970-01-01 00:06:07.000000')),
     (pd.Timestamp('1970-01-01 00:06:21.000000'), pd.Timestamp('1970-01-01 00:06:25.000000')),
     (pd.Timestamp('1970-01-01 00:06:30.000000'), pd.Timestamp('1970-01-01 00:06:34.000000')),
     (pd.Timestamp('1970-01-01 00:06:39.000000'), pd.Timestamp('1970-01-01 00:06:43.000000')),
     (pd.Timestamp('1970-01-01 00:06:48.000000'), pd.Timestamp('1970-01-01 00:06:52.000000')), ],
    [(pd.Timestamp('1970-01-01 00:08:48.000000'), pd.Timestamp('1970-01-01 00:08:52.000000')),
     (pd.Timestamp('1970-01-01 00:08:57.000000'), pd.Timestamp('1970-01-01 00:09:01.000000')),
     (pd.Timestamp('1970-01-01 00:09:06.000000'), pd.Timestamp('1970-01-01 00:09:10.000000')),
     (pd.Timestamp('1970-01-01 00:09:15.000000'), pd.Timestamp('1970-01-01 00:09:19.000000')),
     (pd.Timestamp('1970-01-01 00:09:33.000000'), pd.Timestamp('1970-01-01 00:09:37.000000')),
     (pd.Timestamp('1970-01-01 00:09:42.000000'), pd.Timestamp('1970-01-01 00:09:46.000000')),
     (pd.Timestamp('1970-01-01 00:09:51.000000'), pd.Timestamp('1970-01-01 00:09:55.000000')),
     (pd.Timestamp('1970-01-01 00:10:09.000000'), pd.Timestamp('1970-01-01 00:10:13.000000')),
     (pd.Timestamp('1970-01-01 00:10:18.000000'), pd.Timestamp('1970-01-01 00:10:22.000000')),
     (pd.Timestamp('1970-01-01 00:10:27.000000'), pd.Timestamp('1970-01-01 00:10:31.000000')),
     (pd.Timestamp('1970-01-01 00:10:45.000000'), pd.Timestamp('1970-01-01 00:10:49.000000')),
     (pd.Timestamp('1970-01-01 00:10:54.000000'), pd.Timestamp('1970-01-01 00:10:58.000000')),
     (pd.Timestamp('1970-01-01 00:11:03.000000'), pd.Timestamp('1970-01-01 00:11:07.000000')),
     (pd.Timestamp('1970-01-01 00:11:12.000000'), pd.Timestamp('1970-01-01 00:11:16.000000'))]
]
return_time_ranges = [
    [(pd.Timestamp('1970-01-01 00:02:24.000000'), pd.Timestamp('1970-01-01 00:04:24.000000')),
     (pd.Timestamp('1970-01-01 00:00:36.000000'), pd.Timestamp('1970-01-01 00:00:45.000000')),
     (pd.Timestamp('1970-01-01 00:01:12.000000'), pd.Timestamp('1970-01-01 00:01:21.000000')),
     (pd.Timestamp('1970-01-01 00:01:48.000000'), pd.Timestamp('1970-01-01 00:01:57.000000'))
     ],
    [(pd.Timestamp('1970-01-01 00:06:48.000000'), pd.Timestamp('1970-01-01 00:08:48.000000')),
     (pd.Timestamp('1970-01-01 00:05:00.000000'), pd.Timestamp('1970-01-01 00:05:09.000000')),
     (pd.Timestamp('1970-01-01 00:05:36.000000'), pd.Timestamp('1970-01-01 00:05:45.000000')),
     (pd.Timestamp('1970-01-01 00:06:12.000000'), pd.Timestamp('1970-01-01 00:06:21.000000'))
     ],
    [(pd.Timestamp('1970-01-01 00:11:12.000000'), pd.Timestamp('1970-01-01 00:13:12.000000')),
     (pd.Timestamp('1970-01-01 00:09:24.000000'), pd.Timestamp('1970-01-01 00:09:33.000000')),
     (pd.Timestamp('1970-01-01 00:10:00.000000'), pd.Timestamp('1970-01-01 00:10:09.000000')),
     (pd.Timestamp('1970-01-01 00:10:36.000000'), pd.Timestamp('1970-01-01 00:10:45.000000'))
     ],
]
start_times = [
    pd.Timestamp('1970-01-01 00:04:24.000000'),
    pd.Timestamp('1970-01-01 00:08:48.000000')
]
dfs = df_label_split_with_time_ranges(file_name, start_times, stop_time_ranges, return_time_ranges)

# for df in dfs:
#     print(df)
#     print(dfs[df])

col_indices = ['inclination_change', 'azimuth_change', 'toolface_change']

n_features = len(col_indices)
target_col = 'status'
# for n_timesteps in [10, 50, 100]:
#     for n_layers in [2,3]:
#         for hidden_size in [16, 32]:
#             for batch_size in [1, 4, 16]:
for seed in [72498724, 21985287, 28549854, 12369870, 32758467, 45876586, 63985481,91876365]:
    torch.manual_seed(seed)
    exp_index = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    hidden_size = 64
    n_layers = 2
    n_timesteps = 50  # 选择的时间步长
    batch_size = 16
    output_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化一个空的列表来存储所有时间序列的X和y
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    train_df = []
    test_df = []
    random.seed(seed)
    test_name = random.choice(['df_1', 'df_2', 'df_3'])
    # print(test_name)
    for name, df in dfs.items():
        print(name)
        if name == test_name:
            test_df.append(df)
        else:
            train_df.append(df)
    test_dataframes = test_df
    train_dataframes = train_df
    # 对于每个DataFrame（即每个时间序列），进行如下处理
    for df in train_dataframes:
        # 假设所有DataFrame都有相同的列（即特征）
        # n_features = dataframe.shape[1]

        # 标准化数据
        df['inclination_change'] = df['inclination'].diff().abs()
        df['azimuth_change'] = df['azimuth'].diff().abs()
        df['toolface_change'] = df['toolface'].diff().abs()
        # col_indices = ['inclination_change', 'azimuth_change', 'toolface_change']
        df = df.iloc[1:]
        df = df[df['status'] != -1]
        # 划分特征和标签
        X, y = [], []
        for i in range(n_timesteps, len(df)):
            X.append(df.loc[df.index[i - n_timesteps:i], col_indices].values)
            y.append(df.loc[df.index[i], target_col])

            # 将列表转换为NumPy数组
        X = np.array(X)
        y = np.array(y)
        # 将训练集和测试集添加到全局列表中
        X_train.extend(X)
        Y_train.extend(y)

    for df in test_dataframes:
        # 标准化数据
        df['inclination_change'] = df['inclination'].diff().abs()
        df['azimuth_change'] = df['azimuth'].diff().abs()
        df['toolface_change'] = df['toolface'].diff().abs()
        # col_indices = ['inclination_change', 'azimuth_change', 'toolface_change']

        df = df.iloc[1:]
        df = df[df['status'] != -1]

        # 划分特征和标签
        X, y = [], []
        for i in range(n_timesteps, len(df)):
            X.append(df.loc[df.index[i - n_timesteps:i], col_indices].values)
            y.append(df.loc[df.index[i], target_col])

            # 将列表转换为NumPy数组
        X = np.array(X)
        y = np.array(y)
        # 将训练集和测试集添加到全局列表中
        X_test.extend(X)
        Y_test.extend(y)
    X_test = torch.tensor(np.vstack(X_test), dtype=torch.float32).view(-1, n_timesteps, n_features)
    Y_test = torch.tensor(np.vstack(Y_test), dtype=torch.float32)
    X_train = torch.tensor(np.vstack(X_train), dtype=torch.float32).view(-1, n_timesteps, n_features)
    Y_train = torch.tensor(np.vstack(Y_train), dtype=torch.float32)

    # 创建自定义Dataset类（如果需要的话），这里简单使用TensorDataset
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    # 创建DataLoader对象
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    # 创建LSTM模型
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # 初始化隐藏状态和细胞状态
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

            # 前向传播LSTM
            out, _ = self.lstm(x, (h0, c0))

            # 取最后一个时间步的输出
            out = self.fc(out[:, -1, :])
            return out


    # 初始化模型参数
    input_size = n_features
    hidden_size = 64  # 自定义的隐藏状态维度
    output_size = 1  # 根据任务需要设置输出维度
    num_layers = 2  # 自定义的LSTM层数

    # 实例化模型
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 250
    for epoch in range(num_epochs):
        for inputs, targets in train_dataloader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 测试模型
    # model.eval()
    predictions = []
    target_array = []

    # 不计算梯度
    with torch.no_grad():
        # 假设你有一个DataLoader来加载测试集数据
        for inputs, targets in test_dataloader:
            # 将数据移动到模型所在的设备上
            inputs, targets = inputs.to(device), targets.to(device)

            # 执行前向传播以获取预测结果
            outputs = model(inputs)

            # 获取预测类别的索引（对于多分类问题）
            predicted = (outputs > 0.5).float()

            # 将预测结果添加到列表中
            predictions.extend(predicted.cpu().numpy())
            target_array.extend(targets.cpu().numpy())

        # 将预测结果转换为NumPy数组
    predictions = np.array(predictions)
    target_array = np.array(target_array)
    # 计算准确率
    accuracy = accuracy_score(target_array, predictions)
    print(f'Test Accuracy: {accuracy:.4f}')
    result_dict = {'n_timestep': n_timesteps, 'batch_size': batch_size,
                   'num_layer': n_layers, 'hidden_size': hidden_size, 'seed': seed,
                   'exp_index': exp_index, 'result': accuracy}

    with open('4s_exp_res.csv', 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['n_timestep', 'batch_size', 'num_layer', 'hidden_size',
                                                  'seed', 'exp_index', 'result'])
        writer.writerow(result_dict)



