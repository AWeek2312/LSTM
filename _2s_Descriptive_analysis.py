import pandas as pd
from time_cycle import time_cycle_def
import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
from readin_data import  df_label_split_with_time_ranges
from matplotlib import font_manager
import matplotlib
matplotlib.use('TKAgg')
import seaborn as sns


# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体，或者其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def df_emw(df, col_list):
    columns_to_smooth = col_list

    # 选择这些列并应用指数平滑
    smoothed_columns = df[columns_to_smooth].apply(lambda x: x.ewm(alpha=0.1).mean())

    # 将平滑后的列与原始 DataFrame 中其他未平滑的列合并
    smoothed_df = pd.concat([df.drop(columns=columns_to_smooth), smoothed_columns], axis=1)
    return smoothed_df


def calculate_trajectory(initial_position, initial_orientation, inclinations, azimuths, toolfaces, time_interval,
                         num_steps, accelerations):

    # 初始化轨迹列表
    Trajectory = [initial_position]

    # 初始化方向向量和速度向量
    B = np.array(initial_orientation)
    B = B / np.linalg.norm(B)
    V = np.zeros_like(initial_position)  # 初始速度为零
    GRAVITY_ACCELERATION = 9.81
    # 迭代计算每个时间步的轨迹
    for i in range(0, num_steps):
        # 计算当前步骤的方向向量
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(inclinations[i]), -np.sin(inclinations[i])],
                       [0, np.sin(inclinations[i]), np.cos(inclinations[i])]])
        Ry = np.array([[np.cos(azimuths[i]), 0, np.sin(azimuths[i])],
                       [0, 1, 0],
                       [-np.sin(azimuths[i]), 0, np.cos(azimuths[i])]])
        Rz = np.array([[np.cos(toolfaces[i]), -np.sin(toolfaces[i]), 0],
                       [np.sin(toolfaces[i]), np.cos(toolfaces[i]), 0],
                       [0, 0, 1]])

        # 旋转方向向量
        B = Rz @ Ry @ Rx @ B
        B = B / np.linalg.norm(B)
        a_g = np.array(accelerations[i])
        a = a_g * GRAVITY_ACCELERATION
        # 使用旋转后的方向向量计算位移
        displacement_due_to_velocity = V * time_interval
        displacement_due_to_acceleration = 0.5 * a * time_interval ** 2
        total_displacement = displacement_due_to_velocity + displacement_due_to_acceleration
        V = V + a * time_interval
        # print(total_displacement_without_direction)
        # 更新位置
        next_position = Trajectory[-1] + total_displacement

        # 将下一个位置添加到轨迹列表中
        Trajectory.append(next_position)

    return Trajectory

def df_describe(df, col_list):
    # 选择你想要进行描述性分析的列
    selected_columns = col_list  # 替换为你的列名


    descriptive_stats = df[selected_columns].describe()

    # 输出定制的描述性统计结果
    # 设置pandas打印选项
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列

    # 现在打印DataFrame将显示所有行和列
    print(descriptive_stats)

    # 如果之后想恢复默认设置，可以使用：
    pd.reset_option('^display.', silent=True)


def df_plot_col(df, column_name):
    import matplotlib.pyplot as plt

    # 绘制曲线图
    plt.figure(figsize=(10, 5))  # 设置图形大小
    plt.plot(df.index, df[column_name])
    plt.xlabel('时间步')  # 设置x轴标签
    plt.ylabel('倾角')  # 设置y轴标签
    plt.title(f'数据平滑后倾角随时间变化的曲线')  # 设置标题
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图形


if __name__ == "__main__":
    # 读取xlsx文件
    file_name = '数据文件/2s/50度倾角2s停顿带回退.xlsx'  # 请确保这是您的文件路径
    stop_time_ranges = [
        [(pd.Timestamp('1970-01-01 00:00:00.000000'), pd.Timestamp('1970-01-01 00:00:02.000000')),
         (pd.Timestamp('1970-01-01 00:00:07.000000'), pd.Timestamp('1970-01-01 00:00:9.000000')),
         (pd.Timestamp('1970-01-01 00:00:14.000000'), pd.Timestamp('1970-01-01 00:00:16.000000')),
         (pd.Timestamp('1970-01-01 00:00:21.000000'), pd.Timestamp('1970-01-01 00:00:23.000000')),
         (pd.Timestamp('1970-01-01 00:00:35.000000'), pd.Timestamp('1970-01-01 00:00:37.000000')),
         (pd.Timestamp('1970-01-01 00:00:42.000000'), pd.Timestamp('1970-01-01 00:00:44.000000')),
         (pd.Timestamp('1970-01-01 00:00:49.000000'), pd.Timestamp('1970-01-01 00:00:51.000000')),
         (pd.Timestamp('1970-01-01 00:01:03.000000'), pd.Timestamp('1970-01-01 00:01:05.000000')),
         (pd.Timestamp('1970-01-01 00:01:10.000000'), pd.Timestamp('1970-01-01 00:01:12.000000')),
         (pd.Timestamp('1970-01-01 00:01:17.000000'), pd.Timestamp('1970-01-01 00:01:19.000000')),
         (pd.Timestamp('1970-01-01 00:01:31.000000'), pd.Timestamp('1970-01-01 00:01:33.000000')),
         (pd.Timestamp('1970-01-01 00:01:38.000000'), pd.Timestamp('1970-01-01 00:01:40.000000')),
         (pd.Timestamp('1970-01-01 00:01:45.000000'), pd.Timestamp('1970-01-01 00:01:47.000000')),
         (pd.Timestamp('1970-01-01 00:01:52.000000'), pd.Timestamp('1970-01-01 00:01:54.000000'))],
        [(pd.Timestamp('1970-01-01 00:03:52.000000'), pd.Timestamp('1970-01-01 00:03:54.000000')),
         (pd.Timestamp('1970-01-01 00:03:59.000000'), pd.Timestamp('1970-01-01 00:04:01.000000')),
         (pd.Timestamp('1970-01-01 00:04:06.000000'), pd.Timestamp('1970-01-01 00:04:08.000000')),
         (pd.Timestamp('1970-01-01 00:04:13.000000'), pd.Timestamp('1970-01-01 00:04:15.000000')),
         (pd.Timestamp('1970-01-01 00:04:27.000000'), pd.Timestamp('1970-01-01 00:04:29.000000')),
         (pd.Timestamp('1970-01-01 00:04:34.000000'), pd.Timestamp('1970-01-01 00:04:36.000000')),
         (pd.Timestamp('1970-01-01 00:04:41.000000'), pd.Timestamp('1970-01-01 00:04:43.000000')),
         (pd.Timestamp('1970-01-01 00:04:55.000000'), pd.Timestamp('1970-01-01 00:04:57.000000')),
         (pd.Timestamp('1970-01-01 00:05:02.000000'), pd.Timestamp('1970-01-01 00:05:04.000000')),
         (pd.Timestamp('1970-01-01 00:05:09.000000'), pd.Timestamp('1970-01-01 00:05:11.000000')),
         (pd.Timestamp('1970-01-01 00:05:23.000000'), pd.Timestamp('1970-01-01 00:05:25.000000')),
         (pd.Timestamp('1970-01-01 00:05:30.000000'), pd.Timestamp('1970-01-01 00:05:32.000000')),
         (pd.Timestamp('1970-01-01 00:05:37.000000'), pd.Timestamp('1970-01-01 00:05:39.000000')),
         (pd.Timestamp('1970-01-01 00:05:44.000000'), pd.Timestamp('1970-01-01 00:05:46.000000')), ],
        [(pd.Timestamp('1970-01-01 00:07:44.000000'), pd.Timestamp('1970-01-01 00:07:46.000000')),
         (pd.Timestamp('1970-01-01 00:07:51.000000'), pd.Timestamp('1970-01-01 00:07:53.000000')),
         (pd.Timestamp('1970-01-01 00:07:58.000000'), pd.Timestamp('1970-01-01 00:08:00.000000')),
         (pd.Timestamp('1970-01-01 00:08:05.000000'), pd.Timestamp('1970-01-01 00:08:07.000000')),
         (pd.Timestamp('1970-01-01 00:08:19.000000'), pd.Timestamp('1970-01-01 00:08:21.000000')),
         (pd.Timestamp('1970-01-01 00:08:26.000000'), pd.Timestamp('1970-01-01 00:08:28.000000')),
         (pd.Timestamp('1970-01-01 00:08:33.000000'), pd.Timestamp('1970-01-01 00:08:35.000000')),
         (pd.Timestamp('1970-01-01 00:08:47.000000'), pd.Timestamp('1970-01-01 00:08:49.000000')),
         (pd.Timestamp('1970-01-01 00:08:54.000000'), pd.Timestamp('1970-01-01 00:08:56.000000')),
         (pd.Timestamp('1970-01-01 00:09:01.000000'), pd.Timestamp('1970-01-01 00:09:03.000000')),
         (pd.Timestamp('1970-01-01 00:09:15.000000'), pd.Timestamp('1970-01-01 00:09:17.000000')),
         (pd.Timestamp('1970-01-01 00:09:22.000000'), pd.Timestamp('1970-01-01 00:09:24.000000')),
         (pd.Timestamp('1970-01-01 00:09:29.000000'), pd.Timestamp('1970-01-01 00:09:31.000000')),
         (pd.Timestamp('1970-01-01 00:09:36.000000'), pd.Timestamp('1970-01-01 00:09:38.000000'))]
    ]

    return_time_ranges = [
        [(pd.Timestamp('1970-01-01 00:01:52.000000'), pd.Timestamp('1970-01-01 00:03:52.000000')),
         (pd.Timestamp('1970-01-01 00:00:28.000000'), pd.Timestamp('1970-01-01 00:00:35.000000')),
         (pd.Timestamp('1970-01-01 00:00:56.000000'), pd.Timestamp('1970-01-01 00:01:03.000000')),
         (pd.Timestamp('1970-01-01 00:01:24.000000'), pd.Timestamp('1970-01-01 00:01:31.000000'))
         ],
        [(pd.Timestamp('1970-01-01 00:05:44.000000'), pd.Timestamp('1970-01-01 00:07:44.000000')),
         (pd.Timestamp('1970-01-01 00:04:20.000000'), pd.Timestamp('1970-01-01 00:04:27.000000')),
         (pd.Timestamp('1970-01-01 00:04:48.000000'), pd.Timestamp('1970-01-01 00:04:55.000000')),
         (pd.Timestamp('1970-01-01 00:05:16.000000'), pd.Timestamp('1970-01-01 00:05:23.000000'))
         ],
        [(pd.Timestamp('1970-01-01 00:09:36.000000'), pd.Timestamp('1970-01-01 00:11:36.000000')),
         (pd.Timestamp('1970-01-01 00:08:12.000000'), pd.Timestamp('1970-01-01 00:08:19.000000')),
         (pd.Timestamp('1970-01-01 00:08:40.000000'), pd.Timestamp('1970-01-01 00:08:47.000000')),
         (pd.Timestamp('1970-01-01 00:09:08.000000'), pd.Timestamp('1970-01-01 00:09:15.000000'))
         ],
    ]
    start_times = [
        pd.Timestamp('1970-01-01 00:03:52.000000'),
        pd.Timestamp('1970-01-01 00:07:44.000000')
    ]
    dfs = df_label_split_with_time_ranges(file_name, start_times, stop_time_ranges, return_time_ranges)
    all_df = []
    for df_name, df in dfs.items():
        df = df[df['status'] != -1]
        all_df.append(df)
    # 合并数据
    combined_df = pd.concat(all_df)
    print("###########################################")
    # print(combined_df)
    col_list = ['inclination', 'azimuth', 'toolface']
    df_describe(combined_df, col_list)

    # 选择需要处理的列
    columns_to_plot = ['inclination', 'azimuth', 'toolface']
    column_labels = ['倾角', '方位角', '工具面角']  # 自定义标签

    # 遍历每个列并画出频率直方图
    for i, column in enumerate(columns_to_plot):
        plt.figure()  # 创建新的图形窗口
        sns.histplot(data=combined_df, x=column, kde=True)  # 绘制频率直方图和核密度估计曲线
        plt.xlabel(column_labels[i])  # 设置 x 轴标签为自定义标签
        plt.ylabel('频率')  # 设置 y 轴标签
        plt.title(f"50°倾角2s停顿-{column_labels[i]}频率直方图")  # 设置图标题
        plt.show()  # 显示图形窗口

