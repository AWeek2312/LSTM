import pandas as pd
from time_cycle import time_cycle_def
import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体，或者其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def excel_readin(file_name, pause_duration):
    stage_func = time_cycle_def(pause_duration)
    df = pd.read_excel(file_name, header=None)

    old_columns = df.columns

    # 创建新的列索引，从0开始的整数
    new_columns = range(len(old_columns))

    # 重命名列
    df.columns = new_columns
    mapping = {
        0: 'inclination',
        1: 'azimuth',
        2: 'toolface',
        3: 'clock',
        4: 'acceleration_x',
        5: 'acceleration_y',
        6: 'acceleration_z',
        7: 'mag_x',
        8: 'mag_y',
        9: 'mag_z',
        10: 'mark1',
        11: 'mark2',
    }
    # 使用rename函数和映射关系来更改列名
    df.rename(columns=mapping, inplace=True)

    index_list = df.index.tolist()
    state_list = []
    for index in index_list:
        second = int(index / 10)
        cycle_num, state, node_num = stage_func(second)
        state_list.append(state)
        # print(f"第{second}秒属于：{stage}")
    df.loc[:, 'status'] = state_list
    dfs = {}
    current_group = []

    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        # 如果当前行的state不等于-1，则添加到当前组
        if row['status'] != -1:
            current_group.append(row)
        else:
            # 如果当前行的state等于-1，则将当前组保存为一个新的DataFrame
            if current_group:
                dfs[f'df_{len(dfs) + 1}'] = pd.DataFrame(current_group)
                current_group = []  # 重置当前组

    # 保存最后一个组（如果有的话）
    if current_group:
        dfs[f'df_{len(dfs) + 1}'] = pd.DataFrame(current_group)

    # 打印出所有的DataFrame
    # for name, df_piece in dfs.items():
    #     print(f"DataFrame {name}:")
    #     print(df_piece)
    #     print()

    return dfs


def dat_readin(file_name, pause_duration):
    stage_func = time_cycle_def(pause_duration)
    # 读取.dat文件
    df = pd.read_csv(file_name, sep='\t', header=None)

    df_no_nan = df.dropna(axis=1)

    # 去除第一列（索引列，如果它是DataFrame的一部分）
    df = df_no_nan.drop(df_no_nan.columns[0], axis=1)
    # 获取当前的列索引
    old_columns = df.columns

    # 创建新的列索引，从0开始的整数
    new_columns = range(len(old_columns))

    # 重命名列
    df.columns = new_columns
    mapping = {
        0: 'inclination',
        1: 'toolface',
        2: 'azimuth',
        3: 'acceleration_x',
        4: 'acceleration_y',
        5: 'acceleration_z',
        6: 'mag_x',
        7: 'mag_y',
        8: 'mag_z',
        9: 'time_stamp',
    }

    # 使用rename函数和映射关系来更改列名
    df.rename(columns=mapping, inplace=True)
    index_list = df.index.tolist()
    state_list = []
    for index in index_list:
        second = int(index / 10)
        cycle_num, state, node_num = stage_func(second)
        state_list.append(state)
        # print(f"第{second}秒属于：{stage}")
    df.loc[:, 'status'] = state_list
    dfs = {}
    current_group = []

    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        # 如果当前行的state不等于-1，则添加到当前组
        if row['status'] != -1:
            current_group.append(row)
        else:
            # 如果当前行的state等于-1，则将当前组保存为一个新的DataFrame
            if current_group:
                dfs[f'df_{len(dfs) + 1}'] = pd.DataFrame(current_group)
                current_group = []  # 重置当前组

    # 保存最后一个组（如果有的话）
    if current_group:
        dfs[f'df_{len(dfs) + 1}'] = pd.DataFrame(current_group)

    # 打印出所有的DataFrame
    # for name, df_piece in dfs.items():
    #     print(f"DataFrame {name}:")
    #     print(df_piece)
    #     print()
    return dfs

def excel_readin_whole(file_name, pause_duration):
    stage_func = time_cycle_def(pause_duration)
    df = pd.read_excel(file_name, header=None)

    old_columns = df.columns

    # 创建新的列索引，从0开始的整数
    new_columns = range(len(old_columns))

    # 重命名列
    df.columns = new_columns
    mapping = {
        0: 'inclination',
        1: 'azimuth',
        2: 'toolface',
        3: 'clock',
        4: 'acceleration_x',
        5: 'acceleration_y',
        6: 'acceleration_z',
        7: 'mag_x',
        8: 'mag_y',
        9: 'mag_z',
        10: 'mark1',
        11: 'mark2',
    }
    # 使用rename函数和映射关系来更改列名
    df.rename(columns=mapping, inplace=True)

    index_list = df.index.tolist()
    state_list = []
    for index in index_list:
        second = int(index / 10)
        cycle_num, state, node_num = stage_func(second)
        state_list.append(state)
        # print(f"第{second}秒属于：{stage}")
    df.loc[:, 'status'] = state_list
    return df


def dat_readin_whole(file_name, pause_duration):
    stage_func = time_cycle_def(pause_duration)
    # 读取.dat文件
    df = pd.read_csv(file_name, sep='\t', header=None)

    df_no_nan = df.dropna(axis=1)

    # 去除第一列（索引列，如果它是DataFrame的一部分）
    df = df_no_nan.drop(df_no_nan.columns[0], axis=1)
    # 获取当前的列索引
    old_columns = df.columns

    # 创建新的列索引，从0开始的整数
    new_columns = range(len(old_columns))

    # 重命名列
    df.columns = new_columns
    mapping = {
        0: 'inclination',
        1: 'toolface',
        2: 'azimuth',
        3: 'acceleration_x',
        4: 'acceleration_y',
        5: 'acceleration_z',
        6: 'mag_x',
        7: 'mag_y',
        8: 'mag_z',
        9: 'time_stamp',
    }

    # 使用rename函数和映射关系来更改列名
    df.rename(columns=mapping, inplace=True)
    index_list = df.index.tolist()
    state_list = []
    for index in index_list:
        second = int(index / 10)
        cycle_num, state, node_num = stage_func(second)
        state_list.append(state)
        # print(f"第{second}秒属于：{stage}")
    df.loc[:, 'status'] = state_list

    return df

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
    file_name = '数据文件/4s/30度倾角4s停顿带回退.xlsx'  # 请确保这是您的文件路径
    #df = excel_readin_whole(file_name, 4)
    #print(df.head())
    # 读取dat文件
    # file_name = '8-3-1.dat'
    # df = dat_readin_whole(file_name, 8)

    #col_list = ['inclination', 'toolface', 'azimuth', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'mag_x',
     #           'mag_y',
      #          'mag_z']
    # df_describe(df, col_list)
    col_list = ['inclination',  'azimuth','toolface']

    dfs = excel_readin(file_name, 4)
    train_df = []
    test_df = []
    for name, df in dfs.items():
        if name == 'df_1':
            test_df.append(df)
        else:
            train_df.append(df)
        print(name)
    print("###########################################")
    df_describe(dfs['df_1'], col_list)
        # df = df.head(900)
        # df_describe(df, col_list)
        # df_plot_col(df, 'inclination')
        # smoothed_df = df_emw(df, col_list)
        # df_plot_col(smoothed_df, 'inclination')