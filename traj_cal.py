import numpy as np
import pandas as pd
from time_cycle import time_cycle_def
import scipy.stats as stats
import math
from emw import excel_readin_whole, dat_readin_whole

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
def per_node_distance_check(df):
    # 假设你有一个名为df的DataFrame
    # df = pd.DataFrame(...)

    # 计算DataFrame的行数
    num_rows = len(df)

    # 生成一个索引列表，每九十行一个组
    group_indices = np.arange(0, num_rows, 90)

    # 添加一个分组列到DataFrame中
    df['group'] = pd.cut(np.arange(num_rows), bins=group_indices, labels=False)

    # 使用groupby方法根据分组列来分组
    grouped_df = df.groupby('group')
    dist_arr = []
    # 现在，你可以迭代grouped_df来处理每个组
    for group, group_df in grouped_df:
        print(f"DataFrame pernode moving {name}:")
        inclinations, azimuths, toolfaces, accelerations, states = df_extract(group_df)
        num_steps = len(inclinations)  # 时间步的数量
        time_interval = 0.1
        # step_lengths = [1.0] * num_steps  # 步长列表，每个元素为当前步骤的长度
        Trajectory = calculate_trajectory(initial_position, initial_orientation, inclinations, azimuths, toolfaces, time_interval,
        num_steps, accelerations)
        dist_arr.append(math.sqrt(sum(x ** 2 for x in Trajectory[-1])))
        # plot_traj(Trajectory)
    return dist_arr


def per_node_moving_or_stopping_distance_check(df):
    print(len(df))
    # 创建一个空的字典来存储分割后的DataFrame
    dfs = {}
    current_group = []

    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        # 如果当前行的state不等于-1，则添加到当前组
        if row['status'] != 0:
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
    for name, df_piece in dfs.items():
        print(f"DataFrame pernode moving {name}:")
        inclinations, azimuths, toolfaces, accelerations, states = df_extract(df_piece)
        num_steps = len(inclinations)  # 时间步的数量
        time_interval = 0.1
        Trajectory = calculate_trajectory(initial_position, initial_orientation, inclinations, azimuths, toolfaces,
                                          time_interval, num_steps, accelerations)
        plot_traj(Trajectory)
        print(math.sqrt(sum(x ** 2 for x in Trajectory[-1])))


def df_acceleration_t_check(df):
    acceleration_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z']

    # 定义一个函数来执行t检验并返回结果
    def compare_groups(df, column, status1, status2):
        group1 = df[df['status'] == status1][column]
        group2 = df[df['status'] == status2][column]
        t_statistic, p_value = stats.ttest_ind(group1, group2)
        return t_statistic, p_value

        # 遍历加速度列进行检验

    results = {}
    for column in acceleration_columns:
        t_stat, p_value = compare_groups(df, column, 0, 1)
        results[column] = {'t_statistic': t_stat, 'p_value': p_value}

        # 输出结果
    for column, stats_dict in results.items():
        print(f"对于{column}列:")
        print(f"t统计量: {stats_dict['t_statistic']}")
        print(f"p值: {stats_dict['p_value']}\n")

        # 根据p值判断两组数据是否存在显著差异
        if stats_dict['p_value'] < 0.05:
            print(f"{column}列中，停顿和行进状态下的加速度存在显著差异。")
        else:
            print(f"{column}列中，停顿和行进状态下的加速度不存在显著差异。\n")


def df_acceleration_mod_t_check(df):
    df['acceleration_magnitude'] = np.sqrt(
        df['acceleration_x'] ** 2 +
        df['acceleration_y'] ** 2 +
        df['acceleration_z'] ** 2
    )

    # 对停顿和行进状态下的加速度模长进行t检验
    stopped_magnitude = df[df['status'] == 0]['acceleration_magnitude']
    moving_magnitude = df[df['status'] == 1]['acceleration_magnitude']
    t_statistic, p_value = stats.ttest_ind(stopped_magnitude, moving_magnitude)

    # 输出结果
    print(f"t统计量: {t_statistic}")
    print(f"p值: {p_value}")

    # 根据p值判断两组数据是否存在显著差异
    if p_value < 0.05:
        print("停顿和行进状态下的加速度模长存在显著差异。")
    else:
        print("停顿和行进状态下的加速度模长不存在显著差异。")


def df_extract(df):
    inclinations = df['inclination'].tolist()
    azimuths = df['azimuth'].tolist()
    toolfaces = df['toolface'].tolist()
    accelerations = df[['acceleration_x', 'acceleration_y', 'acceleration_z']].values.tolist()
    states = df['status'].tolist()
    return inclinations, azimuths, toolfaces, accelerations, states


def plot_traj(traj):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 提取轨迹的x, y, z坐标
    xs = [point[0] for point in traj]
    ys = [point[1] for point in traj]
    zs = [point[2] for point in traj]

    # 创建3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制轨迹
    ax.plot(xs, ys, zs, label='Trajectory')

    # 设置图表的标签
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # 设置图表的标题
    ax.set_title('3D Trajectory Visualization')

    # 显示图例
    ax.legend()

    # 显示图表
    plt.show()

def df_angle_threshold_check(df):
    THRESHOLD = 0.5
    # 遍历DataFrame的每一行（除了第一行）
    for i in range(1, len(df)):
        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]
        inclinations = df['inclination'].tolist()
        azimuths = df['azimuth'].tolist()
        toolfaces = df['toolface'].tolist()
        # 计算角度变化
        inclination_change = abs(curr_row['inclination'] - prev_row['inclination'])
        azimuth_change = abs(curr_row['azimuth'] - prev_row['azimuth'])
        toolface_change = abs(curr_row['toolface'] - prev_row['toolface'])

        # 判断是否达到阈值，来定义状态
        if inclination_change < THRESHOLD and azimuth_change < THRESHOLD and toolface_change < THRESHOLD:
            df.at[i, 'status'] = '停顿'
        else:
            df.at[i, 'status'] = '行进'

        # 显示结果
    print(df)


def df_threshold_solve(df):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    file_name = '设备1——8st停顿.xlsx'  # 请确保这是您的文件路径
    df['inclination_change'] = df['inclination'].diff().abs()
    df['azimuth_change'] = df['azimuth'].diff().abs()
    df['toolface_change'] = df['toolface'].diff().abs()

    col_indices = ['inclination_change', 'azimuth_change', 'toolface_change']

    n_features = len(col_indices)
    target_col = 'status'
    df = df.iloc[1:]
    df = df[df['status'] != -1]
    # 标准化数据
    # scaler = MinMaxScaler()
    # dataframe_scaled = scaler.fit_transform(dataframe)
    # dataframe_scaled = pd.DataFrame(dataframe_scaled, index=dataframe.index, columns=dataframe.columns)
    X = df[['inclination_change', 'azimuth_change', 'toolface_change']]
    y = df['status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化逻辑回归分类器
    clf = LogisticRegression()

    # 训练模型
    clf.fit(X_train, y_train)

    # 预测测试集
    y_pred = clf.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # 查看模型系数，这些系数可以解释为类似阈值的决策边界
    print('Coefficients:', clf.coef_)
    print('Intercept:', clf.intercept_)

    # 使用模型系数来推断可能的阈值（这可能需要进一步的解释和理解）
    # 注意：这不是直接的阈值，而是模型学习到的权重，它们共同决定了决策边界

    # # 清理不再需要的列
    # df.drop(['tilt_change', 'azimuth_change', 'tool_face_change', 'status_label'], axis=1, inplace=True)

    # 显示更新后的DataFrame（如果需要）
    print(df)

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

# 初始位置和方向向量
initial_position = (0.0, 0.0, 0.0)  # 初始位置 (x0, y0, z0)
initial_orientation = (1.0, 1.0, 1.0)  # 初始方向向量 (Bx0, By0, Bz0)，向上为正


# # 读取xlsx文件
# file_name = '设备1——8st停顿.xlsx'  # 请确保这是您的文件路径
#dfs = excel_readin(file_name, 8)

if __name__ == "__main__":
    # 读取dat文件
    # file_name = '8-3-1.dat'
    # dfs = dat_readin(file_name, 8)
    file_name = '设备1——8st停顿.xlsx'  # 请确保这是您的文件路径
    df = excel_readin_whole(file_name, 8)
    df_threshold_solve(df)
    # for name, df in dfs.items():
    #     print(name)
    #     inclinations, azimuths, toolfaces, accelerations, states = df_extract(df)
    #     num_steps = len(inclinations)  # 时间步的数量
    #     time_interval = 0.1
    #     # df_acceleration_mod_t_check(df)
    #     # print(num_steps)
    #     # per_node_moving_or_stopping_distance_check(df)
    #     dist_array = per_node_distance_check(df)
    #     print(np.mean(dist_array))
    #     print(np.std(dist_array))
    #     # 计算轨迹
    #     # Trajectory = calculate_trajectory(initial_position, initial_orientation, inclinations, azimuths, toolfaces,
    #     #                                   time_interval, num_steps, accelerations)
    #     # plot_traj(Trajectory)
    #     # print(sum(x ** 2 for x in Trajectory[-1]))
    #     # 打印轨迹
    #     # for point in Trajectory:
    #     #     print(point)

