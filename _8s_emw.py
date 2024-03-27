import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TKAgg')

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体，或者其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def df_label_split_with_time_ranges(file_name, start_times, stop_time_ranges, return_time_ranges):
    df = pd.read_excel(file_name, header=None)

    old_columns = df.columns

    # 创建新的列索引，从0开始的整数
    new_columns = range(len(old_columns))

    # 重命名列
    df.columns = new_columns
    mapping = {
        0: '倾角',
        1: '方位角',
        2: '工具面角',
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

    # 假设你的DataFrame每零点一秒有十个点，总共100行
    freq = '100L'  # 每10毫秒
    periods = len(df)

    # 创建一个从0时刻开始的时间序列
    start_time = pd.Timestamp(0)  # 0时刻
    time_range = pd.date_range(start=start_time, periods=periods, freq=freq)

    # 将时间序列转换为分钟:秒:毫秒的格式
    time_labels = time_range.strftime('%Y-%m-%d %H:%M:%S.%f')
    # 如果需要只保留毫秒的前三位，可以使用字符串切片
    # time_labels = [label[:8] for label in time_labels]

    # 将时间标签添加到DataFrame中
    df['timestamp'] = time_labels
    # df['timestamp'] = df['timestamp'].apply(pd.to_datetime('%M:%S.%f'))
    # print(df)
    # 三个不同的开始时间

    # print(pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f'))
    # 根据开始时间分割DataFrame
    dfs = []
    start_idx = 0  # 从DataFrame的开始处开始
    for start_time in start_times:
        # 找到第一个大于等于开始时间的索引
        idx = (pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f') >= start_time).idxmax()
        if idx is not None:  # 确保idx不是None，以防start_time大于DataFrame中所有时间戳
            split_df = df[start_idx:idx]  # 从上一个开始时间的索引到当前开始时间的索引
            dfs.append(split_df)
            start_idx = idx  # 更新开始索引为当前开始时间的索引

    # 添加从最后一个开始时间到文件末尾的部分
    last_split_df = df[start_idx:]
    dfs.append(last_split_df)
    ret_dfs = []
    df_dict = {}
    # 输出每个分割后的DataFrame来验证结果
    for i, split_df in enumerate(dfs):
        # print(f"df_{i + 1}:")
        # # print(split_df)
        # print("\n")
        temp_df = split_df.copy()
        temp_df['status'] = [np.nan] * len(temp_df)
        for start_time, end_time in stop_time_ranges[i]:
            # 找到位于时间范围内的行的索引
            within_range_idx = ((pd.to_datetime(temp_df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f') >= start_time) &
                                (pd.to_datetime(temp_df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f') <= end_time))
            # 标记这些行为0
            temp_df.loc[within_range_idx, 'status'] = 0
        for start_time, end_time in return_time_ranges[i]:
            # 找到位于时间范围内的行的索引
            within_range_idx = ((pd.to_datetime(temp_df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f') >= start_time) &
                                (pd.to_datetime(temp_df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f') <= end_time))
            # 标记这些行为-1
            temp_df.loc[within_range_idx, 'status'] = -1
        temp_df.loc[temp_df['status'].isnull(), 'status'] = 1
        # pd.set_option('display.max_rows', None)  # 显示所有行
        # pd.set_option('display.max_columns', None)  # 显示所有列
        # print(split_df)
        ret_dfs.append(temp_df)
        df_dict[f"df_{i + 1}"] = temp_df
    return df_dict


def df_emw(df, col_list):
    columns_to_smooth = col_list

    # 选择这些列并应用指数平滑
    smoothed_columns = df[columns_to_smooth].apply(lambda x: x.ewm(alpha=0.1).mean())

    # 将平滑后的列与原始 DataFrame 中其他未平滑的列合并
    smoothed_df = pd.concat([df.drop(columns=columns_to_smooth), smoothed_columns], axis=1)
    return smoothed_df


def df_plot_col_before(df, column_name, num):
    import matplotlib.pyplot as plt
    # 绘制曲线图
    plt.figure(figsize=(10, 5))  # 设置图形大小
    plt.plot(df.index, df[column_name])
    plt.xlabel('时间步')  # 设置x轴标签
    plt.ylabel(column_name)  # 设置y轴标签
    plt.title(f'50°倾角8s停顿-第{num}趟数据平滑前-{column_name}随时间变化曲线')  # 设置标题
    plt.grid(True)  # 显示网格
    plt.savefig(f"C:\\Users\\Aweek\\Desktop\\数据平滑\\8s\\50度\\第{num}趟\\平滑前图像\\{column_name}.png")
    plt.show()  # 显示图形


def df_plot_col_after(df, column_name, num):
    import matplotlib.pyplot as plt
    # 绘制曲线图
    plt.figure(figsize=(10, 5))  # 设置图形大小
    plt.plot(df.index, df[column_name])
    plt.xlabel('时间步')  # 设置x轴标签
    plt.ylabel(column_name)  # 设置y轴标签
    plt.title(f'50°倾角8s停顿-第{num}趟数据平滑后-{column_name}随时间变化曲线')  # 设置标题
    plt.grid(True)  # 显示网格
    plt.savefig(f"C:\\Users\\Aweek\\Desktop\\数据平滑\\8s\\50度\\第{num}趟\\平滑后图像\\{column_name}.png")
    plt.show()  # 显示图形


if __name__ == "__main__":
    # 读取xlsx文件
    file_name = '数据文件/8s/50度倾角8s停顿带回退.xlsx'  # 请确保这是您的文件路径
    stop_time_ranges = [
        [(pd.Timestamp('1970-01-01 00:00:00.000000'), pd.Timestamp('1970-01-01 00:00:08.000000')),
         (pd.Timestamp('1970-01-01 00:00:13.000000'), pd.Timestamp('1970-01-01 00:00:21.000000')),
         (pd.Timestamp('1970-01-01 00:00:26.000000'), pd.Timestamp('1970-01-01 00:00:34.000000')),
         (pd.Timestamp('1970-01-01 00:00:39.000000'), pd.Timestamp('1970-01-01 00:00:47.000000')),
         (pd.Timestamp('1970-01-01 00:01:05.000000'), pd.Timestamp('1970-01-01 00:01:13.000000')),
         (pd.Timestamp('1970-01-01 00:01:18.000000'), pd.Timestamp('1970-01-01 00:01:26.000000')),
         (pd.Timestamp('1970-01-01 00:01:31.000000'), pd.Timestamp('1970-01-01 00:01:39.000000')),
         (pd.Timestamp('1970-01-01 00:01:57.000000'), pd.Timestamp('1970-01-01 00:02:05.000000')),
         (pd.Timestamp('1970-01-01 00:02:10.000000'), pd.Timestamp('1970-01-01 00:02:18.000000')),
         (pd.Timestamp('1970-01-01 00:02:23.000000'), pd.Timestamp('1970-01-01 00:02:31.000000')),
         (pd.Timestamp('1970-01-01 00:02:49.000000'), pd.Timestamp('1970-01-01 00:02:57.000000')),
         (pd.Timestamp('1970-01-01 00:03:02.000000'), pd.Timestamp('1970-01-01 00:03:10.000000')),
         (pd.Timestamp('1970-01-01 00:03:15.000000'), pd.Timestamp('1970-01-01 00:03:23.000000')),
         (pd.Timestamp('1970-01-01 00:03:28.000000'), pd.Timestamp('1970-01-01 00:03:36.000000'))],
        [(pd.Timestamp('1970-01-01 00:05:28.000000'), pd.Timestamp('1970-01-01 00:05:36.000000')),
         (pd.Timestamp('1970-01-01 00:05:41.000000'), pd.Timestamp('1970-01-01 00:05:49.000000')),
         (pd.Timestamp('1970-01-01 00:05:54.000000'), pd.Timestamp('1970-01-01 00:06:02.000000')),
         (pd.Timestamp('1970-01-01 00:06:07.000000'), pd.Timestamp('1970-01-01 00:06:15.000000')),
         (pd.Timestamp('1970-01-01 00:06:33.000000'), pd.Timestamp('1970-01-01 00:06:41.000000')),
         (pd.Timestamp('1970-01-01 00:06:46.000000'), pd.Timestamp('1970-01-01 00:06:54.000000')),
         (pd.Timestamp('1970-01-01 00:05:59.000000'), pd.Timestamp('1970-01-01 00:07:07.000000')),
         (pd.Timestamp('1970-01-01 00:07:25.000000'), pd.Timestamp('1970-01-01 00:07:33.000000')),
         (pd.Timestamp('1970-01-01 00:07:38.000000'), pd.Timestamp('1970-01-01 00:07:46.000000')),
         (pd.Timestamp('1970-01-01 00:07:51.000000'), pd.Timestamp('1970-01-01 00:07:59.000000')),
         (pd.Timestamp('1970-01-01 00:08:17.000000'), pd.Timestamp('1970-01-01 00:08:25.000000')),
         (pd.Timestamp('1970-01-01 00:08:30.000000'), pd.Timestamp('1970-01-01 00:08:38.000000')),
         (pd.Timestamp('1970-01-01 00:08:43.000000'), pd.Timestamp('1970-01-01 00:08:51.000000')),
         (pd.Timestamp('1970-01-01 00:08:56.000000'), pd.Timestamp('1970-01-01 00:09:04.000000')), ],
        [(pd.Timestamp('1970-01-01 00:10:56.000000'), pd.Timestamp('1970-01-01 00:11:04.000000')),
         (pd.Timestamp('1970-01-01 00:11:09.000000'), pd.Timestamp('1970-01-01 00:11:17.000000')),
         (pd.Timestamp('1970-01-01 00:11:22.000000'), pd.Timestamp('1970-01-01 00:11:30.000000')),
         (pd.Timestamp('1970-01-01 00:11:35.000000'), pd.Timestamp('1970-01-01 00:11:43.000000')),
         (pd.Timestamp('1970-01-01 00:12:01.000000'), pd.Timestamp('1970-01-01 00:12:09.000000')),
         (pd.Timestamp('1970-01-01 00:12:14.000000'), pd.Timestamp('1970-01-01 00:12:22.000000')),
         (pd.Timestamp('1970-01-01 00:12:27.000000'), pd.Timestamp('1970-01-01 00:12:35.000000')),
         (pd.Timestamp('1970-01-01 00:12:53.000000'), pd.Timestamp('1970-01-01 00:13:01.000000')),
         (pd.Timestamp('1970-01-01 00:13:06.000000'), pd.Timestamp('1970-01-01 00:13:14.000000')),
         (pd.Timestamp('1970-01-01 00:13:19.000000'), pd.Timestamp('1970-01-01 00:13:27.000000')),
         (pd.Timestamp('1970-01-01 00:13:45.000000'), pd.Timestamp('1970-01-01 00:13:53.000000')),
         (pd.Timestamp('1970-01-01 00:13:58.000000'), pd.Timestamp('1970-01-01 00:14:06.000000')),
         (pd.Timestamp('1970-01-01 00:14:11.000000'), pd.Timestamp('1970-01-01 00:14:19.000000')),
         (pd.Timestamp('1970-01-01 00:14:24.000000'), pd.Timestamp('1970-01-01 00:14:32.000000'))]
    ]

    return_time_ranges = [
        [(pd.Timestamp('1970-01-01 00:03:28.000000'), pd.Timestamp('1970-01-01 00:05:28.000000')),
         (pd.Timestamp('1970-01-01 00:00:52.000000'), pd.Timestamp('1970-01-01 00:01:05.000000')),
         (pd.Timestamp('1970-01-01 00:01:44.000000'), pd.Timestamp('1970-01-01 00:01:57.000000')),
         (pd.Timestamp('1970-01-01 00:02:36.000000'), pd.Timestamp('1970-01-01 00:02:49.000000'))
         ],
        [(pd.Timestamp('1970-01-01 00:08:56.000000'), pd.Timestamp('1970-01-01 00:10:56.000000')),
         (pd.Timestamp('1970-01-01 00:06:20.000000'), pd.Timestamp('1970-01-01 00:06:33.000000')),
         (pd.Timestamp('1970-01-01 00:07:12.000000'), pd.Timestamp('1970-01-01 00:07:25.000000')),
         (pd.Timestamp('1970-01-01 00:08:04.000000'), pd.Timestamp('1970-01-01 00:08:17.000000'))
         ],
        [(pd.Timestamp('1970-01-01 00:14:24.000000'), pd.Timestamp('1970-01-01 00:16:24.000000')),
         (pd.Timestamp('1970-01-01 00:11:48.000000'), pd.Timestamp('1970-01-01 00:12:01.000000')),
         (pd.Timestamp('1970-01-01 00:12:40.000000'), pd.Timestamp('1970-01-01 00:12:53.000000')),
         (pd.Timestamp('1970-01-01 00:13:32.000000'), pd.Timestamp('1970-01-01 00:13:45.000000'))
         ],
    ]
    start_times = [
        pd.Timestamp('1970-01-01 00:04:24.000000'),
        pd.Timestamp('1970-01-01 00:08:48.000000')
    ]
    dfs = df_label_split_with_time_ranges(file_name, start_times, stop_time_ranges, return_time_ranges)
    # 去掉回退的数据
    all_df = []
    for df_name, df in dfs.items():
        df = df[df['status'] != -1]
        all_df.append(df)

    # 遍历all_df中的三个DataFrame变量
    for i, df in enumerate(all_df):
        # 获取列名
        # column_names = df.columns[:3]
        col_list = ['倾角', '方位角', '工具面角']
        # 遍历列名,绘制平滑前的图像
        for col_name in col_list:
            df_plot_col_before(df, col_name, i + 1)
        # 数据平滑
        smoothed_df = df_emw(df, col_list)
        # 绘制平滑后的图像
        for col_name in col_list:
            df_plot_col_after(smoothed_df, col_name, i + 1)

