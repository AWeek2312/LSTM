import pandas as pd
import numpy as np


def df_label_split_with_time_ranges(file_name, start_times, stop_time_ranges, return_time_ranges):
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


if __name__ == "__main__":
    df = pd.read_excel('4s停顿带回退.xlsx', header=None)

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
    start_times = [
        pd.Timestamp('1970-01-01 00:04:24.000000'),
        pd.Timestamp('1970-01-01 00:08:48.000000')
    ]
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

    stop_time_ranges = [
        [(pd.Timestamp('1970-01-01 00:00:00.000000'), pd.Timestamp('1970-01-01 00:00:04.000000')),
        (pd.Timestamp('1970-01-01 00:00:09.000000'), pd.Timestamp('1970-01-01 00:00:13.000000')),
            (pd.Timestamp('1970-01-01 00:00:18.000000'),pd.Timestamp('1970-01-01 00:00:22.000000')),
    (pd.Timestamp('1970-01-01 00:00:27.000000'),pd.Timestamp('1970-01-01 00:00:31.000000')),
    (pd.Timestamp('1970-01-01 00:00:45.000000'),pd.Timestamp('1970-01-01 00:00:49.000000')),
    (pd.Timestamp('1970-01-01 00:00:54.000000'),pd.Timestamp('1970-01-01 00:00:58.000000')),
    (pd.Timestamp('1970-01-01 00:01:03.000000'),pd.Timestamp('1970-01-01 00:01:07.000000')),
    (pd.Timestamp('1970-01-01 00:01:21.000000'),pd.Timestamp('1970-01-01 00:01:25.000000')),
    (pd.Timestamp('1970-01-01 00:01:30.000000'),pd.Timestamp('1970-01-01 00:01:34.000000')),
    (pd.Timestamp('1970-01-01 00:01:39.000000'),pd.Timestamp('1970-01-01 00:01:43.000000')),
    (pd.Timestamp('1970-01-01 00:01:57.000000'),pd.Timestamp('1970-01-01 00:02:01.000000')),
    (pd.Timestamp('1970-01-01 00:02:06.000000'),pd.Timestamp('1970-01-01 00:02:10.000000')),
    (pd.Timestamp('1970-01-01 00:02:15.000000'),pd.Timestamp('1970-01-01 00:02:19.000000')),
    (pd.Timestamp('1970-01-01 00:02:24.000000'),pd.Timestamp('1970-01-01 00:02:28.000000'))],
        [(pd.Timestamp('1970-01-01 00:04:24.000000'), pd.Timestamp('1970-01-01 00:04:28.000000')),
            (pd.Timestamp('1970-01-01 00:04:33.000000'),pd.Timestamp('1970-01-01 00:04:37.000000')),
    (pd.Timestamp('1970-01-01 00:04:42.000000'),pd.Timestamp('1970-01-01 00:04:46.000000')),
    (pd.Timestamp('1970-01-01 00:04:51.000000'),pd.Timestamp('1970-01-01 00:04:55.000000')),
    (pd.Timestamp('1970-01-01 00:05:09.000000'),pd.Timestamp('1970-01-01 00:05:13.000000')),
    (pd.Timestamp('1970-01-01 00:05:18.000000'),pd.Timestamp('1970-01-01 00:05:22.000000')),
    (pd.Timestamp('1970-01-01 00:05:27.000000'),pd.Timestamp('1970-01-01 00:05:31.000000')),
    (pd.Timestamp('1970-01-01 00:05:45.000000'),pd.Timestamp('1970-01-01 00:05:49.000000')),
    (pd.Timestamp('1970-01-01 00:05:54.000000'),pd.Timestamp('1970-01-01 00:05:58.000000')),
    (pd.Timestamp('1970-01-01 00:06:03.000000'),pd.Timestamp('1970-01-01 00:06:07.000000')),
    (pd.Timestamp('1970-01-01 00:06:21.000000'),pd.Timestamp('1970-01-01 00:06:25.000000')),
    (pd.Timestamp('1970-01-01 00:06:30.000000'),pd.Timestamp('1970-01-01 00:06:34.000000')),
    (pd.Timestamp('1970-01-01 00:06:39.000000'),pd.Timestamp('1970-01-01 00:06:43.000000')),
         (pd.Timestamp('1970-01-01 00:06:48.000000'), pd.Timestamp('1970-01-01 00:06:52.000000')), ],
        [(pd.Timestamp('1970-01-01 00:08:48.000000'), pd.Timestamp('1970-01-01 00:08:52.000000')),
            (pd.Timestamp('1970-01-01 00:08:57.000000'),pd.Timestamp('1970-01-01 00:09:01.000000')),
    (pd.Timestamp('1970-01-01 00:09:06.000000'),pd.Timestamp('1970-01-01 00:09:10.000000')),
    (pd.Timestamp('1970-01-01 00:09:15.000000'),pd.Timestamp('1970-01-01 00:09:19.000000')),
    (pd.Timestamp('1970-01-01 00:09:33.000000'),pd.Timestamp('1970-01-01 00:09:37.000000')),
    (pd.Timestamp('1970-01-01 00:09:42.000000'),pd.Timestamp('1970-01-01 00:09:46.000000')),
    (pd.Timestamp('1970-01-01 00:09:51.000000'),pd.Timestamp('1970-01-01 00:09:55.000000')),
    (pd.Timestamp('1970-01-01 00:10:09.000000'),pd.Timestamp('1970-01-01 00:10:13.000000')),
    (pd.Timestamp('1970-01-01 00:10:18.000000'),pd.Timestamp('1970-01-01 00:10:22.000000')),
    (pd.Timestamp('1970-01-01 00:10:27.000000'),pd.Timestamp('1970-01-01 00:10:31.000000')),
    (pd.Timestamp('1970-01-01 00:10:45.000000'),pd.Timestamp('1970-01-01 00:10:49.000000')),
    (pd.Timestamp('1970-01-01 00:10:54.000000'),pd.Timestamp('1970-01-01 00:10:58.000000')),
    (pd.Timestamp('1970-01-01 00:11:03.000000'),pd.Timestamp('1970-01-01 00:11:07.000000')),
         (pd.Timestamp('1970-01-01 00:11:12.000000'), pd.Timestamp('1970-01-01 00:11:16.000000'))]
    ]

    return_time_ranges =[
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
    from traj_cal import df_threshold_solve
    dfs = {}
    # 输出每个分割后的DataFrame来验证结果
    for i, split_df in enumerate(dfs):
        print(f"df_{i + 1}:")
        # print(split_df)
        print("\n")
        split_df['status'] = None
        for start_time, end_time in stop_time_ranges[i]:
            # 找到位于时间范围内的行的索引
            within_range_idx = ((pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f') >= start_time) &
                                (pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f') <= end_time))
            # 标记这些行为0
            split_df.loc[within_range_idx, 'status'] = 0
        for start_time, end_time in return_time_ranges[i]:
            # 找到位于时间范围内的行的索引
            within_range_idx = ((pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f') >= start_time) &
                                (pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f') <= end_time))
            # 标记这些行为-1
            split_df.loc[within_range_idx, 'status'] = -1
        split_df.loc[split_df['status'].isnull(), 'status'] = 1
        # pd.set_option('display.max_rows', None)  # 显示所有行
        # pd.set_option('display.max_columns', None)  # 显示所有列
        df_threshold_solve(split_df)
        print(split_df)
        dfs["df_{i + 1}:"] = split_df


