import pandas as pd
from datetime import datetime
import re

# 定义时间列表
four_second_times = [
    "00:00", "00:09", "00:18", "00:27", "00:36", "00:45", "00:54", "01:03", "01:12", "01:21", "01:31",
    "03:31", "03:40", "03:49", "03:58", "04:07", "04:16", "04:25", "04:34", "04:43", "04:52", "05:01",
    "07:01", "07:10", "07:19", "07:28", "07:37", "07:46", "07:55", "08:04", "08:13", "08:22", "08:31"
]

eight_second_times = [
    '00:13', '00:26', '00:39', '00:52', '01:05', '01:18', '01:31', '01:44', '01:57', '02:10',
    '04:10', '04:23', '04:36', '04:49', '05:02', '05:15', '05:28', '05:41', '05:54', '06:07',
    '06:20', '08:20', '08:33', '08:46', '08:59', '09:12', '09:25', '09:38', '09:51', '10:04',
    '10:17', '10:30']

# def go_stop_label(second_times, ):
#     # 转换时间为秒数的十倍
#     tenfold_seconds = []
#     for time_str in eight_second_times:
#         time_obj = datetime.strptime(time_str, "%M:%S")
#         total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
#         tenfold_seconds.append(total_seconds * 10)
#
#     print(tenfold_seconds)
#
#
# def time_cycle_def(pause_duration):
#     # 定义各个阶段的时间
#     # pause_duration = 4  # 停顿时间（秒）
#     travel_duration = 5  # 行走时间（秒）
#     num_node_total = 11
#     node_duration = pause_duration+travel_duration
#     measure_duration = node_duration * (num_node_total-1)
#     return_duration = 120  # 返回0号点的时间（秒）
#     cycle_duration = measure_duration+return_duration
#     measure_periods = []
#     return_periods = []
#     for cycle_num in range(10):
#         measure_periods.append((cycle_duration*cycle_num, cycle_duration*cycle_num+measure_duration))
#         return_periods.append((cycle_duration*(cycle_num+1)-return_duration, cycle_duration*(cycle_num+1)))
#     pause_periods = []
#     travel_periods = []
#     for measure_period in measure_periods:
#         temp_pause_periods = []
#         temp_travel_periods = []
#         for num_node in range(num_node_total):
#             temp_pause_periods.append((measure_period[0]+num_node*node_duration,
#                                        measure_period[0]+num_node*node_duration+pause_duration))
#             temp_travel_periods.append((measure_period[0]+(num_node+1)*node_duration-travel_duration,
#                                        measure_period[0]+(num_node+1)*node_duration))
#         pause_periods.append(temp_pause_periods)
#         travel_periods.append(temp_travel_periods)
#
#
#
# time_cycle_def(4)


def time_cycle_def(pause_duration):
    # 定义各个阶段的时间
    travel_duration = 5  # 行走时间（秒）
    num_node_total = 11  # 总节点数（包括0号点）
    node_duration = pause_duration + travel_duration  # 每个节点（包括停顿和行走）的持续时间
    measure_duration = node_duration * (num_node_total - 1)  # 测量总时长（不包括返回0号点的时间）
    return_duration = 120  # 返回0号点的时间（秒）
    cycle_duration = measure_duration + return_duration  # 一个完整周期的时间

    # 生成所有周期的列表
    measure_periods = [(cycle_duration * cycle_num, cycle_duration * cycle_num + measure_duration)
                       for cycle_num in range(10)]
    return_periods = [(cycle_duration * (cycle_num + 1) - return_duration, cycle_duration * (cycle_num + 1))
                      for cycle_num in range(10)]

    # 返回给定秒数所属的阶段
    def get_stage(second):
        for cycle_num, (start, end) in enumerate(measure_periods):
            if start <= second < end:
                # 在测量阶段内
                for node_num, (node_start, node_end) in enumerate(zip(
                        [start] + [start + node_duration * i for i in range(1, num_node_total - 1)],
                        [start + node_duration * (i + 1) for i in range(num_node_total - 1)])):
                    if node_start <= second < node_end:
                        # 判断是停顿还是行走
                        if node_start <= second < node_start + pause_duration:
                            # return f'第{cycle_num + 1}周期，在点{node_num}停顿'
                            return cycle_num+1, 0, node_num
                        else:
                            # return f'第{cycle_num + 1}周期，从点{node_num}到点{node_num + 1}行走'
                            return cycle_num+1, 1, node_num+1
        for cycle_num, (start, end) in enumerate(return_periods):
            if start <= second < end:
                # 在返回阶段
                # return f'第{cycle_num + 1}周期，返回0号点'
                return cycle_num+1, -1, -1
                # 如果秒数不在任何周期内，返回未知阶段
        return '未知阶段'

    return get_stage


# # 使用示例
# pause_duration = 4
# stage_func = time_cycle_def(pause_duration)
#
# # 判断某个秒数属于哪个阶段
# for i in range(1000):
#     second = int(i /10)
#     stage = stage_func(second)
#     print(f"第{second}秒属于：{stage}")