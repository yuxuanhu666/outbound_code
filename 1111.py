
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional
import copy
from dataclasses import dataclass
from math import ceil, sqrt
from gurobivisualcode import InteractiveYardVisualizer

import gurobipy as gp
from gurobipy import GRB

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# =================== 数据定义 ====================

# 时间参数
planning_horizon_days = 30
time_step_hours = 12
T = planning_horizon_days * 24 // time_step_hours  # 总时间步数
time_steps = list(range(1, T + 1))
bay_cap = 40  # 每个贝位容量(不管单贝还是两个连续贝都只能放30个箱子)
block_bays_num = 40  # 每个箱区的贝位数

# 箱区布局：4行×4列
rows, cols = 5, 5
total_blocks = rows * cols  # 16个箱区
blocks = list(range(1, total_blocks + 1))

# 为每个箱区设置坐标 (row, col)，行号和列号从1开始
block_coords = {}
for idx, b in enumerate(blocks):
    row = idx // cols + 1
    col = idx % cols + 1
    block_coords[b] = (row, col)

# 泊位位置
berth_positions = {
    'V1': (200, 0),
    'V2': (600, 0),
    'V3': (1000, 0),
    'V4': (1400, 0),
}

# 将 block (row,col) 映射为物理坐标（可按真实间距调整）
block_positions = {}
for b, (r, c) in block_coords.items():
    x = 200 + (c - 1) * 400  # 列间距400，基准x=200
    y = 1000 + (r - 1) * 300  # 行间距200，基准y=1000
    block_positions[b] = (x, y)



distance = {}

for berth_id, (vx, vy) in berth_positions.items():
    for block_id, (bx, by) in block_positions.items():
        d = math.hypot(bx - vx, by - vy)   # 欧氏距离
        distance[(berth_id, block_id)] = d


# 箱区容量配置（每个箱区的最大容量 = 贝位数 × 每个贝位容量）
# 注意：这里的容量是基于20尺箱的标准容量
# 每个20尺箱占1个容量单位，每个40尺/45尺箱占2个容量单位
BLOCK_CAPACITY = {}
for block in blocks:
    # 容量 = 贝位数 × 每个贝位容量
    BLOCK_CAPACITY[block] = block_bays_num * bay_cap


@dataclass
class AllocationGroup:
    group_id: int
    quantity: int
    size: str  # '20ft'/'40ft'/'45ft'
    ship: str
    period: int
    destination: str
    weight_level: str
    deck: str  # 'up' 或 'down'
    ship_bay: Union[int, Tuple[int, int], None] = None  # 单个奇数（20ft）或两奇数组成的tuple（40/45ft）


# 船舶信息
vessel_info = {
    'SHIP1': {'cycle_days': 7, 'berth': 'V1', 'start_time_step': 1},  # 周期 7 天，10 个 period，每个 period 保 5 个 subblocks
    'SHIP2': {'cycle_days': 10, 'berth': 'V1', 'start_time_step': 4},  # 周期 10 天，7 个 period
    # 'SHIP3': {'cycle_days': 7, 'berth': 'V1', 'start_time_step': 6},  # 周期 14 天，5 个 period
    'SHIP4': {'cycle_days': 7, 'berth': 'V2', 'start_time_step': 3},
    'SHIP5': {'cycle_days': 10, 'berth': 'V2', 'start_time_step': 5},
    # 'SHIP6': {'cycle_days': 14, 'berth': 'V3', 'start_time_step': 2},
    'SHIP7': {'cycle_days': 7, 'berth': 'V3', 'start_time_step': 1},
    # 'SHIP8': {'cycle_days': 10, 'berth': 'V4', 'start_time_step': 1},
    'SHIP9': {'cycle_days': 7, 'berth': 'V4', 'start_time_step': 3},

}

vessel_list = list(vessel_info.keys())


def calculate_vessel_schedule(vessel_info, total_time_steps):
    """
    1-based 时间步：time steps = 1 .. total_time_steps（包含）。
    返回的 receiving_period['start']/['end'] 与 loading_period[...] 都是 1-based，
    time_steps 列表也使用闭区间 range(start, end+1)（当 start<=end）。
    """
    CONTAINER_RECEIVING_DAYS = 3
    CONTAINER_RECEIVING_STEPS = CONTAINER_RECEIVING_DAYS * (24 // time_step_hours)  # 3 天 -> 6 步 （与你脚本一致）
    vessel_schedules = {}

    for ship_name, info in vessel_info.items():
        cycle_days = info['cycle_days']
        start_time_step = info.get('start_time_step', 1)
        berth = info.get('berth')

        # 周期的时间步数（每天2个 time step）：例如 cycle_days*2
        cycle_time_steps = cycle_days * (24 // time_step_hours)
        # 装船所用时间步（取周期的 5% 向上取整）
        loading_time_steps = math.ceil(cycle_time_steps * 0.05)

        cycles = []
        current_start = start_time_step
        cycle_num = 1

        # 当 current_start <= total_time_steps 时还有可能产生一个周期（1-based）
        while current_start <= total_time_steps:
            cycle_end = current_start + cycle_time_steps - 1

            # 装船期（在周期末尾）
            loading_start = cycle_end - loading_time_steps + 1
            loading_end = cycle_end

            # 收箱期：装船开始之前的若干步
            receiving_start = loading_start - CONTAINER_RECEIVING_STEPS
            receiving_end = loading_start - 1

            # 裁剪到 1..total_time_steps 范围内，并只在有重叠时记录 time_steps
            # 注意 cycle_start/cycle_end 也要裁剪以保证在规划期内
            actual_cycle_start = max(1, current_start)
            actual_cycle_end = min(cycle_end, total_time_steps)

            actual_loading_start = max(1, loading_start)
            actual_loading_end = min(loading_end, total_time_steps)

            actual_receiving_start = max(1, receiving_start)
            actual_receiving_end = min(receiving_end, total_time_steps)

            # 只当周期的装船时段在规划期内才把该周期记录（与你原逻辑保持一致的策略）
            if actual_loading_start <= total_time_steps and actual_loading_start <= actual_loading_end:
                cycle_info = {
                    'cycle_num': cycle_num,
                    'cycle_start': actual_cycle_start,
                    'cycle_end': actual_cycle_end,
                    'receiving_period': {
                        'start': actual_receiving_start if actual_receiving_start <= actual_receiving_end else None,
                        'end': actual_receiving_end if actual_receiving_start <= actual_receiving_end else None,
                        'time_steps': list(range(actual_receiving_start,
                                                 actual_receiving_end + 1)) if actual_receiving_start <= actual_receiving_end else []
                    },
                    'loading_period': {
                        'start': actual_loading_start if actual_loading_start <= actual_loading_end else None,
                        'end': actual_loading_end if actual_loading_start <= actual_loading_end else None,
                        'time_steps': list(range(actual_loading_start,
                                                 actual_loading_end + 1)) if actual_loading_start <= actual_loading_end else []
                    }
                }
                cycles.append(cycle_info)

            current_start += cycle_time_steps
            cycle_num += 1

        vessel_schedules[ship_name] = {
            'berth': berth,
            'cycle_days': cycle_days,
            'cycle_time_steps': cycle_time_steps,
            'loading_time_steps': loading_time_steps,
            'cycles': cycles
        }

    return vessel_schedules

# 计算船舶调度
schedules = calculate_vessel_schedule(vessel_info, T)
# 全局：为 (ship, period) 预先生成 收箱/装船 的 time steps 集合
sp_receiving_steps = {}
sp_loading_steps = {}

def build_sp_time_sets(schedules):
    for ship, info in schedules.items():
        cycles = info['cycles']
        for c in cycles:
            period = c['cycle_num']      # 1,2,3,...
            sp_key = (ship, period)
            sp_receiving_steps[sp_key] = set(c['receiving_period']['time_steps'])
            sp_loading_steps[sp_key]   = set(c['loading_period']['time_steps'])

# 在 schedules 计算完以后调用一次
build_sp_time_sets(schedules)

def is_conflict(sp1, sp2):
    # sp1, sp2 形如 ('SHIP1', 1)
    rec1 = sp_receiving_steps.get(sp1, set())
    load1 = sp_loading_steps.get(sp1, set())
    rec2 = sp_receiving_steps.get(sp2, set())
    load2 = sp_loading_steps.get(sp2, set())

    # 条件：某一方进箱 与 另一方装船 的时间步有交集 => 冲突
    if rec1 & load2:
        return True
    if rec2 & load1:
        return True
    return False

def print_vessel_schedules(schedules):
    print("\n================= 船舶周期调度明细 =================\n")
    for ship, info in schedules.items():
        print(f"船舶: {ship}")
        print(f"  泊位: {info['berth']}")
        print(f"  周期天数: {info['cycle_days']}  每周期时间步: {info['cycle_time_steps']}")
        print(f"  装船持续步数: {info['loading_time_steps']}")
        print("  --------------------------------------------------")

        cycles = info['cycles']
        if not cycles:
            print("  **无有效周期（loading_period 不在规划范围内）**\n")
            continue

        for c in cycles:
            print(f"  周期 {c['cycle_num']}:")
            print(f"    周期范围: {c['cycle_start']} → {c['cycle_end']}")

            # 收箱期
            r = c['receiving_period']
            if r['start'] is not None:
                print(f"    收箱期(receiving): {r['start']} → {r['end']}")
                print(f"      步: {r['time_steps']}")
            else:
                print("    收箱期(receiving): 无（完全不在规划期内）")

            # 装船期
            l = c['loading_period']
            if l['start'] is not None:
                print(f"    装船期(loading):   {l['start']} → {l['end']}")
                print(f"      步: {l['time_steps']}")
            else:
                print("    装船期(loading):   无（完全不在规划期内）")

            print("  --------------------------------------------------")

        print("")  # 空行分隔船舶

print_vessel_schedules(schedules)

def generate_random_teu_demand(vessel_info, planning_horizon_days):
    """
    随机生成每个船舶每个周期的TEU需求

    参数:
    - vessel_info: 船舶信息字典
    - planning_horizon_days: 计划期天数

    返回:
    - teu_demand: 字典格式的TEU需求 {(ship, period): teu_value}
    - container_details: 字典格式的箱子数量详情 {(ship, period): {'20ft': num, '40ft': num, '45ft': num}}
    """
    random.seed(42)  # 添加这一句，确保每次运行结果相同
    np.random.seed(42)
    teu_demand = {}
    container_details = {}  # 存储各种尺寸箱子的箱量

    # 箱子类型配置
    container_types = {
        '20ft': {'teu_factor': 1, 'proportion': 0.60},  # 20尺箱子占75%
        '40ft': {'teu_factor': 2, 'proportion': 0.30},  # 40尺箱子占23%
        '45ft': {'teu_factor': 2, 'proportion': 0.10},  # 45尺箱子占2%
    }

    for ship, info in vessel_info.items():
        cycle_days = info['cycle_days']

        # 向上取整：包含不完整的周期
        num_periods = ceil(planning_horizon_days / cycle_days)

        print(f"船舶 {ship}: 周期天数={cycle_days}, 计划期={planning_horizon_days}天, 周期数={num_periods}")

        for period in range(1, num_periods + 1):
            # 生成符合正态分布的基础箱子数量
            # 使用正态分布生成，均值1200，标准差sqrt(200)≈14.14
            base_containers = int(np.random.normal(1700, np.sqrt(300)))

            # 确保在指定范围内
            base_containers = max(1500, min(2000, base_containers))

            # 根据箱子类型比例计算实际TEU
            total_teu = 0

            # 20尺箱子数量（75%）
            containers_20ft = int(base_containers * container_types['20ft']['proportion'])
            total_teu += containers_20ft * container_types['20ft']['teu_factor']

            # 40尺箱子数量（23%）
            containers_40ft = int(base_containers * container_types['40ft']['proportion'])
            total_teu += containers_40ft * container_types['40ft']['teu_factor']

            # 45尺箱子数量（2%）
            containers_45ft = int(base_containers * container_types['45ft']['proportion'])
            total_teu += containers_45ft * container_types['45ft']['teu_factor']

            # 调整剩余箱子到20尺类型，确保总数接近目标
            remaining_containers = base_containers - containers_20ft - containers_40ft - containers_45ft
            if remaining_containers > 0:
                containers_20ft += remaining_containers
                total_teu += remaining_containers * container_types['20ft']['teu_factor']

            # 确保TEU在合理范围内
            total_teu = max(200, min(3300, total_teu))  # 考虑到大箱子的TEU会更高

            # 存储TEU需求
            teu_demand[(ship, period)] = total_teu

            # 存储箱子数量详情
            container_details[(ship, period)] = {
                '20ft': containers_20ft,
                '40ft': containers_40ft,
                '45ft': containers_45ft,
                'total': containers_20ft + containers_40ft + containers_45ft
            }

    return teu_demand, container_details


# 生成随机TEU需求和箱子数量详情
teu_demand, container_details = generate_random_teu_demand(vessel_info, planning_horizon_days)

# 提取每个船舶每个周期的45尺箱量
containers_45ft = {}
for (ship, period), details in container_details.items():
    containers_45ft[(ship, period)] = details['45ft']

global_containers_45ft = containers_45ft

# 提取每个船舶每个周期的40尺箱量
containers_40ft = {}
for (ship, period), details in container_details.items():
    containers_40ft[(ship, period)] = details['40ft']
global_containers_40ft = containers_40ft

# 提取每个船舶每个周期的20尺箱量
containers_20ft = {}
for (ship, period), details in container_details.items():
    containers_20ft[(ship, period)] = details['20ft']
global_containers_20ft = containers_20ft


def generate_allocation_groups(vessel_list, vessel_info, containers_20ft, containers_40ft, containers_45ft):
    # 可用的目的港、重量等级
    DEST_LIST = ["PORT_A", "PORT_B", "PORT_C", "PORT_D",
                 "PORT_E", "PORT_F", "PORT_G", "PORT_H",
                 "PORT_I", "PORT_J", "PORT_K", "PORT_L",
                 "PORT_M", "PORT_N", "PORT_O", "PORT_P",
                 "PORT_Q", "PORT_R", "PORT_S", "PORT_T",
                 "PORT_U", "PORT_V", "PORT_W", "PORT_X", "PORT_Y", "PORT_Z",
                 "PORT_aa", "PORT_bb", "PORT_cc", "PORT_dd",
                 "PORT_ee", "PORT_ff", "PORT_gg", "PORT_hh",
                 "PORT_ii", "PORT_jj", "PORT_kk", "PORT_ll",
                 "PORT_mm", "PORT_nn", "PORT_oo", "PORT_pp", ]
    WEIGHT_LEVELS = ['light', 'middle', 'heavy', 'superheavy']

    allocation_dict = defaultdict(int)
    random.seed(123)

    # deck_dict: 保存每个(ship, period, dest) 对应的 deck
    deck_dict = dict()

    # 逐舰-逐周期-逐尺寸分组（生成 allocation_dict）
    for ship in vessel_list:
        periods = {period for (s, period) in containers_20ft.keys() if s == ship}
        for period in periods:
            # 随机选目的港
            dest_count = random.randint(5, 6)
            dests = random.sample(DEST_LIST, dest_count)

            # 为每个 (ship,period,dest) 分配一次 deck
            for dest in dests:
                deck_dict[(ship, period, dest)] = random.choice(['up', 'down'])

            for size, container_dict in [('20ft', containers_20ft), ('40ft', containers_40ft),
                                         ('45ft', containers_45ft)]:
                total_boxes = container_dict.get((ship, period), 0)
                if total_boxes == 0:
                    continue

                avg_per_dest = total_boxes // dest_count
                remain = total_boxes % dest_count
                per_dest_list = [avg_per_dest] * dest_count
                for i in range(remain):
                    per_dest_list[i] += 1

                for dest, box_num in zip(dests, per_dest_list):
                    weights_split = [random.randint(0, box_num) for _ in range(3)]
                    weights_split.sort()
                    counts = [
                        weights_split[0],
                        weights_split[1] - weights_split[0],
                        weights_split[2] - weights_split[1],
                        box_num - weights_split[2]
                    ]
                    for wl, cnt in zip(WEIGHT_LEVELS, counts):
                        if cnt > 0:
                            key = (ship, period, size, dest, wl)
                            allocation_dict[key] += cnt

    pool = dict()

    def get_state(ship, period):
        k = (ship, period)
        if k not in pool:
            pool[k] = {'next_bay': 1, 'single_bays': dict(), 'pair_bays': dict()}
        return pool[k]

    # 将 allocation_dict 重组为以 base_key=(ship,period,size,dest) 为单位的子列表（方便把同一 base 放到同一 bay）
    base_groups = defaultdict(list)  # base_key -> list of (wl, qty)
    for (ship, period, size, dest, wl), qty in allocation_dict.items():
        base_key = (ship, period, size, dest)
        base_groups[base_key].append((wl, qty))

    groups = []
    group_id = 1
    MAX_CAP = 100

    def get_arrival_and_pickup(ship, period):
        """
        从 schedules 里取收箱期开始（arrival_step）和装船期开始（pickup_step）。
        如果 period 超出该船计算的 cycles，取最后一个周期作为兜底（避免 None）。
        """
        sched = schedules.get(ship, {})
        cycles = sched.get('cycles', [])
        if not cycles:
            return None, None
        idx = max(0, min(len(cycles) - 1, period - 1))
        cyc = cycles[idx]
        arrival = cyc.get('receiving_period', {}).get('start')
        pickup = cyc.get('loading_period', {}).get('start')
        # 如果 schedules 是 0-based step（你脚本中用 0 作为起点），保持一致；否则按需 +1
        return arrival, pickup

    for (ship, period, size, dest), wl_list in base_groups.items():
        state = get_state(ship, period)

        # 保持该 base_key 已使用的贝位列表（用于优先尝试放入同一贝位）
        base_used_bays = []  # list of bay_id（int 或 tuple）
        arrival_step, pickup_step = get_arrival_and_pickup(ship, period)

        # 对该 base 下的每个 weight_level 的数量逐个处理
        for wl, qty in wl_list:
            remaining_qty = qty
            # 如果 qty > MAX_CAP，则拆分为多个块（每块 <= MAX_CAP）
            chunks = []
            while remaining_qty > 0:
                take = min(remaining_qty, MAX_CAP)
                chunks.append(take)
                remaining_qty -= take

            for chunk in chunks:
                assigned_bay = None

                # 1) 先尝试 base 已用贝位（同一 ship,period,size,dest）
                for bay in base_used_bays:
                    if isinstance(bay, int):  # single bay
                        cap = state['single_bays'].get(bay, 0)
                        if cap >= chunk and size == '20ft':
                            assigned_bay = bay
                            state['single_bays'][bay] -= chunk
                            break
                    else:  # tuple pair
                        cap = state['pair_bays'].get(bay, 0)
                        if cap >= chunk and size in ('40ft', '45ft'):
                            assigned_bay = bay
                            state['pair_bays'][bay] -= chunk
                            break

                # 2) 若未找到，尝试同 ship-period 的其他已有贝位（同类型）
                if assigned_bay is None:
                    if size == '20ft':
                        # 在已有 single_bays 中找一个剩余 >= chunk 的
                        for bay, cap in state['single_bays'].items():
                            if cap >= chunk:
                                assigned_bay = bay
                                state['single_bays'][bay] -= chunk
                                break
                    else:
                        for bay, cap in state['pair_bays'].items():
                            if cap >= chunk:
                                assigned_bay = bay
                                state['pair_bays'][bay] -= chunk
                                break

                # 3) 若仍未找到，则新建贝位（single 或 pair）
                if assigned_bay is None:
                    if size == '20ft':
                        # 修改：使用连续编号
                        bay_num = state['next_bay']
                        state['next_bay'] += 1  # 下一个编号
                        state['single_bays'][bay_num] = MAX_CAP - chunk
                        assigned_bay = bay_num
                    else:
                        # 修改：使用连续编号的贝位对
                        bay1 = state['next_bay']
                        bay2 = state['next_bay'] + 1
                        state['next_bay'] += 2  # 占用两个连续编号
                        state['pair_bays'][(bay1, bay2)] = MAX_CAP - chunk
                        assigned_bay = (bay1, bay2)

                # 将 assigned_bay 记录为 base 已用贝位（若尚未记录）
                if assigned_bay not in base_used_bays:
                    base_used_bays.append(assigned_bay)

                # 生成 group（字典形式，与 YardSolver.run() 兼容）
                g = {
                    'group_id': group_id,
                    'quantity': int(chunk),
                    'size': size,
                    'ship': ship,
                    'period': period,
                    'destination': dest,
                    'weight_level': wl,
                    'deck': deck_dict.get((ship, period, dest), 'up'),
                    'ship_bay': assigned_bay,
                    'arrival_step': arrival_step,
                    'pickup_step': pickup_step,

                }
                groups.append(g)
                group_id += 1

    return groups


# 用法示例
allocation_groups = generate_allocation_groups(vessel_list, vessel_info, containers_20ft, containers_40ft,
                                               containers_45ft)

class YardSolver:
    """
    YardSolver: 把原来的 solve_with_sliding_windows 封装成类
    使用方法：
        solver = YardSolver(allocation_groups, blocks, block_types, block_positions,
                            berth_positions, vessel_info,
                            block_bays_num=40, bay_cap=30, T=T, window_size=6, ...)
        all_assignments, remaining_state, final_df = solver.run()
    结果保存在 solver.all_assignments / solver.remaining_state / solver.final_df
    """

    def __init__(self,
                 allocation_groups,
                 blocks,
                 block_positions,
                 berth_positions,
                 vessel_info,
                 block_bays_num=40,
                 bay_cap=30,
                 T=None,
                 window_size=2,
                 initial_state=None,
                 verbose=True,
                 time_limit=400,
                 mipgap=0.03,
                 ):
        self.allocation_groups = allocation_groups
        self.blocks = blocks
        self.block_positions = block_positions
        self.berth_positions = berth_positions
        self.vessel_info = vessel_info
        self.block_bays_num = block_bays_num
        self.bay_cap = bay_cap
        self.T = T
        self.window_size = window_size
        self.initial_state = initial_state if initial_state is not None else []
        self.verbose = verbose
        self.time_limit = time_limit
        self.mipgap = mipgap


        # 运行结果占位
        self.all_assignments = []
        self.remaining_state = []
        self.final_df = pd.DataFrame()

    # ------------------ helper: 原 generate_bay_pairs_for_size ------------------
    @staticmethod
    def generate_bay_pairs_for_size(size, block_bays_num):
        """返回给定尺寸允许的贝位对列表 (i,i) 或 (i,i+1)"""
        pairs = []
        if size == '20ft':
            pairs = [(i, i) for i in range(1, block_bays_num + 1)]
        else:  # 40ft 或 45ft
            pairs = [(i, i + 1) for i in range(1, block_bays_num,2)]
            if size == '45ft':
                # 45尺只能放在最左(1,2)或最右(39,40)
                if len(pairs) >= 2:
                    pairs = [pairs[0], pairs[-1]]
        return pairs

    @staticmethod
    def _size_to_teu(sz):
        if sz is None:
            return 1
        if isinstance(sz, str) and ('40' in sz or '45' in sz):
            return 2
        return 1

    # ------------------ run()：把原 solve_with_sliding_windows 的逻辑搬进来 ------------------
    def run(self):
        """运行滑动窗口求解并返回 (all_assignments, remaining_state, final_df)"""
        allocation_groups = self.allocation_groups
        blocks = self.blocks
        block_positions = self.block_positions
        berth_positions = self.berth_positions
        vessel_info = self.vessel_info
        block_bays_num = self.block_bays_num
        bay_cap = self.bay_cap
        T = self.T
        window_size = self.window_size
        initial_state = copy.deepcopy(self.initial_state)
        verbose = self.verbose

        if initial_state is None:
            initial_state = []

        # 建立便于查询的字典：ship->berth_pos
        ship_berth = {s: vessel_info[s]['berth'] for s in vessel_info}

        all_assignments = []  # 输出汇总
        current_state = copy.deepcopy(initial_state)

        # 按 arrival_step 索引 allocation_groups
        groups_by_arrival = defaultdict(list)
        for g in allocation_groups:
            a = g.get('arrival_step')
            if isinstance(a, int):
                groups_by_arrival[a].append(g)
            else:
                print("⚠ 非整数 arrival_step：", g)

        # 窗口滑动（非重叠窗，步长 = window_size）
        for window_start in range(1, T+1, window_size):
            window_end = min(window_start + window_size - 1, T)
            if verbose:
                print(f"\n=== 求解窗口 [{window_start}, {window_end}] ===")

            # 1) 本窗口新到港组
            window_new_groups = []
            for at in range(window_start, window_end + 1):
                window_new_groups.extend(groups_by_arrival.get(at, []))

            # 2) 构造 existing_occupancy（TEU 单位）
            existing_occupancy = defaultdict(int)
            filtered_state = []
            for rec in current_state:
                if rec['pickup_step'] is None or rec['pickup_step'] > window_start:
                    sz = rec.get('size', '20ft')
                    existing_occupancy[(rec['block'], tuple(rec['bp']))] += int(rec['qty']) * self._size_to_teu(sz)
                    if rec['pickup_step'] is None or rec['pickup_step'] > window_start:
                        filtered_state.append(rec)
            current_state = filtered_state

            # 基于 current_state 统计：每个 block 里有哪些 (ship, period) 已经被放进去
            block_sp_used = defaultdict(set)
            for rec in current_state:
                b = rec['block']
                sp = (rec['ship'], rec['period'])  # period 刚才已经建议你在 rec 里加上
                block_sp_used[b].add(sp)

            # 3) 若本窗口无新到港组，则按 pickup 更新 state，跳过建模
            if len(window_new_groups) == 0:
                if verbose:
                    print("本窗口无新到港分配组，跳过求解（但会按取箱时间更新 state）。")
                next_state = []
                for rec in current_state:
                    if rec['pickup_step'] is None or rec['pickup_step'] > window_end:
                        next_state.append(rec)
                current_state = next_state
                continue

            # 4) 构建 Gurobi 模型
            m = gp.Model(f"window_{window_start}_{window_end}")
            # 可选：当窗口较大可能需要更多时间限制，这里用你已有设置，或重新设置
            m.Params.OutputFlag = 1
            m.Params.TimeLimit = 500
            m.Params.MIPGap = 0.03
            # 性能优化
            m.Params.Threads = 3  # 使用所有核心
            m.Params.MIPFocus = 1  # 优先找可行解
            m.Params.Heuristics = 0.1  # 10% 时间用于启发式



            # 改进策略
            m.Params.ImproveStartTime = 60  # 1分钟后开始改进

            # 决策变量：y[g_id, b, bp]（整数箱量）
            y = {}
            # 二进制：z[b, bp] 表示该箱区该贝位对是否被使用
            z = {}
            # group unmet 量（若不可行则允许未分配，惩罚）
            unmet = {}


            # 预先生成每个 group 的可选 (block, bp) 列表

            group_candidates = {}
            for g in window_new_groups:
                gid = g['group_id']
                size = g['size']
                ship = g['ship']
                period = g['period']
                sp_new = (ship, period)
                allowed_bps = [tuple(bp) for bp in self.generate_bay_pairs_for_size(size, block_bays_num)]
                cand = []

                for b in blocks:
                    # 先看这个 block 里历史上已经有哪些 (ship,period)
                    bad_block = False
                    for sp_old in block_sp_used.get(b, set()):
                        if is_conflict(sp_new, sp_old):
                            bad_block = True
                            break
                    if bad_block:
                        continue  # 这个 block 对新 group 来说是非法候选

                    # 不冲突，才把所有贝位对加入候选
                    for bp in allowed_bps:
                        cand.append((b, bp))

                group_candidates[gid] = cand

            # 用于约束：对每个箱区、每个 bp 创建 z 变量（注意同时包括 existing occupancy 中已存在的 bp）
            all_bps_by_block = defaultdict(set)
            # 添加所有可能的 bp（由任何 group 候选或 existing_occupancy 产生）
            for gid, cand in group_candidates.items():
                for b, bp in cand:
                    all_bps_by_block[b].add(tuple(bp))
            for (b, bp), q in existing_occupancy.items():
                all_bps_by_block[b].add(tuple(bp))

            # 创建 z 变量
            for b, bps in all_bps_by_block.items():
                for bp in bps:
                    z[(b, bp)] = m.addVar(vtype=GRB.BINARY, name=f"z_b{b}_bp{bp[0]}_{bp[1]}")

            # 创建 y 变量（仅为 group 的候选(bp, block)）
            for g in window_new_groups:
                gid = g['group_id']
                qty = int(g['quantity'])
                unmet[gid] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=qty, name=f"unmet_{gid}")
                for (b, bp) in group_candidates[gid]:
                    y[(gid, b, tuple(bp))] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=qty,
                                                      name=f"y_{gid}_b{b}_bp{bp[0]}_{bp[1]}")

            m.update()

            # ====== 插入点：在 y/unmet 创建并 m.update() 之后 ======
            # 添加按 block 的二进制指示变量 u[(gid,b)]，以及每个组的 used 标志
            u = {}  # u[(gid,b)] = 0/1 表示 group gid 是否在 block b 使用过
            used = {}  # used[gid] = 0/1 表示 group gid 是否有任何被分配的箱

            for g in window_new_groups:
                gid = g['group_id']
                qty = int(g['quantity'])
                used[gid] = m.addVar(vtype=GRB.BINARY, name=f"used_{gid}")
                # allowed blocks 由 group_candidates 给出（去重）
                allowed_blocks = sorted({b for (b, bp) in group_candidates.get(gid, [])})
                for b in allowed_blocks:
                    u[(gid, b)] = m.addVar(vtype=GRB.BINARY, name=f"u_{gid}_b{b}")

            # 先建一个字典：gid -> (ship, period)
            gid_to_sp = {}
            for g in window_new_groups:
                gid_to_sp[g['group_id']] = (g['ship'], g['period'])

            # 对每个箱区 b，检查本窗口中潜在会用到这个箱区的 group 组合
            for b in blocks:
                # 只考虑在候选里能放进 b 的 group（否则没必要加约束）
                gids_in_b = sorted({
                    gid for (gid2, bb, bp) in y.keys()
                    for gid in [gid2] if bb == b
                })

                # 两两检查是否冲突
                for i in range(len(gids_in_b)):
                    gid1 = gids_in_b[i]
                    sp1 = gid_to_sp[gid1]
                    for j in range(i + 1, len(gids_in_b)):
                        gid2 = gids_in_b[j]
                        sp2 = gid_to_sp[gid2]

                        if is_conflict(sp1, sp2):
                            # 这两个 (船,period) 在时间上有“进箱/装船冲突”，
                            # 那么它们不能同时用同一个箱区 b
                            if (gid1, b) in u and (gid2, b) in u:
                                m.addConstr(
                                    u[(gid1, b)] + u[(gid2, b)] <= 1,
                                    name=f"no_mix_RL_g{gid1}_g{gid2}_b{b}"
                                )

            # ---------- 连续性约束：为每个 group-block 创建物理贝位二进制变量 x[(gid,b,i)] ----------
            # 说明：此处应在 u 和 used 已经被创建后插入

            # 1) 先确定一个合理的 BIGM_group（用于 y <= BIGM_group * x 的线性化）
            BIGM_group = max(int(g['quantity']) for g in window_new_groups) if window_new_groups else 1000

            # 2) 创建 x 变量
            x = {}  # x[(gid,b,i)] = 0/1 表示 group gid 在 block b 使用了物理贝位 i
            for g in window_new_groups:
                gid = g['group_id']
                # allowed blocks 由 group_candidates 给出（去重）
                allowed_blocks = sorted({b for (b, bp) in group_candidates.get(gid, [])})
                for b in allowed_blocks:
                    for i in range(1, block_bays_num + 1):
                        x[(gid, b, i)] = m.addVar(vtype=GRB.BINARY, name=f"x_g{gid}_b{b}_i{i}")

            m.update()

            # 3) 把 y 与 x 互相绑定：
            #    - 若 y[(gid,b,bp)] > 0，则 bp 内每个物理贝位对应的 x 必须为1： y <= BIGM_group * x[(...,i)]
            #    - 若 x[(gid,b,i)] == 1，则在该物理贝位上至少有一个 y 覆盖它： sum_y_on_i >= x
            for (gid, b, bp) in list(y.keys()):
                # bp 是 tuple (i,j)
                i0, j0 = bp[0], bp[1]
                for i in range(i0, j0 + 1):
                    # y <= BIGM * x
                    m.addConstr(y[(gid, b, bp)] <= BIGM_group * x[(gid, b, i)],
                                name=f"link_y_x_g{gid}_b{b}_bp{bp[0]}_{bp[1]}_i{i}")

            # 反向约束：对每个 (gid,b,i) 计算覆盖该 i 的所有 y 的和
            for g in window_new_groups:
                gid = g['group_id']
                allowed_blocks = sorted({b for (b, bp) in group_candidates.get(gid, [])})
                for b in allowed_blocks:
                    for i in range(1, block_bays_num + 1):
                        # 找到所有覆盖 i 的 y 变量
                        sum_y_on_i = gp.quicksum(
                            y[(gid, b, tuple(bp))] for (bb, bp) in group_candidates.get(gid, [])
                            if bb == b and (gid, b, tuple(bp)) in y and (i >= bp[0] and i <= bp[1])
                        )
                        # 若 sum_y_on_i >= 1 则 x 必须为1；若 x 为1 则 sum_y_on_i 至少 1
                        m.addConstr(sum_y_on_i >= x[(gid, b, i)],
                                    name=f"link_sumY_ge_x_g{gid}_b{b}_i{i}")
                        m.addConstr(sum_y_on_i <= BIGM_group * x[(gid, b, i)],
                                    name=f"link_sumY_leM_x_g{gid}_b{b}_i{i}")

            # 4) 把 u[(gid,b)] 跟 x 绑定（若组在该 box 使用，则至少一个物理贝位被占用）
            for g in window_new_groups:
                gid = g['group_id']
                allowed_blocks = sorted({b for (b, bp) in group_candidates.get(gid, [])})
                for b in allowed_blocks:
                    m.addConstr(gp.quicksum(x[(gid, b, i)] for i in range(1, block_bays_num + 1)) >= u[(gid, b)],
                                name=f"link_u_x_low_g{gid}_b{b}")
                    m.addConstr(gp.quicksum(x[(gid, b, i)] for i in range(1, block_bays_num + 1)) <= block_bays_num * u[
                        (gid, b)],
                                name=f"link_u_x_up_g{gid}_b{b}")


            m.update()
            # ---------- 连续性约束 插入结束 ----------

            # t[(gid,b,i)] 表示 x_i 与 x_{i+1} 是否不同（即是否存在边界）
            t = {}
            for g in window_new_groups:
                gid = g['group_id']
                # allowed blocks 由 group_candidates 给出（去重）
                allowed_blocks = sorted({b for (b, bp) in group_candidates.get(gid, [])})
                for b in allowed_blocks:
                    # 为 i = 1..(block_bays_num-1) 创建转变二进制变量
                    for i in range(1, block_bays_num):
                        t[(gid, b, i)] = m.addVar(vtype=GRB.BINARY, name=f"t_g{gid}_b{b}_i{i}")
                        # t >= x_i - x_{i+1}
                        m.addConstr(t[(gid, b, i)] >= x[(gid, b, i)] - x[(gid, b, i + 1)],
                                    name=f"t_ge_diff1_g{gid}_b{b}_i{i}")
                        # t >= x_{i+1} - x_i
                        m.addConstr(t[(gid, b, i)] >= x[(gid, b, i + 1)] - x[(gid, b, i)],
                                    name=f"t_ge_diff2_g{gid}_b{b}_i{i}")
                    # 如果组在该 box 没被使用（u==0），则转变数应为0；若被使用，则转变数 <= 2（即 0/1 段或单一连续段）
                    m.addConstr(gp.quicksum(t[(gid, b, i)] for i in range(1, block_bays_num)) <= 2 * u[(gid, b)],
                                name=f"contiguous_transitions_limit_g{gid}_b{b}")

            m.update()

            # ------------------ 新增：为每个 (gid,b,bp) 引入 v 二进制，表示该组是否使用了该 bp ------------------
            # 插入点：在 y, unmet, u, used, x, t 等变量创建并 m.update() 之后，且在目标函数构建之前

            # BIGM_group 你之前已有（用于 y <= BIGM_group * x），可以复用
            # 如果没有，可使用下面之一：
            # BIGM_group = max(int(g['quantity']) for g in window_new_groups) if window_new_groups else 1000

            v = {}
            for (gid, b, bp) in list(y.keys()):
                # bp 是 tuple
                v[(gid, b, bp)] = m.addVar(vtype=GRB.BINARY, name=f"v_g{gid}_b{b}_bp{bp[0]}_{bp[1]}")

            m.update()

            # 约束：v == 1 <=> y >= 1（因为 y 为整数）
            for (gid, b, bp) in list(y.keys()):
                # 如果该 y 变量存在，则强制：
                # 1) y >= v   （若 v=1 则至少有 1 个箱放在该 bp）
                # 2) y <= BIGM_group * v （若 v=0 则 y 必须为0）
                m.addConstr(y[(gid, b, bp)] >= v[(gid, b, bp)], name=f"link_y_v_low_g{gid}_b{b}_bp{bp[0]}_{bp[1]}")
                m.addConstr(y[(gid, b, bp)] <= BIGM_group * v[(gid, b, bp)],
                            name=f"link_y_v_up_g{gid}_b{b}_bp{bp[0]}_{bp[1]}")

            m.update()


            # 注意：使用最小改动的线性化（二进制变量 z40, z45）
            from collections import defaultdict as _dd

            # group_size 快速查表（window_new_groups 在本作用域内）
            group_size = {g['group_id']: g['size'] for g in window_new_groups}

            # existing sizes（考虑 current_state 中遗留的分配，current_state 在本作用域）
            existing_sizes = _dd(set)
            for rec in current_state:
                try:
                    bp_t = tuple(rec['bp'])
                    blk = rec['block']
                    sz = rec.get('size')
                    if sz:
                        existing_sizes[(blk, bp_t)].add(sz)
                except Exception:
                    # 忽略格式不标准的记录
                    pass

            # 新增二进制变量：z40, z45
            z40 = {}
            z45 = {}
            z20 = {}
            for b, bps in all_bps_by_block.items():
                for bp in bps:
                    z40[(b, bp)] = m.addVar(vtype=GRB.BINARY, name=f"z40_b{b}_bp{bp[0]}_{bp[1]}")
                    z45[(b, bp)] = m.addVar(vtype=GRB.BINARY, name=f"z45_b{b}_bp{bp[0]}_{bp[1]}")
                    z20[(b, bp)] = m.addVar(vtype=GRB.BINARY, name=f"z20_b{b}_bp{bp[0]}_{bp[1]}")

            m.update()


            # 线性化约束：把 y 与 z40/z45 关联起来；并禁止二者同时为1
            for b, bps in all_bps_by_block.items():
                for bp in bps:
                    # 聚合该 bp 上 40ft / 45ft 的 y 变量之和（若没有则为 0）
                    y40_sum = gp.quicksum(
                        y[(gid, b, bp)]
                        for gid in group_size
                        if group_size[gid] == '40ft' and (gid, b, bp) in y
                    )
                    y45_sum = gp.quicksum(
                        y[(gid, b, bp)]
                        for gid in group_size
                        if group_size[gid] == '45ft' and (gid, b, bp) in y
                    )
                    # 聚合 20ft 的 y 之和
                    y20_sum = gp.quicksum(
                        y[(gid, b, bp)]
                        for gid in group_size
                        if group_size[gid] == '20ft' and (gid, b, bp) in y
                    )

                    # 用 bay_cap 做大 M（bay_cap 在外层可用）
                    m.addConstr(y40_sum <= bay_cap * z40[(b, bp)], name=f"link_y40_z40_b{b}_bp{bp[0]}_{bp[1]}")
                    m.addConstr(y45_sum <= bay_cap * z45[(b, bp)], name=f"link_y45_z45_b{b}_bp{bp[0]}_{bp[1]}")
                    m.addConstr(y20_sum <= bay_cap * z20[(b, bp)], name=f"link_y20_z20_b{b}_bp{bp[0]}_{bp[1]}")
                    # 禁止同一 bp 同时出现 40ft 和 45ft
                    # m.addConstr(z40[(b, bp)] + z45[(b, bp)] <= 1, name=f"no_40_45_b{b}_bp{bp[0]}_{bp[1]}")
                    m.addConstr(z20[(b, bp)] + z40[(b, bp)] + z45[(b, bp)] <= 1,
                                name=f"no_mix_sizes_b{b}_bp{bp[0]}_{bp[1]}")
                    # 保持与原 z 的一致性：若 z40/z45 为1，则 z 也应为1
                    if (b, bp) in z:
                        m.addConstr(z[(b, bp)] >= z40[(b, bp)], name=f"z_link_z40_b{b}_bp{bp[0]}_{bp[1]}")
                        m.addConstr(z[(b, bp)] >= z45[(b, bp)], name=f"z_link_z45_b{b}_bp{bp[0]}_{bp[1]}")
                        m.addConstr(z[(b, bp)] >= z20[(b, bp)], name=f"z_link_z20_b{b}_bp{bp[0]}_{bp[1]}")

            # 若已有遗留占用包含 40ft/45ft，则把对应的 z40/z45 强制为 1（以反映已有占用）
            for (b_bp, sizes) in existing_sizes.items():
                b, bp = b_bp
                bp = tuple(bp)
                if '40ft' in sizes:
                    # 若该 (b,bp) 并不存在对应的 z40 变量（理论上不会），忽略
                    if (b, bp) in z40:
                        m.addConstr(z40[(b, bp)] == 1, name=f"force_z40_exist_b{b}_bp{bp[0]}_{bp[1]}")
                if '45ft' in sizes:
                    if (b, bp) in z45:
                        m.addConstr(z45[(b, bp)] == 1, name=f"force_z45_exist_b{b}_bp{bp[0]}_{bp[1]}")
                if '20ft' in sizes:
                    if (b, bp) in z20:
                        m.addConstr(z20[(b, bp)] == 1, name=f"force_z20_exist_b{b}_bp{bp[0]}_{bp[1]}")



            # -------------------- 新增结束 --------------------

            # 连接约束：把 y 与 u / used 关联起来
            for g in window_new_groups:
                gid = g['group_id']
                qty = int(g['quantity'])
                # 所有该组的 y 之和（跨 block, bp）
                sum_y_all = gp.quicksum(
                    y[(gid, b, tuple(bp))] for (b, bp) in group_candidates.get(gid, []) if (gid, b, tuple(bp)) in y
                )
                # used 标志与是否有分配量绑定：若有分配 sum_y_all >=1 => used=1；若 used=1 => sum_y_all <= qty
                m.addConstr(sum_y_all >= used[gid], name=f"link_used_low_{gid}")
                m.addConstr(sum_y_all <= qty * used[gid], name=f"link_used_up_{gid}")

                # 对每个 block，u 与该 block 上的 y 之和关联
                allowed_blocks = sorted({b for (b, bp) in group_candidates.get(gid, [])})
                for b in allowed_blocks:
                    # 该组在 block b 上所有 bp 的 y 和
                    sum_y_b = gp.quicksum(
                        y[(gid, b, tuple(bp))] for (bb, bp) in group_candidates.get(gid, []) if
                        bb == b and (gid, b, tuple(bp)) in y
                    )
                    # 若 u=1 则必须至少放 1 箱；若放箱则 u=1（用两条约束互相绑定）
                    m.addConstr(sum_y_b >= u[(gid, b)], name=f"link_u_low_{gid}_b{b}")
                    m.addConstr(sum_y_b <= qty * u[(gid, b)], name=f"link_u_up_{gid}_b{b}")


            # ====== 插入结束 ======

            # 大 M
            BIGM = bay_cap  # 30

            # 5) 约束：每个 bay_pair 的容量（包含 existing_occupancy）
            group_size_map = {g['group_id']: g['size'] for g in window_new_groups}

            for b, bps in all_bps_by_block.items():
                for bp in bps:
                    # bp 是 tuple，如 (i,i) 或 (i, i+1)
                    bp_len = bp[1] - bp[0] + 1  # 1 或 2（若以后出现更长也通用）
                    exist_q = existing_occupancy.get((b, bp), 0)  # 已经以 TEU 单位存储（见上一步修改）

                    # 把每个分配组 y 的箱数乘以其 TEU（20ft->1, 40/45->2）
                    lhs = gp.quicksum(
                        y[(gid, b, bp)] * (2 if ('40' in group_size_map[gid] or '45' in group_size_map[gid]) else 1)
                        for gid in group_size_map
                        if (gid, b, bp) in y
                    )

                    # 容量按物理贝数缩放：每个物理贝容量为 bay_cap -> 整个 bp 的容量是 bay_cap * bp_len
                    m.addConstr(lhs + exist_q <= bay_cap * bp_len, name=f"cap_b{b}_bp{bp[0]}_{bp[1]}")

            # 6) 约束：若某 bp 上有任何 y > 0，则 z >= 1；反过来 y <= BIGM * z
            for (b, bp) in list(z.keys()):
                bp_len = bp[1] - bp[0] + 1
                # per-bp bigM：最大 TEU 可放入该 bp
                bigM_bp = bay_cap * bp_len
                for g in window_new_groups:
                    gid = g['group_id']
                    if (gid, b, bp) in y:
                        size_factor = 2 if ('40' in g['size'] or '45' in g['size']) else 1
                        # y * size_factor <= bigM_bp * z  等价于 y <= (bigM_bp/size_factor) * z
                        m.addConstr(y[(gid, b, bp)] * size_factor <= bigM_bp * z[(b, bp)],
                                    name=f"link_yz_{gid}_b{b}_bp{bp[0]}_{bp[1]}")
                if existing_occupancy.get((b, bp), 0) > 0:
                    m.addConstr(z[(b, bp)] == 1, name=f"force_z_exist_b{b}_bp{bp[0]}_{bp[1]}")

            # 7) 约束：不同尺寸的分配组的贝位对之间不能有重叠（即同一箱区内，每个实际贝索引 i 最多被使用一次）
            # 我们通过 z 变量来控制：对每个箱区的每个物理 bay index i，相关 bp 的 z 之和 <= 1
            for b in blocks:
                # 考虑箱区 b 中可能出现的 bp（若没有任何候选也可跳过）
                bps = [bp for bp in all_bps_by_block.get(b, set())]
                if not bps:
                    continue
                for i in range(1, block_bays_num + 1):
                    # 找出包含 bay index i 的所有 bp
                    bps_with_i = [bp for bp in bps if i in bp]
                    if not bps_with_i:
                        continue
                    m.addConstr(gp.quicksum(z[(b, bp)] for bp in bps_with_i) <= 1, name=f"no_overlap_b{b}_bay{i}")

            # 8) 约束：单个分配组必须被完全分配（或允许 unmet 松弛）
            for g in window_new_groups:
                gid = g['group_id']
                qty = int(g['quantity'])
                lhs = gp.quicksum(y[(gid, b, bp)] for (b, bp) in group_candidates[gid] if (gid, b, bp) in y)
                m.addConstr(lhs + unmet[gid] == qty, name=f"fulfill_{gid}")

            # 9) 约束：箱区内占用的贝位数不能超过箱区的贝位总数
            # 计算箱区被占用的物理贝数：sum_{bp} z[b,bp] * len(bp) <= block_bays_num
            for b in blocks:
                bps = [bp for bp in all_bps_by_block.get(b, set())]
                if not bps:
                    continue
                m.addConstr(gp.quicksum(z[(b, bp)] * (1 if bp[0] == bp[1] else 2) for bp in bps) <= block_bays_num,
                            name=f"block_bays_limit_b{b}")

            # 10) 额外：若某 group 在某 b,bp 上没有定义 y 变量（因为不在 candidate 列表），就不允许赋值（已由变量集合决定）
            # —— 无需额外约束
            # =======================================归一化目标函数构建（替换原有目标） =====================================
            # 11) 目标函数：距离 + 占用箱区数 + 未分配量惩罚
            # 计算距离代价：每个 y[(g,b,bp)] * distance(ship. berth -> block)
            dist_cost_terms = []

            for (gid, b, bp) in list(y.keys()):
                # 找到 group 对应的 ship
                g_obj = next(g for g in window_new_groups if g['group_id'] == gid)
                ship = g_obj['ship']
                berth = ship_berth.get(ship)
                # 算距离：berth_positions[berth] 到 block_positions[b] 的欧氏距离
                berth_pos = berth_positions[berth]
                block_pos = block_positions[b]
                dist = math.hypot(berth_pos[0] - block_pos[0], berth_pos[1] - block_pos[1])
                # 代价以箱数量 * 距离 为基准
                dist_cost_terms.append(dist * y[(gid, b, bp)])




            block_usage_term = gp.quicksum(u[(gid, b)] for (gid, b) in u.keys())
            # v_sum 表示所有组使用的贝位对总数（即我们想最小化的量）
            bay_usage_term = gp.quicksum(v_val for v_val in v.values())

            gid_list = [g['group_id'] for g in window_new_groups]
            qty = {g['group_id']: int(g['quantity']) for g in window_new_groups}

            # 每个组最多能“占用的箱区个数”的上界：至多把每个组拆到 min(组箱量, 可用箱区数) 个箱区
            allowed_blocks_cnt = {gid: len({b for (b, bp) in group_candidates[gid]}) for gid in gid_list}
            den_block = sum(min(qty[gid], allowed_blocks_cnt[gid]) for gid in gid_list) or 1

            # 每个组最多能“占用的贝位对个数”的上界：至少1箱才能占1个(b,bp)，上界为 min(组箱量, 候选(b,bp)数)
            cand_pairs_cnt = {gid: len(group_candidates[gid]) for gid in gid_list}
            den_bay = sum(min(qty[gid], cand_pairs_cnt[gid]) for gid in gid_list) or 1

            total_qty = sum(qty.values())
            den_dist = max(distance.values()) * sum(qty.values())
            dist_norm = gp.quicksum(dist_cost_terms) / den_dist  # 用常数分母，保持线性目标

            # v_sum 表示所有组使用的贝位对总数（即我们想最小化的量）
            bay_usage_term = gp.quicksum(v_val for v_val in v.values())

            block_usage_norm = block_usage_term / den_block
            bay_usage_norm = bay_usage_term / den_bay
            unmet_norm = gp.quicksum(unmet.values()) / total_qty

            w_dist, w_block, w_bay, w_unmet = 2.0, 1.0, 10.0, 50.0

            m.setObjective(w_dist * dist_norm+ w_block * block_usage_norm+ w_bay * bay_usage_norm+ w_unmet * unmet_norm,
                GRB.MINIMIZE
            )

            # 12) 求解
            m.optimize()

            # 13) 解析结果：将 y 中 >0 的解记入本次分配；并生成下一窗口的 current_state（按 pickup_step 筛选）
            if m.status in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT, GRB.Status.SUBOPTIMAL]:
                # 收集本次分配
                this_window_assignments = []
                for (gid, b, bp), var in y.items():
                    val = int(round(var.X)) if var.X is not None else 0
                    if val > 0:
                        # 找到 group info
                        g_obj = next(g for g in window_new_groups if g['group_id'] == gid)
                        rec = {
                            'group_id': gid,
                            'ship': g_obj['ship'],
                            'period': g_obj['period'],  # <<< 建议加上这一行
                            'size': g_obj['size'],
                            'quantity': g_obj['quantity'],
                            'block': b,
                            'bp': bp,
                            'qty': val,
                            'pickup_step': g_obj.get('pickup_step')
                        }
                        this_window_assignments.append(rec)
                        all_assignments.append(rec)

                # 同时把 current_state (existing occupancy) 中在窗口结束后仍存在的记录带到下一窗口
                next_state = []
                # 先 existing ones
                for rec in current_state:
                    # 如果 pickup_step is None 或者 pickup_step > window_end 则仍然存在
                    if rec['pickup_step'] is None or rec['pickup_step'] > window_end:
                        next_state.append(rec)
                # 再新增的assignments中，只有 pickup_step > window_end 才留到 next_state
                for rec in this_window_assignments:
                    if rec['pickup_step'] is None or rec['pickup_step'] > window_end:
                        next_state.append(rec)

                # 将 next_state 设为 current_state
                current_state = next_state

                # 打印本窗口分配摘要
                if verbose:
                    if len(this_window_assignments) == 0:
                        print("本窗口模型解得但没有新的有效分配（可能都被 unmet 吞掉或被限制）。")
                    else:
                        df_win = pd.DataFrame(this_window_assignments)
                        # 显示关键信息
                        print("本窗口新增分配（部分）：")
                        print(df_win[['group_id', 'ship', 'size', 'quantity', 'block', 'bp', 'qty',
                                      'pickup_step']].head(
                            20).to_string(index=False))
                # 若有 unmet>0，打印警告
                unmet_nonzero = [(gid, int(unmet[gid].X)) for gid in unmet if
                                 unmet[gid].X is not None and unmet[gid].X > 0]
                if unmet_nonzero:
                    print("WARNING: 以下分配组未被完全分配 (unmet > 0)：", unmet_nonzero)

            else:
                # 求解失败
                print(f"窗口 [{window_start},{window_end}] 求解失败，状态码: {m.status}")
                # 为了继续滑动，我们不添加新分配，直接按 current_state 移除在窗口内取走的
                next_state = []
                for rec in current_state:
                    if rec['pickup_step'] is None or rec['pickup_step'] > window_end:
                        next_state.append(rec)
                current_state = next_state
                continue

        # 结束所有窗口，输出总分配表
        if len(all_assignments) == 0:
            print("没有任何分配结果。")
            final_df = pd.DataFrame(
                columns=['group_id', 'ship', 'size', 'quantity', 'block', 'bp', 'qty', 'pickup_step'])
        else:
            final_df = pd.DataFrame(all_assignments)
        print("\n=== 总分配结果（样例前 50 条） ===")
        print(final_df.head(50).to_string(index=False))

        # 保存到对象
        self.all_assignments = all_assignments
        self.remaining_state = current_state
        self.final_df = final_df

        # 返回结果供后续使用
        return all_assignments, current_state, final_df

    # 额外工具方法
    def save_csv(self, path="allocation_results.csv"):
        if self.final_df is not None and not self.final_df.empty:
            self.final_df.to_csv(path, index=False)
        else:
            raise RuntimeError("final_df 为空，请先运行 run() 获取结果。")


solver = YardSolver(
    allocation_groups=allocation_groups,
    blocks=blocks,
    block_positions=block_positions,
    berth_positions=berth_positions,
    vessel_info=vessel_info,
    block_bays_num=block_bays_num,
    bay_cap=bay_cap,
    T=T,
    window_size=2,
    initial_state=None,
    verbose=True
)
all_assignments, remaining_state, final_df = solver.run()

solver.save_csv("allocation_results.csv")

viz = InteractiveYardVisualizer(
    all_assignments=all_assignments,
    allocation_groups=allocation_groups,
    rows=5, cols=4,
    block_bays_num=40,
    bay_cap=40,
    T=T,
    window_size=2,
    figsize=(12,9)
)

viz.show()