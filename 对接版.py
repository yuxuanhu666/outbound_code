
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import random
import math
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Union, Optional
import copy
import pprint
import json
from dataclasses import dataclass
from math import ceil, sqrt
from 对接版可视化 import InteractiveYardVisualizer

import gurobipy as gp
from gurobipy import GRB

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# =================== 数据定义 ====================

# 时间参数

planning_start_time = '2025-12-18 00:00:00'
planning_end_time = '2025-12-23 00:00:00'
planning_start = datetime.strptime(planning_start_time, '%Y-%m-%d %H:%M:%S')
planning_end = datetime.strptime(planning_end_time, '%Y-%m-%d %H:%M:%S')
planning_horizon_days =  (planning_end - planning_start).days
time_step_hours = 4  # 4小时一个时间步
T = planning_horizon_days * 24 // time_step_hours  # 总时间步数
time_steps = list(range(1, T + 1))


# 泊位位置
berth_positions = {
    2173846: {"berthID": "V1", "position": (21661, 78647)},
    2173847: {"berthID": "V2", "position": (35561, 78647)},
    2173848: {"berthID": "V3", "position": (49461, 78647)},
    2173849: {"berthID": "V4", "position": (63361, 78647)},  # 对应的泊位ID是4
}

# 载入JSON文件
with open('../data/yardblockdata.json', 'r', encoding='utf-8') as f:
    yardblock_data = json.load(f)

# 提取箱区信息的函数
def extract_yardblock_info(yardblock_data):
    blocks_info = {}

    for block in yardblock_data['data']['blockList']:
        block_id = block['name']
        bay_count = block['bayCount']  # 贝位数量
        col_count = block['colCount']  # 列数
        tier_count = block['tierCount']  # 每列的层数
        center_x = block['centerX']  # 中心位置X
        center_y = block['centerY']  # 中心位置Y

        block_info = {
            'bay_count': bay_count,
            'col_count': col_count,
            'tier_count': tier_count,
            'bay_cap': col_count * tier_count,
            'block_cap': bay_count * col_count * tier_count,
            'block_position': {'centerX': center_x, 'centerY': center_y}
        }

        # 用block_id作为键，将block_info作为值
        blocks_info[block_id] = block_info

    return blocks_info
# 提取箱区信息
blocks_info = extract_yardblock_info(yardblock_data)
# ✅打印：每个箱区的贝位数/列数/层数
for block_id in sorted(blocks_info.keys()):
    info = blocks_info[block_id]
    print(f"箱区 {block_id}: 贝位数={info['bay_count']}  列数={info['col_count']}  层数={info['tier_count']}")
# 从 blocks_info 中提取 blocks 和 block_positions
blocks = list(blocks_info.keys())
bay_cap_by_block = {b: info['bay_cap'] for b, info in blocks_info.items()}          # col*tier
block_bays_num_by_block = {b: info['bay_count'] for b, info in blocks_info.items()} # bay_count
#
block_positions = {
    block_id: (
        info['block_position']['centerX'],
        info['block_position']['centerY']
    )
    for block_id, info in blocks_info.items()
}

distance = {}

# 遍历泊位和箱区计算欧氏距离
for berth_dbkey, berth_data in berth_positions.items():
    vx, vy = berth_data['position']
    berth_name = berth_data['berthID']
    for block_id, (bx, by) in block_positions.items():  # 箱区中心坐标
        d = math.hypot(bx - vx, by - vy)
        distance[(berth_name, block_id)] = d

# 打印距离字典
for key, value in distance.items():
    print(f"Distance between berth {key[0]} and block {key[1]}: {value}")
# 箱区容量配置（每个箱区的最大容量 = 贝位数 × 每个贝位容量）
# 注意：这里的容量是基于20尺箱的标准容量
# 每个20尺箱占1个容量单位，每个40尺/45尺箱占2个容量单位

BLOCK_CAPACITY = {bid: info['block_cap'] for bid, info in blocks_info.items()}


# 1. 读入原始 json
with open("../data/SHIPVISITID_TEST1220.JSON.json", "r", encoding="utf-8") as f:
    raw2 = json.load(f)

# 2. 提取船舶信息
ship_id = "TEST1220"
vessel_info = {}

# 遍历 shipVisitList 查找 id 为 TEST1220 的船舶
for ship in raw2["data"]["shipVisitList"]:
    if ship["id"] == ship_id:
        # 提取所需的信息
        berth_key = ship["berthPlan"]["berthKey"] if ship["berthPlan"] else None
        qc_num = ship["qcNum"]
        eta_time = ship["etaTime"]
        etd_time = ship["etdTime"]
        pbw_time = ship["pbwTime"]
        pew_time = ship["pewTime"]

        # 构造 vessel_info 字典
        vessel_info[ship["id"]] = {
            "berthKey": berth_key,
            "qcNum": qc_num,
            "etaTime": eta_time,
            "etdTime": etd_time,
            "pbwTime": pbw_time,
            "pewTime": pew_time
        }

# 输出结果
pprint.pprint(vessel_info)
# 3. 如果还需要一个 id 列表
vessel_list = list(vessel_info.keys())


def calculate_vessel_schedule(vessel_info, planning_start_time, planning_end_time, time_step_hours):
    """
    根据船舶的etaTime、etdTime、pbwTime、pewTime计算船舶的调度时间。

    参数：
    - vessel_info (dict): 每艘船舶的信息。
    - planning_start_time (str): 规划期的起始时间，例如 '2025-12-18 00:00:00'。
    - planning_end_time (str): 规划期的结束时间，例如 '2025-12-23 00:00:00'。
    - time_step_hours (int): 每个时间步的小时数，例如4小时。

    返回：
    - vessel_schedules (dict): 每艘船舶的调度信息。
    """
    # 计算总的时间步数
    total_time_steps = planning_horizon_days * 24 // time_step_hours

    vessel_schedules = {}

    for ship_name, info in vessel_info.items():
        # 提取船舶的时间并转换为datetime对象
        eta_time = datetime.strptime(info['etaTime'], '%Y-%m-%d %H:%M:%S')
        etd_time = datetime.strptime(info['etdTime'], '%Y-%m-%d %H:%M:%S')
        pbw_time = datetime.strptime(info['pbwTime'], '%Y-%m-%d %H:%M:%S')
        pew_time = datetime.strptime(info['pewTime'], '%Y-%m-%d %H:%M:%S')

        # 将时间转换为时间步
        def time_to_step(time):
            return int((time - planning_start).total_seconds() // (time_step_hours * 3600)) + 1

        eta_step = time_to_step(eta_time)
        etd_step = time_to_step(etd_time)
        pbw_step = time_to_step(pbw_time)
        pew_step = time_to_step(pew_time)

        # 计算收箱期（从装船前3天开始）
        receiving_start_step = time_to_step(eta_time - timedelta(days=3))
        receiving_end_step = time_to_step(eta_time)

        # 计算装船期（从pbw_time到pew_time）
        loading_start_step = time_to_step(pbw_time)
        loading_end_step = time_to_step(pew_time)

        # 将计算的时间步存储到船舶的调度信息中
        vessel_schedules[ship_name] = {
            'eta_step': eta_step,
            'etd_step': etd_step,
            'pbw_step': pbw_step,
            'pew_step': pew_step,
            'receiving_period': {
                'start': receiving_start_step,
                'end': receiving_end_step,
            },
            'loading_period': {
                'start': loading_start_step,
                'end': loading_end_step,
            }
        }

    return vessel_schedules

# 计算船舶调度
vessel_schedules = calculate_vessel_schedule(vessel_info, planning_start_time, planning_end_time, time_step_hours)

# 全局：按 shipVisitId（即 ship）预先生成 收箱/装船 的 time steps 集合
ship_receiving_steps = {}
ship_loading_steps = {}

def _clamp_step(s: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, s))

# 用 vessel_schedules 中的 receiving_period / loading_period（由 pbw/pew 推出）
for ship, info in vessel_schedules.items():
    r0 = info.get('receiving_period', {}).get('start')
    r1 = info.get('receiving_period', {}).get('end')
    l0 = info.get('loading_period', {}).get('start')
    l1 = info.get('loading_period', {}).get('end')

    if isinstance(r0, int) and isinstance(r1, int):
        r0 = _clamp_step(r0, 1, T)
        r1 = _clamp_step(r1, 1, T)
        ship_receiving_steps[ship] = set(range(min(r0, r1), max(r0, r1) + 1))
    else:
        ship_receiving_steps[ship] = set()

    if isinstance(l0, int) and isinstance(l1, int):
        l0 = _clamp_step(l0, 1, T)
        l1 = _clamp_step(l1, 1, T)
        ship_loading_steps[ship] = set(range(min(l0, l1), max(l0, l1) + 1))
    else:
        ship_loading_steps[ship] = set()

def is_conflict(ship1: str, ship2: str) -> bool:
    """若 ship1 的收箱期 与 ship2 的装船期（或反过来）有交集，则认为冲突。"""
    rec1 = ship_receiving_steps.get(ship1, set())
    load1 = ship_loading_steps.get(ship1, set())
    rec2 = ship_receiving_steps.get(ship2, set())
    load2 = ship_loading_steps.get(ship2, set())
    return bool((rec1 & load2) or (rec2 & load1))

def print_vessel_schedules(schedules):
    print("\n================= 船舶调度明细 =================\n")
    for ship, info in schedules.items():
        print(f"船舶: {ship}")

        # 获取泊位信息
        berth_key = vessel_info.get(ship, {}).get('berthKey', '未知')  # 如果没有找到berthKey，默认为'未知'
        print(f"  泊位: {berth_key}")

        # 输出预计到港时间（eta）、预计离港时间（etd）、预计装船时间（pbw）、预计装船结束时间（pew）
        print(f"  预计到港时间 (eta): {info['eta_step']}")
        print(f"  预计离港时间 (etd): {info['etd_step']}")
        print(f"  预计装船时间 (pbw): {info['pbw_step']}")
        print(f"  预计装船结束时间 (pew): {info['pew_step']}")

        # 输出收箱期的时间步集合
        receiving_period = info.get('receiving_period', {})
        if receiving_period:
            receiving_steps = receiving_period.get('time_steps', [])
            print(f"  收箱期 (receiving): {receiving_period.get('start', '未定义')} → {receiving_period.get('end', '未定义')}")
            print(f"    步: {receiving_steps}")
        else:
            print("  收箱期 (receiving): 无（完全不在规划期内）")

        # 输出装船期的时间步集合
        loading_period = info.get('loading_period', {})
        if loading_period:
            loading_steps = loading_period.get('time_steps', [])
            print(f"  装船期 (loading):   {loading_period.get('start', '未定义')} → {loading_period.get('end', '未定义')}")
            print(f"    步: {loading_steps}")
        else:
            print("  装船期 (loading):   无（完全不在规划期内）")


        print(f"  装船持续步数: {info.get('loading_time_steps', '未计算')}")
        print("  --------------------------------------------------")

        print("")  # 空行分隔船舶

print_vessel_schedules(vessel_schedules)


# 1. 读取 JSON 文件
with open("../data/unit.json", "r", encoding="utf-8") as f:
    raw = json.load(f)


units = raw["data"]["contrUnitList"]  # 这里就是所有箱子列表

# 3. 定义一个函数来根据重量确定重量等级
def get_weight_category(gross_weight):
    if gross_weight is None:  # 如果重量为 None，设置默认值
        return "未知"  # 可以设置为 "未知" 或者其他适当的值
    if gross_weight < 8000:
        return "轻"
    elif gross_weight < 15000:
        return "中"
    else:
        return "重"
def iso_to_size(iso: str) -> str:
    # 你后面有逻辑在用 '20ft/40ft/45ft' 做判断（例如 z40/z45 那段），所以这里转成 ft 更稳
    if not isinstance(iso, str):
        return "20ft"
    if "45" in iso:
        return "45ft"
    if "40" in iso:
        return "40ft"
    return "20ft"
# 2. 用 (isoGroupKey, visitId, visitType, moveTo) 做分组 key
in_groups = defaultdict(list)
out_groups = defaultdict(list)

for u in units :
    ivid = u.get("ibVisitId")
    in_groups[ivid].append(u)

for j in units:
    ovid = j.get("obVisitId")
    out_groups[ovid].append(j)

# 1) 查看有哪些 ibVisitId
# print(list(in_groups.keys()))
# print(list(out_groups.keys()))
# 2) 取某个 ibVisitId 对应的所有箱子（例如 TEST1223）

same1 = in_groups.get("TEST1220", [])

same2 = out_groups["TEST1220"]
print(len(same1), [x["unitId"] for x in same1])
print(len(same2), [x["unitId"] for x in same2])
# 3) 如果只想要 “ibVisitId -> unitId列表”
# result = {k: [x["unitId"] for x in v] for k, v in in_groups.items()}
# 4. 对每个 ibVisitId 下的箱子按要求进行分组
# 5. 创建一个新的列表存储分配组
# 1. 用 defaultdict 来分组箱子，键是属性组合，值是箱子列表
allocation_groups = defaultdict(list)

for unit in same1:
    # 提取箱子属性
    contr_iso = unit.get("contrIso")  # 箱型
    ib_visit_id = unit.get("ibVisitId")  # 船舶艘次号
    pod = unit.get("pod")  # 目的港
    gross_weight = unit.get("grossWeight")  # 重量
    contr_owner_name = unit.get("contrOwnerName")  # 持箱公司

    # 获取重量等级
    weight_category = get_weight_category(gross_weight)

    # 使用这些属性组合成一个唯一的分配组键
    group_key = (contr_iso, ib_visit_id, pod, weight_category, contr_owner_name)

    # 将箱子放入对应的分配组
    allocation_groups[group_key].append(unit)

# 2. 创建分配组，并为每个分配组命名，统计箱量
grouped_result = []
for group_key, units in allocation_groups.items():
    contr_iso, ib_visit_id, pod, weight_category, contr_owner_name = group_key

    # 生成分配组ID，这里用属性的简化拼接生成一个唯一ID
    group_id = f"{contr_iso}_{ib_visit_id}_{pod}_{weight_category}_{contr_owner_name}"

    # 统计箱量
    box_count = len(units)

    # 收集所有箱子的箱号 (unitId)
    unit_ids = [unit["unitId"] for unit in units]
    # 获取船舶的到港时间步和装船时间步（根据 ibVisitId 查找）
    vessel_data = vessel_schedules.get(ib_visit_id, {})

    arrival_step = vessel_data.get('receiving_period', {}).get('start')
    pickup_step = vessel_data.get('pbw_step')
    # 将分配组信息添加到结果列表
    allocation_group = {
        "group_id": group_id,
        "contrISO": contr_iso,
        "ibVisitId": ib_visit_id,
        "ship": ib_visit_id,  # ✅新增：给 YardSolver 用
        "size": iso_to_size(contr_iso),  # ✅新增：给 YardSolver 用（避免后续 g['size'] 再报错）
        "pod": pod,
        "weightCategory": weight_category,
        "contrOwnerName": contr_owner_name,
        "quantity": box_count, # 统计箱量
        "unit_ids": unit_ids , # 保存所有箱子的箱号
        "arrival_step": arrival_step,  # 到港时间步
        "pickup_step": pickup_step  # 装船时间步
    }

    grouped_result.append(allocation_group)



pprint.pprint(grouped_result)
# 用法示例


class YardSolver:
    """
    YardSolver: 把原来的 solve_with_sliding_windows 封装成类
    使用方法：
        solver = YardSolver(allocation_groups, blocks, block_positions,
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
        # ✅ 统一规范化成 dict：每个箱区一个值
        if isinstance(block_bays_num, dict):
            self.block_bays_num_by_block = {b: int(block_bays_num[b]) for b in blocks}
        else:
            self.block_bays_num_by_block = {b: int(block_bays_num) for b in blocks}

        if isinstance(bay_cap, dict):
            self.bay_cap_by_block = {b: int(bay_cap[b]) for b in blocks}
        else:
            self.bay_cap_by_block = {b: int(bay_cap) for b in blocks}

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
    def generate_bay_pairs_for_size(size: str, bay_count: int):
        """返回给定尺寸允许的贝位对列表：(i,i) 或 (i,i+1)。
        bay_count 来自 blocks_info[b]['bay_count']（每个箱区可能不同）。
        """
        if not bay_count or bay_count < 1:
            return []

        # 20ft：单贝位
        if size == '20GP':
            return [(i, i) for i in range(1, bay_count + 1)]

        # 40/45ft：占两个连续贝位（保持你原来的“从 1 开始、步长 2”的规则）
        pairs = [(i, i + 1) for i in range(1, bay_count, 2)]  # 自动保证 i+1 <= bay_count

        # 45ft：只能放最左或最右（不再写死 39/40，而是取 pairs 的两端）
        if size == '45GP':
            if not pairs:
                return []
            if len(pairs) == 1:
                return [pairs[0]]
            return [pairs[0], pairs[-1]]

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
        block_bays_num_by_block = self.block_bays_num_by_block
        bay_cap_by_block = self.bay_cap_by_block
        T = self.T
        window_size = self.window_size
        initial_state = copy.deepcopy(self.initial_state)
        verbose = self.verbose

        if initial_state is None:
            initial_state = []

        # 建立便于查询的字典：ship->berth_pos
        ship_berth_dbkey = {s: vessel_info[s]['berthKey'] for s in vessel_info}

        all_assignments = []  # 输出汇总
        current_state = copy.deepcopy(initial_state)
        for rec in current_state:
            rec2 = rec.copy()
            rec2.setdefault("quantity", rec2.get("qty"))  # 和新分配字段对齐（你的新分配里有 quantity）
            rec2["is_initial"] = True
            all_assignments.append(rec2)


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
            block_ship_used = defaultdict(set)
            for rec in current_state:
                b = rec['block']
                block_ship_used[b].add(rec['ship'])

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
            m.Params.TimeLimit = 800
            m.Params.MIPGap = 0.05
            # 性能优化
            m.Params.Threads = 0  # 使用所有核心
            m.Params.MIPFocus = 1  # 优先找可行解
            m.Params.Heuristics = 0.3  # 10% 时间用于启发式

            # # 改进策略
            # m.Params.ImproveStartTime = 100  # 1分钟后开始改进

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
                size = g['contrISO']
                ship = g['ibVisitId']
                # period = g['period']
                ship_new = ship

                cand = []

                for b in blocks:
                    bay_count = self.block_bays_num_by_block[b]
                    allowed_bps = self.generate_bay_pairs_for_size(size, bay_count)
                    # 先看这个 block 里历史上已经有哪些 (ship,period)
                    bad_block = False
                    for ship_old in block_ship_used.get(b, set()):
                        if is_conflict(ship_new, ship_old):
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
            # used = {}  # used[gid] = 0/1 表示 group gid 是否有任何被分配的箱

            for g in window_new_groups:
                gid = g['group_id']
                qty = int(g['quantity'])
                # used[gid] = m.addVar(vtype=GRB.BINARY, name=f"used_{gid}")
                # allowed blocks 由 group_candidates 给出（去重）
                allowed_blocks = sorted({b for (b, bp) in group_candidates.get(gid, [])})
                for b in allowed_blocks:
                    u[(gid, b)] = m.addVar(vtype=GRB.BINARY, name=f"u_{gid}_b{b}")


            # 先建一个字典：gid -> ship（shipVisitId）
            gid_to_ship = {}
            for g in window_new_groups:
                gid_to_ship[g['group_id']] = g['ship']

            # 对每个箱区 b，检查本窗口中潜在会用到这个箱区的 group 组合
            for b in blocks:
                gids_in_b = sorted({
                    gid for (gid2, bb, bp) in y.keys()
                    for gid in [gid2] if bb == b
                })

                for i in range(len(gids_in_b)):
                    gid1 = gids_in_b[i]
                    ship1 = gid_to_ship[gid1]
                    for j in range(i + 1, len(gids_in_b)):
                        gid2 = gids_in_b[j]
                        ship2 = gid_to_ship[gid2]

                        if is_conflict(ship1, ship2):
                            if (gid1, b) in u and (gid2, b) in u:
                                m.addConstr(
                                    u[(gid1, b)] + u[(gid2, b)] <= 1,
                                    name=f"no_mix_RL_g{gid1}_g{gid2}_b{b}"
                                )

            # 1) 先统计每个 (ship, period) 在候选中可能用到哪些箱区
            sp_to_blocks = defaultdict(set)  # key: (ship, period)  value: {b1, b2, ...}
            for (gid, b) in u.keys():
                sp = gid_to_ship.get(gid)
                if sp is not None:
                    sp_to_blocks[sp].add(b)


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
                    for i in range(1, block_bays_num_by_block[b] + 1):
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
                    for i in range(1, block_bays_num_by_block[b] + 1):
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
                    m.addConstr(gp.quicksum(x[(gid, b, i)] for i in range(1, block_bays_num_by_block[b] + 1)) >= u[(gid, b)],
                                name=f"link_u_x_low_g{gid}_b{b}")
                    m.addConstr(gp.quicksum(x[(gid, b, i)] for i in range(1, block_bays_num_by_block[b] + 1)) <= block_bays_num_by_block[b] * u[
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
                    for i in range(1, block_bays_num_by_block[b]):
                        t[(gid, b, i)] = m.addVar(vtype=GRB.BINARY, name=f"t_g{gid}_b{b}_i{i}")
                        # t >= x_i - x_{i+1}
                        m.addConstr(t[(gid, b, i)] >= x[(gid, b, i)] - x[(gid, b, i + 1)],
                                    name=f"t_ge_diff1_g{gid}_b{b}_i{i}")
                        # t >= x_{i+1} - x_i
                        m.addConstr(t[(gid, b, i)] >= x[(gid, b, i + 1)] - x[(gid, b, i)],
                                    name=f"t_ge_diff2_g{gid}_b{b}_i{i}")
                    # 如果组在该 box 没被使用（u==0），则转变数应为0；若被使用，则转变数 <= 2（即 0/1 段或单一连续段）
                    m.addConstr(gp.quicksum(t[(gid, b, i)] for i in range(1, block_bays_num_by_block[b])) <= 2 * u[(gid, b)],
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

                    # 用 bay_cap_by_block[b] 做大 M（bay_cap_by_block[b] 在外层可用）
                    m.addConstr(y40_sum <= bay_cap_by_block[b] * z40[(b, bp)], name=f"link_y40_z40_b{b}_bp{bp[0]}_{bp[1]}")
                    m.addConstr(y45_sum <= bay_cap_by_block[b] * z45[(b, bp)], name=f"link_y45_z45_b{b}_bp{bp[0]}_{bp[1]}")
                    m.addConstr(y20_sum <= bay_cap_by_block[b] * z20[(b, bp)], name=f"link_y20_z20_b{b}_bp{bp[0]}_{bp[1]}")
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
                # m.addConstr(sum_y_all >= used[gid], name=f"link_used_low_{gid}")
                # m.addConstr(sum_y_all <= qty * used[gid], name=f"link_used_up_{gid}")

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
            # BIGM = bay_cap  # 30

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

                    # 容量按物理贝数缩放：每个物理贝容量为 bay_cap_by_block[b] -> 整个 bp 的容量是 bay_cap_by_block[b] * bp_len
                    m.addConstr(lhs + exist_q <= bay_cap_by_block[b] * bp_len, name=f"cap_b{b}_bp{bp[0]}_{bp[1]}")

            # 6) 约束：若某 bp 上有任何 y > 0，则 z >= 1；反过来 y <= BIGM * z
            for (b, bp) in list(z.keys()):
                bp_len = bp[1] - bp[0] + 1
                # per-bp bigM：最大 TEU 可放入该 bp
                bigM_bp = bay_cap_by_block[b] * bp_len
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
                for i in range(1, block_bays_num_by_block[b] + 1):
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
            # 计算箱区被占用的物理贝数：sum_{bp} z[b,bp] * len(bp) <= block_bays_num_by_block[b]
            for b in blocks:
                bps = [bp for bp in all_bps_by_block.get(b, set())]
                if not bps:
                    continue
                m.addConstr(gp.quicksum(z[(b, bp)] * (1 if bp[0] == bp[1] else 2) for bp in bps) <= block_bays_num_by_block[b],
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
                # 先拿到这个船对应的泊位 dbkey
                berth_dbkey = ship_berth_dbkey.get(ship)
                if berth_dbkey is None:
                    raise ValueError(f"船舶 {ship} 没有对应的 berth_dbkey")

                # 再用 dbkey 到 berth_positions 里取坐标
                berth_info = berth_positions[berth_dbkey]
                berth_pos = berth_info["position"]  # (x, y)
                # 箱区中心坐标
                block_pos = block_positions[b]  # (x, y)
                # 欧氏距离
                dist = math.hypot(berth_pos[0] - block_pos[0],
                                  berth_pos[1] - block_pos[1])

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

            w_dist, w_block, w_bay, w_unmet = 2.0, 1.0, 10.0, 10.0

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

initial_state = [
{'group_id': '-1', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (7, 7), 'qty': 62, 'pickup_step': None},
{'group_id': '-2', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (20, 20), 'qty': 42, 'pickup_step': None},
{'group_id': '-3', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (11, 11), 'qty': 11, 'pickup_step': None},
{'group_id': '-4', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (1, 1), 'qty': 62, 'pickup_step': None},
{'group_id': '-5', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (13, 13), 'qty': 17, 'pickup_step': None},
{'group_id': '-6', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (9, 9), 'qty': 12, 'pickup_step': None},
{'group_id': '-7', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (18, 18), 'qty': 48, 'pickup_step': None},
{'group_id': '-8', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (6, 6), 'qty': 42, 'pickup_step': None},
{'group_id': '-9', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (28, 28), 'qty': 53, 'pickup_step': None},
{'group_id': '-10', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (31, 31), 'qty': 51, 'pickup_step': None},
{'group_id': '-11', 'ship': 'INIT', 'size': '40ft', 'block': '02', 'bp': (31, 32), 'qty': 39, 'pickup_step': None},
{'group_id': '-12', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (14, 14), 'qty': 20, 'pickup_step': None},
{'group_id': '-13', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (15, 15), 'qty': 19, 'pickup_step': None},
{'group_id': '-14', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (7, 7), 'qty': 11, 'pickup_step': None},
{'group_id': '-15', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (10, 10), 'qty': 47, 'pickup_step': None},
{'group_id': '-16', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (17, 17), 'qty': 23, 'pickup_step': None},
{'group_id': '-17', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (16, 16), 'qty': 33, 'pickup_step': None},
{'group_id': '-18', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (9, 9), 'qty': 61, 'pickup_step': None},
{'group_id': '-19', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (12, 12), 'qty': 12, 'pickup_step': None},
{'group_id': '-20', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (19, 19), 'qty': 12, 'pickup_step': None},
{'group_id': '-21', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (29, 29), 'qty': 39, 'pickup_step': None},
{'group_id': '-22', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (9, 9), 'qty': 62, 'pickup_step': None},
{'group_id': '-23', 'ship': 'INIT', 'size': '45ft', 'block': '04', 'bp': (35, 36), 'qty': 34, 'pickup_step': None},
{'group_id': '-24', 'ship': 'INIT', 'size': '40ft', 'block': '04', 'bp': (29, 30), 'qty': 45, 'pickup_step': None},
{'group_id': '-25', 'ship': 'INIT', 'size': '40ft', 'block': '04', 'bp': (11, 12), 'qty': 61, 'pickup_step': None},
{'group_id': '-26', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (22, 22), 'qty': 66, 'pickup_step': None},
{'group_id': '-27', 'ship': 'INIT', 'size': '40ft', 'block': '04', 'bp': (33, 34), 'qty': 47, 'pickup_step': None},
{'group_id': '-28', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (19, 20), 'qty': 59, 'pickup_step': None},
{'group_id': '-29', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (18, 18), 'qty': 28, 'pickup_step': None},
{'group_id': '-30', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (8, 8), 'qty': 6, 'pickup_step': None},
{'group_id': '-31', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (16, 16), 'qty': 11, 'pickup_step': None},
{'group_id': '-32', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (5, 5), 'qty': 18, 'pickup_step': None},
{'group_id': '-33', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (19, 19), 'qty': 63, 'pickup_step': None},
{'group_id': '-34', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (18, 18), 'qty': 55, 'pickup_step': None},
{'group_id': '-35', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (28, 28), 'qty': 44, 'pickup_step': None},
{'group_id': '-36', 'ship': 'INIT', 'size': '40ft', 'block': '04', 'bp': (5, 6), 'qty': 48, 'pickup_step': None},
{'group_id': '-37', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (31, 32), 'qty': 26, 'pickup_step': None},
{'group_id': '-38', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (10, 10), 'qty': 65, 'pickup_step': None},
{'group_id': '-39', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (10, 10), 'qty': 54, 'pickup_step': None},
{'group_id': '-40', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (32, 32), 'qty': 22, 'pickup_step': None},
{'group_id': '-41', 'ship': 'INIT', 'size': '40ft', 'block': '04', 'bp': (13, 14), 'qty': 47, 'pickup_step': None},
{'group_id': '-42', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (21, 21), 'qty': 66, 'pickup_step': None},
{'group_id': '-43', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (24, 24), 'qty': 21, 'pickup_step': None},
{'group_id': '-44', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (25, 26), 'qty': 33, 'pickup_step': None},
{'group_id': '-45', 'ship': 'INIT', 'size': '40ft', 'block': '03', 'bp': (4, 5), 'qty': 66, 'pickup_step': None},
{'group_id': '-46', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (33, 33), 'qty': 4, 'pickup_step': None},
{'group_id': '-47', 'ship': 'INIT', 'size': '40ft', 'block': '02', 'bp': (29, 30), 'qty': 55, 'pickup_step': None},
{'group_id': '-48', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (26, 26), 'qty': 42, 'pickup_step': None},
{'group_id': '-49', 'ship': 'INIT', 'size': '40ft', 'block': '01', 'bp': (5, 6), 'qty': 40, 'pickup_step': None},
{'group_id': '-50', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (17, 18), 'qty': 29, 'pickup_step': None},
{'group_id': '-51', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (10, 10), 'qty': 45, 'pickup_step': None},
{'group_id': '-52', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (20, 20), 'qty': 62, 'pickup_step': None},
{'group_id': '-53', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (20, 20), 'qty': 61, 'pickup_step': None},
{'group_id': '-54', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (25, 25), 'qty': 51, 'pickup_step': None},
{'group_id': '-55', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (30, 30), 'qty': 31, 'pickup_step': None},
{'group_id': '-56', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (22, 22), 'qty': 58, 'pickup_step': None},
{'group_id': '-57', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (19, 19), 'qty': 65, 'pickup_step': None},
{'group_id': '-58', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (21, 21), 'qty': 66, 'pickup_step': None},
{'group_id': '-59', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (34, 34), 'qty': 41, 'pickup_step': None},
{'group_id': '-60', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (32, 32), 'qty': 17, 'pickup_step': None},
{'group_id': '-61', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (33, 33), 'qty': 18, 'pickup_step': None},
{'group_id': '-62', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (14, 14), 'qty': 59, 'pickup_step': None},
{'group_id': '-63', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (27, 27), 'qty': 60, 'pickup_step': None},
{'group_id': '-64', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (21, 21), 'qty': 32, 'pickup_step': None},
{'group_id': '-65', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (29, 29), 'qty': 12, 'pickup_step': None},
{'group_id': '-66', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (27, 27), 'qty': 63, 'pickup_step': None},
{'group_id': '-67', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (6, 6), 'qty': 9, 'pickup_step': None},
{'group_id': '-68', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (25, 25), 'qty': 13, 'pickup_step': None},
{'group_id': '-69', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (11, 11), 'qty': 58, 'pickup_step': None},
{'group_id': '-70', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (14, 14), 'qty': 61, 'pickup_step': None},
{'group_id': '-71', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (15, 15), 'qty': 29, 'pickup_step': None},
{'group_id': '-72', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (28, 28), 'qty': 46, 'pickup_step': None},
{'group_id': '-73', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (7, 7), 'qty': 53, 'pickup_step': None},
{'group_id': '-74', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (13, 13), 'qty': 61, 'pickup_step': None},
{'group_id': '-75', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (14, 14), 'qty': 63, 'pickup_step': None},
{'group_id': '-76', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (11, 11), 'qty': 57, 'pickup_step': None},
{'group_id': '-77', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (23, 23), 'qty': 56, 'pickup_step': None},
{'group_id': '-78', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (12, 12), 'qty': 39, 'pickup_step': None},
{'group_id': '-79', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (24, 24), 'qty': 59, 'pickup_step': None},
{'group_id': '-80', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (17, 17), 'qty': 59, 'pickup_step': None},
{'group_id': '-81', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (3, 3), 'qty': 66, 'pickup_step': None},
{'group_id': '-82', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (17, 17), 'qty': 16, 'pickup_step': None},
{'group_id': '-83', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (16, 16), 'qty': 8, 'pickup_step': None},
{'group_id': '-84', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (15, 15), 'qty': 58, 'pickup_step': None},
{'group_id': '-85', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (13, 13), 'qty': 66, 'pickup_step': None},
{'group_id': '-86', 'ship': 'INIT', 'size': '40ft', 'block': '02', 'bp': (3, 4), 'qty': 39, 'pickup_step': None},
{'group_id': '-87', 'ship': 'INIT', 'size': '40ft', 'block': '02', 'bp': (27, 28), 'qty': 59, 'pickup_step': None},
{'group_id': '-88', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (24, 24), 'qty': 12, 'pickup_step': None},
{'group_id': '-89', 'ship': 'INIT', 'size': '40ft', 'block': '01', 'bp': (9, 10), 'qty': 66, 'pickup_step': None},
{'group_id': '-90', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (18, 18), 'qty': 58, 'pickup_step': None},
{'group_id': '-91', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (8, 8), 'qty': 55, 'pickup_step': None},
{'group_id': '-92', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (17, 17), 'qty': 8, 'pickup_step': None},
{'group_id': '-93', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (26, 26), 'qty': 46, 'pickup_step': None},
{'group_id': '-94', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (15, 16), 'qty': 34, 'pickup_step': None},
{'group_id': '-95', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (13, 13), 'qty': 66, 'pickup_step': None},
{'group_id': '-96', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (21, 21), 'qty': 63, 'pickup_step': None},
{'group_id': '-97', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (30, 30), 'qty': 3, 'pickup_step': None},
{'group_id': '-98', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (35, 35), 'qty': 45, 'pickup_step': None},
{'group_id': '-99', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (4, 4), 'qty': 63, 'pickup_step': None},
{'group_id': '-100', 'ship': 'INIT', 'size': '45ft', 'block': '01', 'bp': (1, 2), 'qty': 29, 'pickup_step': None},
{'group_id': '-101', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (12, 12), 'qty': 53, 'pickup_step': None},
{'group_id': '-102', 'ship': 'INIT', 'size': '45ft', 'block': '04', 'bp': (1, 2), 'qty': 29, 'pickup_step': None},
{'group_id': '-103', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (19, 19), 'qty': 4, 'pickup_step': None},
{'group_id': '-104', 'ship': 'INIT', 'size': '45ft', 'block': '01', 'bp': (35, 36), 'qty': 66, 'pickup_step': None},
{'group_id': '-105', 'ship': 'INIT', 'size': '20ft', 'block': '05', 'bp': (3, 3), 'qty': 1, 'pickup_step': None},
{'group_id': '-106', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (25, 25), 'qty': 32, 'pickup_step': None},
{'group_id': '-108', 'ship': 'INIT', 'size': '40ft', 'block': '02', 'bp': (7, 8), 'qty': 56, 'pickup_step': None},
{'group_id': '-109', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (8, 8), 'qty': 65, 'pickup_step': None},
{'group_id': '-110', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (36, 36), 'qty': 53, 'pickup_step': None},
{'group_id': '-111', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (22, 22), 'qty': 58, 'pickup_step': None},
{'group_id': '-112', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (1, 2), 'qty': 17, 'pickup_step': None},
{'group_id': '-113', 'ship': 'INIT', 'size': '40ft', 'block': '01', 'bp': (7, 8), 'qty': 65, 'pickup_step': None},
{'group_id': '-114', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (11, 11), 'qty': 63, 'pickup_step': None},
{'group_id': '-115', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (9, 9), 'qty': 36, 'pickup_step': None},
{'group_id': '-116', 'ship': 'INIT', 'size': '40ft', 'block': '02', 'bp': (23, 24), 'qty': 66, 'pickup_step': None},
{'group_id': '-117', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (4, 4), 'qty': 49, 'pickup_step': None},
{'group_id': '-118', 'ship': 'INIT', 'size': '20ft', 'block': '04', 'bp': (3, 3), 'qty': 38, 'pickup_step': None},
{'group_id': '-119', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (31, 31), 'qty': 4, 'pickup_step': None},
{'group_id': '-120', 'ship': 'INIT', 'size': '40ft', 'block': '01', 'bp': (26, 27), 'qty': 66, 'pickup_step': None},
{'group_id': '-121', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (15, 15), 'qty': 55, 'pickup_step': None},
{'group_id': '-122', 'ship': 'INIT', 'size': '20ft', 'block': '01', 'bp': (12, 12), 'qty': 66, 'pickup_step': None},
{'group_id': '-123', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (2, 2), 'qty': 66, 'pickup_step': None},
{'group_id': '-124', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (3, 3), 'qty': 66, 'pickup_step': None},
{'group_id': '-125', 'ship': 'INIT', 'size': '20ft', 'block': '03', 'bp': (34, 34), 'qty': 2, 'pickup_step': None},
{'group_id': '-126', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (22, 22), 'qty': 3, 'pickup_step': None},
{'group_id': '-127', 'ship': 'INIT', 'size': '20ft', 'block': '02', 'bp': (5, 5), 'qty': 1, 'pickup_step': None},
{'group_id': '-128', 'ship': 'INIT', 'size': '45ft', 'block': '05', 'bp': (1, 2), 'qty': 1, 'pickup_step': None},
]



solver = YardSolver(
    allocation_groups=grouped_result,
    blocks=blocks,
    block_positions=block_positions,
    berth_positions=berth_positions,
    vessel_info=vessel_info,
    block_bays_num=block_bays_num_by_block,
    bay_cap=bay_cap_by_block,
    T=T,
    window_size=2,
    initial_state=initial_state,
    verbose=True
)
all_assignments, remaining_state, final_df = solver.run()

solver.save_csv("allocation_results.csv")

viz = InteractiveYardVisualizer(
    all_assignments=all_assignments,
    allocation_groups=grouped_result,
    rows=5, cols=4,
    blocks=blocks,                         # ✅真实箱区列表
    block_bays_num=block_bays_num_by_block,# ✅每箱区 bayCount
    bay_cap=bay_cap_by_block,              # ✅每箱区 col*tier
    T=T,
    window_size=2,
    figsize=(12,9),
    planning_start_time="2025-12-18 00:00:00",
    step_hours=time_step_hours
)

viz.show()