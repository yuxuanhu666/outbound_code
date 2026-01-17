import json
from collections import defaultdict
# 输出所有分配组信息
import pprint

# 1. 读取 JSON 文件
with open("../data/unit.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

with open("../data/SHIPVISITID_TEST1220.JSON.json", "r", encoding="utf-8") as f:
    raw2 = json.load(f)


units = raw["data"]["contrUnitList"]  # 这里就是所有箱子列表

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

    # 将分配组信息添加到结果列表
    allocation_group = {
        "group_id": group_id,
        "contrISO": contr_iso,
        "ibVisitId": ib_visit_id,
        "pod": pod,
        "weightCategory": weight_category,
        "contrOwnerName": contr_owner_name,
        "box_count": box_count, # 统计箱量
        "unit_ids": unit_ids  # 保存所有箱子的箱号
    }

    grouped_result.append(allocation_group)



pprint.pprint(grouped_result)

with open("../data/在场箱数据.json", "r", encoding="utf-8") as f:
    data = json.load(f)
# # 输入的JSON数据

# Step 2: Process `fullSlot` and group by block and bay pairs
def process_slot(full_slot):
    # fullSlot format: Y-xx.xx.xx.xx
    parts = full_slot.split('-')[1].split('.')

    # Extract block as the first part of the fullSlot (e.g. "03")
    block = parts[0]  # This gives us the "03" from "03.13.10.02"
    bay = int(parts[1])  # Bay is the second part
    col = int(parts[2])  # Column is the third part
    tier = int(parts[3])  # Tier is the fourth part

    # Process the bay to form the pair
    if bay % 2 == 0:
        bay_pair = (int((bay + 1) / 2), int((bay + 1) / 2) + 1)
    else:
        bay_pair = (bay // 2 + 1, bay // 2 + 1)

    return block, bay_pair


# Step 3: Count boxes per bay pair
counts = defaultdict(int)

for item in data['data']['contrSlotList']:
    full_slot = item["fullSlot"]
    block, bay_pair = process_slot(full_slot)

    key = (block, bay_pair)
    counts[key] += 1


# Step 4: Format the output as required
def format_output(counts):
    output = []
    group_id = -1
    for (block, bay_pair), qty in counts.items():
        output.append({
            "group_id": str(group_id),
            "ship": "INIT",
            "size": "20ft",  # You can modify this part to reflect the container size if necessary
            "block": block,  # Now block is the extracted part from fullSlot
            "bp": bay_pair,
            "qty": qty,
            "pickup_step": None
        })
        group_id -= 1  # Decrementing group_id as per your example

    return output


# Step 5: Get the final result
result = format_output(counts)


print(result)
