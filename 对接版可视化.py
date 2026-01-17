
# =================== 交互式点击查看箱区贝位详情的可视化程序 ===================
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from collections import defaultdict
from 
import math
import pandas as pd
# 颜色帮助：按 group_id 映射固定颜色
import matplotlib.cm as cm



class InteractiveYardVisualizer:
    """
    交互式箱区可视化类（缩略图 + Slider + 点击弹出单区详情）
    参数说明：
        - all_assignments: list[dict] 或 pandas.DataFrame，每条至少包含:
              {'group_id','block','bp' (tuple/list),'qty','size','pickup_step'}
        - allocation_groups: 原始分配组列表（含 group_id -> arrival_step 映射）
        - rows, cols: 缩略图网格行列数（箱区编号按 1..rows*cols）
        - block_bays_num: 每个箱区的物理贝数
        - bay_cap: 单物理贝容量（TEU）
        - T: 总时间步数
        - window_size: 非重叠窗口大小（步长）
        - figsize: 主图画布大小
    """

    def __init__(self,
                 all_assignments,
                 allocation_groups,
                 rows,
                 cols,
                 block_bays_num,
                 bay_cap,
                 T,
                 window_size,
                 blocks=None,
                 figsize=(12,9),
                 planning_start_time=None,   # NEW
                 step_hours =None
                 ):
        self.all_assignments = self._ensure_list_of_dicts(all_assignments)
        self.allocation_groups = allocation_groups
        # ✅使用真实 blocks（否则就从结果里推）
        if blocks is None:
            blocks = sorted({r["block"] for r in self.all_assignments})
        self.blocks = list(blocks)

        self.rows = rows
        self.cols = cols

        # ✅把 bays/cap 统一成 “每箱区一个值”
        if isinstance(block_bays_num, dict):
            self.block_bays_num_by_block = {b: int(block_bays_num[b]) for b in self.blocks}
            self.default_block_bays_num = max(
                self.block_bays_num_by_block.values()) if self.block_bays_num_by_block else 40
        else:
            self.block_bays_num_by_block = {b: int(block_bays_num) for b in self.blocks}
            self.default_block_bays_num = int(block_bays_num)

        if isinstance(bay_cap, dict):
            self.bay_cap_by_block = {b: int(bay_cap[b]) for b in self.blocks}
            self.default_bay_cap = max(self.bay_cap_by_block.values()) if self.bay_cap_by_block else 30
        else:
            self.bay_cap_by_block = {b: int(bay_cap) for b in self.blocks}
            self.default_bay_cap = int(bay_cap)

        self.T = T
        self.window_size = window_size
        self.figsize = figsize

        # NEW: 用于把 step 映射到真实时间
        self.planning_start_dt = pd.to_datetime(planning_start_time) if planning_start_time else None
        self.step_hours = float(step_hours)

        # 预计算 windows 与 assign_to_window
        self.windows, self.assign_to_window = self._map_assignments_to_windows(self.all_assignments,
                                                                               self.allocation_groups,
                                                                               self.T,
                                                                               self.window_size)
        # color map by group_id
        all_gids = sorted({r['group_id'] for r in self.all_assignments})
        palette = (list(cm.get_cmap('tab20').colors)
                   + list(cm.get_cmap('tab20b').colors)
                   + list(cm.get_cmap('tab20c').colors))
        self.color_map = {gid: palette[i % len(palette)] for i, gid in enumerate(all_gids)}

        # 内部绘图状态
        self.fig = None
        self.axes = {}
        self.win_slider = None
        self.current_win_idx = 0
        self.cid = None

    # ------------------ 静态 / 内部工具 ------------------
    def _bay_count(self, block_id):
        return int(self.block_bays_num_by_block.get(block_id, self.default_block_bays_num))

    def _bay_cap(self, block_id):
        return int(self.bay_cap_by_block.get(block_id, self.default_bay_cap))

    def _window_time_str(self, ws, we):
        """steps [ws,we] (含端点) -> 真实时间区间字符串"""
        if self.planning_start_dt is None:
            return None

        # step i 覆盖: [start + (i-1)*H, start + i*H)
        t0 = self.planning_start_dt + pd.Timedelta(hours=(ws - 1) * self.step_hours)
        t1 = self.planning_start_dt + pd.Timedelta(hours=we * self.step_hours)

        return f"{t0:%Y-%m-%d %H:%M}——{t1:%Y-%m-%d %H:%M}"

    @staticmethod
    def size_to_teu(size):
        s = str(size) if size is not None else ""
        return 2 if ("40" in s or "45" in s) else 1
    @staticmethod
    def _ensure_list_of_dicts(all_assignments):
        if isinstance(all_assignments, pd.DataFrame):
            return all_assignments.to_dict('records')
        elif isinstance(all_assignments, list):
            return all_assignments
        else:
            return list(all_assignments) if all_assignments is not None else []

    @staticmethod
    def _map_assignments_to_windows(all_assignments, allocation_groups, T, window_size):
        arrival_map = {g['group_id']: g.get('arrival_step') for g in allocation_groups}
        windows = []
        for ws in range(1, T + 1, window_size):
            we = min(ws + window_size - 1, T)
            windows.append((ws, we))

        assign_to_window = defaultdict(list)
        for rec in all_assignments:
            gid = rec.get('group_id')
            arrival = arrival_map.get(gid)
            pickup = rec.get('pickup_step')
            assigned_win = None
            if isinstance(arrival, int):
                for idx, (ws, we) in enumerate(windows):
                    if ws <= arrival <= we:
                        assigned_win = idx
                        break
            if assigned_win is None and isinstance(pickup, int):
                for idx, (ws, we) in enumerate(windows):
                    if pickup > ws:
                        assigned_win = idx
                        break
            if assigned_win is None:
                assigned_win = 0
            assign_to_window[assigned_win].append(rec)
        return windows, assign_to_window

    @staticmethod
    def _detect_conflicts_in_block(recs, block_bays_num, bay_cap):
        bp_totals = defaultdict(int)
        bp_records = defaultdict(list)
        bay_occupancy_count = [0] * (block_bays_num + 1)
        bay_size_set = defaultdict(set)

        def size_to_teu(size):
            s = str(size) if size is not None else ""
            return 2 if ("40" in s or "45" in s) else 1

        for r in recs:
            bp = tuple(r['bp'])
            qty = int(r.get('qty', 0))
            size = r.get('size')
            teu_qty = qty * size_to_teu(size)
            bp_totals[bp] += qty
            bp_records[bp].append(r)
            i1, i2 = bp
            for i in range(i1, i2 + 1):
                if 1 <= i <= block_bays_num:
                    bay_occupancy_count[i] += 1
                    if size is not None:
                        bay_size_set[i].add(size)

        conflict_bays = [i for i in range(1, block_bays_num + 1) if len(bay_size_set.get(i, set())) > 1]
        overcap_bps = [bp for bp, tot in bp_totals.items() if tot > bay_cap]
        return bp_totals, bay_occupancy_count, bay_size_set, conflict_bays, overcap_bps, bp_records

    # ------------------ 单区详细绘制 ------------------
    def _draw_block_detail(self, block_id, recs, figsize=(14,4)):
        bay_count = self._bay_count(block_id)
        bay_cap = self._bay_cap(block_id)
        bp_totals, bay_occ_count, bay_size_set, conflict_bays, overcap_bps, bp_records = \
            self._detect_conflicts_in_block(recs, bay_count, bay_cap)

        # 颜色按 group_id
        gids = sorted({r['group_id'] for r in recs})
        cmap = cm.get_cmap('tab20')
        color_map = {gid: cmap(i % 20) for i, gid in enumerate(gids)}

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"Block {block_id} 详细贝位分配（贝 1..{bay_count}）", fontsize=14)
        ax.set_xlim(0.5, bay_count + 0.5)
        ax.set_ylim(0, 1.6)
        ax.set_xticks(range(1, bay_count+1))
        ax.set_xticklabels([str(2*i - 1) for i in range(1, bay_count + 1)])
        ax.set_xlabel("Bay index")
        ax.set_ylabel("占用高度 (按 qty / bay_cap 缩放；>1 表示超容)")
        ax.grid(axis='x', linestyle=':', linewidth=0.4)

        if conflict_bays:
            for i in conflict_bays:
                ax.add_patch(patches.Rectangle((i-0.5, 0), 1.0, 1.05, facecolor=(1,0.9,0.9), edgecolor='none', zorder=0))

        for bp, rec_list in bp_records.items():
            i1, i2 = bp
            width = (i2 - i1 + 1) * 0.9
            x = i1 - 0.45
            total_qty = sum(int(r['qty']) for r in rec_list)
            heights = [int(r['qty']) / bay_cap for r in rec_list]
            y0 = 0.0
            for r, h in zip(rec_list, heights):
                gid = r['group_id']
                color = color_map.get(gid, (0.6,0.6,0.6))
                rect = patches.Rectangle((x, y0), width, h, facecolor=color, edgecolor='k', linewidth=0.6, alpha=0.9)
                ax.add_patch(rect)
                if h >= 0.06:
                    ax.text(x + width/2, y0 + h/2, f"{gid}\n{int(r['qty'])}", ha='center', va='center', fontsize=8, color='white')
                y0 += h
            # 超容用红框标出
            if total_qty > bay_cap:
                rect2 = patches.Rectangle((x, 0), width, y0, facecolor='none', edgecolor='red', linewidth=2)
                ax.add_patch(rect2)

            # ----- 双贝时在两奇数之间居中显示偶数编号 -----
            # 如果 bp 覆盖两个物理贝位（i2 == i1+1），则在中点处显示偶数编号 2*i1
            if (i2 - i1 + 1) == 2:
                even_label = str(2 * i1)  # 例如 bp=(1,2) -> 显示 2；bp=(3,4)->显示 6
                # x = i1 + 0.5 为两物理贝位的中点，使用 get_xaxis_transform 使 y 以轴坐标定位（-0.02 表示在 x 轴下方）
                ax.text(i1 + 0.5, -0.02, even_label, transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=8, color='black')


        # 先组织文本行
        text_lines = []
        for r in recs:
            text_lines.append(f"{r['group_id']}: {int(r['qty'])} ({r.get('size', '')}) pk:{r.get('pickup_step')}")

        if len(text_lines) > 0:
            import math
            # 可配置项（按需调整）
            max_lines_per_col = 8  # 每列最大行数，减少会增加列数
            max_cols = 8  # 最大列数，避免太多列影响可读性
            line_height = 0.035  # 每行在 axes 坐标系中的高度
            x_start = 0.01
            x_end = 0.99
            y_start = 0.98  # 从上方开始排版
            max_chars_per_line = 48  # 超长行截断长度

            # 截断过长行，避免超宽
            for i, s in enumerate(text_lines):
                if len(s) > max_chars_per_line:
                    text_lines[i] = s[:max_chars_per_line - 3] + "..."

            # 计算列数与每列行数
            ncols = min(max_cols, max(1, math.ceil(len(text_lines) / max_lines_per_col)))
            per_col = math.ceil(len(text_lines) / ncols)

            total_width = x_end - x_start
            col_width = total_width / ncols
            # 背景高度（稍留边距）
            bg_height = min(0.9, per_col * line_height + 0.03)

            # 背景盒子（画在 axes 内部）
            ax.add_patch(
                patches.Rectangle(
                    (x_start, y_start - bg_height),  # left-bottom in axes coords
                    total_width,
                    bg_height,
                    transform=ax.transAxes,
                    facecolor='white',
                    alpha=0.85,
                    edgecolor='none',
                    zorder=5,
                    clip_on=True
                )
            )

            # 标题行（放在背景内顶部）
            ax.text(x_start + 0.004, y_start - 0.005, "组列表 (group_id: qty, size, pickup_step):",
                    transform=ax.transAxes, fontsize=9, va='top', ha='left', zorder=6, clip_on=True)

            # 逐列绘制文本（轴内坐标）
            for col in range(ncols):
                col_items = text_lines[col * per_col: (col + 1) * per_col]
                x_col = x_start + col * col_width + 0.006
                for r_idx, line in enumerate(col_items):
                    y_pos = y_start - 0.03 - r_idx * line_height  # 下移一点以避开标题
                    if y_pos < 0.02:
                        break
                    ax.text(x_col, y_pos, line,
                            transform=ax.transAxes,
                            fontsize=8,
                            va='top',
                            ha='left',
                            clip_on=True,
                            zorder=6)
        else:
            # 无组时不显示任何文本
            pass

        from matplotlib.lines import Line2D
        sizes_present = sorted({r.get('size') for r in recs if r.get('size') is not None})
        size_color = {'20ft': '#4C72B0', '40ft': '#55A868', '45ft': '#C44E52'}
        legend_elems = []
        for s in sizes_present:
            legend_elems.append(Line2D([0], [0], marker='s', color='w', label=str(s), markerfacecolor=size_color.get(s, '#888888'), markersize=10))
        if legend_elems:
            ax.legend(handles=legend_elems, loc='upper right', fontsize=9)

        plt.show()

    # ------------------ 缩略图概览绘制 ------------------
    def _draw_overview(self, win_idx):
        self.current_win_idx = win_idx
        ws, we = self.windows[win_idx]
        time_str = self._window_time_str(ws, we)
        if time_str:
            title = f"{time_str}\n总体视图 — 窗口 {win_idx} : steps [{ws}, {we}]"
        else:
            title = f"总体视图 — 窗口 {win_idx} : steps [{ws}, {we}]"

        self.fig.suptitle(title, fontsize=14)

        active_by_block = defaultdict(list)
        for idx in range(0, win_idx+1):
            for rec in self.assign_to_window.get(idx, []):
                pstep = rec.get('pickup_step')
                if pstep is None or pstep > ws:
                    recC = rec.copy()
                    recC['bp'] = tuple(recC['bp'])
                    active_by_block[recC['block']].append(recC)

        for b, ax in self.axes.items():
            ax.cla()
            ax.set_title(f"Block {b}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            recs = active_by_block.get(b, [])
            bay_count = self._bay_count(b)
            bay_cap = self._bay_cap(b)
            
            if not recs:
                ax.add_patch(patches.Rectangle((0,0),1,1, facecolor=(0.95,0.95,0.95), edgecolor='k'))
                ax.text(0.5, 0.5, "空", ha='center', va='center', fontsize=12)
            else:
                _, bay_occ_count, _, conflict_bays, overcap_bps, _ = self._detect_conflicts_in_block(recs, bay_count, bay_cap)
                occ_bays = sum(1 for i in range(1, bay_count+1) if bay_occ_count[i] > 0)
                if conflict_bays:
                    bg = (1.0, 0.85, 0.85)
                else:
                    frac = min(1.0, occ_bays / bay_count)
                    bg = (1-frac*0.6, 1-frac*0.6, 1.0 - frac*0.2)
                ax.add_patch(patches.Rectangle((0,0),1,1, facecolor=bg, edgecolor='k'))
                ax.text(0.5, 0.6, f"占贝: {occ_bays}/{bay_count}", ha='center', va='center', fontsize=10)
                if conflict_bays:
                    ax.text(0.5, 0.35, f"冲突贝: {','.join(map(str, conflict_bays[:6]))}{'...' if len(conflict_bays)>6 else ''}", ha='center', va='center', color='red', fontsize=8)
                if overcap_bps:
                    ax.text(0.5, 0.12, "超容 BP 存在", ha='center', va='center', color='red', fontsize=8)

    # ------------------ 事件回调 ------------------
    def _on_click(self, event):
        if event.inaxes is None:
            return
        for b, ax in self.axes.items():
            if event.inaxes == ax:
                win_idx = self.current_win_idx
                ws, we = self.windows[win_idx]
                active = []
                for idx in range(0, win_idx+1):
                    for rec in self.assign_to_window.get(idx, []):
                        pstep = rec.get('pickup_step')
                        if pstep is None or pstep > ws:
                            recC = rec.copy()
                            recC['bp'] = tuple(recC['bp'])
                            active.append(recC)
                block_recs = [r for r in active if r['block'] == b]
                self._draw_block_detail(b, block_recs, figsize=(14,4))
                break

    def _on_slider(self, val):
        idx = int(self.win_slider.val)
        self._draw_overview(idx)
        self.fig.canvas.draw_idle()

    # ------------------ 外部接口 ------------------
    def show(self):
        """创建主界面并显示（会阻塞直到关闭窗口）"""
        self.fig = plt.figure(figsize=self.figsize)
        gs = self.fig.add_gridspec(self.rows, self.cols, wspace=0.5, hspace=0.5, top=0.88, bottom=0.12)
        self.axes = {}
        idx =0
        for r in range(self.rows):
            for c in range(self.cols):
                if idx >=len(self.blocks):
                    break
                b = self.blocks[idx]
                idx += 1

                ax = self.fig.add_subplot(gs[r, c])
                self.axes[b] = ax
                ax.set_title(f"Block {b}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

        ax_slider = self.fig.add_axes([0.15, 0.03, 0.7, 0.05])
        max_win = max(0, len(self.windows) - 1)
        self.win_slider = Slider(ax_slider, 'WindowIdx', 0, max_win, valinit=0, valstep=1)

        # 初始绘制
        self._draw_overview(0)

        # 连接回调
        self.win_slider.on_changed(self._on_slider)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        plt.show()

    def draw_block_at_window(self, block_id, win_idx):
        """外部接口：直接绘制某窗口下单个箱区详情"""
        if win_idx < 0 or win_idx >= len(self.windows):
            raise IndexError("win_idx 超出范围")
        ws, we = self.windows[win_idx]
        active = []
        for idx in range(0, win_idx+1):
            for rec in self.assign_to_window.get(idx, []):
                pstep = rec.get('pickup_step')
                if pstep is None or pstep > ws:
                    recC = rec.copy()
                    recC['bp'] = tuple(recC['bp'])
                    active.append(recC)
        block_recs = [r for r in active if r['block'] == block_id]
        self._draw_block_detail(block_id, block_recs, figsize=(14,4))