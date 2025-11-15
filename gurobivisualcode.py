
# =================== 交互式点击查看箱区贝位详情的可视化程序 ===================
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from collections import defaultdict
import math
import pandas as pd
# 颜色帮助：按 group_id 映射固定颜色
import matplotlib.cm as cm



class InteractiveYardVisualizer:
    """
    交互式箱区可视化类（缩略图 + Slider + 点击弹出单区详情）
    用法示例：
        viz = InteractiveYardVisualizer(
            all_assignments=all_assignments,
            allocation_groups=allocation_groups,
            rows=3, cols=4,
            block_bays_num=40,
            bay_cap=30,
            T=72,
            window_size=6,
            figsize=(12,9)
        )
        viz.show()
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
                 figsize=(12,9)):
        self.all_assignments = self._ensure_list_of_dicts(all_assignments)
        self.allocation_groups = allocation_groups
        self.rows = rows
        self.cols = cols
        self.block_bays_num = block_bays_num
        self.bay_cap = bay_cap
        self.T = T
        self.window_size = window_size
        self.figsize = figsize

        # 预计算 windows 与 assign_to_window
        self.windows, self.assign_to_window = self._map_assignments_to_windows(self.all_assignments,
                                                                               self.allocation_groups,
                                                                               self.T,
                                                                               self.window_size)
        # color map by group_id
        all_gids = sorted({r['group_id'] for r in self.all_assignments})
        cmap = cm.get_cmap('tab20')
        self.color_map = {gid: cmap(i % 20) for i, gid in enumerate(all_gids)}

        # 内部绘图状态
        self.fig = None
        self.axes = {}
        self.win_slider = None
        self.current_win_idx = 0
        self.cid = None

    # ------------------ 静态 / 内部工具 ------------------
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

        for r in recs:
            bp = tuple(r['bp'])
            qty = int(r.get('qty', 0))
            size = r.get('size')
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
        bp_totals, bay_occ_count, bay_size_set, conflict_bays, overcap_bps, bp_records = \
            self._detect_conflicts_in_block(recs, self.block_bays_num, self.bay_cap)

        # 颜色按 group_id
        gids = sorted({r['group_id'] for r in recs})
        cmap = cm.get_cmap('tab20')
        color_map = {gid: cmap(i % 20) for i, gid in enumerate(gids)}

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"Block {block_id} 详细贝位分配（物理贝 1..{self.block_bays_num}）", fontsize=14)
        ax.set_xlim(0.5, self.block_bays_num + 0.5)
        ax.set_ylim(0, 1.6)
        ax.set_xticks(range(1, self.block_bays_num+1))
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
            heights = [int(r['qty']) / self.bay_cap for r in rec_list]
            y0 = 0.0
            for r, h in zip(rec_list, heights):
                gid = r['group_id']
                color = color_map.get(gid, (0.6,0.6,0.6))
                rect = patches.Rectangle((x, y0), width, h, facecolor=color, edgecolor='k', linewidth=0.6, alpha=0.9)
                ax.add_patch(rect)
                if h >= 0.06:
                    ax.text(x + width/2, y0 + h/2, f"{gid}\n{int(r['qty'])}", ha='center', va='center', fontsize=8, color='white')
                y0 += h
            if total_qty > self.bay_cap:
                rect2 = patches.Rectangle((x, 0), width, y0, facecolor='none', edgecolor='red', linewidth=2)
                ax.add_patch(rect2)

        text_y = 1.12
        if len(recs) > 0:
            ax.text(0.01, text_y, "组列表 (group_id: qty, size, pickup_step):", transform=ax.transAxes, fontsize=9, va='bottom')
            text_lines = []
            for r in recs:
                text_lines.append(f"{r['group_id']}: {int(r['qty'])} ({r.get('size','')}) pk:{r.get('pickup_step')}")
            chunk_size = 6
            for i in range(0, len(text_lines), chunk_size):
                line = "  |  ".join(text_lines[i:i+chunk_size])
                ax.text(0.01, text_y - 0.04 - (i//chunk_size)*0.04, line, transform=ax.transAxes, fontsize=8, va='top')

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
        self.fig.suptitle(f"总体视图 — 窗口 {win_idx} : steps [{ws}, {we}]", fontsize=14)

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
            if not recs:
                ax.add_patch(patches.Rectangle((0,0),1,1, facecolor=(0.95,0.95,0.95), edgecolor='k'))
                ax.text(0.5, 0.5, "空", ha='center', va='center', fontsize=12)
            else:
                _, bay_occ_count, _, conflict_bays, overcap_bps, _ = self._detect_conflicts_in_block(recs, self.block_bays_num, self.bay_cap)
                occ_bays = sum(1 for i in range(1, self.block_bays_num+1) if bay_occ_count[i] > 0)
                if conflict_bays:
                    bg = (1.0, 0.85, 0.85)
                else:
                    frac = min(1.0, occ_bays / self.block_bays_num)
                    bg = (1-frac*0.6, 1-frac*0.6, 1.0 - frac*0.2)
                ax.add_patch(patches.Rectangle((0,0),1,1, facecolor=bg, edgecolor='k'))
                ax.text(0.5, 0.6, f"占贝: {occ_bays}/{self.block_bays_num}", ha='center', va='center', fontsize=10)
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
        gs = self.fig.add_gridspec(self.rows, self.cols, wspace=0.5, hspace=0.5, top=0.92, bottom=0.12)
        self.axes = {}
        for r in range(self.rows):
            for c in range(self.cols):
                b = r * self.cols + c + 1
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