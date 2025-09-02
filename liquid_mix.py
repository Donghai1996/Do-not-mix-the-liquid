import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame

# ===================== 可调参数 ===================== #
# ——总体 & 难度—— #
MIN_TOTAL = 6               # 本局最少瓶子数
MAX_TOTAL = 16              # 本局最多瓶子数（支持两排，每排最多 8 个）
N_BOTTLES = 6               # 初始启动时的占位值（会在 _init_bottles 内被随机覆盖为 [6,12]）
LAYERS_PER_BOTTLE = 4       # 每瓶高度 = 4 格
COLOR_VARIETY = (4, 6)      # 颜色种类区间 [min, max]
RANDOM_SEED = None          # 设为整数以复现实验，如 42

# ——窗口尺寸（固定）—— #
WINDOW_W, WINDOW_H = 1000, 860  # 适配两排 0.75 缩放后的瓶子高度
FPS = 60

# ——统一缩放（等比例缩小“概念瓶子”和“瓶子图片”）—— #
SCALE = 0.75  # ☆ 将瓶子整体缩小到原来的 75%

# 原始（未缩放）概念尺寸
BASE_BOTTLE_WIDTH = 90
BASE_BOTTLE_HEIGHT = 360
BASE_INNER_PADDING = 5
BASE_SLOT_GAP = 6
BASE_FRAME_WIDTH = 100

# 计算缩放后的尺寸
BOTTLE_WIDTH = int(round(BASE_BOTTLE_WIDTH * SCALE))
BOTTLE_HEIGHT = int(round(BASE_BOTTLE_HEIGHT * SCALE))
INNER_PADDING = max(1, int(round(BASE_INNER_PADDING * SCALE)))
SLOT_GAP = max(1, int(round(BASE_SLOT_GAP * SCALE)))
FRAME_WIDTH = int(round(BASE_FRAME_WIDTH * SCALE))

# 布局相关（每行最多 6 个；行间距 & 水平间距跟随缩放）
MAX_PER_ROW = 8  # 每行最多 8 个
ROW_VGAP = int(round(80 * SCALE))  # 两行间距更宽一些
H_GAP = int(round(22 * SCALE))
LIFT_OFFSET = -int(round(18 * SCALE))

# ——瓶子图片与概念位置关系—— #
USE_IMAGE_FRAME = True                      # 是否使用本地图片作为瓶子外框（含盖子）
FRAME_PATH = "bottle_frame.png"            # 你的透明 PNG，放同目录
FRAME_HEIGHT_SCALE = 1.16                   # 图片相对概念高度的缩放（1.0 表示与概念等高）
FRAME_ALIGN_RATIO = 0.52                    # 概念中心与图片对齐位置（0~1）；0.5 居中，0.75 图片略靠上

# ===================== 颜色与样式 ===================== #
BG_COLOR = (250, 250, 252)
BORDER_COLOR = (60, 60, 60)
TEXT_COLOR = (40, 40, 40)
LIFTED_TINT = (230, 230, 255)
LOCKED_TINT = (230, 255, 230)
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (180, 180, 180)
BTN_BG = (235, 240, 255)
BTN_BORDER = (120, 140, 200)
INFO_BG = (245, 245, 250)

# 可选调色板（尽量区分明显）
PALETTE = [
    (244, 67, 54),   # red
    (233, 30, 99),   # pink
    (156, 39, 176),  # purple
    (103, 58, 183),  # deep purple
    (25, 118, 210),  # blue (darker，与 light blue 拉开对比)
    (3, 169, 244),   # light blue
    (0, 188, 212),   # cyan
    (0, 150, 136),   # teal
    (76, 175, 80),   # green
    (205, 220, 57),  # lime
    (255, 235, 59),  # yellow
    (255, 193, 7),   # amber
]

# ===================== 瓶子与游戏逻辑 ===================== #
@dataclass
class Bottle:
    layers: List[Optional[int]]  # 自底向上长度 4；值为颜色索引或 None
    state: str = "ground"        # "ground" | "lifted" | "locked"

    # ---------- 读状态 ---------- #
    def fill_level(self) -> int:
        return sum(1 for c in self.layers if c is not None)

    def top_index(self) -> Optional[int]:
        lvl = self.fill_level()
        if lvl == 0:
            return None
        return lvl - 1

    def top_color(self) -> Optional[int]:
        idx = self.top_index()
        return None if idx is None else self.layers[idx]

    def free_space(self) -> int:
        return LAYERS_PER_BOTTLE - self.fill_level()

    def is_empty(self) -> bool:
        return self.fill_level() == 0

    def is_uniform_full(self) -> bool:
        # 锁定规则：当瓶子“满且颜色相同”则锁定
        if self.fill_level() != LAYERS_PER_BOTTLE:
            return False
        first = self.layers[0]
        return all(c == first for c in self.layers)

    def top_block_size(self) -> int:
        # 计算顶部连续同色块的尺寸（0 表示空）
        lvl = self.fill_level()
        if lvl == 0:
            return 0
        color = self.layers[lvl - 1]
        size = 1
        for i in range(lvl - 2, -1, -1):
            if self.layers[i] == color:
                size += 1
            else:
                break
        return size

    # ---------- 写状态 ---------- #
    def push_color(self, color: int, count: int) -> None:
        assert count <= self.free_space()
        lvl = self.fill_level()
        for i in range(count):
            self.layers[lvl + i] = color

    def pop_block(self) -> Tuple[Optional[int], int]:
        # 取出顶部连续同色块，返回 (颜色, 个数)
        lvl = self.fill_level()
        if lvl == 0:
            return None, 0
        color = self.layers[lvl - 1]
        size = self.top_block_size()
        for i in range(size):
            self.layers[lvl - 1 - i] = None
        return color, size

    def try_lock(self) -> None:
        if self.is_uniform_full():
            self.state = "locked"


class Game:
    def __init__(self, n_bottles: int):
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
        # ☆ 每局随机一个总瓶子数（6~12），并在 reset 时重抽
        self.n = random.randint(MIN_TOTAL, MAX_TOTAL)
        self.palette_full = PALETTE[:]
        self.palette = self.palette_full  # 运行时实际截取的颜色集
        self.bottles: List[Bottle] = []
        self.selected: Optional[int] = None  # 抬起的瓶子索引
        self.message = ""
        self.win = False
        self.initial_snapshot = None  # 保存本局初始快照，用于 Reset 回溯
        self.history = []            # ☆ 撤回栈：存放每步操作前的快照，用于 Undo
        self._init_bottles()

    def _calc_empty_bottles(self, n: int) -> int:
        """
        新规则：空瓶数必须严格小于总瓶子数的 1/3。
        - 在允许范围内，优先给 2 个作为“基础”；若上限 < 2（如 n=6），则在 [1..上限] 内随机。
        - 示例：6→1；7/8/9→2；10/11/12→2..3；13/14→2..4；15/16→2..5。
        """
        if n <= 0:
            return 0
        # 最大允许空瓶数，使 empty < n/3 （等价于 floor((n-1)/3)）
        max_allowed = (n - 1) // 3
        if max_allowed <= 0:
            # 在当前范围 n∈[6,12]，n=6 时 max_allowed=1
            return 1 if n >= 6 else 0
        low = 2 if max_allowed >= 2 else 1
        return random.randint(low, max_allowed)

    def _choose_color_count(self, n: int, empty_bottles: int, total_groups: int) -> int:
        """根据关卡规模动态选择颜色数 k：
        - n<=6 时强制 4 色（避免 6 瓶 >4 色可能无解）；
        - n=7~8 时 4~5 色；n=9~10 时 5~6 色；n>=11 时 5~6 色；
        - 还需受上限约束：k ≤ max_colors、k ≤ filled、k ≤ total_groups、k ≤ 调色板大小。
        """
        min_colors, max_colors = COLOR_VARIETY
        filled = n - empty_bottles
        palette_cap = len(self.palette_full)
        # 分段目标区间
        if n <= 6:
            lo, hi = 4, 4
        elif n <= 8:
            lo, hi = 4, 5
        elif n <= 10:
            lo, hi = 5, 6
        else:  # 11~16
            lo, hi = 5, 6
        # 施加硬约束
        hi = min(hi, max_colors, filled, total_groups, palette_cap)
        lo = max(1, min(lo, hi))
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    def _init_bottles(self):
        # ---------------- 关卡生成（保证可通关 & 颜色 4 的倍数）---------------- #
        empty_bottles = self._calc_empty_bottles(self.n)
        filled_bottles = max(0, self.n - empty_bottles)
        total_groups = filled_bottles  # 每个满瓶提供 4 层 = 1 组

        self.bottles = []

        if total_groups == 0:
            for _ in range(self.n):
                self.bottles.append(Bottle(layers=[None]*LAYERS_PER_BOTTLE))
            self._refresh_locks(); return

        # 选择颜色数 k：根据总瓶数/空瓶动态调整，避免 12 瓶仅 4 色过易、6 瓶 >4 色可能无解
        k = self._choose_color_count(self.n, empty_bottles, total_groups)
        self.palette = random.sample(self.palette_full, k)

        # 分配颜色组：每色至少 1 组
        groups_per_color = [1]*k
        remaining = total_groups - k
        for _ in range(remaining):
            groups_per_color[random.randrange(k)] += 1

        # 生成颜色池（每组 4 层）
        pool: List[int] = []
        for color_idx, g in enumerate(groups_per_color):
            pool.extend([color_idx]*(g*LAYERS_PER_BOTTLE))
        random.shuffle(pool)

        # 构建瓶子
        p = 0
        for _ in range(filled_bottles):
            layers = pool[p:p+LAYERS_PER_BOTTLE]
            p += LAYERS_PER_BOTTLE
            self.bottles.append(Bottle(layers=layers, state="ground"))
        for _ in range(empty_bottles):
            self.bottles.append(Bottle(layers=[None]*LAYERS_PER_BOTTLE, state="ground"))

        random.shuffle(self.bottles)
        self._refresh_locks()
        self.message = f"Total {self.n} | Mixed {filled_bottles} | Empty {empty_bottles}"
        # ☆ 记录初始快照，供 Reset 按钮回溯
        self.initial_snapshot = self._make_snapshot()
        # ☆ 清空撤回栈
        self.history = []

    def reset_level(self):
        """☆ 重新生成随机一局（总瓶数 6~12，颜色也重抽）。"""
        self.selected = None
        self.win = False
        self.n = random.randint(MIN_TOTAL, MAX_TOTAL)
        self._init_bottles()

    # ----------- 规则辅助 ----------- #
    def _refresh_locks(self):
        for b in self.bottles:
            if b.state != "locked" and b.is_uniform_full():
                b.state = "locked"

    def _make_snapshot(self):
        """保存当前局面的可还原快照。"""
        return {
            "n": self.n,
            "palette": self.palette[:],
            "bottles": [{"layers": b.layers[:], "state": b.state} for b in self.bottles],
            "selected": self.selected,
            "win": self.win,
            "message": self.message,
        }

    def _restore_snapshot(self, snap):
        """从快照还原局面。"""
        if not snap:
            return
        self.n = snap["n"]
        self.palette = snap["palette"][:]
        self.bottles = [Bottle(layers=rec["layers"][:], state=rec["state"]) for rec in snap["bottles"]]
        self.selected = snap.get("selected")
        self.win = snap.get("win", False)
        self.message = snap.get("message", "")

    def reset_to_initial(self):
        """回到本局初始状态（不改变随机到的总瓶数与颜色选择）。"""
        self._restore_snapshot(getattr(self, "initial_snapshot", None))
        # 回到初始后清空撤回栈
        self.history = []

    def undo(self):
        """撤回上一步：从历史栈弹出最近一次操作前的快照并还原。"""
        if not self.history:
            self.message = "Nothing to undo."
            return
        snap = self.history.pop()
        self._restore_snapshot(snap)
        self.message = "Undone."

    def all_done(self) -> bool:
        # 规则 7：所有瓶子 fill==0 或 locked
        return all(b.is_empty() or b.state == "locked" for b in self.bottles)

    # ----------- 交互逻辑（点击） ----------- #
    def click_bottle(self, idx: int):
        if self.win:
            return
        b = self.bottles[idx]

        if b.state == "locked":
            if self.selected is not None and self.selected != idx:
                a = self.bottles[self.selected]
                a.state = "ground"
                self.selected = None
            return

        if self.selected is None:
            b.state = "lifted"; self.selected = idx; return

        if self.selected == idx:
            b.state = "ground"; self.selected = None; return

        a = self.bottles[self.selected]
        can_pour, pour_cnt = self._can_pour(a, b)
        if can_pour and pour_cnt > 0:
            # ☆ 记录移动前的快照，用于 Undo
            self.history.append(self._make_snapshot())

            color, block = a.top_color(), a.top_block_size()
            move = min(block, b.free_space())
            lvl = a.fill_level()
            for i in range(move):
                a.layers[lvl - 1 - i] = None
            b.push_color(color, move)

            a.state = "ground"; self.selected = None
            a.try_lock(); b.try_lock(); self._refresh_locks()
            if self.all_done():
                self.win = True
                self.message = "You win!"
            return
        else:
            a.state = "ground"; b.state = "lifted"; self.selected = self.bottles.index(b)
            return False, 0
        if b.state == "locked":
            return False, 0
        a_color = a.top_color(); a_block = a.top_block_size(); space = b.free_space()
        if space == 0:
            return False, 0
        if b.is_empty():
            return True, min(a_block, space)
        b_top = b.top_color()
        if b_top == a_color:
            return True, min(a_block, space)
        return False, 0


# ===================== 绘制与主循环 ===================== #
    def _can_pour(self, a: Bottle, b: Bottle) -> Tuple[bool, int]:
        """判定从瓶子 a 倒向瓶子 b 是否允许，并返回本次可倒入的层数上限。"""
        if a.is_empty():
            return False, 0
        if b.state == "locked":
            return False, 0
        a_color = a.top_color()
        a_block = a.top_block_size()
        space = b.free_space()
        if space == 0:
            return False, 0
        if b.is_empty():
            return True, min(a_block, space)
        b_top = b.top_color()
        if b_top == a_color:
            return True, min(a_block, space)
        return False, 0

class View:
    def __init__(self, game: Game):
        pygame.init()
        self.game = game
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Liquid Merge - 点触演示框架")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.big_font = pygame.font.SysFont(None, 36)

        # 概念瓶尺寸（决定液体区域与命中区域）
        self.bottle_w = BOTTLE_WIDTH
        self.bottle_h = BOTTLE_HEIGHT
        self.layer_gap = SLOT_GAP
        self.ground_y = WINDOW_H - 110

        # 资源加载：瓶子外框图片（可选）
        self.frame_img_original = None
        self.frame_img = None
        self.frame_scaled_h = None
        self.frame_w = None
        if USE_IMAGE_FRAME:
            try:
                self.frame_img_original = pygame.image.load(FRAME_PATH).convert_alpha()
            except Exception as e:
                print(f"[WARN] 无法加载瓶子边框图片 {FRAME_PATH}: {e}")
                self.frame_img_original = None

        # 根据概念尺寸缩放外框图片
        if self.frame_img_original is not None:
            self.frame_scaled_h = int(self.bottle_h * FRAME_HEIGHT_SCALE)
            self.frame_w = FRAME_WIDTH
            self.frame_img = pygame.transform.smoothscale(
                self.frame_img_original, (self.frame_w, self.frame_scaled_h)
            )

        # 按钮区域（胜利后显示）
        self.btn_next_rect: Optional[pygame.Rect] = None
        self.btn_exit_rect: Optional[pygame.Rect] = None
        self.btn_giveup_rect: Optional[pygame.Rect] = None
        self.btn_reset_rect: Optional[pygame.Rect] = None
        self.btn_undo_rect: Optional[pygame.Rect] = None

    # ---------- 布局：每行最多 6，出现两行时平均分配（奇数给下行） ---------- #
    def _compute_grid_layout(self):
        """返回：positions, grid_rect
        规则：
        - 每行最多 8 个；
        - 当总数 > 8（出现两行）时，平均分到两行；若为奇数，则把多出来的 1 个放在**下行**。
          例：9 -> [4,5]；10 -> [5,5]；15 -> [7,8]；16 -> [8,8]
        """
        n = self.game.n
        tile_w = max(self.bottle_w, self.frame_w or self.bottle_w)
        hgap = H_GAP

        positions = []
        if n <= 0:
            return positions, pygame.Rect(0, 0, 0, 0)

        if n <= MAX_PER_ROW:
            # 单行：水平居中，且垂直居中到“两排情况下上下两行之间的中线”
            row_w = n * tile_w + (n - 1) * hgap if n > 0 else 0
            start_x = (WINDOW_W - row_w) // 2
            by_bot = self.ground_y - self.bottle_h
            by_top = by_bot - self.bottle_h - ROW_VGAP
            y_slot = (by_top + by_bot) // 2
            for i in range(n):
                x_slot = start_x + i * (tile_w + hgap)
                x_draw = x_slot + (tile_w - self.bottle_w) // 2
                positions.append({
                    "idx": i, "x_slot": x_slot, "y_slot": y_slot,
                    "x_draw": x_draw, "y_draw": y_slot
                })
            grid_rect = pygame.Rect(start_x, y_slot, row_w, self.bottle_h)
            return positions, grid_rect
        else:
            # 双行：平均分配，奇数加到底行
            top_cnt = min(MAX_PER_ROW, n // 2)
            bot_cnt = n - top_cnt
            # 保护：若超出每行上限（理论上 n<=12 不会），截断到上限
            bot_cnt = min(MAX_PER_ROW, bot_cnt)

            counts = [top_cnt, bot_cnt]

            # 顶部第一行的 y 坐标
            top_y = self.ground_y - self.bottle_h * len(counts) - ROW_VGAP * (len(counts) - 1)

            idx = 0
            left_list, right_list = [], []
            for r, cnt in enumerate(counts):
                row_w = cnt * tile_w + (cnt - 1) * hgap if cnt > 0 else 0
                start_x = (WINDOW_W - row_w) // 2
                y_slot = top_y + r * (self.bottle_h + ROW_VGAP)
                for c in range(cnt):
                    x_slot = start_x + c * (tile_w + hgap)
                    x_draw = x_slot + (tile_w - self.bottle_w) // 2
                    positions.append({
                        "idx": idx, "x_slot": x_slot, "y_slot": y_slot,
                        "x_draw": x_draw, "y_draw": y_slot
                    })
                    idx += 1
                left_list.append(start_x)
                right_list.append(start_x + row_w)

            left = min(left_list)
            right = max(right_list)
            height = self.bottle_h * len(counts) + ROW_VGAP * (len(counts) - 1)
            grid_rect = pygame.Rect(left, top_y, right - left, height)
            return positions, grid_rect

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            self._draw()
            self.clock.tick(FPS)

    def _handle_click(self, pos):
        # 底部控制按钮（随时可用）
        if self.btn_undo_rect and self.btn_undo_rect.collidepoint(pos):
            self.game.undo(); return
        if self.btn_reset_rect and self.btn_reset_rect.collidepoint(pos):
            self.game.reset_to_initial(); return
        if self.btn_giveup_rect and self.btn_giveup_rect.collidepoint(pos):
            self.game.reset_level(); return

        # 若胜利，优先判定右侧面板按钮
        if self.game.win:
            if self.btn_next_rect and self.btn_next_rect.collidepoint(pos):
                self.game.reset_level(); return
            if self.btn_exit_rect and self.btn_exit_rect.collidepoint(pos):
                pygame.quit(); sys.exit(); return

        # 其次判定瓶子点击（多排）
        idx = self._hit_test_bottle(pos)
        if idx is not None:
            self.game.click_bottle(idx)
        return
        if self.btn_giveup_rect and self.btn_giveup_rect.collidepoint(pos):
            self.game.reset_level()
            self.btn_next_rect = None
            self.btn_exit_rect = None
            return
        # 若胜利，优先判定右侧面板按钮
        if self.game.win:
            if self.btn_next_rect and self.btn_next_rect.collidepoint(pos):
                self.game.reset_level()  # 仅重新生成随机一局
                self.btn_next_rect = None
                self.btn_exit_rect = None
                return
            if self.btn_exit_rect and self.btn_exit_rect.collidepoint(pos):
                pygame.quit(); sys.exit(); return
        # 其次判定瓶子点击（多排）
        idx = self._hit_test_bottle(pos)
        if idx is not None:
            self.game.click_bottle(idx)
        return
        # 若胜利，优先判定按钮点击
        if self.game.win:
            if self.btn_next_rect and self.btn_next_rect.collidepoint(pos):
                self.game.reset_level()  # 仅重新生成随机一局
                self.btn_next_rect = None
                self.btn_exit_rect = None
                return
            if self.btn_exit_rect and self.btn_exit_rect.collidepoint(pos):
                pygame.quit(); sys.exit(); return
        # 其次判定瓶子点击（多排）
        idx = self._hit_test_bottle(pos)
        if idx is not None:
            self.game.click_bottle(idx)

    def _hit_test_bottle(self, pos) -> Optional[int]:
        positions, _ = self._compute_grid_layout()
        x, y = pos
        tile_w = max(self.bottle_w, self.frame_w or self.bottle_w)
        for d in positions:
            rect = pygame.Rect(d["x_slot"], d["y_slot"], tile_w, self.bottle_h)
            if rect.collidepoint(x, y):
                return d["idx"]
        return None

    def _draw_liquid_layers(self, x: int, y: int, bottle: Bottle):
        # 液体绘制区域宽度 = 概念宽度 - 两侧内缩
        draw_w = max(1, self.bottle_w - 2 * INNER_PADDING)
        base_x = x + INNER_PADDING

        # 计算每层的高度（考虑层间缝）
        total_gap = self.layer_gap * (LAYERS_PER_BOTTLE - 1)
        usable_h = self.bottle_h - total_gap
        slot_h = usable_h // LAYERS_PER_BOTTLE
        remainder = usable_h - slot_h * LAYERS_PER_BOTTLE  # 把余数加到最底层

        current_top = y + self.bottle_h
        for layer_idx in range(LAYERS_PER_BOTTLE):
            # 自底向上绘制
            i = layer_idx  # 0=底层
            h = slot_h + (remainder if i == 0 else 0)
            top_y = current_top - h
            color_index = bottle.layers[i]
            if color_index is not None:
                pygame.draw.rect(
                    self.screen,
                    self.game.palette[color_index],
                    pygame.Rect(base_x, top_y, draw_w, h),
                    border_radius=4,
                )
            # 更新到下一层顶部（再减去层间缝）
            current_top = top_y - self.layer_gap

    def _draw_bottle(self, idx: int, bottle: Bottle, x: int, y: int):
        # 状态底色（仅作高亮背景，图片外框会覆盖在上层）
        if bottle.state == "lifted":
            pygame.draw.rect(self.screen, LIFTED_TINT, pygame.Rect(x, y, self.bottle_w, self.bottle_h), border_radius=10)
        elif bottle.state == "locked":
            pygame.draw.rect(self.screen, LOCKED_TINT, pygame.Rect(x, y, self.bottle_w, self.bottle_h), border_radius=10)

        # 先画液体，再叠加图片外框
        self._draw_liquid_layers(x, y, bottle)

        if self.frame_img is None:
            # 没有外框图片时，画一个默认描边
            pygame.draw.rect(self.screen, BORDER_COLOR, pygame.Rect(x, y, self.bottle_w, self.bottle_h), width=2, border_radius=10)
        else:
            # 计算图片摆放位置（水平居中，对齐比例控制竖直位置）
            img_x = x + (self.bottle_w - self.frame_w) // 2
            img_y = int(y + self.bottle_h * 0.5 - self.frame_scaled_h * FRAME_ALIGN_RATIO)
            self.screen.blit(self.frame_img, (img_x, img_y))

        # 锁定提示
        if bottle.state == "locked":
            done_text = self.font.render("Ready!", True, (0, 120, 0))
            self.screen.blit(done_text, (x + self.bottle_w//2 - done_text.get_width()//2, y - 28))

        # 底部编号
        label = self.font.render(f"{idx + 1}", True, TEXT_COLOR)
        self.screen.blit(label, (x + self.bottle_w // 2 - label.get_width() // 2, y + self.bottle_h + 6))

    def _draw_win_panel(self, grid_rect: pygame.Rect):
        """绘制胜利后的右侧操作面板（下一关 / 退出游戏）。贴在瓶子区域右侧。"""
        panel_w = 240
        panel_h = grid_rect.h
        panel_x = min(WINDOW_W - panel_w - 16, grid_rect.x + grid_rect.w + 24)
        panel_y = grid_rect.y
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

        pygame.draw.rect(self.screen, PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel_rect, width=1, border_radius=12)

        title = self.big_font.render("You win!", True, (0, 130, 0))
        self.screen.blit(title, (panel_x + 20, panel_y + 20))

        # 按钮
        btn_w = panel_w - 40
        btn_h = 44
        gap = 16
        btn_x = panel_x + 20
        btn_y1 = panel_y + 80
        btn_y2 = btn_y1 + btn_h + gap

        # 下一关（仅重生）
        next_rect = pygame.Rect(btn_x, btn_y1, btn_w, btn_h)
        pygame.draw.rect(self.screen, BTN_BG, next_rect, border_radius=10)
        pygame.draw.rect(self.screen, BTN_BORDER, next_rect, width=2, border_radius=10)
        next_txt = self.font.render("Next level", True, (20, 40, 80))
        self.screen.blit(next_txt, (next_rect.centerx - next_txt.get_width()//2, next_rect.centery - next_txt.get_height()//2))

        # 退出游戏
        exit_rect = pygame.Rect(btn_x, btn_y2, btn_w, btn_h)
        pygame.draw.rect(self.screen, BTN_BG, exit_rect, border_radius=10)
        pygame.draw.rect(self.screen, BTN_BORDER, exit_rect, width=2, border_radius=10)
        exit_txt = self.font.render("Exit", True, (20, 40, 80))
        self.screen.blit(exit_txt, (exit_rect.centerx - exit_txt.get_width()//2, exit_rect.centery - exit_txt.get_height()//2))

        # 记录可点击区域
        self.btn_next_rect = next_rect
        self.btn_exit_rect = exit_rect

    def _draw(self):
        self.screen.fill(BG_COLOR)

        # 顶部提示（仅状态文字）
        hud_rect = pygame.Rect(16, 16, 320, 52)
        pygame.draw.rect(self.screen, INFO_BG, hud_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, hud_rect, width=1, border_radius=8)
        txt = self.font.render(
            f"Level of {self.game.n} bottles", True, TEXT_COLOR
        )
        self.screen.blit(txt, (hud_rect.x + 12, hud_rect.y + 14))

        # 底部：Undo(左) / Reset(中) / Give up(右) 三按钮
        btn_h = 48
        btn_w = 140
        left_margin = 16
        right_margin = 16
        bottom_margin = 16
        y = WINDOW_H - bottom_margin - btn_h

        x_left = left_margin
        x_center = WINDOW_W // 2 - btn_w // 2
        x_right = WINDOW_W - right_margin - btn_w

        # Undo（“反悔”）— 左对齐
        undo_rect = pygame.Rect(x_left, y, btn_w, btn_h)
        pygame.draw.rect(self.screen, BTN_BG, undo_rect, border_radius=10)
        pygame.draw.rect(self.screen, BTN_BORDER, undo_rect, width=2, border_radius=10)
        undo_txt = self.font.render("Undo", True, (20, 40, 80))
        self.screen.blit(undo_txt, (undo_rect.centerx - undo_txt.get_width()//2, undo_rect.centery - undo_txt.get_height()//2))

        # Reset — 居中
        reset_rect = pygame.Rect(x_center, y, btn_w, btn_h)
        pygame.draw.rect(self.screen, BTN_BG, reset_rect, border_radius=10)
        pygame.draw.rect(self.screen, BTN_BORDER, reset_rect, width=2, border_radius=10)
        reset_txt = self.font.render("Reset", True, (20, 40, 80))
        self.screen.blit(reset_txt, (reset_rect.centerx - reset_txt.get_width()//2, reset_rect.centery - reset_txt.get_height()//2))

        # Give up — 右对齐
        give_rect = pygame.Rect(x_right, y, btn_w, btn_h)
        pygame.draw.rect(self.screen, BTN_BG, give_rect, border_radius=10)
        pygame.draw.rect(self.screen, BTN_BORDER, give_rect, width=2, border_radius=10)
        give_txt = self.font.render("Give up", True, (20, 40, 80))
        self.screen.blit(give_txt, (give_rect.centerx - give_txt.get_width()//2, give_rect.centery - give_txt.get_height()//2))

        # 保存热点区域
        self.btn_undo_rect = undo_rect
        self.btn_reset_rect = reset_rect
        self.btn_giveup_rect = give_rect

        positions, grid_rect = self._compute_grid_layout()

        # 绘制所有瓶子（多排）
        for d, b in zip(positions, self.game.bottles):
            lift = LIFT_OFFSET if b.state == "lifted" else 0
            self._draw_bottle(d["idx"], b, d["x_draw"], d["y_draw"] + lift)

        # 胜利后绘制右侧操作面板
        if self.game.win:
            self._draw_win_panel(grid_rect)
        else:
            self.btn_next_rect = None
            self.btn_exit_rect = None

        pygame.display.flip()


# ===================== 入口 ===================== #
def main():
    game = Game(N_BOTTLES)
    view = View(game)
    view.run()


if __name__ == "__main__":
    main()
