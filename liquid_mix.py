import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame

# ===================== 可调参数 ===================== #
N_BOTTLES = 6               # 初始瓶子
LAYERS_PER_BOTTLE = 4       # 填充度
COLOR_VARIETY = (4, 6)      # 颜色种类区间 [min, max]
WINDOW_W, WINDOW_H = 1000, 620  # 窗口长宽
FPS = 60
RANDOM_SEED = None          # 设为整数以复现实验，如 42
TOTAL_CAP = 8               # 瓶子上限不超过 8

# ——瓶子尺寸—— #
BOTTLE_WIDTH = 90           # 瓶子宽度
BOTTLE_HEIGHT = 360         # 瓶子高度
INNER_PADDING = 5           # 小于图片宽度
SLOT_GAP = 6                # 每层间隔

# ——瓶子图片与概念位置关系—— #
USE_IMAGE_FRAME = True                      # 本地图片
FRAME_PATH = "bottle_frame.png"             # 透明 PNG
FRAME_WIDTH = 100                           # 缩放宽度
FRAME_HEIGHT_SCALE = 1.16                   # 缩放比例
FRAME_ALIGN_RATIO = 0.52                    # 对齐位置

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
    (25, 118, 210),  # blue
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
        # ☆ 瓶子总数限制 ≤ TOTAL_CAP(8)
        self.n = max(1, min(TOTAL_CAP, n_bottles))
        self.palette_full = PALETTE[:]
        self.palette = self.palette_full  # 运行时实际截取的颜色集
        self.bottles: List[Bottle] = []
        self.selected: Optional[int] = None  # 抬起的瓶子索引
        self.message = ""
        self.win = False
        self._init_bottles()

    def _calc_empty_bottles(self, n: int) -> int:
        """
        空瓶数量小于总瓶子一半
        """
        if n <= 1:
            return 0
        max_under_half = max(0, (n // 2) - 1)
        if n <= 6:
            return min(2, max_under_half)
        else:
            return max_under_half

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

        # 确定颜色数 k（保证每色数量为 4 的倍数，且颜色数在区间内，容量不足时自动缩减）
        min_colors, max_colors = COLOR_VARIETY
        k_upper_bound = min(max_colors, total_groups, len(self.palette_full))
        if total_groups >= min_colors:
            k = random.randint(min_colors, k_upper_bound)
        else:
            k = total_groups
        self.palette = random.sample(self.palette_full, k)  # ☆ 每局随机抽取 k 种颜色（顺序也随机）

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

    def reset_level(self):
        """新关卡随机"""
        self.selected = None
        self.win = False
        self._init_bottles()

    # ----------- 规则辅助 ----------- #
    def _refresh_locks(self):
        for b in self.bottles:
            if b.state != "locked" and b.is_uniform_full():
                b.state = "locked"

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
        else:
            a.state = "ground"; b.state = "lifted"; self.selected = self.bottles.index(b)

    def _can_pour(self, a: Bottle, b: Bottle) -> Tuple[bool, int]:
        if a.is_empty():
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
class View:
    def __init__(self, game: Game):
        pygame.init()
        self.game = game
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Liquid Merge - 小游戏")
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
        # 若胜利，优先判定按钮点击
        if self.game.win:
            if self.btn_next_rect and self.btn_next_rect.collidepoint(pos):
                self.game.reset_level()  # ☆ 仅重新生成随机一局
                self.btn_next_rect = None
                self.btn_exit_rect = None
                return
            if self.btn_exit_rect and self.btn_exit_rect.collidepoint(pos):
                pygame.quit(); sys.exit(); return
        # 其次判定瓶子点击（单排）
        idx = self._hit_test_bottle(pos)
        if idx is not None:
            self.game.click_bottle(idx)

    def _hit_test_bottle(self, pos) -> Optional[int]:
        n = self.game.n
        tile_w = max(self.bottle_w, self.frame_w or self.bottle_w)
        hgap = 22
        total_w = n * tile_w + (n - 1) * hgap
        start_x = (WINDOW_W - total_w) // 2
        x, y = pos
        by = self.ground_y - self.bottle_h
        for i in range(n):
            slot_left = start_x + i * (tile_w + hgap)
            rect = pygame.Rect(slot_left, by, tile_w, self.bottle_h)
            if rect.collidepoint(x, y):
                return i
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

    def _draw_win_panel(self, start_x: int, total_w: int, by: int):
        """通关后操作"""
        panel_w = 240
        panel_h = self.bottle_h
        # 面板尽量贴在瓶子区域右侧；若空间不足则靠右
        panel_x = min(WINDOW_W - panel_w - 16, start_x + total_w + 24)
        panel_y = by
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

        # 顶部提示
        hud_rect = pygame.Rect(16, 16, 360, 52)
        pygame.draw.rect(self.screen, INFO_BG, hud_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, hud_rect, width=1, border_radius=8)
        txt = self.font.render(
            f"Total {self.game.n} bottles (cap=8)", True, TEXT_COLOR
        )
        self.screen.blit(txt, (hud_rect.x + 12, hud_rect.y + 14))

        if self.game.win:
            win_text = self.big_font.render("Win!", True, (0, 150, 0))
            self.screen.blit(win_text, (WINDOW_W // 2 - win_text.get_width() // 2, 70))

        # 单排居中
        n = self.game.n
        tile_w = max(self.bottle_w, self.frame_w or self.bottle_w)
        hgap = 22
        total_w = n * tile_w + (n - 1) * hgap
        start_x = (WINDOW_W - total_w) // 2
        by = self.ground_y - self.bottle_h

        for i, b in enumerate(self.game.bottles):
            slot_left = start_x + i * (tile_w + hgap)
            bx = slot_left + (tile_w - self.bottle_w) // 2
            lift_offset = -18 if b.state == "lifted" else 0
            self._draw_bottle(i, b, bx, by + lift_offset)

        # 胜利后绘制右侧操作面板
        if self.game.win:
            self._draw_win_panel(start_x, total_w, by)
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
