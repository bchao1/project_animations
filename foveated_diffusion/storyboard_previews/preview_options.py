"""
Phase-3 layout option previews — static PNG frames.
Each scene shows one candidate layout for the token-sequence comparison phase.

Options:
  OptionA  — current v7: two centered rows (normal top, foveated bottom), same SIDE_BIG
  OptionB  — scale-to-fill: each row independently fills 80% (different token sizes)
  OptionC  — left-aligned + vertical bar chart on right
  OptionE  — normal split into 2 rows of 18; foveated as 1 row of 12 (RECOMMENDED)
  OptionF  — single-size rows + annotation band with bracket in the middle
"""
from manim import *
from manimpango import register_font

_FONT_DIR = "/home/brianchc/fonts/cmu-serif"
for _f in ["cmunrm.ttf", "cmunbx.ttf", "cmunti.ttf", "cmunbi.ttf"]:
    register_font(f"{_FONT_DIR}/{_f}")

HR_COL = "#448AFF"
LR_COL = "#00C853"
FONT   = "CMU Serif"
FOV_COLORS = [LR_COL]*4 + [HR_COL]*4 + [LR_COL]*4


def _rsq(side, color, fill_alpha=0.82):
    return RoundedRectangle(
        width=side, height=side,
        corner_radius=side * 0.10,
        color=color, fill_color=color, fill_opacity=fill_alpha, stroke_width=1.2,
    )


def _txt(s, font_size=24, color=WHITE, **kw):
    return Text(s, font_size=font_size, color=color, font=FONT, **kw)


def _row(n, cx, y, colors, side, gap_ratio=0.20):
    """Horizontal row of n square tokens centered at (cx, y)."""
    gap   = side * gap_ratio
    total = n * side + (n - 1) * gap
    x0    = cx - total / 2
    g = VGroup()
    for k in range(n):
        g.add(_rsq(side, colors[k]).move_to(
            [x0 + k * (side + gap) + side / 2, y, 0]))
    return g


def _option_label(letter, title, subtitle=""):
    """Top-left badge identifying the option."""
    t = _txt(f"Option {letter}:  {title}", font_size=22, color=YELLOW)
    t.to_corner(UL, buff=0.3)
    if subtitle:
        s = _txt(subtitle, font_size=17, color=GRAY_A)
        s.next_to(t, DOWN, aligned_edge=LEFT, buff=0.08)
        return VGroup(t, s)
    return t


# ── Option A ── current v7 ─────────────────────────────────────────────────────
class OptionA(Scene):
    def construct(self):
        FW = config.frame_width
        SIDE = (FW * 0.80) / (36 + 35 * 0.20)
        PAD  = SIDE * 0.40

        NORM_Y, FOV_Y = 1.7, -2.4
        NORM_LBL_Y = NORM_Y + SIDE / 2 + PAD + 0.18
        FOV_LBL_Y  = FOV_Y  + SIDE / 2 + PAD + 0.18

        norm = _row(36, 0, NORM_Y, [HR_COL]*36, SIDE)
        fov  = _row(12, 0, FOV_Y,  FOV_COLORS,  SIDE)

        norm_lbl = _txt("Normal tokenization: 36 tokens",   font_size=24).move_to([0, NORM_LBL_Y, 0])
        fov_lbl  = _txt("Foveated tokenization: 12 tokens", font_size=24).move_to([0, FOV_LBL_Y,  0])

        badge = _option_label("A", "Current v7 — two centered rows, same token size",
                              "⚠  Foveated row only fills ~1/3 of width; large dead zones above and below")

        self.add(norm, fov, norm_lbl, fov_lbl, badge)


# ── Option B ── scale-to-fill ──────────────────────────────────────────────────
class OptionB(Scene):
    def construct(self):
        FW = config.frame_width
        SIDE_N = (FW * 0.80) / (36 + 35 * 0.20)   # small — 36 tokens fill 80%
        SIDE_F = (FW * 0.80) / (12 + 11 * 0.20)   # large — 12 tokens fill 80%
        PAD_N  = SIDE_N * 0.40
        PAD_F  = SIDE_F * 0.40

        NORM_Y, FOV_Y = 1.3, -1.5
        NORM_LBL_Y = NORM_Y + SIDE_N / 2 + PAD_N + 0.18
        FOV_LBL_Y  = FOV_Y  + SIDE_F / 2 + PAD_F + 0.18

        norm = _row(36, 0, NORM_Y, [HR_COL]*36, SIDE_N)
        fov  = _row(12, 0, FOV_Y,  FOV_COLORS,  SIDE_F)

        norm_lbl = _txt("Normal tokenization: 36 tokens",   font_size=24).move_to([0, NORM_LBL_Y, 0])
        fov_lbl  = _txt("Foveated tokenization: 12 tokens", font_size=24).move_to([0, FOV_LBL_Y,  0])

        badge = _option_label("B", "Scale-to-fill — each row independently fills 80% width",
                              "⚠  Token sizes differ → length comparison harder; foveated tokens appear larger (which is actually meaningful)")

        self.add(norm, fov, norm_lbl, fov_lbl, badge)


# ── Option C ── left-aligned + bar chart ──────────────────────────────────────
class OptionC(Scene):
    def construct(self):
        FW = config.frame_width
        SIDE = (FW * 0.80) / (36 + 35 * 0.20)
        PAD  = SIDE * 0.40

        # Left-align: start at frame left + 5% margin
        LEFT_X = -FW / 2 + FW * 0.05
        NORM_Y, FOV_Y = 1.3, -0.5
        NORM_LBL_Y = NORM_Y + SIDE / 2 + PAD + 0.18
        FOV_LBL_Y  = FOV_Y  + SIDE / 2 + PAD + 0.18

        gap  = SIDE * 0.20
        total_norm = 36 * SIDE + 35 * gap
        total_fov  = 12 * SIDE + 11 * gap

        norm = _row(36, LEFT_X + total_norm / 2, NORM_Y, [HR_COL]*36, SIDE)
        fov  = _row(12, LEFT_X + total_fov  / 2, FOV_Y,  FOV_COLORS,  SIDE)

        norm_lbl = _txt("Normal tokenization: 36 tokens",   font_size=24)\
            .move_to([LEFT_X + total_norm / 2, NORM_LBL_Y, 0])
        fov_lbl  = _txt("Foveated tokenization: 12 tokens", font_size=24)\
            .move_to([LEFT_X + total_fov  / 2, FOV_LBL_Y,  0])

        # Bar chart on the right
        BAR_CX   = FW / 2 - 1.2
        BAR_W    = 0.7
        BAR_MAX  = 3.5
        bar_h_n  = BAR_MAX
        bar_h_f  = BAR_MAX * (12 / 36)
        BAR_BASE = -2.0

        bar_n = Rectangle(width=BAR_W, height=bar_h_n, color=HR_COL,
                          fill_color=HR_COL, fill_opacity=0.7)\
            .move_to([BAR_CX - 0.5, BAR_BASE + bar_h_n / 2, 0])
        bar_f = Rectangle(width=BAR_W, height=bar_h_f, color=LR_COL,
                          fill_color=LR_COL, fill_opacity=0.7)\
            .move_to([BAR_CX + 0.5, BAR_BASE + bar_h_f / 2, 0])
        lbl_n = _txt("36", font_size=20).next_to(bar_n, UP, buff=0.1)
        lbl_f = _txt("12", font_size=20).next_to(bar_f, UP, buff=0.1)
        lbl_axis_n = _txt("Normal",   font_size=16).next_to(bar_n, DOWN, buff=0.1)
        lbl_axis_f = _txt("Foveated", font_size=16).next_to(bar_f, DOWN, buff=0.1)

        badge = _option_label("C", "Left-aligned rows + bar chart on right",
                              "Left-alignment makes length gap obvious; bar chart fills right space")

        self.add(norm, fov, norm_lbl, fov_lbl,
                 bar_n, bar_f, lbl_n, lbl_f, lbl_axis_n, lbl_axis_f, badge)


# ── Option E ── 2×18 normal rows + 1×12 foveated row ─────────────────────────
class OptionE(Scene):
    def construct(self):
        FW = config.frame_width
        SIDE = (FW * 0.80) / (18 + 17 * 0.20)  # 18 tokens per row fill 80%
        PAD  = SIDE * 0.40
        GAP  = SIDE * 0.20
        ROW_SEP = SIDE * 1.30   # vertical gap between the two normal rows

        # Three rows: norm row 1 (top), norm row 2, foveated
        NORM1_Y =  2.5
        NORM2_Y =  NORM1_Y - ROW_SEP
        FOV_Y   = -1.5

        NORM_LBL_Y = NORM1_Y + SIDE / 2 + PAD + 0.18
        FOV_LBL_Y  = FOV_Y   + SIDE / 2 + PAD + 0.18

        norm1 = _row(18, 0, NORM1_Y, [HR_COL]*18, SIDE)
        norm2 = _row(18, 0, NORM2_Y, [HR_COL]*18, SIDE)
        fov   = _row(12, 0, FOV_Y,   FOV_COLORS,  SIDE)

        norm_lbl = _txt("Normal tokenization: 36 tokens",   font_size=24).move_to([0, NORM_LBL_Y, 0])
        fov_lbl  = _txt("Foveated tokenization: 12 tokens", font_size=24).move_to([0, FOV_LBL_Y,  0])

        badge = _option_label("E  ★ RECOMMENDED",
                              "Normal = 2 rows × 18, Foveated = 1 row × 12 — same token size",
                              "Canvas is well-filled; foveated visibly shorter than one normal row")

        self.add(norm1, norm2, fov, norm_lbl, fov_lbl, badge)


# ── Option F ── annotation band with bracket ───────────────────────────────────
class OptionF(Scene):
    def construct(self):
        FW = config.frame_width
        SIDE = (FW * 0.80) / (36 + 35 * 0.20)
        PAD  = SIDE * 0.40

        NORM_Y =  2.4
        FOV_Y  = -0.8
        NORM_LBL_Y = NORM_Y + SIDE / 2 + PAD + 0.18
        FOV_LBL_Y  = FOV_Y  + SIDE / 2 + PAD + 0.18

        norm = _row(36, 0, NORM_Y, [HR_COL]*36, SIDE)
        fov  = _row(12, 0, FOV_Y,  FOV_COLORS,  SIDE)

        norm_lbl = _txt("Normal tokenization: 36 tokens",   font_size=24).move_to([0, NORM_LBL_Y, 0])
        fov_lbl  = _txt("Foveated tokenization: 12 tokens", font_size=24).move_to([0, FOV_LBL_Y,  0])

        # Annotation band in middle
        MID_Y = (NORM_Y + FOV_Y) / 2

        # Measure actual row extents
        gap  = SIDE * 0.20
        norm_half = (36 * SIDE + 35 * gap) / 2
        fov_half  = (12 * SIDE + 11 * gap) / 2

        # Horizontal bracket showing foveated width vs normal width
        brace_y   = MID_Y + 0.25
        line_n = Line([-norm_half, brace_y, 0], [norm_half, brace_y, 0],
                      color=HR_COL, stroke_width=2)
        tick_nl = Line([-norm_half, brace_y - 0.15, 0], [-norm_half, brace_y + 0.15, 0],
                       color=HR_COL, stroke_width=2)
        tick_nr = Line([ norm_half, brace_y - 0.15, 0], [ norm_half, brace_y + 0.15, 0],
                       color=HR_COL, stroke_width=2)
        lbl_n = _txt("36 tokens", font_size=18, color=HR_COL)\
            .move_to([0, brace_y + 0.30, 0])

        brace_y2  = MID_Y - 0.25
        line_f = Line([-fov_half, brace_y2, 0], [fov_half, brace_y2, 0],
                      color=LR_COL, stroke_width=2)
        tick_fl = Line([-fov_half, brace_y2 - 0.15, 0], [-fov_half, brace_y2 + 0.15, 0],
                       color=LR_COL, stroke_width=2)
        tick_fr = Line([ fov_half, brace_y2 - 0.15, 0], [ fov_half, brace_y2 + 0.15, 0],
                       color=LR_COL, stroke_width=2)
        lbl_f = _txt("12 tokens", font_size=18, color=LR_COL)\
            .move_to([0, brace_y2 - 0.30, 0])

        badge = _option_label("F", "Annotation band with span brackets in the middle",
                              "Fills vertical space with a visual ratio indicator")

        self.add(norm, fov, norm_lbl, fov_lbl,
                 line_n, tick_nl, tick_nr, lbl_n,
                 line_f, tick_fl, tick_fr, lbl_f,
                 badge)
