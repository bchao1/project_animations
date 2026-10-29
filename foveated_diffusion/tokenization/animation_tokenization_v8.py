"""
Foveated Tokenization Animation - v8
Changes from v7:
  - Phase 3: normal tokenization shown as 3 rows × 12 tokens (same width as foveated row)
  - Foveated tokenization shown as 1 row × 12 tokens
  - SIDE_BIG sized so 12 tokens fill 80% of canvas width
  - Layout makes 3× token count difference immediately obvious
"""
from manim import *
from manimpango import register_font

_FONT_DIR = "/home/brianchc/fonts/cmu-serif"
for _f in ["cmunrm.ttf", "cmunbx.ttf", "cmunti.ttf", "cmunbi.ttf"]:
    register_font(f"{_FONT_DIR}/{_f}")

IMG_PATH = "/home/brianchc/project_animations/foveated_diffusion/data/image.png"
HR_COL   = "#448AFF"
LR_COL   = "#00C853"
FONT     = "CMU Serif"
N        = 6


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rsq(side, color, fill_alpha=0.82):
    return RoundedRectangle(
        width=side, height=side,
        corner_radius=side * 0.10,
        color=color, fill_color=color, fill_opacity=fill_alpha, stroke_width=1.2,
    )


def _gcell(w, h, color, alpha):
    return RoundedRectangle(
        width=w, height=h,
        corner_radius=min(w, h) * 0.10,
        color=color, fill_color=color, fill_opacity=alpha, stroke_width=1.3,
    )


def _normal_grid(img):
    """6×6 HR cells. Each cell: cw*0.90 wide (margin = cw*0.10 = 10% of HR width)."""
    cw, ch = img.width / N, img.height / N
    g = VGroup()
    for i in range(N):
        for j in range(N):
            g.add(_gcell(cw*0.90, ch*0.90, HR_COL, 0.30).move_to(
                [img.get_left()[0] + (j+0.5)*cw,
                 img.get_top()[1]  - (i+0.5)*ch, 0]))
    return g


def _fov_grid(img):
    """Center 2×2 → 4 HR cells; outer 8 macro-blocks → 8 LR cells.
    LR cell drawn at cw*1.90 wide = 2*cw - 0.10*cw, same absolute margin as HR."""
    cw, ch = img.width / N, img.height / N
    margin = cw * 0.10   # same absolute margin for both HR and LR
    hr, lr = VGroup(), VGroup()
    for bi in range(3):
        for bj in range(3):
            if bi == 1 and bj == 1:
                for di in range(2):
                    for dj in range(2):
                        i, j = bi*2+di, bj*2+dj
                        hr.add(_gcell(cw - margin, ch - margin, HR_COL, 0.42).move_to(
                            [img.get_left()[0] + (j+0.5)*cw,
                             img.get_top()[1]  - (i+0.5)*ch, 0]))
            else:
                lr.add(_gcell(cw*2 - margin, ch*2 - margin, LR_COL, 0.30).move_to(
                    [img.get_left()[0] + (bj*2+1)*cw,
                     img.get_top()[1]  - (bi*2+1)*ch, 0]))
    return hr, lr  # hr:4, lr:8 (row-major, skipping center (1,1))


def _seq_targets(n, cx, y, colors, side):
    """Row of n square tokens; gap = 20% of side, centered at (cx, y)."""
    gap   = side * 0.20
    total = n * side + (n - 1) * gap
    x0    = cx - total / 2
    g = VGroup()
    for k in range(n):
        g.add(_rsq(side, colors[k]).move_to(
            [x0 + k*(side+gap) + side/2, y, 0]))
    return g


def _txt(s, font_size=24, color=WHITE, **kw):
    return Text(s, font_size=font_size, color=color, font=FONT, **kw)


# ── Scene ──────────────────────────────────────────────────────────────────────

class FoveatedTokenization(Scene):

    def construct(self):
        FW   = config.frame_width   # ≈ 14.222
        HALF = FW / 2               # ≈ 7.111
        LCX, RCX = -HALF/2, HALF/2

        IMG_W  = HALF * 0.80        # 80% of half-canvas width ≈ 5.689
        IMG_CY = 0.0

        # Phase-3 layout: 12 tokens per row fill 80% of canvas width
        SIDE_BIG = (FW * 0.80) / (12 + 11 * 0.20)   # ≈ 0.801
        PAD      = SIDE_BIG * 0.40
        ROW_SEP  = SIDE_BIG * 1.30                    # vertical distance between row centers

        # 3 normal rows (top group) + 1 foveated row (bottom)
        NORM1_Y    =  2.4
        NORM2_Y    =  NORM1_Y - ROW_SEP
        NORM3_Y    =  NORM1_Y - 2 * ROW_SEP
        FOV_Y      = -2.2
        NORM_LBL_Y =  NORM1_Y + SIDE_BIG/2 + PAD + 0.18
        FOV_LBL_Y  =  FOV_Y   + SIDE_BIG/2 + PAD + 0.18

        FOV_COLORS = [LR_COL]*4 + [HR_COL]*4 + [LR_COL]*4

        # ── Divider + section titles (instant) ────────────────────────────
        div = Line(UP*4, DOWN*4, color=GRAY_C, stroke_width=1.5)
        ll  = _txt("Normal Tokenization",   font_size=26).move_to([LCX, 3.5, 0])
        rl  = _txt("Foveated Tokenization", font_size=26).move_to([RCX, 3.5, 0])
        self.add(div, ll, rl)

        # ── Both images appear simultaneously ─────────────────────────────
        img_l = ImageMobject(IMG_PATH).set_width(IMG_W).move_to([LCX, IMG_CY, 0])
        img_r = ImageMobject(IMG_PATH).set_width(IMG_W).move_to([RCX, IMG_CY, 0])
        self.play(FadeIn(img_l), FadeIn(img_r))
        self.wait(0.3)

        # ── Both grids overlay simultaneously ─────────────────────────────
        lg           = _normal_grid(img_l)
        rh, rl_cells = _fov_grid(img_r)

        self.play(
            LaggedStart(*[FadeIn(c) for c in lg],
                        lag_ratio=0.02, run_time=1.2),
            LaggedStart(*([FadeIn(c) for c in rh] + [FadeIn(c) for c in rl_cells]),
                        lag_ratio=0.03, run_time=1.3),
        )
        self.wait(0.4)

        # ── Legend: full canvas width, below both images (instant) ────────
        sq_side = 0.24
        hr_sq   = _rsq(sq_side, HR_COL)
        hr_lbl  = _txt("High-resolution tokens", font_size=20, color=WHITE)
        lr_sq   = _rsq(sq_side, LR_COL)
        lr_lbl  = _txt("Low-resolution tokens",  font_size=20, color=WHITE)
        hr_item = VGroup(hr_sq, hr_lbl.next_to(hr_sq, RIGHT, buff=0.2))
        lr_item = VGroup(lr_sq, lr_lbl.next_to(lr_sq, RIGHT, buff=0.2))
        legend  = VGroup(hr_item, lr_item).arrange(RIGHT, buff=0.9)
        legend_y = img_l.get_bottom()[1] - 0.40
        legend.move_to([0, legend_y, 0])
        self.add(legend)
        self.wait(3.0)

        # ── Phase 3: fade out images + UI ─────────────────────────────────
        self.play(
            FadeOut(img_l), FadeOut(img_r),
            FadeOut(div),   FadeOut(ll),   FadeOut(rl),
            FadeOut(legend),
        )

        # ── Single animation: all tokens → 3 normal rows + 1 foveated row ─
        # Normal grid cells (row-major): lg[0..11] → row1, [12..23] → row2, [24..35] → row3
        norm_row1 = _seq_targets(12, 0, NORM1_Y, [HR_COL]*12, SIDE_BIG)
        norm_row2 = _seq_targets(12, 0, NORM2_Y, [HR_COL]*12, SIDE_BIG)
        norm_row3 = _seq_targets(12, 0, NORM3_Y, [HR_COL]*12, SIDE_BIG)
        fov_big   = _seq_targets(12, 0, FOV_Y,   FOV_COLORS,  SIDE_BIG)

        self.play(
            *[Transform(lg[k],          norm_row1[k])  for k in range(12)],
            *[Transform(lg[12+k],       norm_row2[k])  for k in range(12)],
            *[Transform(lg[24+k],       norm_row3[k])  for k in range(12)],
            *[Transform(rl_cells[k],    fov_big[k])    for k in range(4)],
            *[Transform(rh[k],          fov_big[4+k])  for k in range(4)],
            *[Transform(rl_cells[4+k],  fov_big[8+k])  for k in range(4)],
            run_time=1.8,
        )

        # Labels above each group (instant)
        norm_title = _txt("Normal tokenization: 36 tokens",   font_size=24, color=WHITE).move_to([0, NORM_LBL_Y, 0])
        fov_title  = _txt("Foveated tokenization: 12 tokens", font_size=24, color=WHITE).move_to([0, FOV_LBL_Y,  0])
        self.add(norm_title, fov_title)
        self.wait(3.0)
