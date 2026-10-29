"""
Foveated Tokenization Animation - v6
Changes from v5:
  - Both images + grids animate simultaneously
  - LR grid cells use same absolute margin as HR (cw*1.90 wide, not cw*2*0.88)
  - Legend centered at full canvas width (x=0), below both images
  - Final labels: "Normal tokenization: 36 tokens" / "Foveated tokenization: 12 tokens"
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
                # LR spans 2×2 cells; subtract same margin on each outer edge
                lr.add(_gcell(cw*2 - margin, ch*2 - margin, LR_COL, 0.30).move_to(
                    [img.get_left()[0] + (bj*2+1)*cw,
                     img.get_top()[1]  - (bi*2+1)*ch, 0]))
    return hr, lr  # hr:4, lr:8 (row-major, skipping center (1,1))


def _seq_targets(n, cx, y, colors, side):
    """Row of n square tokens; gap = 10% of side, centered at (cx, y)."""
    gap   = side * 0.10
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

        # Phase-3 layout
        SIDE_BIG   = (FW * 0.80) / (36 + 35 * 0.10)  # ≈ 0.288
        PAD        = SIDE_BIG * 0.40
        NORM_Y     =  1.7
        FOV_Y      = -2.4
        NORM_LBL_Y = NORM_Y + SIDE_BIG/2 + PAD + 0.18
        FOV_LBL_Y  = FOV_Y  + SIDE_BIG/2 + PAD + 0.18

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
        lg            = _normal_grid(img_l)
        rh, rl_cells  = _fov_grid(img_r)

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
        hr_lbl  = _txt("High-resolution token", font_size=20, color=WHITE)
        lr_sq   = _rsq(sq_side, LR_COL)
        lr_lbl  = _txt("Low-resolution token",  font_size=20, color=WHITE)
        hr_item = VGroup(hr_sq, hr_lbl.next_to(hr_sq, RIGHT, buff=0.2))
        lr_item = VGroup(lr_sq, lr_lbl.next_to(lr_sq, RIGHT, buff=0.2))
        legend  = VGroup(hr_item, lr_item).arrange(RIGHT, buff=0.9)
        # Place below both images (both have the same bottom y)
        legend_y = img_l.get_bottom()[1] - 0.40
        legend.move_to([0, legend_y, 0])
        self.add(legend)
        self.wait(0.8)

        # ── Phase 3: fade out images + UI ─────────────────────────────────
        self.play(
            FadeOut(img_l), FadeOut(img_r),
            FadeOut(div),   FadeOut(ll),   FadeOut(rl),
            FadeOut(legend),
        )

        # ── Single animation: all tokens → final expanded positions ───────
        norm_big = _seq_targets(36, 0, NORM_Y, [HR_COL]*36, SIDE_BIG)
        fov_big  = _seq_targets(12, 0, FOV_Y,  FOV_COLORS,  SIDE_BIG)

        self.play(
            *[Transform(lg[k],          norm_big[k])   for k in range(36)],
            *[Transform(rl_cells[k],    fov_big[k])    for k in range(4)],
            *[Transform(rh[k],          fov_big[4+k])  for k in range(4)],
            *[Transform(rl_cells[4+k],  fov_big[8+k])  for k in range(4)],
            run_time=1.8,
        )

        # White labels above each sequence (instant)
        norm_title = _txt("Normal tokenization: 36 tokens",   font_size=24, color=WHITE).move_to([0, NORM_LBL_Y, 0])
        fov_title  = _txt("Foveated tokenization: 12 tokens", font_size=24, color=WHITE).move_to([0, FOV_LBL_Y,  0])
        self.add(norm_title, fov_title)
        self.wait(3.0)
