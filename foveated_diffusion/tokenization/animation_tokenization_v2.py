"""
Foveated Tokenization Animation - v2
Conforms to MUST CONFORM guidelines:
  - No overlapping visuals
  - Text appears instantly (no Write/FadeIn for text)
  - CMU Serif font
  - HR tokens = blue, LR tokens = green
  - Square tokens with 10% rounded corners
  - Length comparison shows only token counts
"""
from manim import *

IMG_PATH = "/home/brianchc/project_animations/foveated_diffusion/data/image.png"
HR_COL   = "#448AFF"   # blue  – high-resolution tokens
LR_COL   = "#00C853"   # green – low-resolution tokens
FONT     = "CMU Serif"
N        = 6           # grid dimension


# ── Token / cell helpers ───────────────────────────────────────────────────────

def _rsq(side, color, fill_alpha=0.82):
    """Exactly-square rounded token (corner_radius = 10% of side)."""
    return RoundedRectangle(
        width=side, height=side,
        corner_radius=side * 0.10,
        color=color, fill_color=color, fill_opacity=fill_alpha, stroke_width=1.2,
    )


def _grid_cell(w, h, color, fill_alpha):
    """Grid-overlay cell: rounded rectangle following image cell dimensions."""
    return RoundedRectangle(
        width=w, height=h,
        corner_radius=min(w, h) * 0.10,
        color=color, fill_color=color, fill_opacity=fill_alpha, stroke_width=1.3,
    )


def _normal_grid(img):
    """6×6 uniform HR grid overlaid on img (36 cells, row-major)."""
    cw = img.width  / N
    ch = img.height / N
    cells = VGroup()
    for i in range(N):
        for j in range(N):
            cells.add(_grid_cell(cw * 0.90, ch * 0.90, HR_COL, 0.30).move_to(
                [img.get_left()[0] + (j + 0.5)*cw,
                 img.get_top()[1]  - (i + 0.5)*ch, 0]))
    return cells  # 36 cells, row-major


def _fov_grid(img):
    """Center 2×2 macro-block → 4 HR cells; 8 outer macro-blocks → 8 LR cells."""
    cw = img.width  / N
    ch = img.height / N
    hr, lr = VGroup(), VGroup()
    for bi in range(3):
        for bj in range(3):
            if bi == 1 and bj == 1:                 # center → 4 individual HR cells
                for di in range(2):
                    for dj in range(2):
                        i, j = bi*2+di, bj*2+dj
                        hr.add(_grid_cell(cw*0.90, ch*0.90, HR_COL, 0.42).move_to(
                            [img.get_left()[0] + (j+0.5)*cw,
                             img.get_top()[1]  - (i+0.5)*ch, 0]))
            else:                                    # peripheral → 1 LR 2×2 cell
                lr.add(_grid_cell(cw*2*0.88, ch*2*0.88, LR_COL, 0.30).move_to(
                    [img.get_left()[0] + (bj*2+1)*cw,
                     img.get_top()[1]  - (bi*2+1)*ch, 0]))
    # lr order (row-major, skipping center (1,1)):
    # lr[0..2]: top row   lr[3]: mid-left   lr[4]: mid-right   lr[5..7]: bot row
    return hr, lr  # VGroup(4), VGroup(8)


def _make_seq(n, cx, y, colors, side):
    """Row of n square tokens centered at (cx, y). Gap = 10% of side."""
    gap   = side * 0.10
    total = n * side + (n - 1) * gap
    x0    = cx - total / 2
    g = VGroup()
    for k in range(n):
        g.add(_rsq(side, colors[k]).move_to(
            [x0 + k*(side + gap) + side/2, y, 0]))
    return g


def _txt(text, font_size=24, color=WHITE, **kwargs):
    """Text that uses CMU Serif; added with self.add() to appear instantly."""
    return Text(text, font_size=font_size, color=color, font=FONT, **kwargs)


# ── Scene ──────────────────────────────────────────────────────────────────────

class FoveatedTokenization(Scene):

    def construct(self):
        # Layout constants
        LCX, RCX   = -3.5, 3.5
        IMG_H      = 2.8          # initial image height
        IMG_TOP_Y  = 2.1          # image center-y after move-up
        IMG_H2     = IMG_H * 0.57 # image height after move-up  (≈1.596)
        SEQ_Y      = -2.6         # y-center of flattened sequences
        LBL_Y      = -3.1         # y-center of token-count labels below sequences
        SIDE_SM    = 0.140        # token side in split-screen sequence view (gap=10%→0.014)

        FOV_COLORS = [LR_COL]*4 + [HR_COL]*4 + [LR_COL]*4  # macro-block row-major

        # ── Divider + section titles (instant) ────────────────────────────
        div = Line(UP*4, DOWN*4, color=GRAY_C, stroke_width=1.5)
        ll  = _txt("Normal Tokenization",   font_size=24, weight=BOLD).move_to([LCX, 3.5, 0])
        rl  = _txt("Foveated Tokenization", font_size=24, weight=BOLD).move_to([RCX, 3.5, 0])
        self.add(div, ll, rl)

        # ── Images appear ─────────────────────────────────────────────────
        img_l = ImageMobject(IMG_PATH).set_height(IMG_H).move_to([LCX, 0.2, 0])
        img_r = ImageMobject(IMG_PATH).set_height(IMG_H).move_to([RCX, 0.2, 0])
        self.play(FadeIn(img_l), FadeIn(img_r))
        self.wait(0.3)

        # ── Overlay grids ─────────────────────────────────────────────────
        lg            = _normal_grid(img_l)
        rh, rl_cells  = _fov_grid(img_r)

        self.play(LaggedStart(*[FadeIn(c) for c in lg],
                              lag_ratio=0.02, run_time=1.0))
        self.play(LaggedStart(*[FadeIn(c) for c in rh],
                              lag_ratio=0.06, run_time=0.5))
        self.play(LaggedStart(*[FadeIn(c) for c in rl_cells],
                              lag_ratio=0.06, run_time=0.7))
        self.wait(0.4)

        # ── Images move to top; grid flattens to sequence ─────────────────
        self.play(
            img_l.animate.set_height(IMG_H2).move_to([LCX, IMG_TOP_Y, 0]),
            img_r.animate.set_height(IMG_H2).move_to([RCX, IMG_TOP_Y, 0]),
            run_time=0.75,
        )

        # Build sequence target positions
        norm_seq = _make_seq(36, LCX, SEQ_Y, [HR_COL]*36, SIDE_SM)
        fov_seq  = _make_seq(12, RCX, SEQ_Y, FOV_COLORS,  SIDE_SM)

        # Transform grid cells → sequence squares
        # Normal: lg[k] → norm_seq[k]  (row-major, 36 cells)
        # Fov:    rl_cells[0..3] → fov_seq[0..3]   (first 4 LR)
        #         rh[0..3]       → fov_seq[4..7]   (4 HR)
        #         rl_cells[4..7] → fov_seq[8..11]  (last 4 LR)
        self.play(AnimationGroup(
            *[Transform(lg[k],          norm_seq[k])   for k in range(36)],
            *[Transform(rl_cells[k],    fov_seq[k])    for k in range(4)],
            *[Transform(rh[k],          fov_seq[4+k])  for k in range(4)],
            *[Transform(rl_cells[4+k],  fov_seq[8+k])  for k in range(4)],
            lag_ratio=0.012,
        ), run_time=2.0)

        # Token count labels appear instantly (below sequences, no overlap)
        n_lbl = _txt("36 tokens", font_size=20, color=HR_COL).move_to([LCX, LBL_Y, 0])
        f_lbl = _txt("12 tokens", font_size=20, color=LR_COL).move_to([RCX, LBL_Y, 0])
        self.add(n_lbl, f_lbl)
        self.wait(0.9)

        # ── Phase 3: full-canvas length comparison ────────────────────────
        self.play(
            FadeOut(img_l), FadeOut(img_r),
            FadeOut(div),   FadeOut(ll),   FadeOut(rl),
            FadeOut(n_lbl), FadeOut(f_lbl),
        )

        # Normal: 36 square tokens fill 80% of canvas width
        # With gap=10%: total = 36*s + 35*(s*0.10) = s*(36 + 3.5) = s*39.5
        FW       = config.frame_width   # ≈ 14.222
        SIDE_BIG = (FW * 0.80) / (36 + 35 * 0.10)   # ≈ 0.288

        norm_big = _make_seq(36, 0,  1.2, [HR_COL]*36, SIDE_BIG)
        fov_big  = _make_seq(12, 0, -1.0, FOV_COLORS,  SIDE_BIG)

        # Expand normal sequence (lg → norm_big)
        self.play(
            *[Transform(lg[k], norm_big[k]) for k in range(36)],
            run_time=1.2,
        )
        # Label appears instantly after expansion
        norm_title = _txt("Normal: 36 tokens", font_size=26, color=HR_COL).move_to([0, 2.2, 0])
        self.add(norm_title)

        # Expand foveated sequence
        self.play(
            *[Transform(rl_cells[k],   fov_big[k])   for k in range(4)],
            *[Transform(rh[k],         fov_big[4+k]) for k in range(4)],
            *[Transform(rl_cells[4+k], fov_big[8+k]) for k in range(4)],
            run_time=1.2,
        )
        # Label appears instantly after expansion
        fov_title = _txt("Foveated: 12 tokens", font_size=26, color=LR_COL).move_to([0, -2.1, 0])
        self.add(fov_title)

        self.wait(3.0)
