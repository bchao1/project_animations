"""
Foveated Tokenization Animation - v4
Storyboard:
  Phase 1: Left half — image + 6×6 HR grid
  Phase 2: Right half — image + foveated grid
  Phase 3: Images disappear; tokens morph in ONE single animation directly to
           final expanded positions (normal top, foveated bottom, 80% canvas).

MUST CONFORM:
  - No overlapping visuals
  - Text appears instantly (self.add)
  - CMU Serif font (registered via manimpango from local files)
  - HR = blue, LR = green
  - Square tokens, 10% rounded corners, gap = 10% of side
  - Length labels: token counts only
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
    """Square token with 10% rounded corners."""
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
    """6×6 HR cells, row-major (36 total)."""
    cw, ch = img.width / N, img.height / N
    g = VGroup()
    for i in range(N):
        for j in range(N):
            g.add(_gcell(cw*0.90, ch*0.90, HR_COL, 0.30).move_to(
                [img.get_left()[0] + (j+0.5)*cw,
                 img.get_top()[1]  - (i+0.5)*ch, 0]))
    return g


def _fov_grid(img):
    """Center 2×2 → 4 HR cells; outer 8 macro-blocks → 8 LR cells."""
    cw, ch = img.width / N, img.height / N
    hr, lr = VGroup(), VGroup()
    for bi in range(3):
        for bj in range(3):
            if bi == 1 and bj == 1:
                for di in range(2):
                    for dj in range(2):
                        i, j = bi*2+di, bj*2+dj
                        hr.add(_gcell(cw*0.90, ch*0.90, HR_COL, 0.42).move_to(
                            [img.get_left()[0] + (j+0.5)*cw,
                             img.get_top()[1]  - (i+0.5)*ch, 0]))
            else:
                lr.add(_gcell(cw*2*0.88, ch*2*0.88, LR_COL, 0.30).move_to(
                    [img.get_left()[0] + (bj*2+1)*cw,
                     img.get_top()[1]  - (bi*2+1)*ch, 0]))
    # lr order: (0,0)(0,1)(0,2)(1,0)(1,2)(2,0)(2,1)(2,2)
    return hr, lr


def _seq_targets(n, cx, y, colors, side):
    """Pre-built sequence of square tokens; gap = 10% of side."""
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
        LCX, RCX = -3.5, 3.5
        IMG_H    = 3.0
        IMG_CY   = 0.0

        # Phase-3 expanded layout
        # Normal: 36 tokens fill 80% canvas width → side = FW*0.80 / (36 + 35*0.10)
        FW       = config.frame_width       # ≈ 14.222
        SIDE_BIG = (FW * 0.80) / (36 + 35 * 0.10)   # ≈ 0.288
        EXP_NY   =  1.3    # normal seq center-y
        EXP_FY   = -1.3    # foveated seq center-y

        FOV_COLORS = [LR_COL]*4 + [HR_COL]*4 + [LR_COL]*4

        # ── Labels + divider (instant) ─────────────────────────────────────
        div = Line(UP*4, DOWN*4, color=GRAY_C, stroke_width=1.5)
        ll  = _txt("Normal Tokenization",   font_size=26).move_to([LCX, 3.5, 0])
        rl  = _txt("Foveated Tokenization", font_size=26).move_to([RCX, 3.5, 0])
        self.add(div, ll, rl)

        # ── Phase 1: Left image + 6×6 HR grid ─────────────────────────────
        img_l = ImageMobject(IMG_PATH).set_height(IMG_H).move_to([LCX, IMG_CY, 0])
        self.play(FadeIn(img_l))
        lg = _normal_grid(img_l)
        self.play(LaggedStart(*[FadeIn(c) for c in lg], lag_ratio=0.02, run_time=1.2))
        self.wait(0.5)

        # ── Phase 2: Right image + foveated grid ──────────────────────────
        img_r = ImageMobject(IMG_PATH).set_height(IMG_H).move_to([RCX, IMG_CY, 0])
        self.play(FadeIn(img_r))
        rh, rl_cells = _fov_grid(img_r)
        self.play(LaggedStart(*[FadeIn(c) for c in rh],       lag_ratio=0.06, run_time=0.5))
        self.play(LaggedStart(*[FadeIn(c) for c in rl_cells], lag_ratio=0.06, run_time=0.8))
        self.wait(0.6)

        # ── Phase 3: Images disappear ──────────────────────────────────────
        self.play(
            FadeOut(img_l), FadeOut(img_r),
            FadeOut(div),   FadeOut(ll),   FadeOut(rl),
        )

        # ── Phase 3 (single animation): all tokens morph to final positions ─
        # Normal tokens → top row (36 squares, 80% wide, centered)
        # Fov tokens    → bottom row (12 squares, same SIDE_BIG, centered)
        norm_big = _seq_targets(36, 0, EXP_NY, [HR_COL]*36, SIDE_BIG)
        fov_big  = _seq_targets(12, 0, EXP_FY, FOV_COLORS,  SIDE_BIG)

        self.play(
            # lg[k] row-major → norm_big[k]
            *[Transform(lg[k],          norm_big[k])   for k in range(36)],
            # rl_cells[0..3] → fov_big[0..3]   (first 4 LR)
            *[Transform(rl_cells[k],    fov_big[k])    for k in range(4)],
            # rh[0..3]       → fov_big[4..7]   (4 HR)
            *[Transform(rh[k],          fov_big[4+k])  for k in range(4)],
            # rl_cells[4..7] → fov_big[8..11]  (last 4 LR)
            *[Transform(rl_cells[4+k],  fov_big[8+k])  for k in range(4)],
            run_time=1.8,
        )

        # Labels appear instantly after morph completes
        norm_title = _txt("Normal: 36 tokens",   font_size=28, color=HR_COL).move_to([0,  2.4, 0])
        fov_title  = _txt("Foveated: 12 tokens", font_size=28, color=LR_COL).move_to([0, -2.4, 0])
        self.add(norm_title, fov_title)
        self.wait(3.0)
