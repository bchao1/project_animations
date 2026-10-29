"""
Foveated Tokenization Animation - v3
Based on updated storyboard:
  Phase 1: Split screen — left: image + 6×6 HR grid
  Phase 2: Split screen — right: image + foveated grid
  Phase 3: Images disappear → tokens flatten to compact seqs → expand to 80% canvas

MUST CONFORM guidelines:
  - No overlapping visuals
  - Text appears instantly (self.add, no Write/FadeIn)
  - CMU Serif font (registered directly from font files)
  - HR tokens = blue, LR tokens = green
  - Square tokens with 10% rounded corners, gap = 10% of side
  - Length labels show only token counts
"""
from manim import *
from manimpango import register_font

# ── Register CMU Serif from local font files ───────────────────────────────────
_FONT_DIR = "/home/brianchc/fonts/cmu-serif"
for _f in ["cmunrm.ttf", "cmunbx.ttf", "cmunti.ttf", "cmunbi.ttf"]:
    register_font(f"{_FONT_DIR}/{_f}")

IMG_PATH = "/home/brianchc/project_animations/foveated_diffusion/data/image.png"
HR_COL   = "#448AFF"   # blue  – high-resolution tokens
LR_COL   = "#00C853"   # green – low-resolution tokens
FONT     = "CMU Serif"
N        = 6           # grid dimension


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rsq(side, color, fill_alpha=0.82):
    """Exactly-square token with 10% rounded corners."""
    return RoundedRectangle(
        width=side, height=side,
        corner_radius=side * 0.10,
        color=color, fill_color=color, fill_opacity=fill_alpha, stroke_width=1.2,
    )


def _grid_cell(w, h, color, fill_alpha):
    return RoundedRectangle(
        width=w, height=h,
        corner_radius=min(w, h) * 0.10,
        color=color, fill_color=color, fill_opacity=fill_alpha, stroke_width=1.3,
    )


def _normal_grid(img):
    """6×6 uniform HR grid overlaid on img — 36 cells, row-major."""
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
    """Center 2×2 macro-block → 4 HR cells; 8 outer blocks → 8 LR cells."""
    cw = img.width  / N
    ch = img.height / N
    hr, lr = VGroup(), VGroup()
    for bi in range(3):
        for bj in range(3):
            if bi == 1 and bj == 1:
                for di in range(2):
                    for dj in range(2):
                        i, j = bi*2+di, bj*2+dj
                        hr.add(_grid_cell(cw*0.90, ch*0.90, HR_COL, 0.42).move_to(
                            [img.get_left()[0] + (j + 0.5)*cw,
                             img.get_top()[1]  - (i + 0.5)*ch, 0]))
            else:
                lr.add(_grid_cell(cw*2*0.88, ch*2*0.88, LR_COL, 0.30).move_to(
                    [img.get_left()[0] + (bj*2 + 1)*cw,
                     img.get_top()[1]  - (bi*2 + 1)*ch, 0]))
    # lr order (row-major, skipping (1,1)):
    # lr[0..2]: top row   lr[3]: mid-left   lr[4]: mid-right   lr[5..7]: bot row
    return hr, lr  # VGroup(4), VGroup(8)


def _make_seq(n, cx, y, colors, side):
    """Row of n square tokens, gap = 10% of side, centered at (cx, y)."""
    gap   = side * 0.10
    total = n * side + (n - 1) * gap
    x0    = cx - total / 2
    g = VGroup()
    for k in range(n):
        g.add(_rsq(side, colors[k]).move_to(
            [x0 + k*(side + gap) + side/2, y, 0]))
    return g


def _txt(text, font_size=24, color=WHITE, **kwargs):
    return Text(text, font_size=font_size, color=color, font=FONT, **kwargs)


# ── Scene ──────────────────────────────────────────────────────────────────────

class FoveatedTokenization(Scene):

    def construct(self):
        LCX, RCX  = -3.5,  3.5
        IMG_H     = 3.0          # image height (fills canvas nicely)
        IMG_CY    = 0.0          # image center-y during grid phase
        SEQ_CY    = 0.0          # compact-sequence center-y (intermediate)
        EXP_NY    =  1.3         # expanded normal seq y
        EXP_FY    = -1.3         # expanded foveated seq y
        SIDE_SM   = 0.140        # token side for compact sequences

        FOV_COLORS = [LR_COL]*4 + [HR_COL]*4 + [LR_COL]*4

        # ── Section labels + divider (instant) ────────────────────────────
        div = Line(UP*4, DOWN*4, color=GRAY_C, stroke_width=1.5)
        ll  = _txt("Normal Tokenization",   font_size=24, weight=BOLD).move_to([LCX, 3.5, 0])
        rl  = _txt("Foveated Tokenization", font_size=24, weight=BOLD).move_to([RCX, 3.5, 0])
        self.add(div, ll, rl)

        # ── Phase 1: Left — image + normal 6×6 HR grid ────────────────────
        img_l = ImageMobject(IMG_PATH).set_height(IMG_H).move_to([LCX, IMG_CY, 0])
        self.play(FadeIn(img_l))

        lg = _normal_grid(img_l)
        self.play(LaggedStart(*[FadeIn(c) for c in lg],
                              lag_ratio=0.02, run_time=1.2))
        self.wait(0.5)

        # ── Phase 2: Right — image + foveated grid ────────────────────────
        img_r = ImageMobject(IMG_PATH).set_height(IMG_H).move_to([RCX, IMG_CY, 0])
        self.play(FadeIn(img_r))

        rh, rl_cells = _fov_grid(img_r)
        self.play(LaggedStart(*[FadeIn(c) for c in rh],
                              lag_ratio=0.06, run_time=0.5))
        self.play(LaggedStart(*[FadeIn(c) for c in rl_cells],
                              lag_ratio=0.06, run_time=0.8))
        self.wait(0.6)

        # ── Phase 3a: Images + UI disappear ───────────────────────────────
        self.play(
            FadeOut(img_l), FadeOut(img_r),
            FadeOut(div),   FadeOut(ll),   FadeOut(rl),
        )

        # ── Phase 3b: Flatten grid tokens → compact sequences ─────────────
        # Normal  (left half): 36 HR tokens centered at (LCX, SEQ_CY)
        # Foveated (right half): 4LR+4HR+4LR tokens centered at (RCX, SEQ_CY)
        norm_seq = _make_seq(36, LCX, SEQ_CY, [HR_COL]*36, SIDE_SM)
        fov_seq  = _make_seq(12, RCX, SEQ_CY, FOV_COLORS,  SIDE_SM)

        self.play(AnimationGroup(
            *[Transform(lg[k],          norm_seq[k])   for k in range(36)],
            *[Transform(rl_cells[k],    fov_seq[k])    for k in range(4)],
            *[Transform(rh[k],          fov_seq[4+k])  for k in range(4)],
            *[Transform(rl_cells[4+k],  fov_seq[8+k])  for k in range(4)],
            lag_ratio=0.012,
        ), run_time=2.0)
        self.wait(0.4)

        # ── Phase 3c: Expand sequences to 80% canvas width ────────────────
        # total width with gap=10%: n*s + (n-1)*(s*0.10) = s*(n + (n-1)*0.10)
        # For 36 tokens: s*(36 + 35*0.10) = s*39.5 = FW*0.80  →  s = FW*0.80/39.5
        FW       = config.frame_width   # ≈ 14.222
        SIDE_BIG = (FW * 0.80) / (36 + 35 * 0.10)

        norm_big = _make_seq(36, 0, EXP_NY, [HR_COL]*36, SIDE_BIG)
        fov_big  = _make_seq(12, 0, EXP_FY, FOV_COLORS,  SIDE_BIG)

        # Expand normal
        self.play(
            *[Transform(lg[k], norm_big[k]) for k in range(36)],
            run_time=1.1,
        )
        norm_title = _txt("Normal: 36 tokens", font_size=26, color=HR_COL).move_to([0, 2.3, 0])
        self.add(norm_title)

        # Expand foveated
        self.play(
            *[Transform(rl_cells[k],   fov_big[k])   for k in range(4)],
            *[Transform(rh[k],         fov_big[4+k]) for k in range(4)],
            *[Transform(rl_cells[4+k], fov_big[8+k]) for k in range(4)],
            run_time=1.1,
        )
        fov_title = _txt("Foveated: 12 tokens", font_size=26, color=LR_COL).move_to([0, -2.3, 0])
        self.add(fov_title)

        self.wait(3.0)
