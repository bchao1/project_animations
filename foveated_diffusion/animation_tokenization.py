"""
Foveated Tokenization Animation
Storyboard: split-screen Normal vs Foveated tokenization, then length comparison.
"""
from manim import *

IMG_PATH = "/home/brianchc/project_animations/foveated_diffusion/data/image.png"
HR_COL   = "#00C853"   # green  – high-resolution tokens
LR_COL   = "#448AFF"   # blue   – low-resolution tokens
GOLD_COL = "#FFD600"   # gold   – highlight

N = 6  # grid dimension (6×6)


# ── helpers ───────────────────────────────────────────────────────────────────

def _normal_grid(img):
    """6×6 uniform HR grid overlaid on img (36 cells, row-major)."""
    cw = img.get_width()  / N
    ch = img.get_height() / N
    cells = VGroup()
    for i in range(N):
        for j in range(N):
            cells.add(Rectangle(
                width=cw * .90, height=ch * .90,
                color=HR_COL, fill_color=HR_COL, fill_opacity=.30, stroke_width=1.3,
            ).move_to([img.get_left()[0] + (j + .5)*cw,
                       img.get_top()[1]  - (i + .5)*ch, 0]))
    return cells


def _fov_grid(img):
    """Center 2×2 macro-block → 4 HR cells; remaining 8 macro-blocks → 8 LR cells."""
    cw = img.get_width()  / N
    ch = img.get_height() / N
    hr, lr = VGroup(), VGroup()
    for bi in range(3):
        for bj in range(3):
            if bi == 1 and bj == 1:          # center → 4 individual HR tokens
                for di in range(2):
                    for dj in range(2):
                        i, j = bi*2+di, bj*2+dj
                        hr.add(Rectangle(
                            width=cw * .90, height=ch * .90,
                            color=HR_COL, fill_color=HR_COL, fill_opacity=.40, stroke_width=1.5,
                        ).move_to([img.get_left()[0] + (j + .5)*cw,
                                   img.get_top()[1]  - (i + .5)*ch, 0]))
            else:                            # peripheral → 1 LR token per 2×2 block
                lr.add(Rectangle(
                    width=cw*2 * .88, height=ch*2 * .88,
                    color=LR_COL, fill_color=LR_COL, fill_opacity=.28, stroke_width=1.8,
                ).move_to([img.get_left()[0] + (bj*2 + 1)*cw,
                           img.get_top()[1]  - (bi*2 + 1)*ch, 0]))
    # lr order (row-major, skipping (1,1)):
    # lr[0..2]: top row  (0,0)(0,1)(0,2)
    # lr[3]:    mid-left (1,0)
    # lr[4]:    mid-right(1,2)
    # lr[5..7]: bot row  (2,0)(2,1)(2,2)
    return hr, lr  # VGroup(4), VGroup(8)


def _make_seq(n, cx, y, colors, tok_w, tok_h, gap=0.018):
    """Row of n rectangles centered at (cx, y)."""
    total = n * tok_w + (n - 1) * gap
    x0 = cx - total / 2
    g = VGroup()
    for k in range(n):
        g.add(Rectangle(
            width=tok_w, height=tok_h,
            color=colors[k], fill_color=colors[k], fill_opacity=.82, stroke_width=1.0,
        ).move_to([x0 + k*(tok_w + gap) + tok_w/2, y, 0]))
    return g


# ── Scene ─────────────────────────────────────────────────────────────────────

class FoveatedTokenization(Scene):

    def construct(self):
        LCX, RCX = -3.5, 3.5
        IMG_H     = 2.9
        IMG_TOP_Y = 1.85
        SEQ_Y     = -2.5
        GAP       = 0.018
        TW, TH    = 0.128, 0.22   # small token dims for flattened sequence

        FOV_COLORS = [LR_COL]*4 + [HR_COL]*4 + [LR_COL]*4  # 12-token foveated order

        # ── Divider + section labels ──────────────────────────────────────
        div = Line(UP*4, DOWN*4, color=GRAY_C, stroke_width=1.5)
        ll  = Text("Normal Tokenization",   font_size=24, weight=BOLD).move_to([LCX, 3.5, 0])
        rl  = Text("Foveated Tokenization", font_size=24, weight=BOLD).move_to([RCX, 3.5, 0])
        self.play(Create(div), Write(ll), Write(rl))

        # ── Images appear at center of each half ──────────────────────────
        img_l = ImageMobject(IMG_PATH).set_height(IMG_H).move_to([LCX, 0.15, 0])
        img_r = ImageMobject(IMG_PATH).set_height(IMG_H).move_to([RCX, 0.15, 0])
        self.play(FadeIn(img_l), FadeIn(img_r))
        self.wait(0.35)

        # ── Overlay tokenization grids ───────────────────────────────────
        lg = _normal_grid(img_l)
        rh, rl_cells = _fov_grid(img_r)

        # Left: uniform 6×6 grid
        self.play(LaggedStart(*[FadeIn(c) for c in lg],
                              lag_ratio=0.02, run_time=1.0))
        # Right: HR center then LR periphery
        self.play(
            LaggedStart(*[FadeIn(c) for c in rh],        lag_ratio=0.04, run_time=0.5),
        )
        self.play(
            LaggedStart(*[FadeIn(c) for c in rl_cells],  lag_ratio=0.04, run_time=0.7),
        )

        # Token-type legend (right side, below the grid)
        hr_dot = Square(side_length=0.18, color=HR_COL,
                        fill_color=HR_COL, fill_opacity=0.8, stroke_width=1).move_to([RCX - 1.1, -1.25, 0])
        hr_ltx = Text("HR token (1×1)", font_size=17, color=HR_COL).next_to(hr_dot, RIGHT, buff=0.15)
        lr_dot = Square(side_length=0.26, color=LR_COL,
                        fill_color=LR_COL, fill_opacity=0.6, stroke_width=1).next_to(hr_ltx, RIGHT, buff=0.4)
        lr_ltx = Text("LR token (2×2)", font_size=17, color=LR_COL).next_to(lr_dot, RIGHT, buff=0.15)
        legend = VGroup(hr_dot, hr_ltx, lr_dot, lr_ltx)
        self.play(FadeIn(legend))
        self.wait(0.5)

        # ── Images move to top; tokens flatten to sequence ────────────────
        self.play(
            img_l.animate.set_height(IMG_H * 0.57).move_to([LCX, IMG_TOP_Y, 0]),
            img_r.animate.set_height(IMG_H * 0.57).move_to([RCX, IMG_TOP_Y, 0]),
            FadeOut(legend),
            run_time=0.75,
        )

        # Build small-sequence targets
        norm_seq = _make_seq(36, LCX, SEQ_Y, [HR_COL]*36, TW, TH, GAP)
        fov_seq  = _make_seq(12, RCX, SEQ_Y, FOV_COLORS,  TW, TH, GAP)

        # Transform grid cells → sequence positions
        # Normal: lg[k] → norm_seq[k]  (row-major, all 36)
        # Fov LR[0..3] → fov_seq[0..3], HR[0..3] → fov_seq[4..7], LR[4..7] → fov_seq[8..11]
        self.play(AnimationGroup(
            *[Transform(lg[k],          norm_seq[k])   for k in range(36)],
            *[Transform(rl_cells[k],    fov_seq[k])    for k in range(4)],
            *[Transform(rh[k],          fov_seq[4+k])  for k in range(4)],
            *[Transform(rl_cells[4+k],  fov_seq[8+k])  for k in range(4)],
            lag_ratio=0.012,
        ), run_time=2.0)

        # Token count labels beneath sequences
        n_lbl = Text("36 tokens", font_size=20, color=HR_COL).move_to([LCX, SEQ_Y - 0.35, 0])
        f_lbl = Text("12 tokens", font_size=20, color=LR_COL).move_to([RCX, SEQ_Y - 0.35, 0])
        self.play(Write(n_lbl), Write(f_lbl))
        self.wait(0.8)

        # ── Phase 3: length comparison (full canvas) ──────────────────────
        self.play(
            FadeOut(img_l), FadeOut(img_r),
            FadeOut(div), FadeOut(ll), FadeOut(rl),
            FadeOut(n_lbl), FadeOut(f_lbl),
        )

        # Expanded sizes: normal 36 tokens fill 80% of canvas width
        FW = config.frame_width          # ≈ 14.222
        TW_BIG = (FW * 0.80 - 35*GAP) / 36
        TH_BIG = 0.55

        # Normal: 36 tokens centered at y = 1.2 (top half)
        norm_big = _make_seq(36, 0, 1.2, [HR_COL]*36, TW_BIG, TH_BIG, GAP)
        # Foveated: 12 tokens, same TW_BIG per token → shorter span, centered at y = -1.0
        fov_big  = _make_seq(12, 0, -1.0, FOV_COLORS,  TW_BIG, TH_BIG, GAP)

        norm_title = Text("Normal:   L = 36 tokens",
                          font_size=26, color=HR_COL).move_to([0, 2.2, 0])
        fov_title  = Text("Foveated: L = 12 tokens  (3× fewer)",
                          font_size=26, color=GOLD_COL).move_to([0, -2.1, 0])

        self.play(
            *[Transform(lg[k], norm_big[k]) for k in range(36)],
            Write(norm_title),
            run_time=1.3,
        )
        self.play(
            *[Transform(rl_cells[k],   fov_big[k])   for k in range(4)],
            *[Transform(rh[k],         fov_big[4+k]) for k in range(4)],
            *[Transform(rl_cells[4+k], fov_big[8+k]) for k in range(4)],
            Write(fov_title),
            run_time=1.3,
        )

        # Brace showing the shorter foveated span
        fov_span = fov_big.get_width()
        brace = BraceBetweenPoints(
            [fov_big.get_left()[0], -1.35, 0],
            [fov_big.get_right()[0], -1.35, 0],
            direction=DOWN, color=GOLD_COL,
        )
        brace_lbl = Text(f"{fov_span:.1f} units", font_size=18, color=GOLD_COL)
        brace_lbl.next_to(brace, DOWN, buff=0.1)

        norm_span = norm_big.get_width()
        brace_n = BraceBetweenPoints(
            [norm_big.get_left()[0], 0.85, 0],
            [norm_big.get_right()[0], 0.85, 0],
            direction=DOWN, color=HR_COL,
        )
        brace_n_lbl = Text(f"{norm_span:.1f} units", font_size=18, color=HR_COL)
        brace_n_lbl.next_to(brace_n, DOWN, buff=0.1)

        self.play(
            Create(brace_n), Write(brace_n_lbl),
            Create(brace),   Write(brace_lbl),
        )
        self.wait(3.0)
