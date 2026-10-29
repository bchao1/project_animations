"""
Foveated Diffusion - Project Animation
Manim animation explaining the Foveated Diffusion framework.
Uses only Text (Unicode math) — no LaTeX required.
"""

from manim import *
import numpy as np

IMG_PATH = "/home/brianchc/project_animations/foveated_diffusion/data/image.png"

# ── Color palette ──────────────────────────────────────────────────────────────
HR_COL   = "#00C853"   # green  – high-resolution foveal tokens
LR_COL   = "#448AFF"   # blue   – low-resolution peripheral tokens
DIT_COL  = "#FF6D00"   # orange – DiT block
ENC_COL  = "#AA00FF"   # purple – VAE encoder
DEC_COL  = "#00BFA5"   # teal   – VAE decoder
GOLD_COL = "#FFD600"   # gold   – merge / mask / output


# ── Shared helpers ─────────────────────────────────────────────────────────────
def lbox(label, color, w=1.6, h=1.0, fs=21):
    """Labeled rounded-rectangle box, centered at ORIGIN."""
    box = RoundedRectangle(
        width=w, height=h, corner_radius=0.12,
        color=color, fill_color=color, fill_opacity=0.45, stroke_width=2,
    )
    lbl = Text(label, font_size=fs, color=WHITE, line_spacing=1.2)
    lbl.move_to(box)
    return VGroup(box, lbl)


def arr(start, end, color=WHITE):
    return Arrow(
        start, end, buff=0.08, color=color, stroke_width=2,
        max_tip_length_to_length_ratio=0.15,
    )


# ── Scene 1: Title ─────────────────────────────────────────────────────────────
class S1_Title(Scene):
    def construct(self):
        title = Text("Foveated Diffusion", font_size=62, weight=BOLD)
        title.set_color_by_gradient(WHITE, YELLOW)

        subtitle = Text(
            "Perceptually-Motivated Efficient\nImage & Video Generation",
            font_size=30, color=GRAY_A, line_spacing=1.3,
        )
        subtitle.next_to(title, DOWN, buff=0.5)

        eye_white = Ellipse(width=3.2, height=1.6, color=WHITE, stroke_width=2.5, fill_opacity=0)
        iris  = Circle(radius=0.58, color=BLUE_D, fill_color=BLUE_D, fill_opacity=1, stroke_width=0)
        pupil = Circle(radius=0.26, color=BLACK,  fill_color=BLACK,  fill_opacity=1, stroke_width=0)
        glint = Circle(radius=0.08, color=WHITE,  fill_color=WHITE,  fill_opacity=0.9, stroke_width=0)
        glint.shift(UP * 0.12 + RIGHT * 0.12)
        eye = VGroup(eye_white, iris, pupil, glint)
        eye.next_to(subtitle, DOWN, buff=0.55)

        speedup = VGroup(
            Text("2\u00d7 faster", font_size=28, color=GREEN),
            Text("  images   \u2022   ", font_size=24, color=GRAY_A),
            Text("4\u00d7 faster", font_size=28, color=GREEN),
            Text("  videos", font_size=24, color=GRAY_A),
        ).arrange(RIGHT, buff=0.0)
        speedup.next_to(eye, DOWN, buff=0.45)

        self.play(Write(title, run_time=1.5))
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.play(
            GrowFromCenter(eye_white),
            GrowFromCenter(iris),
            GrowFromCenter(pupil),
            GrowFromCenter(glint),
        )
        self.play(FadeIn(speedup, shift=UP * 0.2))
        self.wait(2.5)


# ── Scene 2: Human Visual System ───────────────────────────────────────────────
class S2_HumanVision(Scene):
    def construct(self):
        title = Text("Human Visual System", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.35)
        self.play(Write(title))

        cp = DOWN * 0.5

        periph = Circle(radius=3.2, color=RED,    fill_color=RED,    fill_opacity=0.08).move_to(cp)
        para   = Circle(radius=2.0, color=YELLOW, fill_color=YELLOW, fill_opacity=0.15).move_to(cp)
        fovea  = Circle(radius=1.0, color=GREEN,  fill_color=GREEN,  fill_opacity=0.35).move_to(cp)

        gaze = VGroup(
            Line(cp + UP * 0.20, cp + DOWN * 0.20, color=WHITE, stroke_width=2.5),
            Line(cp + LEFT * 0.20, cp + RIGHT * 0.20, color=WHITE, stroke_width=2.5),
            Circle(radius=0.07, color=WHITE, stroke_width=2).move_to(cp),
        )

        fov_lbl  = Text("Fovea\n(sharp detail)",   font_size=18, color=GREEN,  line_spacing=1.2)
        para_lbl = Text("Para-fovea\n(good acuity)",font_size=18, color=YELLOW, line_spacing=1.2)
        peri_lbl = Text("Periphery\n(low acuity)",  font_size=18, color=RED,    line_spacing=1.2)
        fov_lbl.move_to(cp + RIGHT * 1.8 + UP * 0.5)
        para_lbl.move_to(cp + LEFT * 2.9 + UP * 0.9)
        peri_lbl.move_to(cp + RIGHT * 3.8 + DOWN * 0.9)

        self.play(GrowFromCenter(periph))
        self.play(GrowFromCenter(para))
        self.play(GrowFromCenter(fovea))
        self.play(FadeIn(gaze))
        self.play(Write(fov_lbl), Write(para_lbl), Write(peri_lbl))

        insight = Text(
            "Key Insight: Allocate fewer tokens to the periphery!",
            font_size=25, color=YELLOW,
        )
        insight.to_edge(DOWN, buff=0.45)
        box = SurroundingRectangle(insight, color=YELLOW, buff=0.12, corner_radius=0.1)
        self.play(Write(insight), Create(box))
        self.wait(2.5)


# ── Scene 3: Foveated Tokenization ────────────────────────────────────────────
class S3_Tokenization(Scene):
    def construct(self):
        title = Text("Foveated Tokenization", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.35)
        self.play(Write(title))

        lion = ImageMobject(IMG_PATH)
        lion.set_height(3.8)
        lion.shift(LEFT * 3.4 + DOWN * 0.15)
        self.play(FadeIn(lion))

        N    = 8
        cell = lion.height / N
        origin = lion.get_corner(UL)
        hr_cells = VGroup()
        lr_cells = VGroup()
        used: set = set()
        center = N // 2

        for i in range(N):
            for j in range(N):
                if (i, j) in used:
                    continue
                di = abs(i - center + 0.5)
                dj = abs(j - center + 0.5)
                dist = max(di, dj)
                cx = origin[0] + (j + 0.5) * cell
                cy = origin[1] - (i + 0.5) * cell
                if dist <= 2.0:
                    sq = Square(
                        side_length=cell * 0.88,
                        color=HR_COL, fill_color=HR_COL, fill_opacity=0.35, stroke_width=1.3,
                    )
                    sq.move_to([cx, cy, 0])
                    hr_cells.add(sq)
                else:
                    if i % 2 == 0 and j % 2 == 0 and i + 1 < N and j + 1 < N:
                        used.update({(i, j), (i+1, j), (i, j+1), (i+1, j+1)})
                        mx = origin[0] + (j + 1) * cell
                        my = origin[1] - (i + 1) * cell
                        sq = Square(
                            side_length=cell * 2 * 0.88,
                            color=LR_COL, fill_color=LR_COL, fill_opacity=0.28, stroke_width=1.6,
                        )
                        sq.move_to([mx, my, 0])
                        lr_cells.add(sq)

        self.play(LaggedStart(*[FadeIn(c) for c in hr_cells], lag_ratio=0.04, run_time=1.0))
        self.play(LaggedStart(*[FadeIn(c) for c in lr_cells], lag_ratio=0.04, run_time=0.9))

        hr_demo = Square(side_length=0.3, color=HR_COL, fill_color=HR_COL, fill_opacity=0.6, stroke_width=1.5)
        hr_demo.shift(RIGHT * 1.2 + UP * 1.5)
        hr_lbl = Text("HR token (1\u00d71)", font_size=19, color=HR_COL)
        hr_lbl.next_to(hr_demo, RIGHT, buff=0.2)
        lr_demo = Square(side_length=0.46, color=LR_COL, fill_color=LR_COL, fill_opacity=0.45, stroke_width=1.5)
        lr_demo.next_to(hr_lbl, RIGHT, buff=0.6)
        lr_lbl = Text("LR token (2\u00d72)", font_size=19, color=LR_COL)
        lr_lbl.next_to(lr_demo, RIGHT, buff=0.2)
        self.play(FadeIn(hr_demo), Write(hr_lbl), FadeIn(lr_demo), Write(lr_lbl))

        formula   = Text("L  =  m  +  (h\u00b7w \u2212 m) / 4", font_size=34)
        m_note    = Text("m = number of HR (foveal) tokens", font_size=21, color=HR_COL)
        save_note = Text("4\u00d7 savings per peripheral token", font_size=21, color=LR_COL)
        l_ineq    = Text("L  \u226a  h\u00b7w", font_size=32, color=YELLOW, weight=BOLD)
        formula.shift(RIGHT * 3.2 + UP * 0.6)
        m_note.next_to(formula, DOWN, buff=0.3)
        save_note.next_to(m_note, DOWN, buff=0.2)
        l_ineq.next_to(save_note, DOWN, buff=0.35)
        self.play(Write(formula))
        self.play(Write(m_note))
        self.play(Write(save_note))
        self.play(Write(l_ineq))
        self.wait(2.5)


# ── Scene 4: Foveated Generation Pipeline ─────────────────────────────────────
class S4_GenerationPipeline(Scene):
    def construct(self):
        title = Text("Foveated Generation", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        cy = -0.15

        noise_box = lbox("Noise",      GRAY_B,  w=1.4,  h=0.95, fs=20)
        noise_box.shift(LEFT * 5.5 + UP * cy)
        noise_sub = Text("z\u2081\u1da0\u1d52\u1d5b ~ N(0,I)", font_size=16, color=GRAY_A)
        noise_sub.next_to(noise_box, DOWN, buff=0.08)

        dit_box = lbox("DiT\nDenoise", DIT_COL, w=1.7, h=1.15, fs=22)
        dit_box.shift(LEFT * 2.9 + UP * cy)

        loop = CurvedArrow(
            dit_box[0].get_top() + RIGHT * 0.35,
            dit_box[0].get_top() + LEFT * 0.35,
            angle=-PI * 0.85, color=YELLOW, stroke_width=2,
        )
        loop_lbl = Text("t: 1\u21920", font_size=14, color=YELLOW)
        loop_lbl.next_to(loop, UP, buff=0.04)

        fov_lbl = Text("z\u2080\u1da0\u1d52\u1d5b", font_size=30, color=GOLD_COL)
        fov_lbl.shift(LEFT * 0.7 + UP * cy)

        split_box = lbox("Split\n(M)", ENC_COL, w=1.25, h=1.0, fs=21)
        split_box.shift(RIGHT * 1.0 + UP * cy)

        zh_lbl = Text("z\u2080\u02b0\u2071", font_size=22, color=HR_COL)
        zh_lbl.shift(RIGHT * 2.65 + UP * (cy + 0.85))
        zh_full = Text("(high-res)", font_size=15, color=HR_COL)
        zh_full.next_to(zh_lbl, RIGHT, buff=0.12)

        zl_lbl = Text("z\u2080\u02e1\u1d52", font_size=22, color=LR_COL)
        zl_lbl.shift(RIGHT * 2.65 + UP * (cy - 0.85))
        zl_full = Text("(low-res)", font_size=15, color=LR_COL)
        zl_full.next_to(zl_lbl, RIGHT, buff=0.12)

        dh_box = lbox("D", DEC_COL, w=0.75, h=0.65, fs=23)
        dh_box.shift(RIGHT * 4.3 + UP * (cy + 0.85))
        dl_box = lbox("D", DEC_COL, w=0.75, h=0.65, fs=23)
        dl_box.shift(RIGHT * 4.3 + UP * (cy - 0.85))

        blend_box = lbox("Blend\n(M')", GOLD_COL, w=1.25, h=1.0, fs=20)
        blend_box.shift(RIGHT * 5.5 + UP * cy)

        out_lbl = Text("x\u1da0\u1d52\u1d5b", font_size=28, color=GOLD_COL)
        out_lbl.next_to(blend_box, RIGHT, buff=0.3)

        a1  = arr(noise_box.get_right(),  dit_box.get_left())
        a2  = arr(dit_box.get_right(),    fov_lbl.get_left())
        a3  = arr(fov_lbl.get_right(),    split_box.get_left())
        a4h = arr(split_box.get_right(),  zh_lbl.get_left(),   HR_COL)
        a4l = arr(split_box.get_right(),  zl_lbl.get_left(),   LR_COL)
        a5h = arr(zh_full.get_right(),    dh_box.get_left(),   HR_COL)
        a5l = arr(zl_full.get_right(),    dl_box.get_left(),   LR_COL)
        a6h = arr(dh_box.get_right(), blend_box[0].get_top()    + LEFT * 0.1, HR_COL)
        a6l = arr(dl_box.get_right(), blend_box[0].get_bottom() + LEFT * 0.1, LR_COL)
        a7  = arr(blend_box.get_right(), out_lbl.get_left(), GOLD_COL)

        blend_eq = Text(
            "x\u1da0\u1d52\u1d5b  =  M'\u2299x\u02b0\u2071  +  (1\u2212M')\u2299Up(x\u02e1\u1d52)",
            font_size=26, color=GRAY_A,
        )
        blend_eq.to_edge(DOWN, buff=0.38)

        self.play(FadeIn(noise_box), Write(noise_sub))
        self.play(GrowArrow(a1), FadeIn(dit_box))
        self.play(Create(loop), Write(loop_lbl))
        self.play(GrowArrow(a2), Write(fov_lbl))
        self.play(GrowArrow(a3), FadeIn(split_box))
        self.play(GrowArrow(a4h), Write(zh_lbl), Write(zh_full),
                  GrowArrow(a4l), Write(zl_lbl), Write(zl_full))
        self.play(GrowArrow(a5h), FadeIn(dh_box), GrowArrow(a5l), FadeIn(dl_box))
        self.play(GrowArrow(a6h), GrowArrow(a6l), FadeIn(blend_box))
        self.play(GrowArrow(a7), Write(out_lbl))
        self.play(Write(blend_eq))
        self.wait(2.5)


# ── Scene 5: Foveated Training Pipeline ───────────────────────────────────────
class S5_TrainingPipeline(Scene):
    def construct(self):
        title = Text("Foveated Training", font_size=44, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        cy_h = 1.1
        cy_l = -0.65
        cy_m = (cy_h + cy_l) / 2

        img_box  = lbox("x",       WHITE,  w=1.1, h=0.85, fs=30)
        img_box[0].set_fill(WHITE, opacity=0.15)
        img_box.shift(LEFT * 5.5 + UP * cy_h)

        down_box = lbox("Down(x)", BLUE_B, w=1.5, h=0.85, fs=20)
        down_box.shift(LEFT * 5.5 + UP * cy_l)

        enc_h = lbox("E", ENC_COL, w=0.8, h=0.75, fs=26)
        enc_h.shift(LEFT * 3.6 + UP * cy_h)
        enc_l = lbox("E", ENC_COL, w=0.8, h=0.75, fs=26)
        enc_l.shift(LEFT * 3.6 + UP * cy_l)

        zh = Text("z\u2080\u02b0\u2071", font_size=24, color=HR_COL)
        zh.shift(LEFT * 2.2 + UP * cy_h)
        zh_note = Text("(high-res)", font_size=15, color=HR_COL)
        zh_note.next_to(zh, RIGHT, buff=0.1)

        zl = Text("z\u2080\u02e1\u1d52", font_size=24, color=LR_COL)
        zl.shift(LEFT * 2.2 + UP * cy_l)
        zl_note = Text("(low-res)", font_size=15, color=LR_COL)
        zl_note.next_to(zl, RIGHT, buff=0.1)

        merge_box = lbox("Merge\n(M)", GOLD_COL, w=1.3, h=0.95, fs=20)
        merge_box.shift(LEFT * 0.5 + UP * cy_m)

        zfov = Text("z\u2080\u1da0\u1d52\u1d5b", font_size=28, color=GOLD_COL)
        zfov.shift(RIGHT * 1.2 + UP * cy_m)

        noise_box = lbox("Add\nNoise", GRAY_B, w=1.25, h=0.95, fs=20)
        noise_box.shift(RIGHT * 2.9 + UP * cy_m)

        dit_box = lbox("DiT\n+LoRA", DIT_COL, w=1.5, h=1.05, fs=21)
        dit_box.shift(RIGHT * 4.6 + UP * cy_m)
        lora_note = Text("post-training", font_size=16, color=ORANGE)
        lora_note.next_to(dit_box, DOWN, buff=0.12)

        loss = Text("\u2112_flow", font_size=36, color=RED)
        loss.shift(RIGHT * 6.3 + UP * cy_m)

        a_img_e  = arr(img_box.get_right(),   enc_h.get_left(),  WHITE)
        a_dn_e   = arr(down_box.get_right(),  enc_l.get_left(),  BLUE_B)
        a_eh_zh  = arr(enc_h.get_right(),     zh.get_left(),     HR_COL)
        a_el_zl  = arr(enc_l.get_right(),     zl.get_left(),     LR_COL)
        a_zh_mg  = arr(zh_note.get_right(),
                       merge_box[0].get_corner(UL) + RIGHT * 0.1 + DOWN * 0.1, HR_COL)
        a_zl_mg  = arr(zl_note.get_right(),
                       merge_box[0].get_corner(DL) + RIGHT * 0.1 + UP * 0.1,  LR_COL)
        a_mg_fov = arr(merge_box.get_right(), zfov.get_left(),   GOLD_COL)
        a_fov_n  = arr(zfov.get_right(),      noise_box.get_left(), GOLD_COL)
        a_n_dit  = arr(noise_box.get_right(), dit_box.get_left(), WHITE)
        a_dit_l  = arr(dit_box.get_right(),   loss.get_left(),   RED)

        self.play(FadeIn(img_box), FadeIn(down_box))
        self.play(GrowArrow(a_img_e), FadeIn(enc_h),
                  GrowArrow(a_dn_e),  FadeIn(enc_l))
        self.play(GrowArrow(a_eh_zh), Write(zh), Write(zh_note),
                  GrowArrow(a_el_zl), Write(zl), Write(zl_note))
        self.play(GrowArrow(a_zh_mg), GrowArrow(a_zl_mg), FadeIn(merge_box))
        self.play(GrowArrow(a_mg_fov), Write(zfov))
        self.play(GrowArrow(a_fov_n), FadeIn(noise_box))
        self.play(GrowArrow(a_n_dit),  FadeIn(dit_box), FadeIn(lora_note))
        self.play(GrowArrow(a_dit_l),  Write(loss))
        self.wait(2.5)


# ── Scene 6: Results ───────────────────────────────────────────────────────────
class S6_Results(Scene):
    def construct(self):
        title = Text("Results & Speedup", font_size=46, weight=BOLD)
        title.to_edge(UP, buff=0.32)
        self.play(Write(title))

        subtitle = Text("vs. Standard DiT at full resolution", font_size=26, color=GRAY_A)
        subtitle.next_to(title, DOWN, buff=0.18)
        self.play(FadeIn(subtitle))

        bar_w = 4.2
        bar_h = 0.72
        lx    = -3.5
        gap   = 0.22

        img_cat = Text("Image\nGeneration", font_size=24, line_spacing=1.2)
        std_img = Rectangle(width=bar_w, height=bar_h,
                            color=BLUE_C, fill_color=BLUE_C, fill_opacity=0.8, stroke_width=0)
        std_img.move_to([lx + bar_w / 2, 1.55, 0])
        fov_img = Rectangle(width=bar_w / 2, height=bar_h,
                            color=GREEN, fill_color=GREEN, fill_opacity=0.85, stroke_width=0)
        fov_img.align_to(std_img, LEFT)
        fov_img.set_y(std_img.get_y() - bar_h - gap)
        std_img_lbl = Text("Standard DiT  (1\u00d7)", font_size=18, color=BLUE_C)
        std_img_lbl.next_to(std_img, RIGHT, buff=0.25)
        fov_img_lbl = Text("Foveated  (2\u00d7 faster)", font_size=18, color=GREEN)
        fov_img_lbl.next_to(fov_img, RIGHT, buff=0.25)
        img_cat.set_y((std_img.get_y() + fov_img.get_y()) / 2)
        img_cat.set_x(-5.8)

        vid_cat = Text("Video\nGeneration", font_size=24, line_spacing=1.2)
        std_vid = Rectangle(width=bar_w, height=bar_h,
                            color=BLUE_C, fill_color=BLUE_C, fill_opacity=0.8, stroke_width=0)
        std_vid.align_to(std_img, LEFT)
        std_vid.set_y(fov_img.get_y() - bar_h - gap * 3)
        fov_vid = Rectangle(width=bar_w / 4, height=bar_h,
                            color=GREEN, fill_color=GREEN, fill_opacity=0.85, stroke_width=0)
        fov_vid.align_to(std_vid, LEFT)
        fov_vid.set_y(std_vid.get_y() - bar_h - gap)
        std_vid_lbl = Text("Standard DiT  (1\u00d7)", font_size=18, color=BLUE_C)
        std_vid_lbl.next_to(std_vid, RIGHT, buff=0.25)
        fov_vid_lbl = Text("Foveated  (4\u00d7 faster)", font_size=18, color=GREEN)
        fov_vid_lbl.next_to(fov_vid, RIGHT, buff=0.25)
        vid_cat.set_y((std_vid.get_y() + fov_vid.get_y()) / 2)
        vid_cat.set_x(-5.8)

        self.play(Write(img_cat), Write(vid_cat))
        self.play(Create(std_img), Write(std_img_lbl))
        self.play(Create(fov_img), Write(fov_img_lbl))
        self.play(Create(std_vid), Write(std_vid_lbl))
        self.play(Create(fov_vid), Write(fov_vid_lbl))

        strats = VGroup(
            Text("\u25cf Randomized masks",      font_size=20, color=GREEN),
            Text("\u25cf Saliency-guided masks", font_size=20, color=LR_COL),
            Text("\u25cf Bounding-box masks",    font_size=20, color=GOLD_COL),
        ).arrange(RIGHT, buff=0.7)
        strats.to_edge(DOWN, buff=0.85)

        key_msg = Text(
            "Comparable perceptual quality  \u2022  Validated via user study",
            font_size=22, color=YELLOW,
        )
        key_msg.to_edge(DOWN, buff=0.3)

        self.play(LaggedStart(*[Write(s) for s in strats], lag_ratio=0.3))
        self.play(Write(key_msg))
        self.wait(2.5)


# ── Master scene ───────────────────────────────────────────────────────────────
class FoveatedDiffusionAnimation(Scene):
    """Single entry-point: plays all 6 scenes in sequence."""

    def construct(self):
        for cls in [
            S1_Title,
            S2_HumanVision,
            S3_Tokenization,
            S4_GenerationPipeline,
            S5_TrainingPipeline,
            S6_Results,
        ]:
            self.clear()
            cls.construct(self)
            self.wait(0.5)
