"""
Foveated Diffusion – Masking Strategies (v1)
Animates randomized-mask training vs. saliency-guided training.
"""
from manim import *
from manimpango import register_font

for _f in ["cmunrm.ttf", "cmunbx.ttf", "cmunti.ttf", "cmunbi.ttf"]:
    register_font(f"/home/brianchc/fonts/cmu-serif/{_f}")

FONT     = "CMU Serif"
RAND_DIR = "/home/brianchc/project_animations/foveated_diffusion/mask_training/data/random_masking"
SAL_DIR  = "/home/brianchc/project_animations/foveated_diffusion/mask_training/data/saliency_masking"

TITLE_Y = 3.2
LBL_BUF = 0.18

_AKW = dict(color=WHITE, stroke_width=2.5, tip_length=0.20,
            max_stroke_width_to_length_ratio=999)


def _t(s, sz=24):
    return Text(s, font=FONT, font_size=sz, color=WHITE)


def _ln(a, b):
    return Line(a, b, color=WHITE, stroke_width=2.5)


def _ar(a, b):
    return Arrow(a, b, buff=0, **_AKW)


class MaskTraining(Scene):

    def construct(self):
        self._p1()
        self._p2()
        self._p3()

    # ── Phase 1: centered input image ─────────────────────────────────────────

    def _p1(self):
        title = _t("Masking Strategies", 42).move_to([0, TITLE_Y, 0])
        img   = ImageMobject(f"{SAL_DIR}/image.jpg").set_width(3.5).move_to(ORIGIN)
        tag   = _t("Input Image", 28).next_to(img, UP, buff=LBL_BUF)

        self.add(title, tag)
        self.play(FadeIn(img), run_time=0.8)
        self.wait(2.0)

        self._title = title
        self._img   = img
        self._tag   = tag

    # ── Phase 2: randomized-mask training ─────────────────────────────────────

    def _p2(self):
        IMG_X, IMG_W = -5.0, 2.6
        SAMP_W, SAMP_X = 1.8, 2.8
        SAMP_YS = [1.9, 0.0, -1.9]
        STEM_X  = -1.0

        # title + image slide to left ─────────────────────────────────────────
        new_title = _t("Randomized-Mask Training", 42).move_to([0, TITLE_Y, 0])
        _ref      = self._img.copy().set_width(IMG_W).move_to([IMG_X, 0, 0])
        new_tag   = _t("Input Image", 22).next_to(_ref, UP, buff=LBL_BUF)

        self.play(
            Transform(self._title, new_title),
            self._img.animate.set_width(IMG_W).move_to([IMG_X, 0, 0]),
            FadeOut(self._tag),
            run_time=0.7,
        )
        self.add(new_tag)
        self._tag = new_tag

        # 3 sample images ──────────────────────────────────────────────────────
        samp_imgs, samp_tags = [], []
        for k, sy in enumerate(SAMP_YS):
            im = ImageMobject(f"{RAND_DIR}/{k:04d}/circle.jpg").set_width(SAMP_W)
            im.move_to([SAMP_X, sy, 0])
            tg = _t(f"Sample {k+1}", 20).next_to(im, DOWN, buff=0.14)
            samp_imgs.append(im)
            samp_tags.append(tg)

        # branching arrows ─────────────────────────────────────────────────────
        img_rx  = IMG_X + IMG_W / 2          # −3.7
        samp_lx = SAMP_X - SAMP_W / 2 - 0.08  # 1.82

        feeder  = _ln([img_rx + 0.05, 0, 0], [STEM_X, 0, 0])
        stem    = _ln([STEM_X, SAMP_YS[-1], 0], [STEM_X, SAMP_YS[0], 0])
        brs     = VGroup(*[
            _ar([STEM_X + 0.04, sy, 0], [samp_lx, sy, 0])
            for sy in SAMP_YS
        ])

        self.play(Create(feeder), Create(stem), run_time=0.4)
        self.play(LaggedStart(*[Create(b) for b in brs], lag_ratio=0.25), run_time=0.7)
        self.play(
            LaggedStart(
                *[FadeIn(Group(im, tg)) for im, tg in zip(samp_imgs, samp_tags)],
                lag_ratio=0.3,
            ),
            run_time=1.0,
        )
        self.wait(0.8)

        # pulsing selection ────────────────────────────────────────────────────
        for k in [2, 0, 1, 2, 0]:
            self.play(Indicate(samp_imgs[k], color=YELLOW, scale_factor=1.08),
                      run_time=0.4)
        self.wait(1.5)

        self._p2o = Group(*samp_imgs, *samp_tags, feeder, stem, brs)

    # ── Phase 3: saliency-guided training ─────────────────────────────────────

    def _p3(self):
        new_title = _t("Saliency-Guided Training", 42).move_to([0, TITLE_Y, 0])
        self.play(
            Transform(self._title, new_title),
            FadeOut(self._p2o),
            run_time=0.7,
        )

        # input image stays at x=−5.0, width=2.6  →  right edge x=−3.7
        img_rx = -5.0 + 2.6 / 2

        # pipeline objects ─────────────────────────────────────────────────────
        sal     = ImageMobject(f"{SAL_DIR}/saliency.png").set_width(2.5).move_to([-1.5, 0, 0])
        sal_tag = _t("Saliency Map", 22).next_to(sal, UP, buff=LBL_BUF)

        circ     = ImageMobject(f"{SAL_DIR}/circle.jpg").set_width(2.0).move_to([3.8, 1.2, 0])
        mask     = ImageMobject(f"{SAL_DIR}/foveation_mask_binary.png").set_width(1.5).move_to([3.8, -1.2, 0])
        circ_tag = _t("Foveated Image", 20).next_to(circ, UP, buff=0.14)
        mask_tag = _t("Fovea Mask",     20).next_to(mask, UP, buff=0.14)

        # arrow 1: image → saliency map ────────────────────────────────────────
        arr1     = _ar([img_rx + 0.08, 0, 0], [sal.get_left()[0] - 0.08, 0, 0])
        arr1_lbl = _t("Saliency\nDetection", 20).next_to(arr1, UP, buff=0.15)

        # branching: saliency → foveated image + mask ──────────────────────────
        sal_rx = sal.get_right()[0]
        st2_x  = 1.8

        feeder2 = _ln([sal_rx + 0.05, 0, 0], [st2_x, 0, 0])
        stem2   = _ln([st2_x, -1.2, 0], [st2_x, 1.2, 0])
        arr2a   = _ar([st2_x + 0.04,  1.2, 0], [circ.get_left()[0] - 0.08,  1.2, 0])
        arr2b   = _ar([st2_x + 0.04, -1.2, 0], [mask.get_left()[0] - 0.08, -1.2, 0])
        arr2_lbl = _t("Thresholding", 20).move_to([(sal_rx + st2_x) / 2, 0.35, 0])

        # animate pipeline step by step ────────────────────────────────────────
        self.play(Create(arr1), run_time=0.5)
        self.add(arr1_lbl)
        self.play(FadeIn(sal), run_time=0.5)
        self.add(sal_tag)
        self.wait(0.3)

        self.play(Create(feeder2), Create(stem2), run_time=0.4)
        self.add(arr2_lbl)
        self.play(Create(arr2a), Create(arr2b), run_time=0.5)
        self.play(FadeIn(circ), FadeIn(mask), run_time=0.6)
        self.add(circ_tag, mask_tag)
        self.wait(3.0)
