"""
Foveated Diffusion – Masking Strategies (v2)
Changes from v1:
- Phase 1: no title, input image 1.5× bigger (5.25 wide)
- Phase 2: paired images (circle.jpg + foveation_mask_pix64_up512.png, 5% gap),
           no sample labels, pulse travels along arrows then pair scales
- Phase 3: linear pipeline, single arrow saliency → foveation_mask_pix64_up512.png only
- All arrows/lines doubled stroke width (5.0 pt)
"""
import numpy as np
from manim import *
from manimpango import register_font

for _f in ["cmunrm.ttf", "cmunbx.ttf", "cmunti.ttf", "cmunbi.ttf"]:
    register_font(f"/home/brianchc/fonts/cmu-serif/{_f}")

FONT     = "CMU Serif"
RAND_DIR = "/home/brianchc/project_animations/foveated_diffusion/mask_training/data/random_masking"
SAL_DIR  = "/home/brianchc/project_animations/foveated_diffusion/mask_training/data/saliency_masking"

TITLE_Y = 3.2
LBL_BUF = 0.18

_AKW = dict(color=WHITE, stroke_width=5.0, tip_length=0.25,
            max_stroke_width_to_length_ratio=999)


def _t(s, sz=24):
    return Text(s, font=FONT, font_size=sz, color=WHITE)


def _ln(a, b):
    return Line(a, b, color=WHITE, stroke_width=5.0)


def _ar(a, b):
    return Arrow(a, b, buff=0, **_AKW)


class MaskTraining(Scene):

    def construct(self):
        self._p1()
        self._p2()
        self._p3()

    # ── Phase 1: centered input image, no title ────────────────────────────────

    def _p1(self):
        img = ImageMobject(f"{SAL_DIR}/image.jpg").set_width(5.25).move_to(ORIGIN)
        tag = _t("Input Image", 28).next_to(img, UP, buff=LBL_BUF)
        self.add(tag)
        self.play(FadeIn(img), run_time=0.8)
        self.wait(2.0)
        self._img   = img
        self._tag   = tag
        self._title = None

    # ── Phase 2: randomized-mask training ─────────────────────────────────────

    def _p2(self):
        IMG_X, IMG_W = -5.0, 2.6
        W_EACH       = 1.1
        SAMP_X       = 3.2
        SAMP_YS      = [1.9, 0.0, -1.9]
        STEM_X       = -1.0

        # Title appears instantly while image slides left
        new_title = _t("Randomized-Mask Training", 42).move_to([0, TITLE_Y, 0])
        _ref      = self._img.copy().set_width(IMG_W).move_to([IMG_X, 0, 0])
        new_tag   = _t("Input Image", 22).next_to(_ref, UP, buff=LBL_BUF)

        self.add(new_title)
        self.play(
            self._img.animate.set_width(IMG_W).move_to([IMG_X, 0, 0]),
            FadeOut(self._tag),
            run_time=0.7,
        )
        self.add(new_tag)
        self._tag   = new_tag
        self._title = new_title

        # Paired samples — created off-scene first so we can read geometry
        gap = W_EACH * 0.05
        samp_pairs = []
        for k, sy in enumerate(SAMP_YS):
            im1 = ImageMobject(f"{RAND_DIR}/{k:04d}/circle.jpg").set_width(W_EACH)
            im2 = ImageMobject(f"{RAND_DIR}/{k:04d}/foveation_mask_pix64_up512.png").set_width(W_EACH)
            pair = Group(im1, im2).arrange(RIGHT, buff=gap)
            pair.move_to([SAMP_X, sy, 0])
            samp_pairs.append(pair)

        # Arrow geometry based on actual pair left edge
        img_rx  = IMG_X + IMG_W / 2               # −3.7
        samp_lx = samp_pairs[0].get_left()[0] - 0.08

        feeder = _ln([img_rx + 0.05, 0, 0], [STEM_X, 0, 0])
        stem   = _ln([STEM_X, SAMP_YS[-1], 0], [STEM_X, SAMP_YS[0], 0])
        brs    = VGroup(*[
            _ar([STEM_X + 0.04, sy, 0], [samp_lx, sy, 0])
            for sy in SAMP_YS
        ])

        self.play(Create(feeder), Create(stem), run_time=0.4)
        self.play(LaggedStart(*[Create(b) for b in brs], lag_ratio=0.25), run_time=0.7)
        self.play(
            LaggedStart(*[FadeIn(p) for p in samp_pairs], lag_ratio=0.3),
            run_time=1.0,
        )
        self.wait(0.8)

        # Pulsing: flash travels along arrow path, then pair bounces (no color change)
        for k in [2, 0, 1, 2, 0]:
            sy = SAMP_YS[k]
            if abs(sy) < 0.01:  # middle branch — straight horizontal
                pts = [
                    np.array([img_rx + 0.05, 0,  0]),
                    np.array([samp_lx,        0,  0]),
                ]
            else:               # top/bottom — feeder → stem → branch
                pts = [
                    np.array([img_rx + 0.05, 0,  0]),
                    np.array([STEM_X,         0,  0]),
                    np.array([STEM_X,         sy, 0]),
                    np.array([samp_lx,        sy, 0]),
                ]
            pulse = VMobject()
            pulse.set_points_as_corners(pts)
            pulse.set_stroke(WHITE, width=12, opacity=1.0)
            pulse.set_fill(opacity=0)

            self.play(ShowPassingFlash(pulse, time_width=0.15), run_time=0.65)
            self.play(samp_pairs[k].animate.scale(1.08), run_time=0.12)
            self.play(samp_pairs[k].animate.scale(1 / 1.08), run_time=0.12)

        self.wait(1.5)

        self._p2o = Group(*samp_pairs, feeder, stem, brs)

    # ── Phase 3: saliency-guided training ─────────────────────────────────────

    def _p3(self):
        new_title = _t("Saliency-Guided Training", 42).move_to([0, TITLE_Y, 0])
        self.play(
            Transform(self._title, new_title),
            FadeOut(self._p2o),
            run_time=0.7,
        )

        # Input image stays at x=−5.0, width=2.6  →  right edge −3.7
        img_rx = -5.0 + 2.6 / 2

        # Saliency map (center)
        sal     = ImageMobject(f"{SAL_DIR}/saliency.png").set_width(2.5).move_to([0, 0, 0])
        sal_tag = _t("Saliency Map", 22).next_to(sal, UP, buff=LBL_BUF)

        # Fovea mask (right) — pix64 version as specified
        fov     = ImageMobject(f"{SAL_DIR}/foveation_mask_pix64_up512.png").set_width(2.5).move_to([4.2, 0, 0])
        fov_tag = _t("Fovea Mask", 22).next_to(fov, UP, buff=LBL_BUF)

        # Arrow 1: input image → saliency map
        arr1     = _ar([img_rx + 0.08, 0, 0], [sal.get_left()[0] - 0.08, 0, 0])
        arr1_lbl = _t("Saliency\nDetection", 20).next_to(arr1, UP, buff=0.15)

        # Arrow 2: saliency map → fovea mask (single, no branching)
        arr2     = _ar([sal.get_right()[0] + 0.08, 0, 0], [fov.get_left()[0] - 0.08, 0, 0])
        arr2_lbl = _t("Thresholding", 20).next_to(arr2, UP, buff=0.15)

        # Animate pipeline step by step
        self.play(Create(arr1), run_time=0.5)
        self.add(arr1_lbl)
        self.play(FadeIn(sal), run_time=0.5)
        self.add(sal_tag)
        self.wait(0.3)

        self.play(Create(arr2), run_time=0.5)
        self.add(arr2_lbl)
        self.play(FadeIn(fov), run_time=0.5)
        self.add(fov_tag)
        self.wait(3.0)
