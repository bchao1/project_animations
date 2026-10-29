"""
Foveated Diffusion – Masking Strategies (v3)
Changes from v2:
- Phase 1 → Phase 2: image slides left FIRST, title appears after (no simultaneous overlap)
- Phase 2: paired images 1.5× bigger (W_EACH=1.65), positions adjusted to avoid overlap
- Phase 3: label "Foveation Mask" (was "Fovea Mask")
- All images have white border (stroke_width=5.0, matching arrow thickness)
- Arrow geometry fixed: exact T-junction alignment, no floating gaps or sticking out
"""
import numpy as np
from manim import *
from manimpango import register_font

for _f in ["cmunrm.ttf", "cmunbx.ttf", "cmunti.ttf", "cmunbi.ttf"]:
    register_font(f"/home/brianchc/fonts/cmu-serif/{_f}")

FONT      = "CMU Serif"
RAND_DIR  = "/home/brianchc/project_animations/foveated_diffusion/mask_training/data/random_masking"
SAL_DIR   = "/home/brianchc/project_animations/foveated_diffusion/mask_training/data/saliency_masking"

TITLE_Y   = 3.2
LBL_BUF   = 0.18
STROKE_W  = 5.0
ARROW_GAP = 0.08   # gap between arrowhead tip and image border

_AKW = dict(color=WHITE, stroke_width=STROKE_W, tip_length=0.25,
            max_stroke_width_to_length_ratio=999)


def _t(s, sz=24):
    return Text(s, font=FONT, font_size=sz, color=WHITE)


def _ln(a, b):
    return Line(a, b, color=WHITE, stroke_width=STROKE_W)


def _ar(a, b):
    return Arrow(a, b, buff=0, **_AKW)


def _img(path, width):
    """Load image and wrap it with a white border rectangle."""
    im  = ImageMobject(path).set_width(width)
    bdr = SurroundingRectangle(im, color=WHITE, stroke_width=STROKE_W, buff=0)
    return Group(im, bdr)


class MaskTraining(Scene):

    def construct(self):
        self._p1()
        self._p2()
        self._p3()

    # ── Phase 1: centered input image, no title ────────────────────────────────

    def _p1(self):
        ig  = _img(f"{SAL_DIR}/image.jpg", 5.25)
        ig.move_to(ORIGIN)
        tag = _t("Input Image", 28).next_to(ig, UP, buff=LBL_BUF)
        self.add(tag)
        self.play(FadeIn(ig), run_time=0.8)
        self.wait(2.0)
        self._ig  = ig
        self._tag = tag

    # ── Phase 2: randomized-mask training ─────────────────────────────────────

    def _p2(self):
        IMG_X, IMG_W = -5.0, 2.6
        W_EACH  = 1.65           # 1.5× v2's 1.1
        SAMP_YS = [1.9, 0.0, -1.9]   # spacing 1.9 > W_EACH 1.65 → 0.25-unit vertical gap
        SAMP_X  = 4.5            # pair centers; right edge ≈ 6.2, within canvas
        STEM_X  = -1.0

        # ── Step 1: image slides left; tag fades out (NO title yet) ───────────
        _ref    = self._ig.copy().set_width(IMG_W).move_to([IMG_X, 0, 0])
        new_tag = _t("Input Image", 22).next_to(_ref, UP, buff=LBL_BUF)

        self.play(
            self._ig.animate.set_width(IMG_W).move_to([IMG_X, 0, 0]),
            FadeOut(self._tag),
            run_time=0.7,
        )
        self.add(new_tag)
        self._tag = new_tag

        # ── Step 2: title appears AFTER image has settled ─────────────────────
        new_title = _t("Randomized-Mask Training", 42).move_to([0, TITLE_Y, 0])
        self.add(new_title)
        self._title = new_title

        # ── Paired sample images (each with white border) ─────────────────────
        gap = W_EACH * 0.05    # 5% intra-pair gap
        samp_pairs = []
        for k, sy in enumerate(SAMP_YS):
            im1  = _img(f"{RAND_DIR}/{k:04d}/circle.jpg", W_EACH)
            im2  = _img(f"{RAND_DIR}/{k:04d}/foveation_mask_pix64_up512.png", W_EACH)
            pair = Group(im1, im2).arrange(RIGHT, buff=gap)
            pair.move_to([SAMP_X, sy, 0])
            samp_pairs.append(pair)

        # ── Arrow geometry — exact T-junction, no floating offsets ────────────
        img_rx  = self._ig.get_right()[0]        # right edge of input image
        samp_lx = samp_pairs[0].get_left()[0]    # left edge of pairs

        feeder = _ln([img_rx,  0, 0], [STEM_X, 0, 0])
        stem   = _ln([STEM_X, SAMP_YS[-1], 0], [STEM_X, SAMP_YS[0], 0])
        brs    = VGroup(*[
            _ar([STEM_X, sy, 0], [samp_lx - ARROW_GAP, sy, 0])
            for sy in SAMP_YS
        ])

        self.play(Create(feeder), Create(stem), run_time=0.4)
        self.play(LaggedStart(*[Create(b) for b in brs], lag_ratio=0.25), run_time=0.7)
        self.play(
            LaggedStart(*[FadeIn(p) for p in samp_pairs], lag_ratio=0.3),
            run_time=1.0,
        )
        self.wait(0.8)

        # ── Pulsing: flash travels along arrow path then pair bounces ──────────
        arrow_end = samp_lx - ARROW_GAP
        for k in [2, 0, 1, 2, 0]:
            sy = SAMP_YS[k]
            if abs(sy) < 0.01:  # middle branch — straight horizontal
                pts = [
                    np.array([img_rx,    0,  0]),
                    np.array([arrow_end, 0,  0]),
                ]
            else:               # top/bottom — feeder → down/up stem → branch
                pts = [
                    np.array([img_rx,    0,  0]),
                    np.array([STEM_X,    0,  0]),
                    np.array([STEM_X,    sy, 0]),
                    np.array([arrow_end, sy, 0]),
                ]
            pulse = VMobject()
            pulse.set_points_as_corners(pts)
            pulse.set_stroke(WHITE, width=12, opacity=1.0)
            pulse.set_fill(opacity=0)

            self.play(ShowPassingFlash(pulse, time_width=0.15), run_time=0.65)
            self.play(samp_pairs[k].animate.scale(1.08),      run_time=0.12)
            self.play(samp_pairs[k].animate.scale(1 / 1.08),  run_time=0.12)

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

        # Input image stays at (−5.0, 0), width=2.6 → right edge = −3.7
        img_rx = self._ig.get_right()[0]

        # Saliency map (center)
        sal     = _img(f"{SAL_DIR}/saliency.png", 2.5)
        sal.move_to([0, 0, 0])
        sal_tag = _t("Saliency Map", 22).next_to(sal, UP, buff=LBL_BUF)

        # Foveation mask (right)
        fov     = _img(f"{SAL_DIR}/foveation_mask_pix64_up512.png", 2.5)
        fov.move_to([4.2, 0, 0])
        fov_tag = _t("Foveation Mask", 22).next_to(fov, UP, buff=LBL_BUF)

        # Arrow 1: input image → saliency map
        arr1     = _ar([img_rx + ARROW_GAP, 0, 0],
                       [sal.get_left()[0] - ARROW_GAP, 0, 0])
        arr1_lbl = _t("Saliency\nDetection", 20).next_to(arr1, UP, buff=0.15)

        # Arrow 2: saliency map → foveation mask
        arr2     = _ar([sal.get_right()[0] + ARROW_GAP, 0, 0],
                       [fov.get_left()[0] - ARROW_GAP, 0, 0])
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
