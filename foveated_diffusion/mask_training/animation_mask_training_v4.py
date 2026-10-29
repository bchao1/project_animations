"""
Foveated Diffusion – Masking Strategies (v4)
Changes from v3:
- Phase 2: "Foveation\\nMask" label below each mask image; W_EACH 1.65→1.3, SAMP_YS→[2.2,0,-2.2],
  TITLE_Y→3.4 so all text (title / pair images / labels) is non-overlapping
- Phase 3: output is circle.jpg + mask side by side (same format as Phase 2) with
  "Foveation\\nMask" label below the mask; saliency map pipeline preserved
- All text guaranteed non-overlapping
"""
import numpy as np
from manim import *
from manimpango import register_font

for _f in ["cmunrm.ttf", "cmunbx.ttf", "cmunti.ttf", "cmunbi.ttf"]:
    register_font(f"/home/brianchc/fonts/cmu-serif/{_f}")

FONT      = "CMU Serif"
RAND_DIR  = "/home/brianchc/project_animations/foveated_diffusion/mask_training/data/random_masking"
SAL_DIR   = "/home/brianchc/project_animations/foveated_diffusion/mask_training/data/saliency_masking"

TITLE_Y   = 3.4      # raised from 3.2 to clear top pair
LBL_BUF   = 0.18
STROKE_W  = 5.0
ARROW_GAP = 0.08

_AKW = dict(color=WHITE, stroke_width=STROKE_W, tip_length=0.25,
            max_stroke_width_to_length_ratio=999)


def _t(s, sz=24):
    return Text(s, font=FONT, font_size=sz, color=WHITE)


def _ln(a, b):
    return Line(a, b, color=WHITE, stroke_width=STROKE_W)


def _ar(a, b):
    return Arrow(a, b, buff=0, **_AKW)


def _img(path, width):
    """ImageMobject with white border rectangle."""
    im  = ImageMobject(path).set_width(width)
    bdr = SurroundingRectangle(im, color=WHITE, stroke_width=STROKE_W, buff=0)
    return Group(im, bdr)


def _pair(circ_path, mask_path, w_each):
    """Return (pair_group, im2) where pair_group = Group(im1, im2) arranged LEFT→RIGHT."""
    gap  = w_each * 0.05
    im1  = _img(circ_path, w_each)
    im2  = _img(mask_path, w_each)
    pair = Group(im1, im2).arrange(RIGHT, buff=gap)
    return pair, im2


class MaskTraining(Scene):

    def construct(self):
        self._p1()
        self._p2()
        self._p3()

    # ── Phase 1: centered input image ─────────────────────────────────────────

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
        W_EACH  = 1.3
        gap     = W_EACH * 0.05
        pair_w  = 2 * W_EACH + gap           # ≈ 2.665
        SAMP_YS = [2.2, 0.0, -2.2]           # spacing > W_EACH + label height → no overlap
        SAMP_X  = 4.5
        STEM_X  = -1.0
        MASK_LBL_SZ = 14                      # "Foveation\nMask" font size

        # ── Step 1: image slides left, then title appears ──────────────────────
        _ref    = self._ig.copy().set_width(IMG_W).move_to([IMG_X, 0, 0])
        new_tag = _t("Input Image", 22).next_to(_ref, UP, buff=LBL_BUF)

        self.play(
            self._ig.animate.set_width(IMG_W).move_to([IMG_X, 0, 0]),
            FadeOut(self._tag),
            run_time=0.7,
        )
        self.add(new_tag)
        self._tag = new_tag

        new_title = _t("Randomized-Mask Training", 42).move_to([0, TITLE_Y, 0])
        self.add(new_title)
        self._title = new_title

        # ── Paired sample images + "Foveation\nMask" label below each mask ────
        # Layout check (square mask, W_EACH=1.3):
        #   pair top at sy=2.2: 2.2+0.65=2.85 | title bottom (42pt@3.4): ≈3.10 → gap 0.25 ✓
        #   label bottom at sy=2.2: 2.2-0.65-0.12-~0.23=~1.2 | next pair top 0+0.65=0.65 → gap 0.55 ✓
        samp_groups = []
        for k, sy in enumerate(SAMP_YS):
            pair, im2 = _pair(
                f"{RAND_DIR}/{k:04d}/circle.jpg",
                f"{RAND_DIR}/{k:04d}/foveation_mask_pix64_up512.png",
                W_EACH,
            )
            pair.move_to([SAMP_X, sy, 0])
            mask_lbl = _t("Foveation\nMask", MASK_LBL_SZ).next_to(im2, DOWN, buff=0.12)
            samp_groups.append(Group(pair, mask_lbl))

        # ── Arrow geometry — exact T-junction ─────────────────────────────────
        img_rx    = self._ig.get_right()[0]
        samp_lx   = SAMP_X - pair_w / 2          # geometric left edge of pairs
        arrow_end = samp_lx - ARROW_GAP

        feeder = _ln([img_rx, 0, 0], [STEM_X, 0, 0])
        stem   = _ln([STEM_X, SAMP_YS[-1], 0], [STEM_X, SAMP_YS[0], 0])
        brs    = VGroup(*[
            _ar([STEM_X, sy, 0], [arrow_end, sy, 0])
            for sy in SAMP_YS
        ])

        self.play(Create(feeder), Create(stem), run_time=0.4)
        self.play(LaggedStart(*[Create(b) for b in brs], lag_ratio=0.25), run_time=0.7)
        self.play(
            LaggedStart(*[FadeIn(g) for g in samp_groups], lag_ratio=0.3),
            run_time=1.0,
        )
        self.wait(0.8)

        # ── Pulsing flash ──────────────────────────────────────────────────────
        for k in [2, 0, 1, 2, 0]:
            sy = SAMP_YS[k]
            if abs(sy) < 0.01:
                pts = [np.array([img_rx, 0, 0]), np.array([arrow_end, 0, 0])]
            else:
                pts = [
                    np.array([img_rx,     0,  0]),
                    np.array([STEM_X,     0,  0]),
                    np.array([STEM_X,    sy,  0]),
                    np.array([arrow_end, sy,  0]),
                ]
            pulse = VMobject()
            pulse.set_points_as_corners(pts)
            pulse.set_stroke(WHITE, width=12, opacity=1.0)
            pulse.set_fill(opacity=0)
            self.play(ShowPassingFlash(pulse, time_width=0.15), run_time=0.65)
            self.play(samp_groups[k].animate.scale(1.08),     run_time=0.12)
            self.play(samp_groups[k].animate.scale(1 / 1.08), run_time=0.12)

        self.wait(1.5)

        self._p2o = Group(*samp_groups, feeder, stem, brs)

    # ── Phase 3: saliency-guided training ─────────────────────────────────────

    def _p3(self):
        W_EACH  = 1.3
        gap     = W_EACH * 0.05
        pair_w  = 2 * W_EACH + gap
        SAMP_X  = 4.5                         # same x as Phase 2 pairs
        MASK_LBL_SZ = 14

        new_title = _t("Saliency-Guided Training", 42).move_to([0, TITLE_Y, 0])
        self.play(
            Transform(self._title, new_title),
            FadeOut(self._p2o),
            run_time=0.7,
        )

        img_rx = self._ig.get_right()[0]

        # ── Saliency map (center stage) ────────────────────────────────────────
        sal     = _img(f"{SAL_DIR}/saliency.png", 2.5)
        sal.move_to([0, 0, 0])
        sal_tag = _t("Saliency Map", 22).next_to(sal, UP, buff=LBL_BUF)

        # ── Output: circle + mask pair, same format as Phase 2 ────────────────
        pair, im2 = _pair(
            f"{SAL_DIR}/circle.jpg",
            f"{SAL_DIR}/foveation_mask_pix64_up512.png",
            W_EACH,
        )
        pair.move_to([SAMP_X, 0, 0])
        mask_lbl  = _t("Foveation\nMask", MASK_LBL_SZ).next_to(im2, DOWN, buff=0.12)
        out_group = Group(pair, mask_lbl)

        # ── Arrows ─────────────────────────────────────────────────────────────
        samp_lx  = SAMP_X - pair_w / 2
        arr1     = _ar([img_rx + ARROW_GAP, 0, 0],
                       [sal.get_left()[0] - ARROW_GAP, 0, 0])
        arr1_lbl = _t("Saliency\nDetection", 20).next_to(arr1, UP, buff=0.15)

        arr2     = _ar([sal.get_right()[0] + ARROW_GAP, 0, 0],
                       [samp_lx - ARROW_GAP, 0, 0])
        arr2_lbl = _t("Thresholding", 20).next_to(arr2, UP, buff=0.15)

        # ── Animate pipeline ───────────────────────────────────────────────────
        self.play(Create(arr1), run_time=0.5)
        self.add(arr1_lbl)
        self.play(FadeIn(sal), run_time=0.5)
        self.add(sal_tag)
        self.wait(0.3)

        self.play(Create(arr2), run_time=0.5)
        self.add(arr2_lbl)
        self.play(FadeIn(out_group), run_time=0.5)
        self.wait(3.0)
