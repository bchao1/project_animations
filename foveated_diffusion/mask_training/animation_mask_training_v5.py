"""
Foveated Diffusion – Masking Strategies (v5)
Changes from v4:
- Phase 2: circle.jpg removed; only mask images, stacked vertically.
  W_EACH=1.8 (maximised, verified no-overlap).  SAMP_YS=[1.3,-0.7,-2.7], spacing=2.0.
  Single shared "Foveation Mask" label (22pt) above the top mask, centered on the column.
- Phase 3: circle.jpg removed; mask only, width=2.5 (matches saliency map).
  "Foveation Mask" label (22pt) above mask, same format as "Saliency Map" label.
- All text non-overlapping (verified per-element).
"""
import numpy as np
from manim import *
from manimpango import register_font

for _f in ["cmunrm.ttf", "cmunbx.ttf", "cmunti.ttf", "cmunbi.ttf"]:
    register_font(f"/home/brianchc/fonts/cmu-serif/{_f}")

FONT      = "CMU Serif"
RAND_DIR  = "/home/brianchc/project_animations/foveated_diffusion/mask_training/data/random_masking"
SAL_DIR   = "/home/brianchc/project_animations/foveated_diffusion/mask_training/data/saliency_masking"

TITLE_Y   = 3.4
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
        # Mask images only (no circle). Maximise size within canvas.
        # Canvas ±4.0 tall. Title bottom ≈ 3.10, label+buff consumes ~0.56 above top mask.
        # Available for 3 masks + 2 gaps: 3*W + 2*0.2 ≤ (3.10-0.56) - (-3.60) ≈ 6.14
        # → W = (6.14-0.4)/3 ≈ 1.91 → use W=1.8 for comfortable margin on all sides.
        W_EACH  = 1.8
        SAMP_X  = 4.5
        SAMP_YS = [1.3, -0.7, -2.7]   # spacing=2.0 > W_EACH+inter_gap=2.0 (no overlap)
        STEM_X  = -1.0

        # ── Transition: image slides left, title appears after ─────────────────
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

        # ── Three mask images (no circles) ────────────────────────────────────
        samp_masks = []
        for k, sy in enumerate(SAMP_YS):
            mask = _img(
                f"{RAND_DIR}/{k:04d}/foveation_mask_pix64_up512.png",
                W_EACH,
            )
            mask.move_to([SAMP_X, sy, 0])
            samp_masks.append(mask)

        # Single shared label centered above the top mask (column header)
        fov_lbl = _t("Foveation Mask", 22).next_to(samp_masks[0], UP, buff=LBL_BUF)

        # ── Arrow geometry ─────────────────────────────────────────────────────
        img_rx    = self._ig.get_right()[0]
        mask_lx   = SAMP_X - W_EACH / 2          # geometric left edge of masks
        arrow_end = mask_lx - ARROW_GAP

        feeder = _ln([img_rx, 0, 0], [STEM_X, 0, 0])
        stem   = _ln([STEM_X, SAMP_YS[-1], 0], [STEM_X, SAMP_YS[0], 0])
        brs    = VGroup(*[
            _ar([STEM_X, sy, 0], [arrow_end, sy, 0])
            for sy in SAMP_YS
        ])

        self.play(Create(feeder), Create(stem), run_time=0.4)
        self.play(LaggedStart(*[Create(b) for b in brs], lag_ratio=0.25), run_time=0.7)
        self.play(
            LaggedStart(*[FadeIn(m) for m in samp_masks], lag_ratio=0.3),
            run_time=1.0,
        )
        self.add(fov_lbl)   # label appears instantly after masks are in place
        self.wait(0.8)

        # ── Pulsing (scales individual mask, label stays fixed) ────────────────
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
            self.play(samp_masks[k].animate.scale(1.08),     run_time=0.12)
            self.play(samp_masks[k].animate.scale(1 / 1.08), run_time=0.12)

        self.wait(1.5)

        self._p2o = Group(*samp_masks, fov_lbl, feeder, stem, brs)

    # ── Phase 3: saliency-guided training ─────────────────────────────────────

    def _p3(self):
        # Mask width=2.5 matches saliency map; label format identical to "Saliency Map"
        FOV_X   = 4.2
        FOV_W   = 2.5

        new_title = _t("Saliency-Guided Training", 42).move_to([0, TITLE_Y, 0])
        self.play(
            Transform(self._title, new_title),
            FadeOut(self._p2o),
            run_time=0.7,
        )

        img_rx = self._ig.get_right()[0]

        # Saliency map
        sal     = _img(f"{SAL_DIR}/saliency.png", 2.5)
        sal.move_to([0, 0, 0])
        sal_tag = _t("Saliency Map", 22).next_to(sal, UP, buff=LBL_BUF)

        # Foveation mask (mask only, width=2.5, same as saliency)
        fov     = _img(f"{SAL_DIR}/foveation_mask_pix64_up512.png", FOV_W)
        fov.move_to([FOV_X, 0, 0])
        fov_tag = _t("Foveation Mask", 22).next_to(fov, UP, buff=LBL_BUF)

        # Arrows
        arr1     = _ar([img_rx + ARROW_GAP, 0, 0],
                       [sal.get_left()[0] - ARROW_GAP, 0, 0])
        arr1_lbl = _t("Saliency\nDetection", 20).next_to(arr1, UP, buff=0.15)

        arr2     = _ar([sal.get_right()[0] + ARROW_GAP, 0, 0],
                       [fov.get_left()[0] - ARROW_GAP, 0, 0])
        arr2_lbl = _t("Thresholding", 20).next_to(arr2, UP, buff=0.15)

        # Animate pipeline
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
