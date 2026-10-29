"""Style 01 — Control chips.
Header formula  WORLD MODELS = video model + [Camera] + [Hands] + [Time] + [Memory]; the active control chip lights up
and animates its icon. Demo video on a rounded card, paper figures in a right column, lower-third caption.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm_common import *  # noqa

VID_C = np.array([-1.95, 0.05, 0])
FIG_X, LEFT_X, HEADER_Y, TR = 4.97, -6.65, 3.45, 0.34


def camera_path_overlay(card):
    """Top-down camera path drawn over the CameraCtrl II clip, synced to its playback."""
    box = RoundedRectangle(width=1.9, height=1.5, corner_radius=0.1).set_fill(BLACK, 0.62).set_stroke(WHITE, 1, opacity=0.4)
    box.move_to(card.pos() + np.array([-card.w_u / 2 + 1.12, -card.h_u / 2 + 0.92, 0]))
    o = box.get_center()
    path = VMobject().set_points_smoothly([o + np.array(p) for p in
                                           [(-0.4, -0.45, 0), (-0.3, -0.1, 0), (-0.1, 0.18, 0), (0.3, 0.4, 0)]])
    path.set_stroke(C_CAM, 3.5)
    lab = T("camera path", 17, FG).move_to(o + np.array([0.3, -0.5, 0]))
    prog = ValueTracker(0.0)
    drawn = always_redraw(lambda: path.copy().pointwise_become_partial(path, 0, max(prog.get_value(), 1e-3)))

    def frustum():
        a = max(prog.get_value(), 1e-3)
        p = path.point_from_proportion(a)
        q = path.point_from_proportion(min(a + 0.02, 1.0)) - path.point_from_proportion(max(a - 0.02, 0.0))
        tri = Polygon(ORIGIN, [0.24, 0.13, 0], [0.24, -0.13, 0]).set_stroke(FG, 2).set_fill(C_CAM, 0.55)
        return tri.rotate(np.arctan2(q[1], q[0]), about_point=ORIGIN).shift(p)
    return VGroup(box, drawn, always_redraw(frustum), lab), prog


class S01_Chips(TimedScene):
    def construct(self):
        self.camera.background_color = BG
        head = T("WORLD MODELS", 26, FG, weight=HEAVY).move_to([LEFT_X, HEADER_Y, 0], aligned_edge=LEFT)
        chips = VGroup(*[Chip(k) for k in range(4)]).arrange(RIGHT, buff=0.42)
        chips.move_to([6.75, HEADER_Y, 0], aligned_edge=RIGHT)
        plus = VGroup(*[T("+", 22, DIM_TXT).move_to((chips[i].get_right() + chips[i + 1].get_left()) / 2) for i in range(3)])
        gen = T("=   video model   +", 22, MUTED).next_to(chips, LEFT, buff=0.26)
        if gen.get_left()[0] < head.get_right()[0] + 0.3:
            gen = T("=", 24, MUTED).next_to(chips, LEFT, buff=0.26)
        uline = Line(LEFT, RIGHT).set_stroke(C_CAM, 3).set_width(chips[0].width - 0.2).next_to(chips[0], DOWN, buff=0.1)

        cards = [VideoCard(s["key"], s["speed"], box=(9.4, 5.3), center=VID_C) for s in SEGS]
        figs = [fig_stack(s).move_to([FIG_X, VID_C[1], 0]) for s in SEGS]
        caps = [caption(s).move_to([LEFT_X, -3.3, 0], aligned_edge=LEFT) for s in SEGS]
        overlay, prog = camera_path_overlay(cards[0])

        c0 = cards[0].start(self)
        c0.y.set_value(VID_C[1] - 0.25)
        chips[0].style("on", instant=True)
        self.play_t(c0.alpha.animate.set_value(1), c0.y.animate.set_value(VID_C[1]),
                    FadeIn(head, shift=RIGHT * 0.2), FadeIn(gen), FadeIn(chips, lag_ratio=0.15), FadeIn(plus),
                    GrowFromCenter(uline), FadeIn(figs[0], shift=UP * 0.25), FadeIn(caps[0], shift=UP * 0.2),
                    FadeIn(overlay[0]), run_time=0.42)
        self.add(*overlay[1:])
        self.play_until(BOUNDS[1] - TR / 2, prog.animate(rate_func=linear).set_value(1.0))

        for k in range(1, 4):
            old, new = cards[k - 1], cards[k]
            new.start(self)
            new.y.set_value(VID_C[1] - 0.3)
            anims = [old.alpha.animate.set_value(0), old.y.animate.set_value(VID_C[1] + 0.3),
                     new.alpha.animate.set_value(1), new.y.animate.set_value(VID_C[1]),
                     FadeOut(figs[k - 1], shift=UP * 0.25), FadeIn(figs[k], shift=UP * 0.25),
                     FadeOut(caps[k - 1], shift=UP * 0.2), FadeIn(caps[k], shift=UP * 0.2),
                     uline.animate.set_width(chips[k].width - 0.2).next_to(chips[k], DOWN, buff=0.1).set_stroke(SEGS[k]["color"])]
            anims += chips[k - 1].style("done") + chips[k].style("on")
            if k == 1:
                anims.append(FadeOut(overlay, shift=UP * 0.3))
            self.play_t(*anims, run_time=TR)
            self.remove(*old.mobs())
            if k < 3:
                self.goto(BOUNDS[k + 1] - TR / 2)

        self.goto(9.45)
        wide = Line(LEFT, RIGHT).set_stroke(FG, 3).set_width(chips.width).next_to(chips, DOWN, buff=0.1)
        fin = [uline.animate.become(wide)]
        for ch in chips:
            fin += ch.style("all")
        self.play_t(*fin, run_time=0.35)
        self.goto(10.0)
