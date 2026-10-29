"""Style 06 — Paper -> demo.
Split screen: the paper figure on a white sheet (left) feeds an arrow — carrying the control's animated icon — into the
generated world (right). Title top-left; a four-step control tracker along the bottom.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm_common import *  # noqa

FIG_C = np.array([-4.35, -0.15, 0])
VID_C = np.array([2.95, -0.15, 0])
TR = 0.34


class S06_PaperToDemo(TimedScene):
    def construct(self):
        self.camera.background_color = BG
        cards = [VideoCard(s["key"], s["speed"], box=(7.7, 4.6), center=VID_C) for s in SEGS]
        figs = []
        for s in SEGS:
            st = fig_stack(s, width=4.6, max_h=4.3, label_size=17, buff=0.18)
            figs.append(st.move_to(FIG_C))
        lab_paper = T("from the paper", 18, MUTED).move_to([FIG_C[0], -2.78, 0])
        lab_world = T("generated world", 18, MUTED).move_to([VID_C[0], -2.78, 0])
        caps = [caption(s, size=38, venue_size=26, max_w=12.6).move_to([-6.7, 3.3, 0], aligned_edge=LEFT) for s in SEGS]

        def arrow(k):
            return Arrow([-1.85, -0.15, 0], [-0.9, -0.15, 0], buff=0, stroke_width=6, tip_length=0.22,
                         max_tip_length_to_length_ratio=0.4).set_color(SEGS[k]["color"])
        icons = [LiveIcon(k, SEGS[k]["color"], s=1.6, width=3).move_to([-1.37, 0.42, 0]) for k in range(4)]

        # bottom tracker: 4 labelled nodes on a line
        xs = np.linspace(-4.5, 4.5, 4)
        rail = Line([xs[0], -3.4, 0], [xs[-1], -3.4, 0]).set_stroke(DIM, 3)
        nodes = VGroup(*[Dot([x, -3.4, 0], radius=0.09, color=DIM) for x in xs])
        nlabs = VGroup(*[T(s["control"], 18, DIM_TXT).next_to(nodes[i], DOWN, buff=0.1) for i, s in enumerate(SEGS)])
        self.t_prog = ValueTracker(0.0)
        prog_line = always_redraw(lambda: Line([xs[0], -3.4, 0], [xs[0] + 1e-3 + (xs[-1] - xs[0]) * np.clip(self.t_prog.get_value(), 0, 1), -3.4, 0])
                                  .set_stroke(FG, 3))

        c0 = cards[0].start(self)
        arr = arrow(0)
        self.add(rail, prog_line, nodes, nlabs)
        self.play_t(FadeIn(figs[0], shift=RIGHT * 0.3), c0.alpha.animate.set_value(1), FadeIn(lab_paper), FadeIn(lab_world),
                    FadeIn(caps[0], shift=UP * 0.2), nodes[0].animate.set_color(SEGS[0]["color"]).scale(1.4),
                    nlabs[0].animate.set_color(FG), run_time=0.35)
        self.play_t(GrowArrow(arr), FadeIn(icons[0].activate(), scale=0.5), run_time=0.3)
        for k in range(1, 4):
            self.play_until(BOUNDS[k] - TR / 2, self.t_prog.animate(rate_func=linear).set_value((k - 0.5) / 3))
            old, new = cards[k - 1], cards[k]
            new.start(self)
            self.play_t(old.alpha.animate.set_value(0), new.alpha.animate.set_value(1),
                        FadeOut(figs[k - 1], shift=LEFT * 0.4), FadeIn(figs[k], shift=RIGHT * 0.4),
                        FadeOut(caps[k - 1], shift=UP * 0.2), FadeIn(caps[k], shift=UP * 0.2),
                        Transform(arr, arrow(k)), FadeOut(icons[k - 1], scale=0.5), FadeIn(icons[k].activate(), scale=0.5),
                        nodes[k - 1].animate.scale(1 / 1.4), nodes[k].animate.set_color(SEGS[k]["color"]).scale(1.4),
                        nlabs[k - 1].animate.set_color("#B4BBC7"), nlabs[k].animate.set_color(FG),
                        self.t_prog.animate.set_value(k / 3), run_time=TR)
            self.remove(*old.mobs())
        self.goto(9.55)
        self.play_t(*[n.animate.set_color(SEGS[i]["color"]) for i, n in enumerate(nodes)],
                    *[l.animate.set_color(FG) for l in nlabs], run_time=0.3)
        self.goto(10.0)
