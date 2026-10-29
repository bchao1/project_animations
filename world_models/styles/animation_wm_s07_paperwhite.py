"""Style 07 — Paper white.
Light theme (off-white page, ink text, deeper accents). Title top-left with a coloured venue pill, demo on a card with a
soft shadow, paper figures on the right, and a 3b1b number-line style tracker whose dot walks across the four controls.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm_common import *  # noqa

PAGE, INK, SOFT = "#F7F6F2", "#1F2328", "#6B7280"
ACC = ["#1C7ED6", "#E8590C", "#0CA678", "#7048E8"]
VID_C = np.array([-1.9, -0.05, 0])
TR = 0.34


def shadow(w, h, r, pos):
    g = VGroup()
    for i, op in enumerate([0.05, 0.05, 0.04, 0.03]):
        d = 0.04 * (i + 1)
        g.add(RoundedRectangle(width=w + 2 * d, height=h + 2 * d, corner_radius=r + d).set_stroke(width=0)
              .set_fill(BLACK, op).move_to(pos + np.array([0.05, -0.08, 0])))
    return g.set_z_index(-3)  # under the video card (VideoCard sits at z -2)


class S07_PaperWhite(TimedScene):
    def construct(self):
        self.camera.background_color = PAGE
        cards = [VideoCard(s["key"], s["speed"], box=(9.2, 5.0), center=VID_C, radius=0.16, border=None) for s in SEGS]
        shadows = [shadow(c.w_u, c.h_u, 0.16, VID_C) for c in cards]

        def title(k):
            s = SEGS[k]
            t = T(s["title"], 36, INK, weight=BOLD)
            if t.width > 10.6:
                t.set_width(10.6)
            v = T(s["venue"], 22, WHITE, weight=BOLD)
            pill = RoundedRectangle(width=v.width + 0.36, height=0.42, corner_radius=0.21).set_fill(ACC[k], 1).set_stroke(width=0)
            pv = VGroup(pill, v.move_to(pill))
            g = VGroup(t, pv).arrange(RIGHT, buff=0.3)
            if g.width > 13.4:
                g = VGroup(t, pv).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            return g.move_to([-6.65, 3.3, 0], aligned_edge=LEFT)
        titles = [title(k) for k in range(4)]
        figs = []
        for k, s in enumerate(SEGS):
            st = fig_stack(s, width=3.5, max_h=4.0, label_size=16, label_color=SOFT)
            figs.append(st.move_to([4.95, -0.05, 0]))

        # tracker
        x0, x1, y = -5.4, 5.4, -3.45
        line = Line([x0, y, 0], [x1, y, 0]).set_stroke(SOFT, 2, opacity=0.5)
        xs = np.linspace(x0 + 0.9, x1 - 0.9, 4)
        ticks = VGroup(*[Line([x, y - 0.1, 0], [x, y + 0.1, 0]).set_stroke(SOFT, 2) for x in xs])
        tl = VGroup(*[VGroup(LiveIcon(k, SOFT, s=0.85, width=2.2), T(SEGS[k]["control"], 18, SOFT)).arrange(RIGHT, buff=0.08)
                      .next_to(ticks[k], DOWN, buff=0.08) for k in range(4)])
        wm = T("World Models", 20, INK, weight=HEAVY).move_to([x0 - 0.05, y, 0], aligned_edge=RIGHT).shift(LEFT * 0.15)
        wm.set_x(-6.65 + wm.width / 2)
        line.put_start_and_end_on([wm.get_right()[0] + 0.25, y, 0], [x1 + 1.2, y, 0])
        dot = Dot([xs[0], y, 0], radius=0.11, color=ACC[0])

        c0 = cards[0].start(self)
        tl[0][0].color_ = ACC[0]
        tl[0][0].activate()
        self.play_t(c0.alpha.animate.set_value(1), FadeIn(shadows[0]), FadeIn(titles[0], shift=UP * 0.2),
                    FadeIn(figs[0], shift=LEFT * 0.3), Create(line), FadeIn(ticks), FadeIn(tl), FadeIn(wm), GrowFromCenter(dot),
                    tl[0][1].animate.set_color(INK), run_time=0.42)
        for k in range(1, 4):
            self.goto(BOUNDS[k] - TR / 2)
            old, new = cards[k - 1], cards[k]
            new.start(self)
            new.w.set_value(new.w_u * 0.96)
            tl[k][0].color_ = ACC[k]
            tl[k][0].activate()
            tl[k - 1][0].live = False
            self.play_t(old.alpha.animate.set_value(0), new.alpha.animate.set_value(1), new.w.animate.set_value(new.w_u),
                        Transform(shadows[0], shadows[k]),
                        FadeOut(titles[k - 1], shift=UP * 0.2), FadeIn(titles[k], shift=UP * 0.2),
                        FadeOut(figs[k - 1], shift=UP * 0.25), FadeIn(figs[k], shift=UP * 0.25),
                        dot.animate.move_to([xs[k], y, 0]).set_color(ACC[k]),
                        tl[k - 1][1].animate.set_color(SOFT), tl[k][1].animate.set_color(INK), run_time=TR)
            self.remove(*old.mobs())
        self.goto(9.55)
        span = Line([xs[0], y, 0], [xs[-1], y, 0]).set_stroke(INK, 4)
        for k in range(4):
            tl[k][0].color_, tl[k][0].live = ACC[k], True
        self.play_t(Create(span), *[tl[k][1].animate.set_color(ACC[k]) for k in range(4)], run_time=0.3)
        self.goto(10.0)
