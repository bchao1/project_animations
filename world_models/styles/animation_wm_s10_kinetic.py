"""Style 10 — Kinetic type.
Full-bleed demo, dimmed while the paper title types on word by word in big heavy sans; the title then glides down to a
lower-third, the footage brightens, and the paper figure pops in top-right. A thin progress line runs along the bottom.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm_common import *  # noqa

TR = 0.26
HOLD = 0.85   # big-title time per paper before it shrinks


class S10_Kinetic(TimedScene):
    def big_title(self, k):
        s = SEGS[k]
        lines = VGroup(*[T(l, 62, WHITE, weight=HEAVY) for l in s["lines"]]).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        if lines.width > 12.4:
            lines.set_width(12.4)
        venue = T(s["venue"], 34, s["color"], weight=BOLD)
        g = VGroup(lines, venue).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        return g.move_to([-6.3, 0.2, 0], aligned_edge=LEFT)

    def construct(self):
        self.camera.background_color = "#07080B"
        cards = []
        for s in SEGS:
            cover = s["key"] in ("camctrl2", "genreal")
            cards.append(VideoCard(s["key"], s["speed"], box=(FW, FH), radius=0, border=None, cover=cover))
        band = fade_band(height=3.2, max_op=0.95, power=1.0)
        prog = ValueTracker(0.0)

        def pline():
            p = prog.get_value()
            k = min(int(p * 4), 3)
            return Line([-FW / 2, -FH / 2 + 0.04, 0], [-FW / 2 + FW * max(p, 1e-3), -FH / 2 + 0.04, 0]).set_stroke(SEGS[k]["color"], 6)
        line = always_redraw(pline)
        tag = T("WORLD MODELS", 22, WHITE, weight=HEAVY).move_to([-6.6, 3.55, 0], aligned_edge=LEFT)
        tag_bg = fade_band(height=1.2, y_bottom=FH / 2, max_op=0.5, up=False)

        def pip(k):
            st = fig_stack(SEGS[k], width=2.9, max_h=2.8, label_size=14, label_color="#D8DEE9", buff=0.12)
            back = RoundedRectangle(width=st.width + 0.26, height=st.height + 0.26, corner_radius=0.12)
            back.set_fill(BLACK, 0.6).set_stroke(WHITE, 1, opacity=0.3).move_to(st)
            return Group(back, st).move_to([6.85, 3.3, 0], aligned_edge=UR)

        self.add(band, tag_bg, line, tag)
        prev = None
        for k, s in enumerate(SEGS):
            t0 = BOUNDS[k]
            self.goto(max(t0 - (TR / 2 if k else 0), self.t))
            c = cards[k].start(self)
            c.dim.set_value(0.72 if s["key"] == "spmem" else 0.6)
            big = self.big_title(k)
            words = VGroup(*[w for l in big[0] for w in l]) if False else big[0]
            outs = []
            if prev is not None:
                pc, psmall, ppip = prev
                outs = [pc.alpha.animate.set_value(0), FadeOut(psmall, shift=DOWN * 0.2), FadeOut(ppip)]
            self.play_t(*outs, c.alpha.animate.set_value(1), prog.animate(rate_func=linear).set_value((self.t + TR) / 10),
                        run_time=TR)
            if prev is not None:
                self.remove(*prev[0].mobs())
            # type the title on (letters of each line, staggered) + venue
            self.play_t(AddTextLetterByLetter(big[0][0], time_per_char=0.012, run_time=0.3) if False else
                        LaggedStart(*[FadeIn(l, shift=UP * 0.25) for l in big[0]], lag_ratio=0.35),
                        FadeIn(big[1], shift=UP * 0.2), prog.animate(rate_func=linear).set_value((self.t + 0.35) / 10),
                        run_time=0.35)
            self.play_t(prog.animate(rate_func=linear).set_value((self.t + HOLD - 0.35) / 10), run_time=HOLD - 0.35)
            # glide to lower third, brighten, figure in
            if s["key"] == "genreal":  # headset inset lives bottom-left in this clip; caption goes right, two lines
                small = caption(s, size=36, venue_size=26, two_lines=True, align=RIGHT).move_to([6.6, -2.95, 0], aligned_edge=RIGHT)
            else:
                small = caption(s, size=36, venue_size=26, max_w=12.2)
                small.move_to([-6.55, -3.2, 0], aligned_edge=LEFT)
            p = pip(k)
            self.play_t(Transform(big, small), c.dim.animate.set_value(0.0), FadeIn(p, shift=LEFT * 0.3),
                        prog.animate(rate_func=linear).set_value((self.t + 0.35) / 10), run_time=0.35)
            end = BOUNDS[k + 1] - (TR / 2 if k < 3 else 0)
            self.play_until(end, prog.animate(rate_func=linear).set_value(end / 10))
            prev = (c, big, p)
        self.goto(10.0)
