"""Style 03 — Equation + brace.
A 3b1b-style formula across the top,  frames = G( past | camera, hands, time, memory ),  defines a world model. For
each paper its argument lights up and a brace slides under it, pointing down into the demo video.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm_common import *  # noqa

VID_C = np.array([-1.55, -0.05, 0])
TR = 0.34


class S03_Equation(TimedScene):
    def construct(self):
        self.camera.background_color = BG
        sz = 40
        lhs = T("next frames", sz, FG)
        eq = T("=", sz, MUTED)
        G = T("G", sz + 4, FG, slant=ITALIC, weight=BOLD)
        lp, rp = T("(", sz + 6, MUTED), T(")", sz + 6, MUTED)
        past = T("past", sz, MUTED)
        bar = T("|", sz, MUTED)
        args = [T(s["control"].lower(), sz, DIM_TXT) for s in SEGS]
        commas = [T(",", sz, DIM_TXT) for _ in range(3)]
        seq = [lhs, eq, G, lp, past, bar]
        for i, a in enumerate(args):
            seq.append(a)
            if i < 3:
                seq.append(commas[i])
        seq.append(rp)
        formula = VGroup(*seq).arrange(RIGHT, buff=0.16, aligned_edge=DOWN)
        mid = lhs.get_center()[1]
        for m in (eq, lp, rp, bar):
            m.set_y(mid)
        for c in commas:
            c.shift(LEFT * 0.1)
        formula.move_to([0, 3.35, 0])
        tag = T("WORLD MODEL", 18, MUTED, weight=HEAVY).next_to(formula, UP, buff=0.12).align_to(formula, LEFT)
        tag.shift(DOWN * 0.05)
        if tag.get_top()[1] > 3.95:
            formula.shift(DOWN * (tag.get_top()[1] - 3.9))
            tag.shift(DOWN * (tag.get_top()[1] - 3.9))

        def brace_for(k):
            b = Brace(args[k], DOWN, buff=0.08, sharpness=1.2).set_color(SEGS[k]["color"])
            return b

        cards = [VideoCard(s["key"], s["speed"], box=(8.9, 4.75), center=VID_C) for s in SEGS]
        for c in cards:  # align tops just under the formula
            c.y.set_value(2.45 - c.h_u / 2)
        figs = [fig_stack(s, width=3.35, max_h=3.9).move_to([5.1, 0.0, 0]) for s in SEGS]
        caps = [caption(s, size=34, venue_size=25) for s in SEGS]
        for k, cap in enumerate(caps):
            cap.move_to([-6.0, 2.45 - cards[k].h_u - 0.62, 0], aligned_edge=LEFT)

        # a thin connector from the brace tip down to the card
        def connector(k):
            b = brace_for(k)
            return DashedLine(b.get_bottom() + DOWN * 0.05, [b.get_bottom()[0], 2.47, 0], dash_length=0.05).set_stroke(SEGS[k]["color"], 2)

        c0 = cards[0].start(self)
        brace = brace_for(0)
        conn = connector(0)
        self.play_t(Write(formula, run_time=0.45), FadeIn(tag), c0.alpha.animate.set_value(1),
                    FadeIn(figs[0], shift=LEFT * 0.3), FadeIn(caps[0], shift=UP * 0.2), run_time=0.45)
        self.play_t(args[0].animate.set_color(SEGS[0]["color"]), GrowFromCenter(brace), Create(conn), run_time=0.3)

        for k in range(1, 4):
            self.goto(BOUNDS[k] - TR / 2)
            old, new = cards[k - 1], cards[k]
            new.start(self)
            y_new = new.y.get_value()
            new.y.set_value(y_new - 0.3)
            self.play_t(old.alpha.animate.set_value(0), new.alpha.animate.set_value(1), new.y.animate.set_value(y_new),
                        args[k - 1].animate.set_color("#B4BBC7"), args[k].animate.set_color(SEGS[k]["color"]),
                        Transform(brace, brace_for(k)), Transform(conn, connector(k)),
                        FadeOut(figs[k - 1], shift=UP * 0.25), FadeIn(figs[k], shift=UP * 0.25),
                        FadeOut(caps[k - 1], shift=UP * 0.2), FadeIn(caps[k], shift=UP * 0.2), run_time=TR)
            self.remove(*old.mobs())

        # finale: every argument lit, brace spans all four controls
        self.goto(9.5)
        big = Brace(VGroup(*args), DOWN, buff=0.08).set_color(FG)
        self.play_t(*[a.animate.set_color(SEGS[i]["color"]) for i, a in enumerate(args)],
                    Transform(brace, big), FadeOut(conn), run_time=0.35)
        self.goto(10.0)
