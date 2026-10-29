"""Style 08 — Concept glyph -> video.
Each paper opens with a big hand-drawn-feeling 3b1b glyph of its idea (a camera frustum flying a path; a hand skeleton
closing on an object; a time-vs-camera plot where the dot climbs the camera axis with time frozen; a point cloud that
accumulates along a loop). The demo video then grows out of the glyph, which shrinks into a corner badge.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm_common import *  # noqa

VID_C = np.array([-1.7, 0.2, 0])
GLYPH_C = np.array([-1.7, 0.35, 0])
BADGE_C = np.array([-6.25, 2.55, 0])
FIG_X = 5.0
INTRO = 0.72   # glyph-alone time per paper
GROW = 0.36


class S08_GlyphToVideo(TimedScene):
    def construct(self):
        self.camera.background_color = BG
        cards = [VideoCard(s["key"], s["speed"], box=(9.6, 5.25), center=VID_C) for s in SEGS]
        figs = [fig_stack(s, width=3.35, max_h=4.0).move_to([FIG_X, 0.2, 0]) for s in SEGS]
        caps = [caption(s, size=34, venue_size=25).move_to([-6.5, -3.3, 0], aligned_edge=LEFT) for s in SEGS]
        head = T("WORLD MODELS", 20, MUTED, weight=HEAVY).move_to([6.8, 3.65, 0], aligned_edge=RIGHT)
        self.add(head)
        prev = None
        for k, s in enumerate(SEGS):
            t0 = BOUNDS[k]
            g, prog = glyph(k, size=3.3, center=GLYPH_C)
            word = T(s["control"].lower(), 44, s["color"], weight=BOLD).move_to(GLYPH_C + DOWN * 2.05)
            ring = Circle(radius=2.3).set_stroke(s["color"], 2, opacity=0.5).set_fill(BG, 0.94).move_to(GLYPH_C)
            ring_follow = ring.copy()
            self.goto(t0)
            outs = []
            if prev is not None:
                pc, pb, pf, pcap = prev
                outs = [pc.alpha.animate.set_value(0), FadeOut(pb), FadeOut(pf, shift=UP * 0.25), FadeOut(pcap, shift=UP * 0.2)]
            # glyph beat
            self.add(ring)
            self.play_t(*outs, FadeIn(word, shift=UP * 0.2), GrowFromCenter(ring), FadeIn(figs[k], shift=LEFT * 0.3),
                        FadeIn(caps[k], shift=UP * 0.2), prog.animate(rate_func=linear).set_value(0.18), run_time=0.3)
            self.add(g)  # glyph appears once the backing disc covers the outgoing clip
            if prev is not None:
                self.remove(*prev[0].mobs())
            self.play_t(prog.animate(rate_func=linear).set_value(0.55), run_time=INTRO - 0.3)
            # video grows out of the glyph; glyph shrinks into a badge
            c = cards[k].start(self)
            c.w.set_value(3.0)
            c.x.set_value(GLYPH_C[0]), c.y.set_value(GLYPH_C[1])
            badge_back = Circle(radius=0.62).set_fill(BG, 0.9).set_stroke(s["color"], 2.5).move_to(BADGE_C)
            self.play_t(c.alpha.animate.set_value(1), c.w.animate.set_value(c.w_u), c.x.animate.set_value(VID_C[0]),
                        c.y.animate.set_value(VID_C[1]), FadeOut(word), FadeOut(g, scale=0.4), Transform(ring, badge_back),
                        prog.animate(rate_func=linear).set_value(0.7), run_time=GROW)
            self.bring_to_front(ring)
            badge_prog = prog
            # glyph lives on (small) inside the badge while the demo plays
            g.clear_updaters()
            small, sp = glyph(k, size=0.95, prog=badge_prog, center=BADGE_C)
            self.remove(g)
            self.add(small)
            self.play_until(BOUNDS[k + 1], badge_prog.animate(rate_func=linear).set_value(1.0))
            prev = (c, VGroup(ring, small), figs[k], caps[k])
        self.goto(10.0)
