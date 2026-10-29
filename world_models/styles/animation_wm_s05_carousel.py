"""Style 05 — Carousel.
All four demos sit on a horizontal track; the active one is centred and bright, neighbours peek in dimmed. The track
slides left for each paper. Paper figure is a sticker on the card's corner; caption centred below with page dots.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm_common import *  # noqa

CARD_W, CARD_H, STEP, Y = 9.0, 5.06, 9.75, 0.55
TR = 0.42


class S05_Carousel(TimedScene):
    def construct(self):
        self.camera.background_color = BG
        cards = [VideoCard(s["key"], s["speed"], box=(CARD_W, CARD_H), center=[k * STEP, Y, 0], radius=0.16)
                 for k, s in enumerate(SEGS)]
        off = ValueTracker(0.0)   # track offset (in cards)
        act = [ValueTracker(1.0 if k == 0 else 0.0) for k in range(4)]

        def track(m):
            for k, c in enumerate(cards):
                d = k - off.get_value()
                c.x.set_value(d * STEP)
                a = act[k].get_value()
                c.w.set_value(c.w_u * (0.88 + 0.12 * a))
                c.dim.set_value(0.6 * (1 - a))
        driver = Mobject().add_updater(track)
        self.add(driver)
        for c in cards:
            c.start(self, alpha=0)
            c.playing = False
        cards[0].playing = True

        caps = []
        for s in SEGS:
            t = T(s["title"], 36, FG, weight=BOLD)
            v = T(s["venue"], 26, s["color"])
            caps.append(VGroup(t, v).arrange(DOWN, buff=0.14).move_to([0, -2.72, 0]))
        dots = VGroup(*[Dot(radius=0.07, color=DIM) for _ in range(4)]).arrange(RIGHT, buff=0.22).move_to([0, -3.62, 0])
        head = VGroup(T("WORLD MODELS", 22, FG, weight=HEAVY)).move_to([0, 3.62, 0])

        def sticker(k):
            st = fig_stack(SEGS[k], width=2.6, max_h=2.5, label_size=14, label_color="#D8DEE9", buff=0.1)
            back = RoundedRectangle(width=st.width + 0.26, height=st.height + 0.26, corner_radius=0.12)
            back.set_fill("#1A1D26", 0.95).set_stroke(SEGS[k]["color"], 2).move_to(st)
            g = Group(back, st)
            corner = np.array([CARD_W / 2 + 0.35, Y - CARD_H / 2 + 0.35, 0])
            return g.move_to(corner, aligned_edge=DR)
        stickers = [sticker(k) for k in range(4)]
        icon = [LiveIcon(k, SEGS[k]["color"], s=1.3, width=3) for k in range(4)]
        for k, ic in enumerate(icon):
            ic.move_to([-CARD_W / 2 + 0.55, Y + CARD_H / 2 - 0.5, 0])
        ic_back = Circle(radius=0.36).set_fill(BLACK, 0.6).set_stroke(SEGS[0]["color"], 2).move_to(icon[0])

        self.play_t(*[c.alpha.animate.set_value(1) for c in cards], FadeIn(caps[0], shift=UP * 0.2), FadeIn(dots),
                    dots[0].animate.set_color(SEGS[0]["color"]).scale(1.3), FadeIn(head), FadeIn(stickers[0], shift=UP * 0.3),
                    FadeIn(ic_back), FadeIn(icon[0].activate()), run_time=0.4)
        for k in range(1, 4):
            self.goto(BOUNDS[k] - TR / 2)
            cards[k].playing = True
            self.play_t(off.animate.set_value(k), act[k - 1].animate.set_value(0), act[k].animate.set_value(1),
                        FadeOut(caps[k - 1], shift=LEFT * 0.6), FadeIn(caps[k], shift=LEFT * 0.6),
                        FadeOut(stickers[k - 1], shift=LEFT * 0.8), FadeIn(stickers[k], shift=LEFT * 0.8),
                        dots[k - 1].animate.set_color(SEGS[k - 1]["color"]).set_opacity(0.6).scale(1 / 1.3),
                        dots[k].animate.set_color(SEGS[k]["color"]).scale(1.3),
                        FadeOut(icon[k - 1]), FadeIn(icon[k].activate()), ic_back.animate.set_stroke(SEGS[k]["color"]),
                        run_time=TR)
            cards[k - 1].playing = False
        self.goto(9.55)
        self.play_t(*[d.animate.set_color(SEGS[i]["color"]).set_opacity(1) for i, d in enumerate(dots)],
                    head[0].animate.set_color(WHITE).scale(1.12), run_time=0.3)
        self.goto(10.0)
