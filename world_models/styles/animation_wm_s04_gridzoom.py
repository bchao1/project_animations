"""Style 04 — Grid zoom.
Opens on a 2x2 wall of all four demos with WORLD MODELS in the middle, then a virtual camera zooms into each paper in
turn (neighbours peek in, dimmed) and finally pulls back out to the full wall.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm_common import *  # noqa

CW, CH, GAP = 6.85, 3.853, 0.18
CENTERS = [np.array([-(CW + GAP) / 2, (CH + GAP) / 2, 0]), np.array([(CW + GAP) / 2, (CH + GAP) / 2, 0]),
           np.array([-(CW + GAP) / 2, -(CH + GAP) / 2, 0]), np.array([(CW + GAP) / 2, -(CH + GAP) / 2, 0])]
Z = 11.6 / CW       # zoom so the active cell is 11.6 units wide
SCREEN_Y = 0.42     # where the zoomed cell sits vertically


class S04_GridZoom(TimedScene):
    def construct(self):
        self.camera.background_color = BG
        cards = [VideoCard(s["key"], s["speed"], box=(CW, CH), center=CENTERS[k], radius=0.1, max_w=CW * Z, loop=True)
                 for k, s in enumerate(SEGS)]
        zt, fx, fy = ValueTracker(1.0), ValueTracker(0.0), ValueTracker(0.0)
        focus = [ValueTracker(0.0) for _ in range(4)]  # 1 = in focus (undimmed)

        def cam_update(m):
            z, f = zt.get_value(), np.array([fx.get_value(), fy.get_value(), 0])
            for k, c in enumerate(cards):
                p = (CENTERS[k] - f) * z
                c.x.set_value(p[0]), c.y.set_value(p[1]), c.w.set_value(c.w_u * z)
                c.dim.set_value(0.62 * (1 - focus[k].get_value()) * min((z - 1) / (Z - 1), 1))
        driver = Mobject().add_updater(cam_update)
        self.add(driver)
        for c in cards:
            c.start(self, alpha=1)
        # labels on the wall
        tags = VGroup(*[VGroup(LiveIcon(k, SEGS[k]["color"], s=0.9), T(SEGS[k]["control"], 20, FG)).arrange(RIGHT, buff=0.1)
                        for k in range(4)])
        for k, tg in enumerate(tags):
            back = RoundedRectangle(width=tg.width + 0.3, height=0.42, corner_radius=0.12).set_fill(BLACK, 0.6).set_stroke(width=0)
            tg.add_to_back(back.move_to(tg))
            sx, sy = (-1 if k % 2 == 0 else 1), (1 if k < 2 else -1)   # outer corner of each cell
            tg.move_to(CENTERS[k] + np.array([sx * (CW / 2 - 0.2), sy * (CH / 2 - 0.2), 0]),
                       aligned_edge=(UL if k == 0 else UR if k == 1 else DL if k == 2 else DR))
            tg[1].activate()
        title = T("WORLD MODELS", 44, FG, weight=HEAVY)
        tback = RoundedRectangle(width=title.width + 0.8, height=title.height + 0.5, corner_radius=0.2)
        tback.set_fill(BG, 0.92).set_stroke(FG, 1.5, opacity=0.6)
        title_g = VGroup(tback, title)

        def fpoint(k):
            return CENTERS[k] - np.array([0, SCREEN_Y, 0]) / Z

        caps = []
        for s in SEGS:
            t = T(s["title"], 30, FG, weight=BOLD)
            v = T(s["venue"], 26, s["color"])
            dot = T("·", 30, MUTED)
            line = VGroup(t, dot, v).arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
            if line.width > 13.4:
                line.set_width(13.4)
            caps.append(line.move_to([0, -3.62, 0]))
        figs = []
        for s in SEGS:
            st = fig_stack(s, width=2.5, max_h=2.6, label_size=14, label_color="#D8DEE9", buff=0.1)
            back = RoundedRectangle(width=st.width + 0.24, height=st.height + 0.24, corner_radius=0.1)
            back.set_fill(BLACK, 0.6).set_stroke(WHITE, 1, opacity=0.3).move_to(st)
            figs.append(Group(back, st).move_to([6.55, SCREEN_Y + CH * Z / 2 - 0.2, 0], aligned_edge=UR))

        band = fade_band(height=1.6, color=BG, max_op=0.92)
        self.add(band, tags)
        self.play_t(FadeIn(title_g, scale=0.9), run_time=0.3)
        self.goto(0.55)
        # zoom into paper 0
        cards[0].clock = 0.0
        self.play_t(zt.animate.set_value(Z), fx.animate.set_value(fpoint(0)[0]), fy.animate.set_value(fpoint(0)[1]),
                    focus[0].animate.set_value(1), FadeOut(title_g, scale=1.4), FadeOut(tags),
                    FadeIn(caps[0], shift=UP * 0.2), FadeIn(figs[0], shift=LEFT * 0.3), run_time=0.5)
        for k in range(1, 4):
            self.goto(BOUNDS[k] - 0.2)
            cards[k].clock = 0.0
            self.play_t(fx.animate.set_value(fpoint(k)[0]), fy.animate.set_value(fpoint(k)[1]),
                        focus[k - 1].animate.set_value(0), focus[k].animate.set_value(1),
                        FadeOut(caps[k - 1], shift=UP * 0.2), FadeIn(caps[k], shift=UP * 0.2),
                        FadeOut(figs[k - 1]), FadeIn(figs[k], shift=LEFT * 0.3), run_time=0.42)
        # pull back to the wall
        self.goto(9.2)
        for k in range(4):
            focus[k].set_value(1) if k == 3 else None
        self.play_t(zt.animate.set_value(1.0), fx.animate.set_value(0), fy.animate.set_value(0),
                    *[focus[k].animate.set_value(1) for k in range(3)],
                    FadeOut(caps[3], shift=DOWN * 0.2), FadeOut(figs[3]), FadeIn(title_g, scale=0.9), FadeIn(tags),
                    run_time=0.55)
        self.goto(10.0)
