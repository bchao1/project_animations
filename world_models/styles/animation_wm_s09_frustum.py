"""Style 09 — Camera frustum.
A wireframe camera on the left projects a frustum onto the demo video — "a world model renders what the camera sees".
Frustum rays recolour and a light pulse travels down them at every paper change. The paper figure sits under the camera.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm_common import *  # noqa

VID_C = np.array([1.75, 0.3, 0])
APEX = np.array([-4.55, 1.55, 0])
TR = 0.34


def camera_body(color):
    body = RoundedRectangle(width=1.05, height=0.72, corner_radius=0.1)
    lens = Polygon([0.52, 0.16, 0], [0.9, 0.32, 0], [0.9, -0.32, 0], [0.52, -0.16, 0])
    reel1, reel2 = Circle(radius=0.2).move_to([-0.25, 0.58, 0]), Circle(radius=0.2).move_to([0.2, 0.58, 0])
    g = VGroup(body, lens, reel1, reel2).set_stroke(color, 3).set_fill(BG, 1)
    return g.shift(APEX - np.array([0.9, 0, 0]))


class S09_Frustum(TimedScene):
    def construct(self):
        self.camera.background_color = BG
        cards = [VideoCard(s["key"], s["speed"], box=(9.6, 5.3), center=VID_C, radius=0.08) for s in SEGS]

        def rays(k, h=None):
            c = cards[k]
            hh = c.h_u if h is None else h
            corners = [VID_C + np.array([sx * c.w_u / 2, sy * hh / 2, 0]) for sx, sy in [(-1, 1), (-1, -1)]]
            far = [VID_C + np.array([-c.w_u / 2, sy * hh / 2, 0]) for sy in (1, -1)]
            ls = VGroup(*[Line(APEX, p) for p in corners])
            ls.set_stroke(SEGS[k]["color"], 2.5, opacity=0.9)
            fill = Polygon(APEX, far[0], far[1]).set_stroke(width=0).set_fill(SEGS[k]["color"], 0.08)
            return VGroup(fill, ls)
        cam = camera_body(SEGS[0]["color"])
        ico = LiveIcon(0, SEGS[0]["color"], s=1.0).move_to(cam[0])
        head = T("WORLD MODELS", 24, FG, weight=HEAVY).move_to([-6.65, 3.55, 0], aligned_edge=LEFT)
        sub = T("render what the camera sees — and more", 18, MUTED).next_to(head, RIGHT, buff=0.3).align_to(head, DOWN)
        figs = [fig_stack(s, width=3.2, max_h=2.75, label_size=15, buff=0.12).move_to([-5.05, -1.05, 0]) for s in SEGS]
        for f in figs:
            if f.height > 3.6:
                f.set_height(3.6)
            f.move_to([-5.05, -0.95, 0])
        caps = [caption(s, size=34, venue_size=25, max_w=12.4).move_to([-6.65, -3.35, 0], aligned_edge=LEFT) for s in SEGS]

        c0 = cards[0].start(self)
        r = rays(0)
        self.play_t(Create(cam), FadeIn(head), FadeIn(sub), c0.alpha.animate.set_value(1), Create(r),
                    FadeIn(figs[0], shift=UP * 0.2), FadeIn(caps[0], shift=UP * 0.2), run_time=0.45)
        for k in range(1, 4):
            self.goto(BOUNDS[k] - TR / 2)
            old, new = cards[k - 1], cards[k]
            new.start(self)
            nr = rays(k)
            flash = [ShowPassingFlash(l.copy().set_stroke(WHITE, 6), time_width=0.5) for l in nr[1]]
            self.play_t(old.alpha.animate.set_value(0), new.alpha.animate.set_value(1), Transform(r, nr), *flash,
                        cam.animate.set_stroke(SEGS[k]["color"]),
                        FadeOut(figs[k - 1], shift=UP * 0.2), FadeIn(figs[k], shift=UP * 0.2),
                        FadeOut(caps[k - 1], shift=UP * 0.2), FadeIn(caps[k], shift=UP * 0.2), run_time=TR)
            self.remove(*old.mobs())
        self.goto(9.55)
        self.play_t(*[ShowPassingFlash(l.copy().set_stroke(WHITE, 6), time_width=0.6) for l in r[1]], run_time=0.4)
        self.goto(10.0)
