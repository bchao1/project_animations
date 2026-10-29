"""Style 02 — Full-bleed stories.
Each demo fills the frame; four progress bars across the top fill in the paper's colour (story-style), a dark ramp
carries a large lower-left caption, and the paper figure sits as a picture-in-picture card top-right.
Papers change with a horizontal whip-slide.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm_common import *  # noqa

TR = 0.36


class S02_FullBleed(TimedScene):
    def construct(self):
        self.camera.background_color = "#07080B"
        cards = []
        for s in SEGS:
            cover = s["key"] in ("camctrl2", "genreal")
            if s["key"] == "bullettime":  # 2.37:1 side-by-side; keep it whole and clear of the bars/caption
                cards.append(VideoCard(s["key"], s["speed"], box=(13.6, 5.7), center=[0, 0.35, 0], radius=0.1, border=None))
            else:
                cards.append(VideoCard(s["key"], s["speed"], box=(FW, FH), radius=0, border=None, cover=cover))
        band = fade_band(height=3.6, max_op=0.95, power=1.0)
        top_band = fade_band(height=1.5, y_bottom=FH / 2, max_op=0.75, up=False, power=1.2)
        caps = []
        for s in SEGS:
            kick = T("WORLD MODELS", 18, WHITE, weight=HEAVY).set_opacity(0.8)
            if s["key"] == "genreal":  # headset inset lives bottom-left in this clip; caption goes right, two lines
                cap = VGroup(kick, caption(s, size=42, venue_size=28, two_lines=True, align=RIGHT))
                cap.arrange(DOWN, buff=0.16, aligned_edge=RIGHT).move_to([6.6, -2.75, 0], aligned_edge=RIGHT)
            else:
                cap = VGroup(kick, caption(s, size=42, venue_size=28, max_w=12.4)).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
                cap.move_to([-6.55, -2.95, 0], aligned_edge=LEFT)
            caps.append(cap)

        # story progress bars
        bw, gap, y = (FW - 0.8 - 3 * 0.12) / 4, 0.12, 3.72
        xs = [-FW / 2 + 0.4 + bw / 2 + i * (bw + gap) for i in range(4)]
        tracks = VGroup(*[Line([x - bw / 2, y, 0], [x + bw / 2, y, 0]).set_stroke(WHITE, 4, opacity=0.28) for x in xs])
        fills = [ValueTracker(0.0) for _ in range(4)]

        def bar(i):
            return always_redraw(lambda: Line([xs[i] - bw / 2, y, 0], [xs[i] - bw / 2 + bw * max(fills[i].get_value(), 1e-3), y, 0])
                                 .set_stroke(SEGS[i]["color"], 4))
        bars = VGroup(*[bar(i) for i in range(4)])
        labels = VGroup(*[VGroup(LiveIcon(i, WHITE, s=0.8, width=2.2), T(SEGS[i]["control"], 18, WHITE))
                          .arrange(RIGHT, buff=0.1).move_to([xs[i] - bw / 2, y - 0.3, 0], aligned_edge=LEFT) for i in range(4)])
        for i, l in enumerate(labels):
            l.set_opacity(0.45)

        def pip(k):
            w = 2.5 if SEGS[k]["key"] == "bullettime" else 3.0
            st = fig_stack(SEGS[k], width=w, max_h=3.0, label_size=15, label_color="#D8DEE9", buff=0.14)
            back = RoundedRectangle(width=st.width + 0.3, height=st.height + 0.3, corner_radius=0.12)
            back.set_fill(BLACK, 0.55).set_stroke(WHITE, 1, opacity=0.3).move_to(st)
            return Group(back, st).move_to([6.9, 3.05, 0], aligned_edge=UR)
        pips = [pip(k) for k in range(4)]

        c0 = cards[0].start(self)
        self.add(band, top_band, tracks, bars)
        labels[0][0].activate()
        self.play_t(c0.alpha.animate.set_value(1), FadeIn(caps[0], shift=UP * 0.3), FadeIn(labels),
                    FadeIn(pips[0], shift=LEFT * 0.3), run_time=0.4)
        labels[0].set_opacity(1)
        seg_start = 0.0
        for k in range(4):
            end = BOUNDS[k + 1] - (TR / 2 if k < 3 else 0)
            self.play_until(end, fills[k].animate(rate_func=linear).set_value(1.0))
            if k == 3:
                break
            old, new = cards[k], cards[k + 1]
            new.start(self, alpha=1)
            new.x.set_value(FW)
            labels[k + 1][0].activate()
            self.play_t(old.x.animate.set_value(-FW), new.x.animate.set_value(0),
                        FadeOut(caps[k], shift=LEFT * 1.0), FadeIn(caps[k + 1], shift=LEFT * 1.0),
                        FadeOut(pips[k], shift=LEFT * 0.5), FadeIn(pips[k + 1], shift=LEFT * 0.5),
                        labels[k].animate.set_opacity(0.75), labels[k + 1].animate.set_opacity(1),
                        run_time=TR, rate_func=smooth)
            labels[k][0].live = False
            self.remove(*old.mobs())
        self.goto(10.0)
