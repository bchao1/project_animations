"""LoT Diffusion — V2 v4 (post 2), 21.4 s: image and video side by side — quadtree subdivision → noisy token layout →
image / first frame → mean-colour layout → (video stacks in time) → both settle. No captions (user, 2026-10-01).

Background frames: render_frames_v4.py → data/generated/frames_v4/ (q_common_v4.py). This scene adds the panel names
and the counts: "4,096 tokens · 100% of 4,096" / "75,600 tokens · 100% of 75,600" on the full grid, running down to
the layout while the noisy tokens form (as V1); at the end a copy of each % flies onto its tick.
v4: a purple "≈N× faster" line under each count (shared lookup), fading in when the counts reach the layout.
Render: ./render.sh l|h animation_quadtree_settle_v4.py
"""
from manim import *
import numpy as np
from PIL import Image
import os

import q_common_v4 as Q

config.background_color = BLACK

FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
DIM = "#8A8A8A"
AMBER = "#FFCA28"
PURPLE = "#E040FB"                     # STYLE_NOTES highlight 2: speedups
MATHCH = set("×⊙→≈∼−∈𝒩ℒ𝔼′≪≤≥‖·↓⊤–")
TICK = 0.12


def mathify(s):
    return "".join(
        f'<span font_family="{MATHFONT}">{c}</span>' if c in MATHCH else c for c in s)


def tx(s, size=30, color=WHITE, bold=False):
    if bold:
        s = f"<b>{s}</b>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


def count_text(n, full):
    """Two lines: "1,546 tokens" over "37.7% of 4,096" (number and % in amber)."""
    p = "100%" if n == full else f"{100 * n / full:.1f}%"
    return VGroup(tx(f'<span foreground="{AMBER}"><b>{n:,}</b></span> tokens', 28),
                  tx(f'<span foreground="{AMBER}"><b>{p}</b></span> of {full:,}', 24)).arrange(DOWN, buff=0.08)


NF = int(round(Q.DURATION * Q.FPS))
NL = len(Q.corgi_rects())                                  # 1,546
NV = sum(len(f) for f in Q.ST.decode())                    # 27,795
FULL_L = Q.N * Q.N
FULL_V = Q.ST.T * Q.ST.H * Q.ST.W


class LoTQuadtreeSettle(Scene):
    def construct(self):
        clock = lambda: self.renderer.time
        bg = ImageMobject(os.path.join(Q.FRAMES, "0000.png")).set(width=config.frame_width).move_to(ORIGIN)
        last = {"i": 0}

        def upd(m, dt):   # dt updater: keeps manim from treating the waits as static
            i = min(int(round(clock() * Q.FPS)), NF - 1)
            if i != last["i"]:
                last["i"] = i
                m.pixel_array = np.asarray(Image.open(os.path.join(Q.FRAMES, f"{i:04d}.png")).convert("RGBA"))
        bg.add_updater(upd)
        self.add(bg)

        heads = [tx("image", 30).move_to([Q.L_C[0], Q.HEAD_Y, 0]), tx("video", 30).move_to([Q.R_C[0], Q.HEAD_Y, 0])]
        heads[1].align_to(heads[0], UP)                    # same top: "image" has a descender, "video" not
        self.add(*heads)

        spec = [(NL, FULL_L, Q.L_C[0]), (NV, FULL_V, Q.R_C[0])]
        speeds = [tx(f"≈{Q.speedup(NL):.1f}× faster", 28, PURPLE, bold=True),
                  tx(f"≈{Q.speedup(NV, modality='video'):.1f}× faster", 28, PURPLE, bold=True)]
        ref = VGroup(count_text(FULL_L, FULL_L), speeds[0].copy()).arrange(DOWN, buff=0.12)
        cy = Q.COUNT_Y + ref[0].get_center()[1] - ref.get_center()[1]     # count's centre in the 3-line label
        sy = Q.COUNT_Y + ref[1].get_center()[1] - ref.get_center()[1]
        vals = [ValueTracker(full) for _, full, _ in spec]
        labels = [always_redraw(lambda k=k: count_text(int(round(vals[k].get_value())), spec[k][1])
                                .move_to([spec[k][2], cy, 0])) for k in range(2)]
        self.add(*labels)
        self.wait(Q.COUNT[0] - clock())
        self.play(*[vals[k].animate.set_value(spec[k][0]) for k in range(2)],
                  run_time=Q.COUNT[1] - Q.COUNT[0], rate_func=smooth)
        for k in range(2):                                 # freeze: static text from here on
            self.remove(labels[k])
            labels[k] = count_text(spec[k][0], spec[k][1]).move_to([spec[k][2], cy, 0])
            self.add(labels[k])
            speeds[k].move_to([spec[k][2], sy, 0])
        self.play(*[FadeIn(sp, shift=0.08 * UP) for sp in speeds], run_time=0.4)

        lx = Q.L_C[0] - Q.L_SIDE / 2
        top = Q.L_C[1] + Q.L_SIDE / 2

        def v_tick(level):
            return Q.v2_from_v14(Q.ST.project(clock() - Q.PRE + Q.VOL0, Q.ST.ytick(level)[1]))

        events = [(Q.LIQ0, lambda: self.add(Line([lx - TICK, top, 0], [lx, top, 0], color=DIM, stroke_width=2),
                                            tx("100%", 20, DIM).next_to([lx - TICK, top, 0], LEFT, buff=0.06))),
                  (Q.GHOST1, lambda: self.add(tx("100%", 20, DIM).next_to(v_tick(Q.ST.H), LEFT, buff=0.08)))]

        def collapse():
            y = Q.left_level_y(NL / Q.N)
            self.add(Line([lx - TICK, y, 0], [lx, y, 0], color=AMBER, stroke_width=2.5),
                     DashedLine([lx, y, 0], [lx + Q.L_SIDE, y, 0], color=AMBER, stroke_width=1.5,
                                dash_length=0.06, stroke_opacity=0.7))
            pl = tx(f"{100 * NL / FULL_L:.1f}%", 26, AMBER, bold=True).next_to([lx - TICK, y, 0], LEFT, buff=0.06)
            pv = tx(f"{100 * NV / FULL_V:.1f}%", 26, AMBER, bold=True).next_to(v_tick(NV / (Q.ST.W * Q.ST.T)), LEFT,
                                                                               buff=0.08)
            self.play(TransformFromCopy(labels[0][1], pl), TransformFromCopy(labels[1][1], pv),
                      run_time=Q.COLLAPSE[1] - Q.COLLAPSE[0], rate_func=smooth)
        events.append((Q.COLLAPSE[0], collapse))
        for t0, fn in sorted(events, key=lambda e: e[0]):
            if t0 > clock():
                self.wait(t0 - clock())
            fn()
        self.wait(Q.DURATION - clock())
