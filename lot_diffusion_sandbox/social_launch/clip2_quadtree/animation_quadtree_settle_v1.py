"""LoT Diffusion — V2 (post 2), 16.8 s: quadtree of an image and of a video side by side, then both settle.

Background frames (render_frames_v1.py → data/generated/frames_v1/): left the corgi's quadtree → mean-colour layout →
2D settle; right latent frame 1's quadtree → spacetime v16 colour volume (stack in time → shrink → 3D settle).
This scene adds the panel names, token counts, captions, the per-frame footnote, the 100% ticks and the fractions.
Render: ./render.sh l|h
"""
from manim import *
import numpy as np
from PIL import Image
import os

import q_common_v1 as Q

config.background_color = BLACK

FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
DIM = "#8A8A8A"
AMBER = "#FFCA28"
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈𝒩ℒ𝔼′≪≤≥‖·↓⊤–")
TICK = 0.12


def mathify(s):
    return "".join(
        f'<span font_family="{MATHFONT}">{c}</span>' if c in MATHCH else c for c in s)


def tx(s, size=30, color=WHITE, bold=False):
    if bold:
        s = f"<b>{s}</b>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


def fmt(n):
    return f"{n:,}"


NF = int(round(Q.DURATION * Q.FPS))
NL = len(Q.corgi_rects())                                  # 1,546
NV = sum(len(f) for f in Q.ST.decode())                    # 27,795
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
        events = [(t0, lambda s=s: self.set_caption(s)) for t0, s in Q.CAPS]

        # panel names, left-aligned with the thumbnails that appear under them
        (ix, iy), th = Q.IMG_TH
        (vx, vy), _ = Q.VID_TH
        img_l, vid_l = ix - th / 2, vx - th * 8 / 9
        names = VGroup(tx("image", 26).next_to([img_l, iy + th / 2 + 0.08, 0], UR, buff=0).shift(LEFT * 0.0),
                       tx("video", 26).next_to([vid_l, vy + th / 2 + 0.08, 0], UR, buff=0))
        for m, x in zip(names, (img_l, vid_l)):
            m.align_to([x, 0, 0], LEFT)
        events.append((0.0, lambda: self.add(names)))

        def count(n, full, x):
            g = VGroup(tx(f"{fmt(n)} tokens", 28, AMBER, bold=True),
                       tx(f"of {fmt(full)} at full resolution", 18, DIM)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            return g.move_to([x, iy, 0], aligned_edge=LEFT)
        cl = count(NL, Q.N * Q.N, ix + th / 2 + 0.25)
        cv = count(NV, FULL_V, vx + th * 8 / 9 + 0.25)
        events.append((Q.THUMBS[0], lambda: self.play(FadeIn(cl), FadeIn(cv), run_time=0.5)))

        foot = tx("each slab is one latent frame: no token spans time", 18, DIM).move_to([Q.R_C[0], -2.85, 0])
        events.append((Q.PRE + 2.8, lambda: self.add(foot)))
        events.append((Q.SHRINK[0], lambda: self.play(FadeOut(foot), run_time=0.3)))

        lx = Q.L_C[0] - Q.L_SIDE / 2
        top = Q.L_C[1] + Q.L_SIDE / 2

        def ticks_left():
            self.add(Line([lx - TICK, top, 0], [lx, top, 0], color=DIM, stroke_width=2),
                     tx("100%", 18, DIM).next_to([lx - TICK, top, 0], LEFT, buff=0.06))
        events.append((Q.LIQ0, ticks_left))

        def v_tick(level):
            return Q.v2_from_v14(Q.ST.project(clock() - Q.PRE, Q.ST.ytick(level)[1]))
        events.append((Q.GHOST1, lambda: self.add(tx("100%", 18, DIM).next_to(v_tick(Q.ST.H), LEFT, buff=0.08))))

        def collapse():
            lev = NL / Q.N
            y = Q.left_level_y(lev)
            self.add(Line([lx - TICK, y, 0], [lx, y, 0], color=AMBER, stroke_width=2.5),
                     DashedLine([lx, y, 0], [lx + Q.L_SIDE, y, 0], color=AMBER, stroke_width=1.5,
                                dash_length=0.06, stroke_opacity=0.7))
            pl = tx(f"{100 * NL / (Q.N * Q.N):.1f}%", 24, AMBER, bold=True).next_to([lx - TICK, y, 0], LEFT, buff=0.06)
            pv = tx(f"{100 * NV / FULL_V:.1f}%", 24, AMBER, bold=True).next_to(v_tick(NV / (Q.ST.W * Q.ST.T)), LEFT,
                                                                               buff=0.08)
            self.play(ReplacementTransform(cl[0], pl), FadeOut(cl[1]), ReplacementTransform(cv[0], pv), FadeOut(cv[1]),
                      run_time=Q.COLLAPSE[1] - Q.COLLAPSE[0], rate_func=smooth)
        events.append((Q.COLLAPSE[0], collapse))

        self.cap = None
        for t0, fn in sorted(events, key=lambda e: e[0]):
            if t0 > clock():
                self.wait(t0 - clock())
            fn()
        self.wait(Q.DURATION - clock())

    def set_caption(self, s):
        if self.cap is not None:
            self.remove(self.cap)
        self.cap = tx(s, 28).move_to([0, CAPTION_Y, 0])
        self.add(self.cap)
