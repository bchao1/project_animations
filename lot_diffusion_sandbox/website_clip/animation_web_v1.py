"""LoT Diffusion — website clip v1, 5.0 s, 1024 × 1024: giraffe depth map (token layout) → noisy tokens → image →
mean-colour layout → fluid settle. Text: "1,027 of 4,096 tokens" from frame 0; the 100% tick when the fall starts;
the % on the settled level and the purple speedup at the end. No "depth map" label (user, 2026-10-06).
Frames: render_frames_v1.py. Render: ./render.sh l|h v1
"""
from manim import *
import numpy as np
from PIL import Image
import os

import web_common_v1 as C

config.pixel_width = config.pixel_height = C.W
config.frame_height = config.frame_width = 8.0
config.background_color = BLACK

FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
DIM = "#8A8A8A"
AMBER = "#FFCA28"
PURPLE = "#E040FB"
MATHCH = set("×≈·–")
TICK = 0.14


def mathify(s):
    return "".join(f'<span font_family="{MATHFONT}">{c}</span>' if c in MATHCH else c for c in s)


def tx(s, size=30, color=WHITE, bold=False):
    if bold:
        s = f"<b>{s}</b>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


TK = C.S.Tokens(C.ITEM)
NF = int(round(C.DURATION * C.FPS))


class LoTWebsite(Scene):
    def construct(self):
        clock = lambda: self.renderer.time
        bg = ImageMobject(os.path.join(C.FRAMES, "0000.png")).set(width=config.frame_width).move_to(ORIGIN)
        last = {"i": 0}

        def upd(m, dt):
            i = min(int(round(clock() * C.FPS)), NF - 1)
            if i != last["i"]:
                last["i"] = i
                m.pixel_array = np.asarray(Image.open(os.path.join(C.FRAMES, f"{i:04d}.png")).convert("RGBA"))
        bg.add_updater(upd)
        self.add(bg)

        lx, top = C.px_to_m(C.X0, C.Y0)
        rx, bot = C.px_to_m(C.X0 + C.TP, C.Y0 + C.TP)
        count = tx(f'<span foreground="{AMBER}"><b>{TK.n:,}</b></span> of {C.FULL:,} tokens', 34)
        speed = tx(f"≈{C.speedup(TK.n):.1f}× faster", 36, PURPLE, bold=True)
        lab = VGroup(count, speed).arrange(DOWN, buff=0.14)
        lab.next_to([0, bot, 0], DOWN, buff=C.LABEL_GAP / 128)
        self.add(count)

        self.wait(C.FALL0 - clock())
        self.add(Line([lx - TICK, top, 0], [lx, top, 0], color=DIM, stroke_width=2.5),
                 tx("100%", 24, DIM).next_to([lx - TICK, top, 0], LEFT, buff=0.06))

        self.wait(C.COLLAPSE[0] - clock())
        y = bot + TK.level / C.N * (top - bot)
        self.add(Line([lx - TICK, y, 0], [lx, y, 0], color=AMBER, stroke_width=3),
                 DashedLine([lx, y, 0], [rx, y, 0], color=AMBER, stroke_width=2, dash_length=0.08,
                            stroke_opacity=0.7))
        pct = tx(f"{100 * TK.frac:.1f}%", 28, AMBER, bold=True).next_to([lx - TICK, y, 0], LEFT, buff=0.06)
        self.play(FadeIn(pct, shift=0.1 * DOWN), FadeIn(speed, shift=0.08 * UP), run_time=C.COLLAPSE[1] - C.COLLAPSE[0])
        self.wait(C.DURATION - clock())
