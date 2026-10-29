"""LoT Diffusion — website clip v3, 8.0 s, 1024 × 1024: giraffe depth map (token layout) → noisy tokens → image →
mean-colour layout → fluid settle.
v3 (user, 2026-10-06): 8 s, slower beats (web_common_v3); look unchanged.
v2 (user, 2026-10-06): the image fills the frame (16-px border); all text overlaid on it in dark rounded boxes —
count box top-centre from frame 0, "100%" inside the top-left corner when the fall starts, "25.1%" just above the
settled-level line, speedup box under the count box at the end. No "depth map" label.
Frames: render_frames_v3.py. Render: ./render.sh l|h v3
"""
from manim import *
import numpy as np
from PIL import Image
import os

import web_common_v3 as C

config.pixel_width = config.pixel_height = C.W
config.frame_height = config.frame_width = 8.0
config.background_color = BLACK

FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
DIM = "#B0B0B0"
AMBER = "#FFCA28"
PURPLE = "#E040FB"
MATHCH = set("×≈·–")
U = 1 / 128                            # manim units per pixel


def mathify(s):
    return "".join(f'<span font_family="{MATHFONT}">{c}</span>' if c in MATHCH else c for c in s)


def tx(s, size=30, color=WHITE, bold=False):
    if bold:
        s = f"<b>{s}</b>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


def boxed(m, pad=14, opacity=0.62):
    r = RoundedRectangle(width=m.width + 2 * pad * U, height=m.height + 1.6 * pad * U, corner_radius=10 * U,
                         fill_color=BLACK, fill_opacity=opacity, stroke_width=0).move_to(m)
    return VGroup(r, m)


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
        count = boxed(tx(f'<span foreground="{AMBER}"><b>{TK.n:,}</b></span> of {C.FULL:,} tokens', 38))
        count.move_to([0, top - C.PAD * U - count.height / 2, 0])
        speed = boxed(tx(f"≈{C.speedup(TK.n):.1f}× faster", 40, PURPLE, bold=True))
        speed.next_to(count, DOWN, buff=10 * U)
        self.add(count)

        self.wait(C.FALL0 - clock())
        hund = tx("100%", 26, DIM)
        hund.move_to([lx + C.PAD * U + hund.width / 2 + 6 * U, top - C.PAD * U - hund.height / 2 - 4 * U, 0])
        self.add(hund)

        self.wait(C.COLLAPSE[0] - clock())
        y = bot + TK.level / C.N * (top - bot)
        line = DashedLine([lx, y, 0], [rx, y, 0], color=AMBER, stroke_width=3, dash_length=0.1, stroke_opacity=0.9)
        pct = boxed(tx(f"{100 * TK.frac:.1f}%", 34, AMBER, bold=True), pad=10)
        pct.move_to([lx + C.PAD * U + pct.width / 2, y + 10 * U + pct.height / 2, 0])
        self.add(line)
        self.play(FadeIn(pct, shift=0.1 * DOWN), FadeIn(speed, shift=0.08 * DOWN), run_time=C.COLLAPSE[1] - C.COLLAPSE[0])
        self.wait(C.DURATION - clock())
