"""LoT Diffusion — V1 (post 1), 16.9 s: control signal → token layout → generated image → token layout → settle.

The six images of ../../settle2d_clip with a new opening: each tile starts as its control signal (depth map,
semantic mask or bounding box), sweeps into its LoT token layout in the signal's colours, generates the image inside
the token boxes, then continues as settle2d v3 (image → mean-colour layout → shrink → physics → fraction → speedup).
Tiles: render_tiles_v1.py → data/generated/tiles_v1/. This scene adds signal names, counts, ticks, %, speedups, captions.
Render: ./render.sh l|h
"""
from manim import *
import numpy as np
from PIL import Image
import os

import signal_common_v1 as C

config.background_color = BLACK

FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
DIM = "#8A8A8A"
AMBER = "#FFCA28"
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈𝒩ℒ𝔼′≪≤≥‖·↓⊤–")


def mathify(s):
    return "".join(
        f'<span font_family="{MATHFONT}">{c}</span>' if c in MATHCH else c for c in s)


def tx(s, size=30, color=WHITE, bold=False):
    if bold:
        s = f"<b>{s}</b>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


def fmt(n):
    return f"{n:,}"


TOKS = [C.B.Tokens(i) for i in range(C.NI)]
NF = int(round(C.DURATION * C.FPS))
TICK = 0.12


class LoTSignalSettle(Scene):
    def construct(self):
        clock = lambda: self.renderer.time
        bg = ImageMobject(os.path.join(C.TILES, "0000.png")).set(width=config.frame_width).move_to(ORIGIN)
        last = {"i": 0}

        def upd(m, dt):   # dt updater: keeps manim from treating the waits as static
            i = min(int(round(clock() * C.FPS)), NF - 1)
            if i != last["i"]:
                last["i"] = i
                m.pixel_array = np.asarray(Image.open(os.path.join(C.TILES, f"{i:04d}.png")).convert("RGBA"))
        bg.add_updater(upd)
        self.add(bg)

        events = [(t0, lambda s=s: self.set_caption(s)) for t0, s in C.CAPS]
        left = [C.CENTERS[i][0] - C.TILE / 2 for i in range(C.NI)]
        below = [C.CENTERS[i][1] - C.TILE / 2 - 0.2 for i in range(C.NI)]
        top = [C.CENTERS[i][1] + C.TILE / 2 for i in range(C.NI)]

        names = [tx(C.SIGNAL_NAME[C.ITEMS[i][1]], 22, DIM).move_to([C.CENTERS[i][0], below[i], 0])
                 for i in range(C.NI)]
        events.append((C.SIG_IN[0], lambda: self.add(*names)))
        counts = [tx(f"{fmt(tk.n)} tokens", 22, AMBER, bold=True).move_to([C.CENTERS[i][0], below[i], 0])
                  for i, tk in enumerate(TOKS)]

        def names_to_counts():
            self.play(*[FadeTransform(a, b) for a, b in zip(names, counts)], run_time=0.4)
        events.append((C.CUE_TOK[1] + C.STAGGER * (C.NI - 1), names_to_counts))

        def ticks_100():
            g = VGroup()
            for i in range(C.NI):
                y = top[i]
                g.add(Line([left[i] - TICK, y, 0], [left[i], y, 0], color=DIM, stroke_width=2),
                      tx("100%", 18, DIM).next_to([left[i] - TICK, y, 0], LEFT, buff=0.06))
            self.add(g)
        events.append((C.LIQUEFY0, ticks_100))

        def collapse():
            anims, extra = [], VGroup()
            for i, tk in enumerate(TOKS):
                y = C.level_y(i, tk.level)
                extra.add(Line([left[i] - TICK, y, 0], [left[i], y, 0], color=AMBER, stroke_width=2.5),
                          DashedLine([left[i], y, 0], [left[i] + C.TILE, y, 0], color=AMBER, stroke_width=1.5,
                                     dash_length=0.06, stroke_opacity=0.7))
                pct = tx(f"{100 * tk.frac:.1f}%", 22, AMBER, bold=True).next_to([left[i] - TICK, y, 0], LEFT,
                                                                               buff=0.06)
                anims.append(FadeTransform(counts[i], pct))
            self.add(extra)
            self.play(*anims, run_time=C.COLLAPSE[1] - C.COLLAPSE[0], rate_func=smooth)
        events.append((C.COLLAPSE[0], collapse))

        def speeds():
            for i, tk in enumerate(TOKS):
                self.add(tx(f"≈{C.speedup(tk.n):.1f}× speedup", 22, WHITE, bold=True)
                         .move_to([C.CENTERS[i][0], below[i], 0]))
            self.add(tx("speedup ≈ fit of the paper's LoT-FLUX.2 4B timings at each token count", 18, DIM)
                     .set_y(3.72).to_edge(RIGHT, buff=0.35))
        events.append((C.SPEED_T, speeds))

        self.cap = None
        for t0, fn in sorted(events, key=lambda e: e[0]):
            if t0 > clock():
                self.wait(t0 - clock())
            fn()
        self.wait(C.DURATION - clock())

    def set_caption(self, s):
        if self.cap is not None:
            self.remove(self.cap)
        self.cap = tx(s, 28).move_to([0, CAPTION_Y, 0])
        self.add(self.cap)
