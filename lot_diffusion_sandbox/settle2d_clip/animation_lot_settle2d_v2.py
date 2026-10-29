"""Level-of-Token Diffusion — 14.5-s X clip: six images settle into their token fraction (2D version of
../spacetime_clip v13), v2.

v2 (user feedback on v1): the pile flows out to a flat top by physics alone (sim2d_v2.py: mixed disk sizes, the
slope push stays on), no lattice snap.

Six LoT-FLUX.2 4B images with 25–35% of the 4,096 full-resolution tokens. Each is generated (noise → image), turned
into its LoT layout (token fill = mean image colour), every token shrinks to one 1 × 1 token, and the squares fall
and flow flat under 2D particle physics; the pile's height is n / 64 rows = the token fraction. The count then
collapses onto a tick at that height and the speedup appears under the image.

Data, timeline, geometry: settle2d_common_v2.py. Tile frames (images, tokens, frames): render_tiles_v2.py →
data/generated/tiles_v2/ (physics: sim2d_v2.py). This scene adds counts, ticks, percentages, speedups and captions.
Render (see render.sh):
    python sim2d_v2.py && python render_tiles_v2.py && manim -qh --media_dir media animation_lot_settle2d_v2.py LoTSettle2D
"""
from manim import *
import numpy as np
from PIL import Image
import os

import settle2d_common_v2 as C

config.background_color = BLACK

# ------------------------------------------------------------------ style (as ../spacetime_clip v13)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
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


TOKS = [C.Tokens(i) for i in range(C.NI)]
NF = int(round(C.DURATION * C.FPS))
TICK = 0.12


class LoTSettle2D(Scene):
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

        counts = [tx(f"{fmt(tk.n)} tokens", 22, AMBER, bold=True).move_to([C.CENTERS[i][0], below[i], 0])
                  for i, tk in enumerate(TOKS)]
        events.append((C.TOKENIZE[1] + C.STAGGER * (C.NI - 1), lambda: self.add(*counts)))

        def ticks_100():
            g = VGroup()
            for i in range(C.NI):
                y = top[i]
                g.add(Line([left[i] - TICK, y, 0], [left[i], y, 0], color=DIM, stroke_width=2),
                      tx("100%", 18, DIM).next_to([left[i] - TICK, y, 0], LEFT, buff=0.06))
            self.add(g)
        events.append((C.LIQUEFY[0], ticks_100))

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
