"""LoT Diffusion — V1 v5 (post 1), 17.5 s: control signal → token layout → noisy token layout → image → layout → settle.

v2 (user feedback on v1): columns by signal (depth map | semantic mask | bounding box), the signal on screen from
frame 0, holds between the steps, no captions / footnotes / speedups — only the column names and one count per tile.
The count reads "4,096 tokens · 100% of 4,096" on the full-resolution grid and counts down to the layout's tokens
during the sweep; at the end the % lands on a tick at the settled height.
v3: a purple "≈1.7× faster" line under every count (shared lookup ../../speedup/lot_speedup.py), fading in when
the count reaches the layout.
v4: only "N tokens" under each tile (4,096 on the full grid, counting down to the layout); after the settle the %
lands on the left tick and the speedup fades in under the tile.
v5 (user, 2026-10-01): the count always names the full-resolution total: "2,047 of 4,096 tokens".
Tiles: render_tiles_v4.py → data/generated/tiles_v4/ (physics sim_v2.py, same items). Render: ./render.sh l|h v4
"""
from manim import *
import numpy as np
from PIL import Image
import os

import signal_common_v4 as C

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


def count_text(n):
    return tx(f'<span foreground="{AMBER}"><b>{n:,}</b></span> of {C.FULL:,} tokens', 26)


def speed_text(n):
    return tx(f"≈{C.speedup(n):.1f}× faster", 28, PURPLE, bold=True)


TOKS = [C.Tokens(i) for i in range(C.NI)]
NF = int(round(C.DURATION * C.FPS))


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

        heads = [tx(name, 26).move_to([*C.px_to_m(C.COL_X[c], C.HEAD_Y), 0]) for c, name in enumerate(C.COLUMNS)]
        for h in heads[1:]:
            h.align_to(heads[0], UP)                       # same top: descenders differ between the names
        self.add(*heads)

        geo = []
        for i in range(C.NI):
            x0, y0 = C.tile_origin_px(i)
            lx, top = C.px_to_m(x0, y0)
            _, bot = C.px_to_m(x0, y0 + C.TP)
            geo.append(dict(lx=lx, top=top, bot=bot, cx=C.px_to_m(C.COL_X[i // 2], 0)[0],
                            ly=C.px_to_m(0, y0 + C.TP + C.LABEL_DY)[1]))

        # counts: 4,096 on the full grid → n during the sweep (each tile on its own stagger)
        vals = [ValueTracker(C.FULL) for _ in range(C.NI)]
        ref = VGroup(count_text(C.FULL), speed_text(C.FULL)).arrange(DOWN, buff=0.1)
        cdy = ref.get_center()[1] - ref[0].get_center()[1]      # count line: upper half of the 2-line label
        sdy = ref.get_center()[1] - ref[1].get_center()[1]
        labels = [always_redraw(lambda i=i: count_text(int(round(vals[i].get_value())))
                                .move_to([geo[i]["cx"], geo[i]["ly"] - cdy, 0])) for i in range(C.NI)]
        self.add(*labels)
        if C.SWEEP[0] > clock():
            self.wait(C.SWEEP[0] - clock())
        d = C.STAGGER * (C.NI - 1)
        self.play(*[vals[i].animate(rate_func=smooth, run_time=C.SWEEP[1] - C.SWEEP[0]).set_value(tk.n)
                    for i, tk in enumerate(TOKS)], run_time=C.SWEEP[1] - C.SWEEP[0] + d)
        for i, tk in enumerate(TOKS):                          # freeze: static text from here on
            self.remove(labels[i])
            labels[i] = count_text(tk.n).move_to([geo[i]["cx"], geo[i]["ly"] - cdy, 0])
            self.add(labels[i])

        self.wait(C.LIQUEFY0 - clock())
        for g in geo:
            self.add(Line([g["lx"] - TICK, g["top"], 0], [g["lx"], g["top"], 0], color=DIM, stroke_width=2),
                     tx("100%", 20, DIM).next_to([g["lx"] - TICK, g["top"], 0], LEFT, buff=0.06))

        self.wait(C.COLLAPSE[0] - clock())
        anims = []
        for i, (g, tk) in enumerate(zip(geo, TOKS)):
            y = g["bot"] + tk.level / C.N * (g["top"] - g["bot"])
            self.add(Line([g["lx"] - TICK, y, 0], [g["lx"], y, 0], color=AMBER, stroke_width=2.5),
                     DashedLine([g["lx"], y, 0], [g["lx"] + (g["top"] - g["bot"]), y, 0], color=AMBER,
                                stroke_width=1.5, dash_length=0.06, stroke_opacity=0.7))
            pct = tx(f"{100 * tk.frac:.1f}%", 26, AMBER, bold=True).next_to([g["lx"] - TICK, y, 0], LEFT, buff=0.06)
            sp = speed_text(tk.n).move_to([g["cx"], g["ly"] - sdy, 0])
            anims += [FadeIn(pct, shift=0.1 * DOWN), FadeIn(sp, shift=0.08 * UP)]
        self.play(*anims, run_time=C.COLLAPSE[1] - C.COLLAPSE[0])
        self.wait(C.DURATION - clock())
