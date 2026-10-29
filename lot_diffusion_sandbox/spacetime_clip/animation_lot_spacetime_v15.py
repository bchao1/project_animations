"""Level-of-Token Diffusion — 13.8-s X clip: a space-time view of a video's per-frame LoT layouts, v15.

v15 = v14 for any site video ($LOT_ITEM): the "fine tokens follow …" caption comes from SUBJECT and the settle
caption carries the item's own token fraction (render_batch.sh renders the 10-video batch).

v14 (user feedback on v13): green tokens have 50%-opaque faces before the collapse (render_volume_v9.py).

v13 (user feedback on v12): no ease into a lattice slab — the cubes keep moving under the physics to the end, and the
y axis shows 36.8% at the mean level.

v12 (user feedback on v11: the trickle did not move like real particles): the settle is a GPU particle simulation
(sim_physics.py, position-based dynamics: gravity, contacts, walls, friction high → ~0, hydrostatic push, viscosity);
the rested cubes then snap ≤ ~1 cell into the exact lattice slab.


v10 (user feedback on v9): after the stacked view, every token shrinks to the size of one 1 × 1 token (keeping its
colour), the cubes fall straight down into piles and flatten into one slab; the token count then collapses onto the
y axis at the slab's height: 27,795 of 75,600 tokens = 36.8% of the full-resolution height (the wireframe).

Data, timeline and token motion: spacetime_common_v7.py (site showcase item 31, LoT-Wan2.1 14B, depth-of-field
layout, 21 latent frames × 45 × 80 tokens, exact leaves from ../web/showcase.json). The 3D part (tokens, octant,
ticks, wireframe) is rendered by render_volume_v9.py into data/generated/vol9_31/; this scene adds the video, the
current latent frame's 2D layout, captions, the count and the y-axis tick labels. Every token is one latent frame
thick; stacking, piling and flattening are displays of the per-frame layouts and their token count.

Render (see render.sh):
    python sim_physics_v2.py && python render_volume_v9.py && manim -qh --media_dir media animation_lot_spacetime_v14.py LoTSpaceTime
"""
from manim import *
import numpy as np
from PIL import Image, ImageDraw
import os

from spacetime_common_v7 import (CAPS, CLIP, COLLAPSE, DURATION, FPS, GEN, GHOST as GHOST_T, H, ITEM, REGION, T, VOL,
                                 W, decode, latent_of, project, video_frame, ytick)

VOL = VOL.replace("vol8_", "vol9_")      # render_volume_v9 / v10 output (50%-opaque green faces)

# per-video caption: what the 1 × 1 tokens follow (site layout legends: mask / bbox classes, VRS error, depth)
SUBJECT = {0: "follow the hands", 3: "follow the cars, people and lanes", 4: "follow the cars, lanes and lights",
           10: "follow the arm and the block", 14: "follow the warrior", 16: "follow the cars and lanes",
           22: "follow the guitarist", 26: "track the detailed scenery", 31: "follow her", 37: "follow him",
           39: "follow him"}
_n = sum(len(f) for f in decode())
CAPS = [CAPS[0], (CAPS[1][0], f"Stacked in time, the fine tokens {SUBJECT[ITEM]}."), CAPS[2],
        (CAPS[3][0], f"…and let them settle: {100 * _n / (T * H * W):.1f}% of the full-resolution tokens.")]

config.background_color = BLACK

# ------------------------------------------------------------------ style (as ../teaser_5min v3)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
HR, LR, LR3 = "#448AFF", "#00C853", "#1B7F3B"
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
AMBER = "#FFCA28"
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈𝒩ℒ𝔼′≪≤≥‖·↓⊤")


def mathify(s):
    return "".join(
        f'<span font_family="{MATHFONT}">{c}</span>' if c in MATHCH else c for c in s)


def tx(s, size=30, color=WHITE, bold=False):
    if bold:
        s = f"<b>{s}</b>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


def framed(m, color=WHITE, sw=1.5):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw)


def fmt(n):
    return f"{n:,}"


# ------------------------------------------------------------------ 2D layouts of the latent frames
def _hex(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


AREA_RGB = {1: _hex(HR), 2: tuple((a + b) // 2 for a, b in zip(_hex(HR), _hex(LR))), 4: _hex(LR),
            8: tuple((a + b) // 2 for a, b in zip(_hex(LR), _hex(LR3))), 16: _hex(LR3)}
FRAMES = decode()
COUNTS = [len(L) for L in FRAMES]
TOTAL = sum(COUNTS)                                   # 27,795 for item 31


def build_layouts(s=16):
    os.makedirs(os.path.join(GEN, f"lay_{ITEM}"), exist_ok=True)
    for k, L in enumerate(FRAMES):
        p = os.path.join(GEN, f"lay_{ITEM}", f"{k:02d}.png")
        if os.path.exists(p):
            continue
        im = Image.new("RGB", (W * s, H * s))
        dr = ImageDraw.Draw(im)
        m = int(round(0.1 * s))
        for y0, y1, x0, x1 in L:
            dr.rounded_rectangle([x0 * s + m, y0 * s + m, x1 * s - m - 1, y1 * s - m - 1],
                                 radius=max(1, int(0.1 * s * min(y1 - y0, x1 - x0))),
                                 fill=AREA_RGB[(y1 - y0) * (x1 - x0)])
        im.save(p)


build_layouts()

# left column
LX, PW = -4.75, 4.0
PH = PW * 9 / 16
VID_Y, LAY_Y = 1.95, -0.85


class LoTSpaceTime(Scene):
    def construct(self):
        fs = sorted(os.listdir(CLIP))
        vid_f = [np.asarray(Image.open(os.path.join(CLIP, f)).convert("RGBA").resize((960, 540))) for f in fs]
        lay_f = [np.asarray(Image.open(os.path.join(GEN, f"lay_{ITEM}", f"{k:02d}.png")).convert("RGBA").resize((960, 540)))
                 for k in range(T)]
        clock = lambda: self.renderer.time

        # volume
        cx, cy, w, h = REGION
        vol = ImageMobject(os.path.join(VOL, "0000.png")).set(width=w).move_to([cx, cy, 0])
        last = {"i": 0}

        def upd_vol(m, dt):   # dt updater: keeps manim from treating the waits as static
            i = min(int(round(clock() * FPS)), int(DURATION * FPS) - 1)
            if i != last["i"]:
                last["i"] = i
                m.pixel_array = np.asarray(Image.open(os.path.join(VOL, f"{i:04d}.png")).convert("RGBA"))
        vol.add_updater(upd_vol)
        self.add(vol)

        # video + the current latent frame's layout
        vid = ImageMobject(vid_f[0]).set(width=PW).move_to([LX, VID_Y, 0])
        lay = ImageMobject(lay_f[0]).set(width=PW).move_to([LX, LAY_Y, 0])
        vid.add_updater(lambda m, dt: setattr(m, "pixel_array", vid_f[video_frame(clock())]))
        lay.add_updater(lambda m, dt: setattr(m, "pixel_array", lay_f[latent_of(video_frame(clock()))]))
        self.add(vid, framed(vid, GHOST), lay)
        self.add(tx("generated video", 22, DIM).move_to([LX, VID_Y + PH / 2 + 0.22, 0]))
        cache = {}

        def label():
            k = latent_of(video_frame(clock()))
            if k not in cache:
                cache[k] = tx(f"latent frame {k + 1} of {T} · {fmt(COUNTS[k])} tokens", 20, DIM)
            return cache[k].copy().move_to([LX, LAY_Y + PH / 2 + 0.2, 0])
        self.add(always_redraw(label))

        # time axis on the volume's front-left edge, after the tilt
        events = []
        for t0, s in CAPS:
            events.append((t0, lambda s=s: self.set_caption(s)))
        foot = tx("per-frame layouts stacked in time: each slab is one latent frame, no token spans time",
                  18, DIM).set_y(3.72).to_edge(RIGHT, buff=0.35)
        events.append((2.8, lambda: self.add(foot)))
        # the footnote describes the space-time view only; the shrink / fall / flatten pools the count
        events.append((CAPS[2][0], lambda: self.play(FadeOut(foot), run_time=0.3)))
        count = VGroup(tx(f"{fmt(TOTAL)} tokens", 30, AMBER, bold=True),
                       tx(f"of {fmt(T * H * W)} at full resolution", 20, DIM)).arrange(DOWN, buff=0.08)
        count.move_to([LX, -2.6, 0])
        level = TOTAL / (W * T)                       # slab height (cells) = 16.545 of H = 45
        frac = f"{100 * TOTAL / (T * H * W):.1f}%"     # 36.8%

        def at_tick(lv, m):                          # label just left of a y-axis tick (camera at rest by then)
            return m.next_to(project(clock(), ytick(lv)[1]), LEFT, buff=0.1)

        def collapse():
            pct = at_tick(level, tx(frac, 28, AMBER, bold=True))
            self.play(ReplacementTransform(count[0], pct), FadeOut(count[1]),
                      run_time=COLLAPSE[1] - COLLAPSE[0], rate_func=smooth)

        events.append((CAPS[3][0], lambda: self.add(count)))
        events.append((GHOST_T[1], lambda: self.add(at_tick(H, tx("100%", 22, DIM)))))
        events.append((COLLAPSE[0], collapse))
        self.cap = None
        for t0, fn in sorted(events, key=lambda e: e[0]):
            if t0 > clock():
                self.wait(t0 - clock())
            fn()
        self.wait(DURATION - clock())

    def set_caption(self, s):
        if self.cap is not None:
            self.remove(self.cap)
        self.cap = tx(s, 28).move_to([0, CAPTION_Y, 0])
        self.add(self.cap)
