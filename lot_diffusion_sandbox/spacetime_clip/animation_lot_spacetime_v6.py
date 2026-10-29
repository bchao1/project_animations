"""Level-of-Token Diffusion — 10-s X clip: a space-time view of a video's per-frame LoT layouts, v6 (v5 + footnote right-aligned inside the frame).

Data and timeline: spacetime_common.py (site showcase item 31, LoT-Wan2.1 14B, depth-of-field layout, 21 latent
frames × 45 × 80 tokens, exact leaves from ../web/showcase.json). The 3D volume is rendered by render_volume.py
(moderngl, 60 fps) into data/generated/vol_31/; this scene composites it with the generated video, the current
latent frame's 2D layout, captions and the time axis. Every token is one latent frame thick; the tube is the
stacking of per-frame layouts (footnote on screen).

Render (see render.sh):
    python render_volume.py && manim -qh --media_dir media animation_lot_spacetime_v6.py LoTSpaceTime
"""
from manim import *
import numpy as np
from PIL import Image, ImageDraw
import os

from spacetime_common import (CAPS, CLIP, DURATION, FPS, GEN, H, REGION, T, VOL, VOL_H, W, TILT, decode,
                              latent_of, project, video_frame, world_x, world_z)

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
TOTAL = sum(COUNTS)                                   # 27,795


def build_layouts(s=16):
    os.makedirs(os.path.join(GEN, "lay"), exist_ok=True)
    for k, L in enumerate(FRAMES):
        p = os.path.join(GEN, "lay", f"{k:02d}.png")
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
        lay_f = [np.asarray(Image.open(os.path.join(GEN, "lay", f"{k:02d}.png")).convert("RGBA").resize((960, 540)))
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
        def t_axis():
            """Screen-vertical time arrow just left of the volume's projected silhouette (never over it)."""
            t = clock()
            if t < TILT[1]:
                return VGroup()
            xs = (world_x(0), world_x(W))
            zs = (world_z(0), world_z(H))
            bot = [project(t, (x, 0.0, z)) for x in xs for z in zs]
            top = [project(t, (x, VOL_H, z)) for x in xs for z in zs]
            x0 = min(p[0] for p in bot + top) - 0.35
            p0 = np.array([x0, min(p[1] for p in bot), 0.0])
            p1 = np.array([x0, max(p[1] for p in top), 0.0])
            a = Arrow(p0, p1, buff=0, color=DIM, stroke_width=3, max_tip_length_to_length_ratio=0.08,
                      tip_length=0.16)
            return VGroup(a, tx("time", 22, DIM).next_to(p1, UP, buff=0.1))

        events = []
        for t0, s in CAPS:
            events.append((t0, lambda s=s: self.set_caption(s)))
        events.append((TILT[1], lambda: self.add(always_redraw(t_axis))))
        events.append((2.8, lambda: self.add(
            tx("per-frame layouts stacked in time: each slab is one latent frame, no token spans time",
               18, DIM).set_y(3.72).to_edge(RIGHT, buff=0.35))))   # top strip above the volume, inside the frame
        events.append((CAPS[2][0], lambda: self.add(
            VGroup(tx(f"{fmt(TOTAL)} tokens", 30, AMBER, bold=True),
                   tx(f"of {fmt(T * H * W)} at full resolution", 20, DIM)).arrange(DOWN, buff=0.08)
            .move_to([LX, -2.6, 0]))))
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
