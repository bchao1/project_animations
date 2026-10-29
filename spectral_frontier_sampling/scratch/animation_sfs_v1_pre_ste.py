"""Spectral Frontier Sampling (SFS): 3b1b-style teaser and method video, v1.

Source: the CVPR 2027 draft in /home/brianchc/prompt-adaptive-speed-cvpr-2027/submission/adaptive-speed-cvpr2027/
(Secs. 1 and 3). Data: data/sfs_data.npz (scratch/prep_data.py) and data/capture/ (scratch/capture.py), both from
real FLUX.2 [klein] 2048^2 runs (N = 50 steps). See storyboard.md.

Render (see render.sh):
    ./render.sh l teaser        # T00 ... T04
    ./render.sh l method        # M00 ... M07
"""
from manim import *
import numpy as np
from PIL import Image
import matplotlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CAP = os.path.join(DATA, "capture")

config.background_color = BLACK
config.text_dir = os.path.join(HERE, "media", "texts", str(os.getpid()))   # parallel renders race on one cache
CAPTIONS = True

# ------------------------------------------------------------------ style (as the SPEED animation)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
SIG = "#58C4DD"        # signal (3b1b blue)
NOI = "#FC6255"        # noise / waste (3b1b red)
OURS = "#FFCA28"       # frontier / ours
SPD = "#B0BEC5"        # SPEED staircases
DIM = "#8A8A8A"
GOOD = "#83C167"       # 3b1b green
HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×→≈∼−∈≤≥‖·∝≲⋯²")


def mathify(s):
    return "".join(f'<span font_family="{MATHFONT}">{c}</span>' if c in MATHCH else c for c in s)


def sub(x):
    return f'<span rise="-3000"><small><small>{x}</small></small></span>'


def sup(x):
    return f'<span rise="4500"><small><small>{x}</small></small></span>'


def col(x, c):
    return f'<span foreground="{c}">{x}</span>'


def tx(s, size=30, color=WHITE, bold=False, italic=False):
    if bold:
        s = f"<b>{s}</b>"
    if italic:
        s = f"<i>{s}</i>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


XT = "<i>x</i>" + sub("<i>t</i>")
X0H = "<i>x̂</i>" + sub("0")


def rgba(a):
    a = np.clip(a, 0.0, 1.0)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    out = np.empty(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = (a * 255 + 0.5).astype(np.uint8)
    out[..., 3] = 255
    return out


def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size is not None and im.size[0] != size:
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im) / 255.0


def arr_img(a, h, nearest=False):
    m = ImageMobject(rgba(a))
    if nearest:
        m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    m.height = h
    return m


def framed(m, color=GREY_B, sw=1.2):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw)


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw, max_tip_length_to_length_ratio=0.2, tip_length=0.16)


MAGMA = matplotlib.colormaps["magma"]

# ------------------------------------------------------------------ data
Z = np.load(os.path.join(DATA, "sfs_data.npz"))
NSTEP, NB = 50, 128
CY = Z["compute_of_bin"]                 # compute-axis value of bin f (f = 0..128)
FB = np.arange(len(CY), dtype=float)
NAMES = {2580: "forest at sunset", 2348: "busy street", 1026: "border collie", 577: "kitchen interior",
         548: "barn in a field"}
PROMPTS = [p for p in (2580, 2348, 1026, 577, 548) if f"p{p}_P_x0" in Z.files]
LO = -1.6      # about log10(0.5th percentile of the final power) - 0.6, as spectrogram.py (all prompts)
HI = 0.6       # about the final power at bin 20: keeps bins 32-128 (the ones that matter) in the bright range
YMIN = float(np.interp(0.5, FB, CY))
YMAX = float(CY[-1] + 0.5 * (CY[-1] - CY[-2]))


def cy(f):
    return float(np.interp(f, FB, CY))


def spec_rgb(p, hpx=420, cw=12):
    """[hpx, 50*cw, 3] magma image of log10 x0hat power, rows on the compute axis (top = bin 128)."""
    P = Z[f"p{p}_P_x0"]
    ys = np.linspace(YMAX, YMIN, hpx)
    b = np.clip(np.round(np.interp(ys, CY, FB)), 1, NB).astype(int)
    L = np.log10(np.maximum(np.nan_to_num(P[:, b]), 10 ** LO))          # [steps, rows]
    c = MAGMA(np.clip((L - LO) / (HI - LO), 0, 1))[..., :3]              # [steps, rows, 3]
    return np.repeat(c.transpose(1, 0, 2), cw, axis=1)


def ceil_of(p, arm):
    return np.asarray(Z[f"p{p}_{arm}"], float)


def speedup(p, arm):
    return float(Z[f"p{p}_{arm}_x"])


# ------------------------------------------------------------------ spectrogram widget
class Spectrogram:
    """Axes + magma image (x = step, y = frequency on the compute axis). Use .p(step, bin) for points."""

    def __init__(self, p, center=(0.9, 0.3, 0), w=7.4, h=4.6, cw=12):
        self.w, self.h, self.cw = w, h, cw
        self.ax = Axes(x_range=[-0.5, NSTEP - 0.5, 10], y_range=[YMIN, YMAX, 1], x_length=w, y_length=h, tips=False,
                       axis_config={"stroke_width": 2, "color": GREY_B, "include_ticks": False})
        self.ax.move_to(center)
        self.rgb = spec_rgb(p, cw=cw)
        self.img = ImageMobject(rgba(self.rgb))
        self.img.stretch_to_fit_width(w)
        self.img.stretch_to_fit_height(h)
        self.img.move_to(self.ax.c2p(NSTEP / 2 - 0.5, (YMIN + YMAX) / 2))
        labs = VGroup()
        for s in (0, 10, 20, 30, 40, 49):
            q = self.p(s, 0.5)
            labs.add(Line(q, q + DOWN * 0.08, stroke_width=2, color=GREY_B),
                     tx(str(s), 18, GREY_B).next_to(q, DOWN, buff=0.13))
        for f in (32, 64, 96, 128):
            q = self.p(-0.5, f)
            labs.add(Line(q, q + LEFT * 0.08, stroke_width=2, color=GREY_B),
                     tx(str(f), 18, GREY_B).next_to(q, LEFT, buff=0.12))
        self.ticks = labs
        self.xlab = tx("denoising step", 22, GREY_B).next_to(self.ax, DOWN, buff=0.45)
        self.ylab = VGroup(tx("frequency", 22, GREY_B), tx("(height = cost of one step)", 16, DIM)) \
            .arrange(DOWN, buff=0.06).rotate(PI / 2).next_to(self.ax, LEFT, buff=0.55)
        self.frame = Rectangle(width=w, height=h, stroke_color=GREY_B, stroke_width=1.2).move_to(self.img)

    def p(self, step, f):
        return self.ax.c2p(step, cy(f))

    def labels(self):
        return VGroup(self.ticks, self.xlab, self.ylab)

    def stair_pts(self, c):
        pts = [self.p(-0.5, c[0])]
        for i in range(NSTEP):
            pts.append(self.p(i + 0.5, c[i]))
            if i + 1 < NSTEP and c[i + 1] != c[i]:
                pts.append(self.p(i + 0.5, c[i + 1]))
        return pts

    def stair(self, c, color, sw=4, dashed=False):
        m = VMobject(stroke_color=color, stroke_width=sw)
        m.set_points_as_corners(self.stair_pts(c))
        return DashedVMobject(m, num_dashes=90, dashed_ratio=0.6) if dashed else m

    def area(self, c, color, opacity):
        pts = self.stair_pts(c) + [self.p(NSTEP - 0.5, 0.5), self.p(-0.5, 0.5)]
        return Polygon(*pts, stroke_width=0, fill_color=color, fill_opacity=opacity)

    def between(self, upper, lower, color, opacity):
        lower = np.minimum(lower, upper)
        pts = self.stair_pts(upper) + self.stair_pts(lower)[::-1]
        return Polygon(*pts, stroke_width=0, fill_color=color, fill_opacity=opacity)

    def hatch(self, upper, lower, color=NOI):
        """Red diagonal hatching (raster, same pixel grid as the image) between two staircases, plus its outline.
        The spectrogram stays visible between the stripes."""
        hpx, wpx = self.rgb.shape[:2]
        ys = np.linspace(YMAX, YMIN, hpx)
        b = np.interp(ys, CY, FB)[:, None]
        st = np.minimum(np.arange(wpx) // self.cw, NSTEP - 1)
        up, lo = np.asarray(upper)[st][None], np.minimum(lower, upper)[st][None]
        inside = (b > lo) & (b <= up)
        rr, cc = np.mgrid[0:hpx, 0:wpx]
        stripe = ((rr + cc) % 16) < 5
        a = np.zeros((hpx, wpx, 4), np.uint8)
        rgb = (np.array(color_to_rgb(color)) * 255).astype(np.uint8)
        a[..., :3] = rgb
        a[..., 3] = np.where(inside, np.where(stripe, 235, 40), 0)
        m = ImageMobject(a)
        m.stretch_to_fit_width(self.w)
        m.stretch_to_fit_height(self.h)
        m.move_to(self.img)
        edge = self.between(upper, lower, color, 0).set_stroke(color, 2)
        return Group(m, edge)

    def set_rgb(self, rgb):
        self.img.pixel_array = rgba(rgb)


def colorbar(h=2.2, w=0.22):
    v = np.linspace(1, 0, 200)[:, None]
    a = np.repeat(MAGMA(v)[..., :3], 8, axis=1)
    m = ImageMobject(rgba(a))
    m.stretch_to_fit_width(w)
    m.stretch_to_fit_height(h)
    t = tx("strong signal", 16, GREY_B).next_to(m, UP, buff=0.1)
    b = tx("no signal yet", 16, GREY_B).next_to(m, DOWN, buff=0.1)
    return Group(m, framed(m, GREY_B, 1), t, b)


# ------------------------------------------------------------------ scene base
class Base(Scene):
    def setup(self):
        self.cap = None
        self.hdr = None

    def header(self, s):
        if self.hdr is not None:
            self.remove(self.hdr)
        self.hdr = tx(s, 24, DIM).to_corner(UL, buff=0.35).set_y(HEADER_Y)
        if CAPTIONS:
            self.add(self.hdr)

    def _hold_caption(self):
        if self.cap is not None:
            left = self.cap_need - (self.renderer.time - self.cap_t0)
            if left > 0.05:
                self.wait(left)

    @staticmethod
    def _split(s):
        opened, spaces, n, k = 0, [], 0, 0
        while k < len(s):
            if s[k] == "<":
                end = s.index(">", k)
                opened += -1 if s[k + 1] == "/" else 1
                k = end + 1
                continue
            if opened == 0 and s[k] == " ":
                spaces.append((k, n))
            n += 1
            k += 1
        if not spaces:
            return [s]
        k = min(spaces, key=lambda q: abs(q[1] - n / 2))[0]
        return [s[:k], s[k + 1:]]

    def caption(self, s, wait=None, size=28, hold=True):
        """Show a caption. hold=True waits until it has been read (len/15 + 1 s) before returning."""
        if not CAPTIONS:
            if wait:
                self.wait(wait)
            return None
        self._hold_caption()
        if self.cap is not None:
            self.remove(self.cap)
        c = tx(s, size)
        if c.width > 13.2:
            c = VGroup(*[tx(q, size - 2) for q in self._split(s)]).arrange(DOWN, buff=0.12)
            if c.width > 13.2:
                c.scale_to_fit_width(13.2)
        c.move_to([0, CAPTION_Y, 0])
        if c.get_bottom()[1] < -3.85:
            c.shift(UP * (-3.85 - c.get_bottom()[1]))
        self.cap = c
        self.cap_t0 = self.renderer.time
        self.cap_need = len(re.sub(r"<[^>]+>", "", s)) / 15.0 + 1.0
        self.add(c)
        if hold:
            self._hold_caption()
        if wait:
            self.wait(wait)
        return c

    def uncaption(self):
        self._hold_caption()
        if self.cap is not None:
            self.remove(self.cap)
            self.cap = None

    def section(self, name):
        self.next_section(name)

    def clear_all(self, keep=(), run_time=0.7):
        self.uncaption()
        keep = set(keep) | {self.hdr}
        mobs = [m for m in self.mobjects if m not in keep]
        for m in mobs:
            m.clear_updaters()
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)


def x0_frames(src, p):
    d = os.path.join(CAP, f"{src}_{p:06d}")
    if not os.path.exists(os.path.join(d, "x0_49.png")):
        return None
    return [load_rgb(os.path.join(d, f"x0_{i:02d}.png"), 384) for i in range(NSTEP)]


def final_img(p, arm="cont2x"):
    return load_rgb(os.path.join(DATA, f"final_{arm}_{p:06d}.png"), 384)


IMG_C = np.array([-5.45, 0.45, 0])
IMG_S = 2.8


# ================================================================== TEASER
class T00_Title(Base):
    def construct(self):
        t = tx("Spectral Frontier Sampling", 60, bold=True).move_to([0, 0.5, 0])
        s = tx("Online, per-sample resolution for diffusion models", 30, GREY_B).next_to(t, DOWN, buff=0.35)
        self.play(FadeIn(t, shift=UP * 0.2), run_time=0.8)
        self.add(s)
        self.wait(2.2)
        self.play(FadeOut(t), FadeOut(s), run_time=0.6)


class T01_Spectrogram(Base):
    def construct(self):
        self.header("How diffusion builds an image")
        P = 2580
        sp = Spectrogram(P)
        frames = x0_frames("full", P) or [final_img(P, "fullres")] * NSTEP
        pic = arr_img(frames[0], IMG_S).move_to(IMG_C)
        pf = framed(pic)
        plab = tx("clean guess " + X0H + ", full-size run", 18, GREY_B).next_to(pic, DOWN, buff=0.15)
        k = ValueTracker(0)
        black = np.zeros_like(sp.rgb)

        def reveal(mob):
            n = int(round(k.get_value() * sp.cw))
            a = black.copy()
            a[:, :n] = sp.rgb[:, :n]
            mob.pixel_array = rgba(a)
        sp.set_rgb(black)
        cb = colorbar().move_to([5.6, 0.3, 0])
        self.play(FadeIn(pic), Create(pf), Create(sp.frame), run_time=0.6)
        self.add(plab, sp.labels())
        self.caption("Diffusion turns noise into an image in 50 steps.")
        self.section("build")
        sp.img.add_updater(reveal)
        pic.add_updater(lambda m: setattr(m, "pixel_array",
                                          rgba(frames[min(NSTEP - 1, int(k.get_value()))])))
        cursor = always_redraw(lambda: Line(sp.p(k.get_value() - 0.5, 0.5), sp.p(k.get_value() - 0.5, 128.5),
                                            color=WHITE, stroke_width=2))
        stepn = always_redraw(lambda: tx(f"step {min(NSTEP, int(k.get_value()))}", 22)
                              .next_to(pic, UP, buff=0.15))
        self.add(sp.img, cursor, stepn, cb)
        self.caption("At each step, we measure how much power the image has at each frequency.")
        self.caption("Low frequencies (coarse shapes) appear first.", hold=False)
        self.play(k.animate.set_value(30), run_time=5.0, rate_func=linear)
        self.caption("Fine detail (high frequencies) appears only near the end.", hold=False)
        self.play(k.animate.set_value(NSTEP), run_time=4.5, rate_func=linear)
        sp.img.clear_updaters()
        pic.clear_updaters()
        self.remove(cursor)
        self.caption("Black means no signal yet. Yet a full-size model pays for all of it.", wait=0.5)


def teaser_base(scene, P=2580):
    sp = Spectrogram(P)
    pic = arr_img(final_img(P, "fullres"), IMG_S).move_to(IMG_C)
    scene.add(pic, framed(pic), sp.img, sp.frame, sp.labels(), colorbar().move_to([5.6, 0.3, 0]))
    return sp, pic


class T02_Waste(Base):
    def construct(self):
        self.header("Where the compute goes")
        P = 2580
        sp, pic = teaser_base(self, P)
        self.add(tx("forest at sunset", 18, GREY_B).next_to(pic, DOWN, buff=0.15))
        full = Rectangle(width=sp.w, height=sp.h, stroke_width=0, fill_color=WHITE, fill_opacity=0.14).move_to(sp.img)
        self.play(FadeIn(full), run_time=0.6)
        self.caption("Height shows the cost of one step. Area shows total compute.")
        self.caption("A full-size model pays for the whole box at every step.")
        self.play(FadeOut(full), run_time=0.5)

        def leg(i, name, x, color, dashed):
            ln = Line(ORIGIN, RIGHT * 0.5, stroke_color=color, stroke_width=4)
            if dashed:
                ln = DashedLine(ORIGIN, RIGHT * 0.5, color=color, stroke_width=4, dash_length=0.08)
            row = VGroup(ln, tx(name, 20, color), tx(f"{x:.1f}× faster", 20, color)).arrange(RIGHT, buff=0.2)
            row.move_to(sp.p(0.5, 125), aligned_edge=LEFT).shift(DOWN * 0.42 * i)   # black upper-left corner
            return row

        self.section("speed2")
        c2, c3, co = ceil_of(P, "speed2"), ceil_of(P, "speed3"), ceil_of(P, "cont2x")
        s2 = sp.stair(c2, SPD, 4, dashed=True)
        self.play(Create(s2), run_time=1.2)
        l2 = leg(0, "SPEED, 2 stages", speedup(P, "speed2"), SPD, True)
        self.add(l2)
        self.caption("SPEED runs early steps on a small grid, then grows it once.")
        w2 = sp.hatch(c2, co)
        self.play(FadeIn(w2), run_time=0.8)
        self.caption(col("Red", NOI) + ": compute spent on frequencies that are still black, with no signal yet.")

        self.section("speed3")
        s3 = sp.stair(c3, SPD, 4, dashed=True)
        w3 = sp.hatch(c3, co)
        l3 = leg(1, "SPEED, 3 stages", speedup(P, "speed3"), SPD, True)
        self.play(Transform(s2, s3), FadeOut(w2), FadeIn(w3), l2.animate.set_opacity(0.45), run_time=1.4)
        self.add(l3)
        self.caption("More stages waste less. But each stage is one more hand-picked choice.")
        self.caption("And each step up still adds a full octave of frequencies that are mostly noise.")

        self.section("ours")
        so = sp.stair(co, OURS, 5)
        lo_ = leg(2, "Ours", speedup(P, "cont2x"), OURS, False)
        self.play(Create(so), run_time=2.0)
        self.add(lo_)
        self.caption("Our grid grows with the " + col("spectral frontier", OURS)
                     + ": the highest frequency that has signal.")
        self.play(FadeOut(w3), l3.animate.set_opacity(0.45), run_time=1.4)
        self.caption(f"The red waste is gone. This image needs {speedup(P, 'cont2x'):.1f}× less compute than full size.")
        self.wait(0.5)


class T03_PerSample(Base):
    def construct(self):
        self.header("One schedule per image")
        order = [p for p in (2580, 2348, 1026) if p in PROMPTS]
        P = order[0]
        sp, pic = teaser_base(self, P)
        name = tx(NAMES[P], 18, GREY_B).next_to(pic, DOWN, buff=0.15)
        self.add(name)
        ghost = sp.stair(ceil_of(P, "speed3"), SPD, 3, dashed=True).set_opacity(0.5)
        gl = tx("SPEED (fixed)", 18, SPD).set_opacity(0.7).next_to(sp.p(30, 128), DOWN, buff=0.12)
        so = sp.stair(ceil_of(P, "cont2x"), OURS, 5)
        self.add(ghost, gl, so)

        def kmark(p):
            K0 = int(Z[f"p{p}_cont2x_K"][0])
            ln = DashedLine(sp.p(K0 - 0.5, 0.5), sp.p(K0 - 0.5, 128.5), color=OURS, stroke_width=2, dash_length=0.08)
            lb = tx(f"grows at step {K0}", 20, OURS).next_to(sp.p(K0 - 0.5, 128.5), UP, buff=0.1)
            return VGroup(ln, lb)
        km = kmark(P)
        self.add(km)
        self.caption("Our staircase follows the frontier of this image.")
        self.wait(0.3)
        caps = {2348: "A busy, detailed image needs high frequencies early: the grid grows sooner.",
                1026: "A smooth image has little fine detail: the grid stays small longer."}
        for p in order[1:]:
            self.section(f"to_{p}")
            a, b = sp.rgb.copy(), spec_rgb(p, cw=sp.cw)
            pa, pb = final_img(P, "fullres"), final_img(p, "fullres")
            al = ValueTracker(0)
            sp.img.add_updater(lambda m: setattr(m, "pixel_array", rgba((1 - al.get_value()) * a + al.get_value() * b)))
            pic.add_updater(lambda m: setattr(m, "pixel_array", rgba((1 - al.get_value()) * pa + al.get_value() * pb)))
            new_so = sp.stair(ceil_of(p, "cont2x"), OURS, 5)
            new_km = kmark(p)
            new_name = tx(NAMES[p], 18, GREY_B).next_to(pic, DOWN, buff=0.15)
            self.play(al.animate.set_value(1), Transform(so, new_so), Transform(km, new_km),
                      FadeTransform(name, new_name), run_time=2.2)
            name = new_name
            sp.img.clear_updaters()
            pic.clear_updaters()
            sp.rgb = b
            P = p
            self.caption(caps.get(p, ""))
        self.caption("SPEED uses one schedule for every image. Ours finds one per image, while it samples.")
        self.wait(0.5)


class T04_End(Base):
    def construct(self):
        t = tx("Spectral Frontier Sampling", 48, bold=True).move_to([0, 1.6, 0])
        a = tx("1.7× faster than a 3-stage schedule", 34, OURS)
        b = tx("4.7–5.0× faster than full resolution", 34, OURS)
        c = tx("HPSv3 −0.19 vs. the 3-stage schedule · no training", 24, GREY_B)
        d = tx("FLUX.2 [klein], 2048 × 2048", 22, DIM)
        g = VGroup(a, b, c, d).arrange(DOWN, buff=0.3).next_to(t, DOWN, buff=0.6)
        self.play(FadeIn(t), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in g], lag_ratio=0.25), run_time=1.6)
        self.wait(3.0)
        self.play(FadeOut(VGroup(t, g)), run_time=0.6)


# ================================================================== METHOD
RUN = 2348 if os.path.exists(os.path.join(DATA, "method_002348.npz")) else 577
MZ = np.load(os.path.join(DATA, f"method_{RUN:06d}.npz"))
LG = np.log10(np.e)
FIRES = [i for i in range(NSTEP) if MZ["fire"][i]]
I1 = FIRES[0]                                    # first growth decision
ILATE = FIRES[-2] if len(FIRES) > 1 else I1      # a late one, where the residual fraction u is clearly < 1
THR = MZ["lps"] * LG                             # log10 P*(sigma_{i+1}; delta): power needed to beat the noise
MLOG = MZ["m"] * LG                              # mean log10 power of generated images (bins 2..128)
PFIN = np.log10(MZ["P_final"])                   # this image's finished power (known only after sampling)
ZM = float(MZ["z"])
BINS = np.arange(2, NB + 1)


def e10(k):
    return "10" + sup(("−" if k < 0 else "") + str(abs(k)))


def cap_frame(src, i, p=RUN, size=384):
    return load_rgb(os.path.join(CAP, f"cont_{p:06d}", f"{src}_{i:02d}.png"), size)


def frontier_bin(logp, thr):
    above = np.where(logp[2:] >= thr)[0]
    return int(BINS[above[-1]]) if len(above) else 2


class LogPlot:
    """log-log spectrum axes: x = frequency bin xb[0]..xb[1], y = log10 power. p(bin, log10 P) -> point."""

    def __init__(self, center=(1.5, 0.25, 0), w=7.8, h=4.7, yr=(-1.3, 2.5), xb=(2, 128)):
        self.xb, self.yr = xb, yr
        self.xr = (np.log10(xb[0]), np.log10(xb[1]))
        self.ax = Axes(x_range=[self.xr[0], self.xr[1], 1], y_range=[0, yr[1] - yr[0], 1], x_length=w, y_length=h,
                       tips=False, axis_config={"stroke_width": 2, "color": GREY_B, "include_ticks": False})
        self.ax.move_to(center)
        labs = VGroup()
        xt = (2, 4, 8, 16, 32, 64, 128) if xb[0] < 8 else (8, 16, 24, 32, 48, 64, 96, 128)
        for b in xt:
            if not xb[0] <= b <= xb[1]:
                continue
            q = self.p(b, yr[0])
            labs.add(Line(q, q + DOWN * 0.08, stroke_width=2, color=GREY_B),
                     tx(str(b), 18, GREY_B).next_to(q, DOWN, buff=0.12))
        for v in range(int(np.ceil(yr[0])), int(np.floor(yr[1])) + 1):
            q = self.p(xb[0], v)
            labs.add(Line(q, q + LEFT * 0.08, stroke_width=2, color=GREY_B),
                     tx(e10(v), 18, GREY_B).next_to(q, LEFT, buff=0.12))
        self.ticks = labs
        self.xlab = tx("frequency (cycles per image)", 22, GREY_B).next_to(self.ax, DOWN, buff=0.5)
        self.ylab = tx("power", 22, GREY_B).rotate(PI / 2).next_to(self.ax, LEFT, buff=0.7)

    def inx(self, b):
        return self.xb[0] <= b <= self.xb[1]

    def p(self, b, y):
        return self.ax.c2p(np.log10(b), float(np.clip(y, self.yr[0], self.yr[1])) - self.yr[0])   # x axis at the bottom

    def parts(self):
        return VGroup(self.ax, self.ticks, self.xlab, self.ylab)

    def curve(self, bins, ys, color, sw=3, dashed=False, opacity=1.0):
        pts = [self.p(b, y) for b, y in zip(bins, ys) if np.isfinite(y) and self.inx(b)]
        m = VMobject(stroke_color=color, stroke_width=sw, stroke_opacity=opacity)
        m.set_points_as_corners(pts)
        return DashedVMobject(m, num_dashes=60, dashed_ratio=0.55) if dashed else m

    def band(self, bins, lo, hi, color, opacity=0.25):
        bins = np.asarray(bins)
        ok = np.isfinite(lo) & np.isfinite(hi) & (bins >= self.xb[0]) & (bins <= self.xb[1])
        b, lo, hi = bins[ok], np.asarray(lo)[ok], np.asarray(hi)[ok]
        pts = [self.p(x, y) for x, y in zip(b, hi)] + [self.p(x, y) for x, y in zip(b[::-1], lo[::-1])]
        return Polygon(*pts, stroke_width=0, fill_color=color, fill_opacity=opacity)

    def hline(self, y, color=NOI, sw=3):
        return DashedLine(self.p(self.xb[0], y), self.p(self.xb[1], y), color=color, stroke_width=sw, dash_length=0.1)

    def vline(self, b, color=WHITE, sw=2, dashed=True):
        a, c = self.p(b, self.yr[0]), self.p(b, self.yr[1])
        return DashedLine(a, c, color=color, stroke_width=sw, dash_length=0.08) if dashed else \
            Line(a, c, color=color, stroke_width=sw)

    def signal_area(self, logp, thr, color=SIG, opacity=0.3):
        """area between the curve and the threshold line where the curve is above it."""
        fb = frontier_bin(logp, thr)
        if fb <= 2:
            return VMobject()
        b = np.arange(2, fb + 1)
        return self.band(b, np.full(len(b), thr), logp[b], color, opacity)


def tokgrid(n, size, center, new_from=None, color=SIG, new_color=NOI):
    """n/8 x n/8 squares (one per 8 x 8 latent block). Squares outside the old n/8 block get new_color."""
    k = n // 8
    c = size / k
    g = VGroup()
    for i in range(k):
        for j in range(k):
            isnew = new_from is not None and (i >= new_from // 8 or j >= new_from // 8)
            g.add(Square(side_length=0.82 * c, stroke_width=0, fill_color=new_color if isnew else color,
                         fill_opacity=0.85).move_to(np.array(center) + np.array([(j + 0.5) * c - size / 2,
                                                                                 size / 2 - (i + 0.5) * c, 0])))
    return g


def step_state(pl, i):
    """mobjects of the estimator at step i: ring dots (mapped), the estimate curve, its band."""
    n = int(MZ["n"][i])
    ok = MZ["ok"][i]
    mp = MZ["mapped"][i] * LG
    dots = VGroup(*[Dot(pl.p(w, mp[w]), radius=0.055, color=SIG) for w in range(NB + 1) if ok[w] and pl.inx(w)])
    t = np.array([b for b in range(n // 2, min(n, NB) + 1) if np.isfinite(MZ["est"][i, b])])   # candidates <= 2n
    if len(t) == 0:
        return dots, VMobject(), VMobject(), n
    e, sd = MZ["est"][i, t] * LG, MZ["sd"][i, t] * LG
    return dots, pl.curve(t, e, OURS, 3), pl.band(t, e - sd, e + sd, OURS, 0.22), n


def test_value(i, b):
    """log10 of the tested power at boundary bin b: estimate + log u - z sd (u measured at the grid's own boundary,
    g(in-grid SNR) beyond it)."""
    n = int(MZ["n"][i])
    u = MZ["u"][i] if b == n // 2 else MZ["ug"][i, b]
    return (MZ["est"][i, b] + np.log(u) - ZM * MZ["sd"][i, b]) * LG


class M00_Title(Base):
    def construct(self):
        t = tx("Spectral Frontier Sampling", 56, bold=True).move_to([0, 1.5, 0])
        s = tx("How it works", 32, GREY_B).next_to(t, DOWN, buff=0.3)
        items = VGroup(tx("1 · The spectral frontier", 28), tx("2 · When to grow: online power estimation", 28),
                       tx("3 · To which size: frontier tracking", 28)).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        items.next_to(s, DOWN, buff=0.7)
        self.play(FadeIn(t, shift=UP * 0.2), run_time=0.8)
        self.add(s)
        self.play(LaggedStart(*[FadeIn(m) for m in items], lag_ratio=0.3), run_time=1.5)
        self.wait(2.5)
        self.play(FadeOut(VGroup(t, s, items)), run_time=0.6)


class M01_Primer(Base):
    def construct(self):
        self.header("1 · The spectral frontier")
        pl = LogPlot()
        pic = arr_img(final_img(RUN, "fullres"), 2.6).move_to([-5.5, 0.7, 0])
        self.add(pic, framed(pic), tx(NAMES[RUN], 18, GREY_B).next_to(pic, DOWN, buff=0.15))
        self.play(Create(pl.ax), run_time=0.6)
        self.add(pl.ticks, pl.xlab, pl.ylab)
        sc = pl.curve(BINS, PFIN[2:], SIG, 4)
        self.play(Create(sc), run_time=1.5)
        self.add(tx("finished image", 20, SIG).next_to(pl.p(3, PFIN[3]), RIGHT, buff=0.2).shift(UP * 0.25))
        self.caption("Power falls with frequency. Coarse shapes are strong. Fine detail is weak.")

        self.section("noise")
        steps = np.arange(NSTEP)
        k = ValueTracker(8)
        thr = lambda: float(np.interp(k.get_value(), steps, THR))  # noqa: E731
        line = always_redraw(lambda: pl.hline(thr()))
        lab = always_redraw(lambda: tx("noise level", 20, NOI).next_to(pl.p(128, thr()), UP, buff=0.08)
                            .align_to(pl.p(128, 0), RIGHT))
        stepl = always_redraw(lambda: tx(f"step {int(round(k.get_value()))} of 50", 22)
                              .next_to(pl.ax, UP, buff=0.1).align_to(pl.ax, RIGHT))
        self.play(Create(line), run_time=0.6)
        self.add(lab, stepl)
        self.caption("During sampling, noise hides every frequency whose power is below this line.")
        area = always_redraw(lambda: pl.signal_area(PFIN, thr()))
        fdot = always_redraw(lambda: Dot(pl.p(frontier_bin(PFIN, thr()), thr()), radius=0.08, color=OURS))
        fl = always_redraw(lambda: pl.vline(frontier_bin(PFIN, thr()), OURS, 2))
        flab = always_redraw(lambda: tx("frontier", 22, OURS).next_to(pl.p(frontier_bin(PFIN, thr()), pl.yr[1]),
                                                                      DOWN, buff=0.1))
        self.add(area, fl, fdot, flab)
        self.caption("Frequencies above the line have emerged from the noise.")
        self.caption("The highest one is the " + col("spectral frontier", OURS) + ".")
        self.section("sweep")
        self.caption("As sampling runs, the noise shrinks. The line drops, and the frontier moves right.", hold=False)
        kend = next(j for j in range(NSTEP) if frontier_bin(PFIN, THR[j]) >= 96)
        self.play(k.animate.set_value(kend), run_time=6.0, rate_func=linear)
        self.caption("A frequency counts as signal once (1 − <i>t</i>)²<i>P</i> / <i>t</i>² passes a small "
                     "tolerance <i>δ</i>.")
        for m in (line, lab, stepl, area, fdot, fl, flab):
            m.clear_updaters()

        self.section("spectrogram")
        self.clear_all()
        if f"p{RUN}_P_x0" in Z.files:
            sp = Spectrogram(RUN)
            self.add(sp.img, sp.frame, sp.labels(), colorbar().move_to([5.6, 0.3, 0]))
            fr = np.array([frontier_bin(PFIN, THR[i]) for i in range(NSTEP)], float)
            self.play(Create(sp.stair(fr, OURS, 3)), run_time=2.5)
            self.caption("Each step has one frontier. Over all steps, it traces the edge of the colored region.")
            self.caption("A grid that follows this edge pays for signal, and only for signal.")


class M02_Grid(Base):
    def construct(self):
        self.header("1 · The spectral frontier")
        pl = LogPlot(center=(1.9, 0.25, 0), w=7.4)
        g = tokgrid(64, 2.4, (-5.2, 0.7, 0))
        self.add(pl.parts(), pl.curve(BINS, PFIN[2:], SIG, 4))
        self.play(LaggedStart(*[FadeIn(q) for q in g], lag_ratio=0.01), run_time=0.8)
        glab = tx("grid side <i>n</i> = 64", 22).next_to(g, DOWN, buff=0.2)
        self.add(glab)
        self.caption("Sampling on a small grid is cheaper. But a grid of side <i>n</i> holds only frequencies up to "
                     "<i>n</i>/2.")
        bl = pl.vline(32, WHITE, 3, dashed=False)
        grey = Rectangle(width=pl.p(128, 0)[0] - pl.p(32, 0)[0], height=pl.ax.height, stroke_width=0,
                         fill_color=GREY_D, fill_opacity=0.45).move_to(pl.ax).align_to(pl.p(32, 0), LEFT)
        blab = tx("boundary <i>n</i>/2", 20).next_to(pl.p(32, pl.yr[1]), DOWN, buff=0.1).shift(LEFT * 0.75)
        self.play(Create(bl), FadeIn(grey), run_time=0.8)
        notg = tx("not on the grid", 20, GREY_B).move_to(grey).shift(UP * 1.6)
        self.add(blab, notg)

        self.section("too_small")
        late = THR[next(k for k in range(NSTEP) if frontier_bin(PFIN, THR[k]) >= 56)]
        ln = pl.hline(late)
        lost = pl.band(np.arange(32, frontier_bin(PFIN, late) + 1), np.full(frontier_bin(PFIN, late) - 31, late),
                       PFIN[32:frontier_bin(PFIN, late) + 1], SIG, 0.6)
        self.play(Create(ln), FadeIn(lost), run_time=0.8)
        self.caption("Boundary below the frontier: the grid drops real signal.")
        self.play(FadeOut(lost), FadeOut(ln), run_time=0.5)

        self.section("too_big")
        early = THR[next(k for k in range(NSTEP) if frontier_bin(PFIN, THR[k]) >= 40)]
        ln2 = pl.hline(early)
        fb = frontier_bin(PFIN, early)
        g2 = tokgrid(256, 2.4, (-5.2, 0.7, 0))
        bl2 = pl.vline(128, WHITE, 3, dashed=False)
        waste = Rectangle(width=pl.p(128, 0)[0] - pl.p(fb, 0)[0], height=pl.ax.height, stroke_width=0,
                          fill_color=NOI, fill_opacity=0.3).move_to(pl.ax).align_to(pl.p(fb, 0), LEFT)
        self.play(FadeOut(grey), FadeOut(notg), Transform(g, g2), Transform(bl, bl2), Create(ln2),
                  FadeTransform(glab, tx("grid side <i>n</i> = 256", 22).next_to(g, DOWN, buff=0.2)), run_time=1.0)
        self.play(FadeIn(waste), run_time=0.6)
        self.caption("Boundary far above the frontier: the grid pays for frequencies that are only noise.")
        self.section("two_questions")
        self.caption("So at every step, our sampler answers two questions:")
        q = VGroup(tx("1.  When to grow?", 30, OURS), tx("2.  To which size?", 30, OURS)) \
            .arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to([1.9, 0.4, 0])
        bg = BackgroundRectangle(q, color=BLACK, fill_opacity=0.85, buff=0.3)
        self.play(FadeIn(bg), FadeIn(q), run_time=0.6)
        self.wait(1.5)


def cross_step(logp32):
    k = np.where(THR <= logp32)[0]
    return int(k[0]) if len(k) else NSTEP - 1


class M03_Problem(Base):
    def construct(self):
        self.header("2 · When to grow")
        pl = LogPlot()
        self.add(pl.parts())
        b0 = 32
        others = [p for p in (RUN, 2580, 1026) if f"p{p}_P_final" in Z.files]
        cols = dict(zip(others, [SIG, "#FF8FB1", "#8BD3C7"]))
        curves, labs = VGroup(), VGroup()
        for p in others:
            lp = np.log10(Z[f"p{p}_P_final"])
            curves.add(pl.curve(BINS, lp[2:], cols.get(p, SIG), 3))
        mean = pl.curve(BINS, MLOG[2:], WHITE, 3, dashed=True)
        self.play(LaggedStart(*[Create(c) for c in curves], lag_ratio=0.3), run_time=1.8)

        self.play(Create(mean), run_time=0.8)
        self.add(tx("dashed: average image", 18).next_to(pl.p(3, pl.yr[0]), UP, buff=0.15).align_to(pl.p(2.2, 0), LEFT))
        self.caption("Each image has its own spectrum. SPEED uses one average spectrum for all.")
        bl = pl.vline(b0, WHITE, 3, dashed=False)
        self.play(Create(bl), run_time=0.5)
        self.add(tx("boundary of grid 64", 20).next_to(pl.p(b0, pl.yr[1]), DOWN, buff=0.1).shift(LEFT * 1.0))

        self.section("crossings")
        k = ValueTracker(8)
        line = always_redraw(lambda: pl.hline(float(np.interp(k.get_value(), np.arange(NSTEP), THR))))
        stepl = always_redraw(lambda: tx(f"step {int(round(k.get_value()))}", 22)
                              .next_to(pl.ax, UP, buff=0.1).align_to(pl.ax, RIGHT))
        self.add(line, stepl)
        self.caption("A 64 grid can grow once the noise line falls below the power at its boundary.", hold=False)
        events = [(cross_step(np.log10(Z[f"p{p}_P_final"][b0])), np.log10(Z[f"p{p}_P_final"][b0]),
                   cols.get(p, SIG), NAMES[p]) for p in others]
        events.append((cross_step(MLOG[b0]), MLOG[b0], WHITE, "average"))
        events.sort()
        head = tx("ready to grow at:", 20, GREY_B).move_to(pl.p(70, 2.25))
        self.add(head)
        for j, (ks, y, c, nm) in enumerate(events):
            self.play(k.animate.set_value(ks), run_time=max(0.4, 0.25 * (ks - k.get_value())), rate_func=linear)
            d = Dot(pl.p(b0, y), radius=0.08, color=c)
            row = VGroup(Dot(radius=0.06, color=c), tx(f"{nm}: step {ks}", 20, c)).arrange(RIGHT, buff=0.12)
            row.next_to(head, DOWN, buff=0.15 + 0.36 * j).align_to(head, LEFT)
            self.add(d, row)
        line.clear_updaters()
        stepl.clear_updaters()
        self.caption("Each image is ready at its own step. One fixed time is too early for some and too late "
                     "for others.")

        self.section("unknown")
        self.caption("But this power belongs to the " + col("finished", SIG) + " image. During sampling, we do not "
                     "have it.")
        q = tx("?", 60, SIG).move_to(pl.p(80, 0.3))
        self.play(*[c.animate.set_stroke(opacity=0.25) for c in curves], FadeIn(q), run_time=0.8)
        self.caption("And at the boundary, the small grid still sees mostly noise.")
        self.caption("So we estimate it from frequencies that are already clear.")


class M04_Estimate(Base):
    def construct(self):
        self.header("2 · When to grow: online power estimation")
        i = I1
        n = int(MZ["n"][i])
        pl = LogPlot(center=(1.9, 0.25, 0), w=7.4)
        xt = arr_img(cap_frame("xt", i), 1.9).move_to([-5.4, 1.45, 0])
        x0 = arr_img(cap_frame("x0", i), 1.9).move_to([-5.4, -1.15, 0])
        lt = tx("noisy state " + XT + f", grid {n}", 18, GREY_B).next_to(xt, UP, buff=0.1)
        l0 = tx("clean guess " + X0H, 18, GREY_B).next_to(x0, UP, buff=0.1)
        self.play(FadeIn(xt), Create(framed(xt)), run_time=0.5)
        self.add(lt, framed(xt))
        self.caption(f"Step {i}. The sampler is on a {n} × {n} grid.")
        self.section("x0hat")
        a = arrow(xt.get_bottom(), x0.get_top() + UP * 0.25, buff=0.08)
        self.play(GrowArrow(a), FadeIn(x0), run_time=0.8)
        self.add(framed(x0), l0)
        self.caption("The model predicts a velocity <i>v</i>. One subtraction gives a clean guess: "
                     + X0H + " = " + XT + " − <i>t</i>·<i>v</i>.")

        self.section("rings")
        self.play(FadeIn(pl.parts()), run_time=0.6)
        C, rel, ok = MZ["C"][i], MZ["rel"][i], MZ["ok"][i]
        obs = [w for w in range(NB + 1) if np.isfinite(C[w]) and C[w] > 0]
        good, bad = VGroup(), VGroup()
        for w in obs:
            y = np.log10(C[w])
            e = min(rel[w] * LG, 0.6)
            g = VGroup(Line(pl.p(w, y - e), pl.p(w, y + e), stroke_width=2, color=SIG if ok[w] else DIM),
                       Dot(pl.p(w, y), radius=0.055, color=SIG if ok[w] else DIM))
            (good if ok[w] else bad).add(g)
        self.play(LaggedStart(*[FadeIn(m) for m in list(good) + list(bad)], lag_ratio=0.08), run_time=1.2)
        wmax = max(obs)
        self.add(tx(f"rings 2 to {wmax}", 20, SIG).next_to(pl.p(5, pl.yr[0]), UP, buff=0.15))
        self.caption("Low frequencies settle first. We measure their power ring by ring, with an error bar.")
        if len(bad):
            self.play(FadeOut(bad), run_time=0.6)
            self.caption("Rings that are still too noisy are dropped.")

        self.section("vae")
        mp = MZ["mapped"][i] * LG
        moved = VGroup(*[Dot(pl.p(w, mp[w]), radius=0.055, color=SIG) for w in obs if ok[w]])
        self.play(Transform(good, moved), run_time=1.2)
        self.caption("The small grid is not a shrunk copy of the large one: the autoencoder changes its power.")
        self.caption("A map, measured once for each grid size, converts it to the full-size scale.")

        self.section("correlate")
        mean = pl.curve(BINS, MLOG[2:], WHITE, 2, dashed=True)
        self.play(Create(mean), run_time=0.8)
        self.add(tx("average image", 20).next_to(pl.p(90, MLOG[90]), DOWN, buff=0.2))
        devs = VGroup(*[Line(pl.p(w, MLOG[w]), pl.p(w, mp[w]), color=OURS, stroke_width=3) for w in obs if ok[w]])
        self.play(Create(devs), run_time=0.8)
        self.caption("Each ring sits above or below the average image. Here, the image is "
                     + ("above" if np.nanmean([mp[w] - MLOG[w] for w in obs if ok[w]]) > 0 else "below") + " it.")
        self.caption("Nearby frequencies tend to move together, so the low rings predict higher ones.")
        dots, cur, band, _ = step_state(pl, i)
        self.play(FadeIn(band), Create(cur), run_time=1.5)
        self.add(tx("estimate", 20, OURS).next_to(cur.get_end(), UP, buff=0.15))
        truth = pl.curve(BINS, PFIN[2:], SIG, 2, dashed=True, opacity=0.6)
        self.play(Create(truth), run_time=1.0)
        self.add(tx("true power (known only after sampling)", 18, SIG).next_to(pl.p(128, PFIN[128]), DOWN,
                                                                                  buff=0.25).align_to(pl.p(128, 0),
                                                                                                      RIGHT))
        self.caption("The estimate lands near the true power. It needs no extra model runs and no training.")

        self.section("residual")
        self.clear_all(keep=[])
        self.residual_bars()
        self.section("margin")
        self.margin(i)

    def residual_bars(self):
        self.header("2 · When to grow: only the new part counts")
        bars = VGroup()
        for j, i in enumerate((I1, ILATE)):
            u = float(MZ["u"][i])
            h = 3.0
            base = np.array([-2.0 + 4.0 * j, -1.6, 0])
            pred = Rectangle(width=1.1, height=h * (1 - u), stroke_width=0, fill_color=SIG, fill_opacity=0.8)
            new = Rectangle(width=1.1, height=h * u, stroke_width=0, fill_color=OURS, fill_opacity=0.85)
            new.move_to(base, aligned_edge=DOWN)
            pred.next_to(new, UP, buff=0)
            box = Rectangle(width=1.1, height=h, stroke_color=GREY_B, stroke_width=1.5).move_to(base, aligned_edge=DOWN)
            lab = tx(f"step {i}", 22).next_to(box, DOWN, buff=0.15)
            ul = tx(f"new part <i>u</i> = {u:.2f}", 20, OURS).next_to(box, RIGHT, buff=0.15).align_to(new, DOWN)
            bars.add(VGroup(box, new, pred, lab, ul))
        title = tx("power at the grid boundary", 22, GREY_B).move_to([0, 2.2, 0])
        self.add(title)
        self.play(FadeIn(bars[0]), run_time=0.7)
        self.caption("When the grid grows, the new frequencies start as noise. Their signal must be made from "
                     "scratch.")
        self.caption("But edges and textures span nearby frequencies. The bigger grid can predict part of it.")
        leg = VGroup(VGroup(Square(0.2, fill_color=SIG, fill_opacity=0.8, stroke_width=0),
                            tx("predicted from the small grid", 20, SIG)).arrange(RIGHT, buff=0.15),
                     VGroup(Square(0.2, fill_color=OURS, fill_opacity=0.85, stroke_width=0),
                            tx("new: only this part is lost", 20, OURS)).arrange(RIGHT, buff=0.15)) \
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to([4.6, 1.2, 0])
        self.add(leg)
        self.caption("So the test uses only the new part, the " + col("residual power", OURS) + " <i>V</i> = <i>u</i>·<i>P</i>.")
        self.play(FadeIn(bars[1]), run_time=0.7)
        self.caption("Late in sampling, more of it is predictable. The residual is smaller, so the grid can wait "
                     "longer.")
        self.clear_all()

    def margin(self, i):
        self.header("2 · When to grow: the test")
        n = int(MZ["n"][i])
        b = n // 2
        e, sd = MZ["est"][i, b] * LG, MZ["sd"][i, b] * LG
        pl = zoom_plot(i, xb=(16, 128))
        dots, cur, band, _ = step_state(pl, i)
        bl = pl.vline(b, WHITE, 3, dashed=False)
        self.add(pl.parts(), dots, cur, band, bl)
        self.add(tx(f"grid boundary {b}", 20).next_to(pl.p(b, pl.yr[1]), DR, buff=0.1))
        self.caption("Zoom in on the boundary. We have an estimate, with an error bar.")
        eb = VGroup(Line(pl.p(b, e - sd), pl.p(b, e + sd), color=OURS, stroke_width=4),
                    Line(pl.p(b, e - sd) + LEFT * 0.1, pl.p(b, e - sd) + RIGHT * 0.1, color=OURS, stroke_width=4),
                    Line(pl.p(b, e + sd) + LEFT * 0.1, pl.p(b, e + sd) + RIGHT * 0.1, color=OURS, stroke_width=4))
        pe = Dot(pl.p(b, e), radius=0.09, color=OURS)
        self.play(Create(eb), FadeIn(pe), run_time=0.6)
        v = e + np.log10(MZ["u"][i])
        lowb = test_value(i, b)
        self.play(pe.animate.move_to(pl.p(b, v)), run_time=0.6)
        self.caption(f"Keep only the new part (<i>u</i> = {MZ['u'][i]:.2f}).")
        self.play(pe.animate.move_to(pl.p(b, lowb)), run_time=0.8)
        tag = tx("tested value", 20, OURS).next_to(pl.p(b, lowb), LEFT, buff=0.25)
        self.add(tag)
        self.caption(f"To be safe, take the low end of the error bar ({ZM:.1f} standard errors down).")
        thr = pl.hline(THR[i])
        self.play(Create(thr), run_time=0.6)
        self.add(tx("noise level", 20, NOI).next_to(pl.p(pl.xb[0], THR[i]), UP, buff=0.08)
                 .align_to(pl.p(pl.xb[0], 0), LEFT).shift(RIGHT * 0.1))
        ok = lowb >= THR[i]
        verdict = tx("above the noise: grow" if ok else "below the noise: wait", 30, GOOD if ok else NOI) \
            .next_to(pl.ax, UP, buff=0.15).align_to(pl.ax, RIGHT)
        self.play(FadeIn(verdict, shift=UP * 0.1), run_time=0.4)
        self.caption("If even the low end is above the noise level, the boundary has emerged. The grid grows.")


def zoom_plot(i, xb=(8, 128), center=(1.2, 0.25, 0), w=8.4, steps=None):
    """spectrum axes zoomed on the boundary test: y range covers the threshold(s) and the estimates."""
    steps = [i] if steps is None else steps
    vals = [THR[j] for j in steps]
    for j in steps:
        n = int(MZ["n"][j])
        t = [b for b in range(n // 2, min(n, NB) + 1) if np.isfinite(MZ["est"][j, b])]
        vals += list((MZ["est"][j, t] + MZ["sd"][j, t]) * LG) + list((MZ["est"][j, t] - MZ["sd"][j, t]) * LG)
    lo, hi = min(vals) - 0.35, max(vals) + 0.45
    return LogPlot(center=center, w=w, yr=(lo, hi), xb=xb)


class M05_Rule(Base):
    def construct(self):
        self.header("2 · When to grow: one test per step")
        b = int(MZ["n"][I1]) // 2
        first = min(i for i in range(NSTEP) if np.isfinite(MZ["est"][i, int(MZ["n"][i]) // 2]))
        steps = list(range(max(first, I1 - 6), I1 + 1))
        pl = zoom_plot(I1, xb=(16, 128), steps=steps)
        self.add(pl.parts(), pl.vline(b, WHITE, 3, dashed=False),
                 tx(f"grid boundary {b}", 20).next_to(pl.p(b, pl.yr[1]), DR, buff=0.1))

        def state(i):
            dots, cur, band, _ = step_state(pl, i)
            lowb = test_value(i, b)
            ok = lowb >= THR[i]
            return VGroup(band, cur, pl.hline(THR[i]), Dot(pl.p(b, lowb), radius=0.09, color=OURS),
                          tx(f"step {i}:  " + ("grow" if ok else "wait"), 30, GOOD if ok else NOI)
                          .next_to(pl.ax, UP, buff=0.15).align_to(pl.ax, RIGHT))
        cur = state(steps[0])
        self.add(cur, tx("noise level (red) drops every step", 20, NOI).next_to(pl.ax, UP, buff=0.2)
                 .align_to(pl.ax, LEFT))
        self.caption("Each step: measure the low rings, estimate the boundary, test it.", hold=False)
        for i in steps[1:]:
            self.wait(0.5)
            self.play(Transform(cur, state(i)), run_time=0.7)
        self.caption(f"At step {I1}, the tested value passes the noise level. The rule fires.")
        self.caption("All inputs are already in the sampler. The cost is a few FFTs and one tiny linear solve.")


class M06_Tracking(Base):
    def construct(self):
        self.header("3 · To which size: frontier tracking")
        # pick a growth step where some candidates pass and one fails, and one where all pass
        ex = []
        for i in FIRES:
            n, nxt = int(MZ["n"][i]), int(MZ["next"][i])
            k = len([c for c in MZ["ladder"] if n < c < nxt])
            ex.append((i, n, nxt, k))
        mid = next((e for e in ex if e[3] >= 1 and e[2] < 256), ex[0])
        full = next((e for e in ex if e[2] == 256 and e[3] >= 2), None)
        self.track(*mid[:2], intro=True)
        if full is not None:
            self.clear_all()
            self.header("3 · To which size: frontier tracking")
            self.track(*full[:2], intro=False)
        self.caption("The number of stages and their sizes are no longer choices. They come out of sampling.")

        self.section("cap")
        self.clear_all()
        self.header("3 · To which size: a safety cap")
        a = tokgrid(64, 1.6, (-4.2, 0.4, 0))
        bb = tokgrid(256, 3.2, (2.6, 0.4, 0))
        x = tx("×", 80, NOI).move_to(bb)
        self.add(a, tx("64", 22).next_to(a, DOWN, buff=0.2))
        self.play(GrowArrow(arrow(a.get_right(), bb.get_left())), FadeIn(bb), run_time=0.8)
        l256 = tx("256", 22).next_to(bb, DOWN, buff=0.2)
        self.add(l256)
        self.caption("Early on, a high estimate can make every larger grid look ready.")
        self.play(FadeIn(x), run_time=0.4)
        self.caption("Without a cap, 3 of 80 test images jumped from 64 to 256 too early, and ran slower.")
        c2 = tokgrid(128, 2.2, (2.6, -0.1, 0))
        self.play(FadeOut(bb), FadeOut(x), FadeOut(l256), FadeIn(c2), run_time=0.8)
        self.add(tx("at most 128", 22, GOOD).next_to(c2, DOWN, buff=0.2).shift(DOWN * 0.45))
        self.caption("So one growth step at most doubles the grid. This fixes all 3 and changes no other schedule.")

    def track(self, i, n, intro):
        self.section(f"track_{i}")
        pl = zoom_plot(i, xb=(n // 2 * 0.8, 128), center=(1.9, 0.25, 0), w=7.4)
        dots, cur, band, _ = step_state(pl, i)
        size = 2.2
        g = tokgrid(n, size * n / 256 * 1.6, (-5.2, 0.7, 0))
        g.move_to([-5.2, 0.7, 0])
        glab = tx(f"step {i}: grid {n}", 22).next_to(g, DOWN, buff=0.2)
        self.add(pl.parts(), cur, band, pl.hline(THR[i]), g, glab,
                 pl.vline(n // 2, WHITE, 3, dashed=False),
                 tx(f"boundary {n // 2}", 18).next_to(pl.p(n // 2, pl.yr[1]), DR, buff=0.1))
        if intro:
            self.caption(f"At step {i}, the rule fired on grid {n}. A fixed ladder would now double the grid.")
            self.caption("Instead, we test the boundary of every larger grid, in steps of 8, with the same test.",
                         hold=False)
        else:
            self.caption(f"Later, at step {i} on grid {n}:", hold=False)
        chosen = 256
        for c in [int(c) for c in MZ["ladder"] if n < c <= min(2 * n, 256)]:
            bc = c // 2
            if not np.isfinite(MZ["est"][i, bc]):
                continue
            tv = test_value(i, bc)
            ok = tv >= THR[i]
            colr = GOOD if ok else NOI
            d = Dot(pl.p(bc, tv), radius=0.08, color=colr)
            tk = Line(pl.p(bc, pl.yr[0]), pl.p(bc, pl.yr[0]) + UP * 0.25, color=colr, stroke_width=4)
            self.play(FadeIn(d), Create(tk), run_time=0.35)
            self.add(tx(str(c), 16, colr).next_to(tk, UP, buff=0.05))
            if not ok:
                chosen = c
                break
        if chosen < 256:
            self.caption(f"Grid {chosen} is the first whose boundary is still noise. Its boundary sits just above the "
                         "frontier, so we grow to it.")
        else:
            self.caption("Here every candidate has emerged, so the grid goes straight to full size.")
        g2 = tokgrid(chosen, size * chosen / 256 * 1.6, (-5.2, 0.7, 0), new_from=n)
        g2.move_to(g, aligned_edge=UL)
        self.play(FadeIn(g2), FadeOut(g), FadeTransform(glab, tx(f"grid {chosen}", 22).next_to(g2, DOWN, buff=0.2)),
                  run_time=1.0)
        if intro:
            self.caption("The new frequencies (red) start as pure noise. Sampling then continues on the bigger grid.")


class M07_Loop(Base):
    def construct(self):
        self.header("Putting it together")
        P = RUN if f"p{RUN}_P_x0" in Z.files else 2580
        sp = Spectrogram(P)
        pic = arr_img(final_img(P, "fullres"), IMG_S).move_to(IMG_C)
        self.add(pic, framed(pic), tx(NAMES[P], 18, GREY_B).next_to(pic, DOWN, buff=0.15),
                 sp.img, sp.frame, sp.labels(), colorbar().move_to([5.6, 0.3, 0]))
        ghost = sp.stair(ceil_of(P, "speed3"), SPD, 3, dashed=True).set_opacity(0.5)
        self.add(ghost, tx("SPEED (fixed)", 18, SPD).set_opacity(0.7).next_to(sp.p(30, 128), DOWN, buff=0.12))
        co = ceil_of(P, "cont2x")
        k = ValueTracker(1)

        def partial():
            m = int(round(k.get_value()))
            c = co.copy()
            pts = sp.stair_pts(c)
            xs = sp.p(m - 0.5, 0.5)[0]
            keep = [q for q in pts if q[0] <= xs + 1e-6]
            v = VMobject(stroke_color=OURS, stroke_width=5)
            if len(keep) >= 2:
                v.set_points_as_corners(keep)
            return v
        st = always_redraw(partial)
        gl = always_redraw(lambda: tx(f"grid {int(2 * co[min(NSTEP - 1, int(round(k.get_value())) - 1)])}", 22, OURS)
                           .next_to(sp.p(min(NSTEP - 1, k.get_value()) - 0.5, co[min(NSTEP - 1, int(round(k.get_value())) - 1)]),
                                    UP, buff=0.1))
        self.add(st, gl)
        self.caption("Repeat at every step: estimate, test, track.", hold=False)
        self.play(k.animate.set_value(NSTEP), run_time=7.0, rate_func=linear)
        st.clear_updaters()
        gl.clear_updaters()
        self.caption(f"The staircase follows this image’s frontier: {speedup(P, 'cont2x'):.1f}× less compute than full "
                     f"size, against {speedup(P, 'speed3'):.1f}× for SPEED.")

        self.section("results")
        self.clear_all()
        self.header("Results on FLUX.2 [klein], 2048 × 2048")
        pairs = [p for p in (2580, 1026) if os.path.exists(os.path.join(DATA, f"final_speed3_{p:06d}.png"))]
        g = Group()
        for j, p in enumerate(pairs):
            for c, arm in enumerate(("speed3", "cont2x")):
                im = arr_img(final_img(p, arm), 2.3).move_to([-4.8 + 2.45 * c + 5.1 * j, 0.9, 0])
                g.add(im, framed(im))
                g.add(tx("SPEED, 3 stages" if arm == "speed3" else "Ours", 20, SPD if arm == "speed3" else OURS)
                      .next_to(im, UP, buff=0.12))
        g.move_to([0, 0.9, 0])
        self.play(FadeIn(g), run_time=0.8)
        nums = VGroup(tx("1.7× faster than a 3-stage schedule · 4.7–5.0× faster than full resolution", 26, OURS),
                      tx("HPSv3 −0.19 vs. the 3-stage schedule · ImageReward and HPSv2.1 unchanged · no training",
                         20, GREY_B)).arrange(DOWN, buff=0.18).move_to([0, -1.85, 0])
        self.add(nums)
        self.caption("Similar quality, with far less compute.")
        self.wait(1.0)
