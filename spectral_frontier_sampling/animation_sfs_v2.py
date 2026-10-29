"""Spectral Frontier Sampling (SFS): 3b1b-style teaser and method video, v2.

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

# ------------------------------------------------------------------ layout grid (all scenes)
COL_X = -5.4                  # centre of the left column (images, grids, legends)
COL_L = -6.85                 # left edge of left-column legends
PLOT_L = -2.4                 # left edge of every plot's axes
SPEC_R, LOG_R = 4.6, 5.6      # right edge: spectrograms (colour bar at CBAR_X), spectrum plots
CBAR_X = 5.75
PLOT_Y = 0.35                 # vertical centre of every plot
PLOT_H = 4.5

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


SNR_LO, SNR_HI = -2.5, 2.0     # log10 SNR colour range of frequencies that passed the delta test (others: black)


def log_snr(P, sig):
    """log10 SNR of x_t = (1 - sigma)^2 P / sigma^2 (model-free, from the finished power P), clipped."""
    with np.errstate(divide="ignore"):
        return np.clip(np.log10((1 - sig) ** 2 * np.maximum(P, 1e-30) / sig ** 2), SNR_LO, SNR_HI)


def snr_color(L, emerged=None):
    """magma from dark purple up for frequencies that passed SPEED's delta = 0.01 test; pure black for the rest."""
    c = MAGMA(0.18 + 0.82 * np.clip((L - SNR_LO) / (SNR_HI - SNR_LO), 0, 1))[..., :3]
    return c if emerged is None else np.where(emerged[..., None], c, 0.0)


def spec_rgb(p, hpx=420, cw=12):
    """[hpx, 50*cw, 3] magma image of log10 SNR of x_t per step and bin (full-res run, finished power P_final),
    rows on the compute axis (top = bin 128)."""
    Pf, sig = Z[f"p{p}_P_final"], Z[f"p{p}_sigmas"][:NSTEP]
    ys = np.linspace(YMAX, YMIN, hpx)
    b = np.clip(np.round(np.interp(ys, CY, FB)), 1, NB).astype(int)
    L = log_snr(Pf[b][None, :], sig[:, None])                            # [steps, rows]
    em = np.log10(np.maximum(Pf[b], 1e-30))[None, :] >= THR[:, None]     # the white contour's test, per cell
    return np.repeat(snr_color(L, em).transpose(1, 0, 2), cw, axis=1)


def row_bins(hpx=420):
    ys = np.linspace(YMAX, YMIN, hpx)
    return np.clip(np.round(np.interp(ys, CY, FB)), 1, NB).astype(int)


def ceil_of(p, arm):
    return np.asarray(Z[f"p{p}_{arm}"], float)


def speedup(p, arm):
    return float(Z[f"p{p}_{arm}_x"])


# ------------------------------------------------------------------ spectrogram widget
class Spectrogram:
    """Axes + magma image (x = step, y = frequency on the compute axis). Use .p(step, bin) for points."""

    def __init__(self, p, center=((PLOT_L + SPEC_R) / 2, PLOT_Y, 0), w=SPEC_R - PLOT_L, h=PLOT_H, cw=12):
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
                     tx(str(s), 18, GREY_B).next_to(q, DOWN, buff=0.18))
        for f in (32, 64, 96, 128):
            q = self.p(-0.5, f)
            labs.add(Line(q, q + LEFT * 0.08, stroke_width=2, color=GREY_B),
                     tx(str(f), 18, GREY_B).next_to(q, LEFT, buff=0.18))
        self.ticks = labs
        self.xlab = tx("denoising step", 22, GREY_B).next_to(self.ax, DOWN, buff=0.45)
        self.ylab = VGroup(tx("frequency", 20, GREY_B), tx("(height = cost of one step)", 16, DIM)) \
            .arrange(RIGHT, buff=0.15, aligned_edge=DOWN).next_to(self.ax, UP, buff=0.15).align_to(self.ax, LEFT) \
            .shift(LEFT * 0.45)
        self.frame = Rectangle(width=w, height=h, stroke_color=GREY_B, stroke_width=1.2).move_to(self.img)
        self.frame.set_z_index(3)          # draw the border over the image

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


def colorbar(h=2.4, w=0.22, delta_line=True):
    """colour bar: magma = SNR of x_t for frequencies that passed the delta test (ticks 0.01, 1, 100), and a
    black box for the rest (noise), plus the delta = 0.01 legend entry."""
    v = np.linspace(1, 0, 200)[:, None]
    a = np.repeat(MAGMA(0.18 + 0.82 * v)[..., :3], 8, axis=1)
    m = ImageMobject(rgba(a))
    m.stretch_to_fit_width(w)
    m.stretch_to_fit_height(h)
    g = Group(m, framed(m, GREY_B, 1))
    for val, lab in ((-2, "0.01"), (0, "1"), (2, "100")):
        y = m.get_bottom()[1] + h * (val - SNR_LO) / (SNR_HI - SNR_LO)
        q = np.array([m.get_right()[0], y, 0])
        g.add(Line(q, q + RIGHT * 0.08, color=GREY_B, stroke_width=2),
              tx(lab, 16, GREY_B).next_to(q, RIGHT, buff=0.18))
    blk = Square(side_length=w, fill_color=BLACK, fill_opacity=1, stroke_color=GREY_B, stroke_width=1)
    blk.next_to(m, DOWN, buff=0.12)
    g.add(blk, tx("noise", 15, GREY_B).next_to(blk, RIGHT, buff=0.12))
    g.add(tx("SNR of " + XT, 18, GREY_B).next_to(m, UP, buff=0.15))
    m_lab = tx("signal", 14, GREY_B).next_to(m, LEFT, buff=0.15).align_to(m, UP)
    g.add(m_lab)
    if delta_line:
        ln = DashedLine(ORIGIN, RIGHT * 0.45, color=WHITE, stroke_width=2.5, dash_length=0.06)
        lg = VGroup(ln, VGroup(tx("signal emerges", 15, GREY_B), tx("(<i>δ</i> = 0.01)", 15, GREY_B))
                    .arrange(DOWN, buff=0.1, aligned_edge=LEFT)).arrange(RIGHT, buff=0.12)
        g.add(lg.next_to(blk, DOWN, buff=0.25).align_to(m, LEFT).shift(LEFT * 0.3))
    return g


def frontier_steps(p):
    """per step: highest bin whose finished power passes SPEED's delta = 0.01 test, P >= P*(sigma_{k+1}) (the
    white dashed 'signal emerges' contour)."""
    lp = np.log10(Z[f"p{p}_P_final"])
    return np.array([frontier_bin(lp, THR[k]) for k in range(NSTEP)], float)


# ------------------------------------------------------------------ scene base
QA = os.environ.get("SFS_QA") == "1"     # layout check: log text overlaps, crossings and off-frame text
QA_PAD = 0.04                             # text boxes closer than this to another element count as touching


def _bbox(m, pad=0.0):
    lo, hi = m.get_corner(DL), m.get_corner(UR)
    return lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad


def _hit(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class Base(Scene):
    def setup(self):
        self.cap = None
        self.hdr = None
        self._qa_n = 0
        self._qa_seen = set()

    def play(self, *args, **kwargs):
        super().play(*args, **kwargs)
        self._qa("play")

    def wait(self, *args, **kwargs):
        super().wait(*args, **kwargs)
        self._qa("wait")

    def _qa(self, where):
        """log every text box that overlaps another text box, is crossed by a stroke, sits on an image, or leaves
        the frame. Text with a background rectangle may sit on images on purpose."""
        if not QA:
            return
        self._qa_n += 1
        fam = []
        for m in self.mobjects:
            fam += m.get_family()
        texts = [m for m in fam if isinstance(m, MarkupText) and m.width > 0.01 and m.get_fill_opacity() > 0.05]
        tset = set()
        for t in texts:
            tset.update(id(q) for q in t.get_family())
        imgs = [m for m in fam if isinstance(m, ImageMobject)]
        strokes = [m for m in fam if isinstance(m, VMobject) and id(m) not in tset and m.get_stroke_width() > 0
                   and m.get_stroke_opacity() > 0.05 and len(m.points) > 0 and not isinstance(m, MarkupText)]
        out = []
        boxes = [(t, _bbox(t)) for t in texts]
        name = lambda t: re.sub(r"<[^>]+>", "", t.original_text)[:40]  # noqa: E731
        for i, (t, bx) in enumerate(boxes):
            if bx[0] < -7.05 or bx[2] > 7.05 or bx[1] < -3.97 or bx[3] > 3.97:
                out.append(f"OFF-FRAME  '{name(t)}'")
            for u, by in boxes[i + 1:]:
                if _hit(_bbox(t, QA_PAD), by):
                    out.append(f"TEXT/TEXT  '{name(t)}'  x  '{name(u)}'")
            inner = _bbox(t, QA_PAD)
            for sm in strokes:
                pts = sm.points
                if len(pts) > 400:
                    pts = pts[:: len(pts) // 400]
                if np.any((pts[:, 0] > inner[0]) & (pts[:, 0] < inner[2]) & (pts[:, 1] > inner[1]) &
                          (pts[:, 1] < inner[3])):
                    out.append(f"STROKE     '{name(t)}'  crossed by {type(sm).__name__}")
                    break
            if not any(isinstance(q, BackgroundRectangle) for q in t.submobjects):
                for im in imgs:
                    if _hit(inner, _bbox(im)):
                        out.append(f"IMAGE      '{name(t)}'  on an image")
                        break
        # images: no partial overlaps (an overlay that sits inside another image is fine), nothing off-frame
        ib = [(im, _bbox(im)) for im in imgs]
        inside = lambda a, b: a[0] >= b[0] - 1e-3 and a[1] >= b[1] - 1e-3 and a[2] <= b[2] + 1e-3 and a[3] <= b[3] + 1e-3  # noqa: E731,E501
        for j, (im, bx) in enumerate(ib):
            for im2, by in ib[j + 1:]:
                if _hit(bx, by) and not inside(bx, by) and not inside(by, bx):
                    out.append(f"IMG/IMG    {bx[0]:.2f},{bx[1]:.2f} x {by[0]:.2f},{by[1]:.2f}")
        for m in self.mobjects:
            if m is self.cap or m.width < 1e-3:
                continue
            bx = _bbox(m)
            if bx[0] < -7.1 or bx[2] > 7.1 or bx[1] < -4.0 or bx[3] > 4.0:
                out.append(f"OFF-FRAME  {type(m).__name__} {bx[0]:.2f},{bx[1]:.2f},{bx[2]:.2f},{bx[3]:.2f}")
        new = [o for o in out if o not in self._qa_seen]
        self._qa_seen.update(new)
        if new:
            with open(os.path.join(HERE, "scratch", "qa", f"layout_{type(self).__name__}.txt"), "a") as f:
                for o in new:
                    f.write(f"[{self._qa_n:03d} {where}] {o}\n")

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


IMG_C = np.array([COL_X, 0.45, 0])
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


def xt_frames(src, p, size=384):
    d = os.path.join(CAP, f"{src}_{p:06d}")
    if not os.path.exists(os.path.join(d, "xt_49.png")):
        return None
    return [load_rgb(os.path.join(d, f"xt_{i:02d}.png"), size) for i in range(NSTEP)]


def p2d_final(p):
    f = os.path.join(CAP, f"full_{p:06d}", "steps.npz")
    if os.path.exists(f):
        z = np.load(f)
        if "P2D_final" in z.files:
            return z["P2D_final"]
    return None


def snr2d_rgb(P2D, sig, thr):
    """centered 2D log10 SNR map of x_t at noise level sig; black where the power fails the delta test thr."""
    if sig >= 1:
        return np.zeros(P2D.shape + (3,))
    return snr_color(log_snr(P2D, sig), np.log10(np.maximum(P2D, 1e-30)) >= thr)


class T01_Spectrogram(Base):
    def construct(self):
        self.header("How diffusion builds an image")
        P = 2580
        K0 = 40
        sp = Spectrogram(P)
        sig = Z[f"p{P}_sigmas"]
        frames = xt_frames("full", P) or [final_img(P, "fullres")] * NSTEP
        P2D = p2d_final(P)
        S = 2.25
        pic = arr_img(frames[0], S).move_to([COL_X, 1.45, 0])
        spec = arr_img(np.zeros((256, 256, 3)), S).move_to([COL_X, -1.15, 0])
        lab_s = tx("spectrum", 20).next_to(spec, DOWN, buff=0.1)
        lab_g = tx("spectrogram: one column per step", 22).next_to(sp.frame, UP, buff=0.55).align_to(sp.frame, RIGHT)
        rb = row_bins(sp.rgb.shape[0])[:, None, None]
        st = ValueTracker(0)         # the step shown in the image and the spectrum; columns < st are filled
        hold = ValueTracker(-1)      # a column left empty except bins <= fr (the ring demo)
        fr = ValueTracker(0)

        def cur():
            return int(min(NSTEP - 1, max(0, np.floor(st.get_value()))))

        def reveal(mob):
            n = int(np.floor(st.get_value())) + 1
            a = np.zeros_like(sp.rgb)
            a[:, :n * sp.cw] = sp.rgb[:, :n * sp.cw]
            c = int(hold.get_value())
            if c >= 0:
                sl = slice(c * sp.cw, (c + 1) * sp.cw)
                a[:, sl] = np.where(rb <= fr.get_value(), sp.rgb[:, sl], 0.0)
            mob.pixel_array = rgba(a)

        def upd_pic(m):
            m.pixel_array = rgba(frames[cur()])

        def upd_spec(m):
            if P2D is not None:
                m.pixel_array = rgba(snr2d_rgb(P2D, float(sig[cur()]), THR[cur()]))
        sp.set_rgb(np.zeros_like(sp.rgb))
        stepl = always_redraw(lambda: tx("image " + XT + f", step {cur()}", 20).next_to(pic, UP, buff=0.1))
        cursor = always_redraw(lambda: Line(sp.p(cur() + 0.5, 0.5), sp.p(cur() + 0.5, 128.5), color=WHITE,
                                            stroke_width=2))
        # the whole layout is on screen from the start, and the spectrogram fills while the sampler runs
        self.play(FadeIn(pic), FadeIn(spec), Create(framed(pic)), Create(framed(spec)), Create(sp.frame),
                  FadeIn(sp.labels()), FadeIn(lab_g), FadeIn(colorbar().move_to([CBAR_X, 0.2, 0])), run_time=0.8)
        self.add(framed(pic), framed(spec), lab_s, stepl, sp.img, cursor)
        for m, f in ((pic, upd_pic), (spec, upd_spec), (sp.img, reveal)):
            m.add_updater(f)
        self.caption("Diffusion turns noise into an image in 50 steps. Watch the image, its spectrum, and the "
                     "record on the right.", hold=False)
        self.play(st.animate.set_value(K0), run_time=6.0, rate_func=linear)

        self.section("spectrum")
        r = S / 2
        cen = spec.get_center()
        p_coarse = Arrow(cen + UP * (r - 0.3) + LEFT * 0.35, cen + UP * 0.08 + LEFT * 0.05, buff=0, color=WHITE,
                         stroke_width=2.5, max_tip_length_to_length_ratio=0.2, tip_length=0.12)
        p_fine = Arrow(cen + DOWN * (r - 0.45) + RIGHT * (r - 0.75), cen + DOWN * (r - 0.12) + RIGHT * (r - 0.12),
                       buff=0, color=WHITE, stroke_width=2.5, max_tip_length_to_length_ratio=0.35, tip_length=0.1)
        t_coarse = tx("coarse", 16).next_to(p_coarse.get_start(), LEFT, buff=0.1)
        t_fine = tx("fine", 16).next_to(p_fine.get_start(), LEFT, buff=0.1)
        for t in (t_coarse, t_fine):
            t.add_background_rectangle(color=BLACK, opacity=0.6, buff=0.03)
        self.play(GrowArrow(p_coarse), GrowArrow(p_fine), FadeIn(t_coarse), FadeIn(t_fine), run_time=0.6)
        self.caption("The spectrum shows every frequency of the image: coarse shapes in the center, fine detail at "
                     "the edge.")
        self.caption("Color is the signal-to-noise ratio (SNR). Black means the noise still hides that frequency.")
        self.caption(f"At step {K0}, the image still looks like noise. But the center of the spectrum already "
                     "shows signal.")
        self.play(*[FadeOut(m) for m in (p_coarse, p_fine, t_coarse, t_fine)], run_time=0.4)

        self.section("one_column")
        hold.set_value(K0)
        fr.set_value(0)
        ring = always_redraw(lambda: Circle(radius=max(fr.get_value(), 1) / 128 * S / 2, color=WHITE,
                                            stroke_width=2.5).move_to(spec))
        ya = (sp.p(0, 64)[1] + sp.p(0, 96)[1]) / 2           # between two tick labels
        link = VGroup(Arrow([spec.get_right()[0] + 0.1, ya, 0], [sp.ax.get_left()[0] - 0.08, ya, 0], buff=0,
                            color=WHITE, stroke_width=3, max_tip_length_to_length_ratio=0.12, tip_length=0.15))
        link.add(tx("ring → cell", 16).next_to(link[0], UP, buff=0.08))
        cell = always_redraw(lambda: Square(side_length=0.14, color=WHITE, stroke_width=2.5)
                             .move_to(sp.p(K0, max(fr.get_value(), 1))))
        self.add(ring, cell)
        self.play(GrowArrow(link[0]), FadeIn(link[1]), run_time=0.5)
        self.caption(f"How is the column for step {K0} made? Average the spectrum over one ring: that gives one "
                     "cell.", hold=False)
        self.play(fr.animate.set_value(30), run_time=1.6, rate_func=smooth)
        self.wait(1.2)
        self.caption("Grow the ring, and the cells fill the whole column.", hold=False)
        self.play(fr.animate.set_value(NB), run_time=3.0, rate_func=linear)
        self.play(FadeOut(link), run_time=0.3)
        self.remove(ring, cell)
        hold.set_value(-1)

        self.section("rest")
        self.caption("Coarse shapes rise above the noise first. Fine detail rises only near the end.", hold=False)
        self.play(st.animate.set_value(NSTEP - 1), run_time=3.5, rate_func=linear)
        for m in (sp.img, pic, spec, stepl):
            m.clear_updaters()
        self.remove(cursor)
        cont = sp.stair(frontier_steps(P), WHITE, 2.5, dashed=True)
        self.play(Create(cont), run_time=1.5)
        self.caption("The dashed line is the edge where frequencies pass the tolerance <i>δ</i> = 0.01. Below it: "
                     "signal. Above it: noise.")
        self.caption("All the black area is noise. A full-resolution model pays for it anyway.", wait=0.5)


def side_legend(rows, top=-1.6, left=COL_L):
    """rows of (line mobject, text, colour) stacked under the thumbnail in the left column."""
    g = VGroup(*[VGroup(ln, tx(t, 18, c)).arrange(RIGHT, buff=0.15) for ln, t, c in rows]) \
        .arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    return g.move_to([left, top, 0], aligned_edge=UL)


def legline(color, sw=4, opacity=1.0):
    return Line(ORIGIN, RIGHT * 0.45, stroke_color=color, stroke_width=sw, stroke_opacity=opacity)


def teaser_base(scene, P=2580):
    sp = Spectrogram(P)
    pic = arr_img(final_img(P, "fullres"), IMG_S).move_to(IMG_C)
    sp.cont = sp.stair(frontier_steps(P), WHITE, 2.5, dashed=True)
    scene.add(pic, framed(pic), sp.img, sp.frame, sp.labels(), colorbar().move_to([CBAR_X, 0.2, 0]), sp.cont)
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
        self.caption("A full-resolution model pays for the whole box at every step.")
        self.play(FadeOut(full), run_time=0.5)

        head = tx("compute vs. full resolution:", 16, GREY_B).move_to([COL_L, -1.55, 0], aligned_edge=UL)
        self.add(head)

        def leg(i, name, x, color, dashed):
            row = VGroup(legline(color), tx(name, 18, color), tx(f"{x:.1f}× less", 18, color)).arrange(RIGHT, buff=0.15)
            return row.next_to(head, DOWN, buff=0.2 + 0.4 * i).align_to(head, LEFT)

        self.section("speed2")
        c2, c3, co = ceil_of(P, "speed2"), ceil_of(P, "speed3"), ceil_of(P, "cont2x")
        s2 = sp.stair(c2, SPD, 3.5)
        self.play(Create(s2), run_time=1.2)
        l2 = leg(0, "SPEED, 2 stages", speedup(P, "speed2"), SPD, False)
        self.add(l2)
        self.caption("SPEED runs early steps on a small grid, then grows it once.")
        w2 = sp.hatch(c2, co)
        self.play(FadeIn(w2), run_time=0.8)
        self.caption(col("Red", NOI) + " marks compute spent on frequencies that are still dark: the noise still hides "
                     "them.")

        self.section("speed3")
        s3 = sp.stair(c3, SPD, 3.5)
        w3 = sp.hatch(c3, co)
        l3 = leg(1, "SPEED, 3 stages", speedup(P, "speed3"), SPD, False)
        self.play(Transform(s2, s3), FadeOut(w2), FadeIn(w3), l2.animate.set_opacity(0.45), run_time=1.4)
        self.add(l3)
        self.caption("More stages waste less. But each stage is one more hand-picked choice.")
        self.caption("Each step up doubles the grid. It adds many new frequencies at once, and most are still noise.")

        self.section("ours")
        so = sp.stair(co, OURS, 5)
        lo_ = leg(2, "Ours", speedup(P, "cont2x"), OURS, False)
        self.play(Create(so), run_time=2.0)
        self.add(lo_)
        self.caption("Our grid grows with the " + col("spectral frontier", OURS)
                     + ": the highest frequency that has signal.")
        self.play(FadeOut(w3), l3.animate.set_opacity(0.45), run_time=1.4)
        self.caption(f"The red waste is gone. This image needs {speedup(P, 'cont2x'):.1f}× less compute than full resolution.")
        self.wait(0.5)


class T03_PerSample(Base):
    def construct(self):
        self.header("One schedule per image")
        order = [p for p in (2580, 2348, 1026) if p in PROMPTS]
        P = order[0]
        sp, pic = teaser_base(self, P)
        name = tx(NAMES[P], 18, GREY_B).next_to(pic, DOWN, buff=0.15)
        self.add(name)
        ghost = sp.stair(ceil_of(P, "speed3"), SPD, 3).set_stroke(opacity=0.5)
        gl = side_legend([(legline(SPD, 3, 0.5), "SPEED (fixed)", SPD),
                          (legline(OURS, 5), "ours (this image)", OURS)])
        so = sp.stair(ceil_of(P, "cont2x"), OURS, 5)
        self.add(ghost, gl, so)

        def kmark(p):
            K0 = int(Z[f"p{p}_cont2x_K"][0])
            ln = DashedLine(sp.p(K0 - 0.5, 0.5), sp.p(K0 - 0.5, 128.5), color=OURS, stroke_width=2, dash_length=0.08)
            lb = tx(f"grows at step {K0}", 20, OURS).next_to(sp.p(K0 - 0.5, 128.5), UP, buff=0.55)
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
            new_cont = sp.stair(frontier_steps(p), WHITE, 2.5, dashed=True)
            new_km = kmark(p)
            new_name = tx(NAMES[p], 18, GREY_B).next_to(pic, DOWN, buff=0.15)
            self.play(al.animate.set_value(1), Transform(so, new_so), Transform(km, new_km), Transform(sp.cont, new_cont),
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

    def __init__(self, center=((PLOT_L + LOG_R) / 2, PLOT_Y, 0), w=LOG_R - PLOT_L, h=PLOT_H, yr=(-1.3, 2.5),
                 xb=(2, 128)):
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
                     tx(str(b), 18, GREY_B).next_to(q, DOWN, buff=0.18))
        for v in range(int(np.ceil(yr[0])), int(np.floor(yr[1])) + 1):
            q = self.p(xb[0], v)
            labs.add(Line(q, q + LEFT * 0.08, stroke_width=2, color=GREY_B),
                     tx(e10(v), 18, GREY_B).next_to(q, LEFT, buff=0.18))
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
        pic = arr_img(final_img(RUN, "fullres"), 2.6).move_to([COL_X, 0.7, 0])
        plab = tx("finished image", 18, GREY_B).next_to(pic, DOWN, buff=0.15)
        self.add(pic, framed(pic), plab)
        self.play(Create(pl.ax), run_time=0.6)
        self.add(pl.ticks, pl.xlab, pl.ylab)
        sc = pl.curve(BINS, PFIN[2:], SIG, 4)
        self.play(Create(sc), run_time=1.5)
        self.add(VGroup(legline(SIG, 4), tx("power of the finished image", 20, SIG)).arrange(RIGHT, buff=0.15)
                 .next_to(pl.ax, UP, buff=0.15).align_to(pl.ax, LEFT))
        self.caption("Power falls with frequency. Coarse shapes are strong. Fine detail is weak.")

        self.section("noise")
        steps = np.arange(NSTEP)
        k = ValueTracker(8)
        thr = lambda: float(np.interp(k.get_value(), steps, THR))  # noqa: E731
        line = always_redraw(lambda: pl.hline(thr()))
        lab = always_redraw(lambda: tx("noise level", 20, NOI).next_to(pl.p(2, thr()), UP, buff=0.1)
                            .align_to(pl.p(2, 0), LEFT).shift(RIGHT * 0.1))
        stepl = always_redraw(lambda: tx(f"step {int(round(k.get_value()))} of 50", 22)
                              .next_to(pl.ax, UP, buff=0.1).align_to(pl.ax, RIGHT))
        frames = xt_frames("full", RUN, size=256)
        # two views on the left: the noisy state x_t, and the part of the image that has emerged (the finished
        # image keeping only the frequencies above the noise line)
        fin = final_img(RUN, "fullres")
        Fz = np.fft.fft2(fin, axes=(0, 1))
        ff = np.fft.fftfreq(fin.shape[0]) * fin.shape[0]
        rr = np.sqrt(ff[:, None] ** 2 + ff[None] ** 2)
        cache = {}

        def emerged(fb):
            if fb not in cache:
                cache[fb] = np.clip(np.real(np.fft.ifft2(Fz * (rr <= fb)[..., None], axes=(0, 1))), 0, 1)
            return cache[fb]
        small = 2.0
        top = np.array([COL_X, 1.55, 0])
        bot = np.array([COL_X, -1.15, 0])
        pic_n = arr_img(frames[8] if frames is not None else fin, small).move_to(top)
        pic_e = arr_img(emerged(frontier_bin(PFIN, THR[8])), small).move_to(bot)
        if frames is not None:
            pic_n.add_updater(lambda m: setattr(m, "pixel_array", rgba(frames[int(round(k.get_value()))])))
        pic_e.add_updater(lambda m: setattr(m, "pixel_array", rgba(emerged(frontier_bin(PFIN, thr())))))
        ln_n = always_redraw(lambda: tx("noisy image " + XT + f", step {int(round(k.get_value()))}", 17, GREY_B)
                             .next_to(pic_n, UP, buff=0.1))
        ln_e = tx("what has emerged so far", 17, OURS).next_to(pic_e, DOWN, buff=0.1)
        self.play(FadeOut(pic), FadeOut(plab), *[FadeOut(m) for m in self.mobjects
                                                  if isinstance(m, SurroundingRectangle)], run_time=0.4)
        self.add(pic_n, framed(pic_n), pic_e, framed(pic_e, OURS), ln_n, ln_e)
        self.play(Create(line), run_time=0.6)
        self.add(lab, stepl)
        self.caption("During sampling, the image is noisy. Noise hides every frequency whose power is below this "
                     "line.")
        area = always_redraw(lambda: pl.signal_area(PFIN, thr()))
        fdot = always_redraw(lambda: Dot(pl.p(frontier_bin(PFIN, thr()), thr()), radius=0.08, color=OURS))
        fl = always_redraw(lambda: pl.vline(frontier_bin(PFIN, thr()), OURS, 2))
        flab = always_redraw(lambda: tx("frontier", 22, OURS).next_to(pl.p(frontier_bin(PFIN, thr()), thr()),
                                                                      UR, buff=0.12))
        self.add(area, fl, fdot, flab)
        self.caption("Frequencies above the line have emerged from the noise. Bottom left: the image with only "
                     "those frequencies.")
        self.caption("The highest one is the " + col("spectral frontier", OURS) + ".")
        self.section("sweep")
        self.caption("As sampling runs, the noise shrinks. The line drops, and the frontier moves right.", hold=False)
        kend = next(j for j in range(NSTEP) if frontier_bin(PFIN, THR[j]) >= 96)
        self.play(k.animate.set_value(kend), run_time=6.0, rate_func=linear)
        self.caption("A frequency counts as signal when its signal-to-noise ratio, (1 − <i>t</i>)²<i>P</i> / <i>t</i>², "
                     "passes a small tolerance <i>δ</i>.")
        for m in (line, lab, stepl, area, fdot, fl, flab, pic_n, pic_e, ln_n):
            m.clear_updaters()

        self.section("spectrogram")
        self.clear_all()
        if f"p{RUN}_P_x0" in Z.files:
            sp = Spectrogram(RUN)
            self.add(sp.img, sp.frame, sp.labels(), colorbar(delta_line=False).move_to([CBAR_X, 0.2, 0]))
            self.add(tx("spectrogram (SNR of " + XT + ")", 22).next_to(sp.frame, UP, buff=0.55).align_to(sp.frame, RIGHT))
            fr = np.array([frontier_bin(PFIN, THR[i]) for i in range(NSTEP)], float)
            self.play(Create(sp.stair(fr, OURS, 3)), run_time=2.5)
            self.caption("Each step has one frontier. Over all steps, it traces the edge where the SNR passes "
                         "<i>δ</i>.")
            self.caption("A grid that follows this edge pays for signal, and only for signal.")


class M02_Grid(Base):
    def construct(self):
        self.header("1 · The spectral frontier")
        pl = LogPlot()
        g = tokgrid(64, 2.4, (COL_X, 0.7, 0))
        self.add(pl.parts(), pl.curve(BINS, PFIN[2:], SIG, 4))
        self.play(LaggedStart(*[FadeIn(q) for q in g], lag_ratio=0.01), run_time=0.8)
        glab = tx("grid side <i>n</i> = 64", 22).next_to(g, DOWN, buff=0.2)
        self.add(glab)
        self.caption("Sampling on a small grid is cheaper. But a grid of side <i>n</i> holds only frequencies up to "
                     "<i>n</i>/2.")
        bl = pl.vline(32, WHITE, 3, dashed=False)
        grey = Rectangle(width=pl.p(128, 0)[0] - pl.p(32, 0)[0], height=pl.ax.height, stroke_width=0,
                         fill_color=GREY_D, fill_opacity=0.45).move_to(pl.ax).align_to(pl.p(32, 0), LEFT)
        blab = tx("grid boundary <i>n</i>/2 = 32", 20).next_to(pl.p(32, pl.yr[1]), DL, buff=0.1)
        self.play(Create(bl), FadeIn(grey), run_time=0.8)
        notg = tx("not on the grid", 20, GREY_B).move_to(grey).shift(UP * 1.6)
        self.add(blab, notg)

        def scene_at(k, fill_lo, fill_hi, color, text, text_at):
            thr = THR[k]
            fb = frontier_bin(PFIN, thr)
            ln = pl.hline(thr)
            nl = tx(f"noise level at step {k}", 18, NOI).next_to(pl.p(2, thr), DOWN, buff=0.08) \
                .align_to(pl.p(2, 0), LEFT).shift(RIGHT * 0.1)
            fl = pl.vline(fb, OURS, 2)
            fd = Dot(pl.p(fb, thr), radius=0.08, color=OURS)
            ft = tx("frontier", 20, OURS).next_to(pl.p(fb, pl.yr[1]), DR, buff=0.1)
            lo, hi = fill_lo(fb), fill_hi(fb)
            if color == SIG:
                b = np.arange(lo, hi + 1)
                area = pl.band(b, np.full(len(b), thr), PFIN[b], SIG, 0.7)
            else:
                area = Rectangle(width=pl.p(hi, 0)[0] - pl.p(lo, 0)[0], height=pl.ax.height, stroke_width=0,
                                 fill_color=NOI, fill_opacity=0.35).move_to(pl.ax).align_to(pl.p(lo, 0), LEFT)
            if color == SIG:
                lab = tx(text, 20, color).move_to(text_at(pl, fb, thr))
                ar = Arrow(lab.get_right() + RIGHT * 0.12, area.get_center(), buff=0.05, color=color, stroke_width=3,
                           max_tip_length_to_length_ratio=0.15)
            else:
                lab = VGroup(*[tx(t, 20, color) for t in text.split(", ")]).arrange(DOWN, buff=0.08)
                lab.next_to(pl.p(128, pl.yr[1] - 0.9), LEFT, buff=0.2)
                ar = VMobject()
            return VGroup(ln, nl, fl, fd, ft), VGroup(area, lab, ar)

        self.section("too_small")
        k1 = next(k for k in range(NSTEP) if frontier_bin(PFIN, THR[k]) >= 80)
        base1, show1 = scene_at(k1, lambda fb: 32, lambda fb: fb, SIG, "real signal the grid drops",
                                lambda pl, fb, thr: pl.p(7, thr - 0.5))
        self.play(FadeIn(base1), FadeOut(notg), run_time=0.8)
        self.caption(f"Later, at step {k1}, the noise level is lower. The frontier has moved past the boundary.")
        self.play(FadeIn(show1), run_time=0.6)
        self.caption("Between the boundary and the frontier, the grid drops real signal. It grows too late.")
        self.play(FadeOut(base1), FadeOut(show1), run_time=0.5)

        self.section("too_big")
        k2 = next(k for k in range(NSTEP) if frontier_bin(PFIN, THR[k]) >= 40)
        g2 = tokgrid(256, 2.4, (COL_X, 0.7, 0))
        bl2 = pl.vline(128, WHITE, 3, dashed=False)
        self.play(FadeOut(grey), FadeOut(blab), Transform(g, g2), Transform(bl, bl2),
                  FadeTransform(glab, tx("grid side <i>n</i> = 256", 22).next_to(g, DOWN, buff=0.2)), run_time=1.0)
        self.caption("Now take the full grid, 256. It holds every frequency.")
        base2, show2 = scene_at(k2, lambda fb: fb, lambda fb: 128, NOI, "paid for, but only noise",
                                lambda pl, fb, thr: pl.p(72, pl.yr[1] - 0.6))
        self.play(FadeIn(base2), run_time=0.8)
        self.caption(f"At step {k2}, the frontier is only at {frontier_bin(PFIN, THR[k2])}.")
        self.play(FadeIn(show2), run_time=0.6)
        self.caption("Above the frontier, the grid pays for frequencies that are still noise. It grows too early.")
        self.section("two_questions")
        self.clear_all()
        q = VGroup(tx("So at every step, our sampler answers two questions:", 30),
                   tx("1.  When to grow the grid?", 36, OURS), tx("2.  To which size?", 36, OURS)) \
            .arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to([0, 0.3, 0])
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in q], lag_ratio=0.4), run_time=1.5)
        self.wait(2.5)


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
        self.caption("Each image has its own spectrum (its power at each frequency). SPEED uses one average spectrum for all.")
        bl = pl.vline(b0, WHITE, 3, dashed=False)
        self.play(Create(bl), run_time=0.5)
        self.add(tx("boundary of grid 64", 20).next_to(pl.p(b0, pl.yr[1]), DOWN, buff=0.1).shift(LEFT * 1.0))

        self.section("crossings")
        k = ValueTracker(8)
        line = always_redraw(lambda: pl.hline(float(np.interp(k.get_value(), np.arange(NSTEP), THR))))
        stepl = always_redraw(lambda: tx(f"step {int(round(k.get_value()))}", 22)
                              .next_to(pl.ax, UP, buff=0.1).align_to(pl.ax, RIGHT))
        self.add(line, stepl)
        self.caption("A grid of side 64 can grow when the noise level falls below the power at its boundary.", hold=False)
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
        pl = LogPlot()
        xt = arr_img(cap_frame("xt", i), 1.9).move_to([COL_X, 1.45, 0])
        x0 = arr_img(cap_frame("x0", i), 1.9).move_to([COL_X, -1.15, 0])
        lt = tx("noisy state " + XT + f", grid {n}", 18, GREY_B).next_to(xt, UP, buff=0.1)
        l0 = tx("clean guess " + X0H, 18, GREY_B).next_to(x0, UP, buff=0.1)
        self.play(FadeIn(xt), Create(framed(xt)), run_time=0.5)
        self.add(lt, framed(xt))
        self.caption(f"Step {i}. The sampler is on a {n} × {n} grid.")
        self.section("x0hat")
        a = arrow(xt.get_bottom(), l0.get_top(), buff=0.08)
        self.play(GrowArrow(a), FadeIn(x0), run_time=0.8)
        self.add(framed(x0), l0)
        self.caption("The model predicts a velocity <i>v</i> (the direction from noise to image). One subtraction gives a clean "
                     "guess: "
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
        self.caption("Low frequencies settle first. We measure their power on rings (all frequencies at one radius), with error "
                     "bars.")
        if len(bad):
            self.play(FadeOut(bad), run_time=0.6)
            self.caption("We drop the rings that are still too noisy.")

        self.section("vae")
        mp = MZ["mapped"][i] * LG
        self.caption("These values are on the small grid. The small grid is not a shrunk copy of the large one.")
        self.caption("The autoencoder (the image-to-latent network) changes the power when the image size changes.")
        saved = [m for m in self.mobjects if m is not self.hdr and m is not self.cap]
        self.play(*[FadeOut(m) for m in saved], run_time=0.6)
        self.vae_scatter(i)
        self.play(*[FadeIn(m) for m in saved], run_time=0.6)
        moved = VGroup(*[Dot(pl.p(w, mp[w]), radius=0.055, color=SIG) for w in obs if ok[w]])
        self.caption("We do the same for every ring, with its own line.", hold=False)
        self.play(Transform(good, moved), run_time=1.5)
        self.add(tx("now on the full-resolution scale", 18, SIG).next_to(moved, UP, buff=0.25))
        self.wait(1.0)

        self.section("correlate")
        mean = pl.curve(BINS, MLOG[2:], WHITE, 2, dashed=True)
        self.play(Create(mean), run_time=0.8)
        def legrow(m, txt, color):
            return VGroup(m, tx(txt, 18, color)).arrange(RIGHT, buff=0.15)
        leg = VGroup(legrow(DashedLine(ORIGIN, RIGHT * 0.5, color=WHITE, stroke_width=2, dash_length=0.07),
                            "average image", WHITE)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        leg.next_to(pl.p(14, pl.yr[1]), DR, buff=0.15)
        self.add(leg)
        devs = VGroup(*[Line(pl.p(w, MLOG[w]), pl.p(w, mp[w]), color=OURS, stroke_width=3) for w in obs if ok[w]])
        self.play(Create(devs), run_time=0.8)
        self.caption("Each ring sits above or below the average image. Here, the image is "
                     + ("above" if np.nanmean([mp[w] - MLOG[w] for w in obs if ok[w]]) > 0 else "below") + " it.")
        self.caption("Nearby frequencies tend to move together, so the low rings predict higher ones.")
        dots, cur, band, _ = step_state(pl, i)
        self.play(FadeIn(band), Create(cur), run_time=1.5)
        r2 = legrow(VGroup(Rectangle(width=0.5, height=0.18, stroke_width=0, fill_color=OURS, fill_opacity=0.3),
                           Line(LEFT * 0.25, RIGHT * 0.25, color=OURS, stroke_width=3)), "estimate, with error bar",
                    OURS)
        self.add(r2.next_to(leg, DOWN, buff=0.12, aligned_edge=LEFT))
        truth = pl.curve(BINS, PFIN[2:], SIG, 2, dashed=True, opacity=0.6)
        self.play(Create(truth), run_time=1.0)
        r3 = legrow(DashedLine(ORIGIN, RIGHT * 0.5, color=SIG, stroke_width=2, dash_length=0.07),
                    "true power (known only after sampling)", SIG)
        self.add(r3.next_to(r2, DOWN, buff=0.12, aligned_edge=LEFT))
        self.caption("The estimate lands near the true power. It needs no extra model runs and no training.")

        self.section("residual")
        self.clear_all(keep=[])
        self.residual()
        self.section("margin")
        self.margin(i)

    def vae_scatter(self, i):
        V = np.load(os.path.join(DATA, "vae_scatter.npz"))
        ring = int(V["ring"])
        xc = float(np.log10(MZ["C"][i, ring]))
        line = lambda x: float(V["m"] + (x - V["m"] - V["off"]) / V["lam"])  # noqa: E731
        xs = np.r_[V["coarse"], xc]
        ys = np.r_[V["full"], line(xc)]
        x0, x1 = xs.min() - 0.15, xs.max() + 0.15
        y0, y1 = ys.min() - 0.15, ys.max() + 0.15
        ax = Axes(x_range=[0, x1 - x0, 1], y_range=[0, y1 - y0, 1], x_length=6.4, y_length=4.4, tips=False,
                  axis_config={"stroke_width": 2, "color": GREY_B, "include_ticks": False}).move_to([(PLOT_L + LOG_R) / 2 + 0.4, PLOT_Y, 0])
        q = lambda x, y: ax.c2p(x - x0, y - y0)  # noqa: E731
        ticks = VGroup()
        for v in np.arange(np.ceil(x0 * 2) / 2, x1, 0.5):
            pt = q(v, y0)
            ticks.add(Line(pt, pt + DOWN * 0.08, color=GREY_B, stroke_width=2),
                      tx(f"{10 ** v:.2g}", 16, GREY_B).next_to(pt, DOWN, buff=0.18))
        for v in np.arange(np.ceil(y0 * 2) / 2, y1, 0.5):
            pt = q(x0, v)
            ticks.add(Line(pt, pt + LEFT * 0.08, color=GREY_B, stroke_width=2),
                      tx(f"{10 ** v:.2g}", 16, GREY_B).next_to(pt, LEFT, buff=0.18))
        xl = tx(f"power on the 64 grid (ring {ring})", 20, GREY_B).next_to(ax, DOWN, buff=0.5)
        yl = tx("power at full resolution", 20, GREY_B).rotate(PI / 2).next_to(ax, LEFT, buff=0.75)
        self.play(Create(ax), FadeIn(ticks), FadeIn(xl), FadeIn(yl), run_time=0.6)
        dots = VGroup(*[Dot(q(a, b), radius=0.028, color=GREY_B, fill_opacity=0.55)
                        for a, b in zip(V["coarse"], V["full"])])
        self.caption("Before sampling, we encoded 600 images at both sizes, once. Each dot is one image, at ring "
                     f"{ring}.", hold=False)
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.002), run_time=1.5)
        self.wait(1.5)
        inv = lambda y: float(V["m"] + V["off"] + V["lam"] * (y - V["m"]))  # noqa: E731
        xa, xb_ = max(x0, inv(y0)), min(x1, inv(y1))
        fit = Line(q(xa, line(xa)), q(xb_, line(xb_)), color=OURS, stroke_width=4)
        self.play(Create(fit), run_time=0.8)
        self.caption("A straight line links the two sizes. We measure one line per ring and per grid size.")
        v1 = DashedLine(q(xc, y0), q(xc, line(xc)), color=SIG, stroke_width=3, dash_length=0.08)
        h1 = DashedLine(q(xc, line(xc)), q(x0, line(xc)), color=SIG, stroke_width=3, dash_length=0.08)
        d0 = Dot(q(xc, y0), radius=0.08, color=SIG)
        self.play(FadeIn(d0), run_time=0.3)
        self.add(tx(f"today, step {i}", 18, SIG).next_to(d0, UR, buff=0.1).shift(RIGHT * 0.1))
        self.play(Create(v1), run_time=0.6)
        self.play(Create(h1), run_time=0.6)
        d1 = Dot(q(x0, line(xc)), radius=0.08, color=SIG)
        self.add(d1)
        self.caption("Today’s value on the small grid reads off the line as a full-resolution value.")
        self.play(*[FadeOut(m) for m in (ax, ticks, xl, yl, dots, fit, v1, h1, d0, d1)],
                  *[FadeOut(m) for m in self.mobjects if isinstance(m, MarkupText) and m is not self.hdr
                    and m is not self.cap], run_time=0.6)

    def residual(self):
        self.header("2 · When to grow: only the new part counts")
        # ---- cartoon: edge vs random texture
        rng = np.random.default_rng(4)
        K = np.arange(1, 17)
        hE = 1.0 / K
        hT = np.r_[rng.uniform(0.35, 0.8, 8), rng.uniform(0.05, 0.8, 8)]
        xx = np.linspace(0, 1, 300)
        sigE = np.tanh((xx - 0.5) / 0.02)
        tex = np.convolve(rng.standard_normal(330), np.ones(12) / 12, mode="same")[15:315]
        sigT = tex / np.abs(tex).max()
        panels = Group()
        stems_hi = []
        for j, (name, sg, h, pat) in enumerate((("an edge", sigE, hE, hE),
                                                ("random texture", sigT, hT, np.full(16, hT[:8].mean())))):
            cx = -3.4 + 6.8 * j
            sa = Axes(x_range=[0, 1, 1], y_range=[-1.2, 1.2, 1], x_length=4.6, y_length=1.1, tips=False,
                      axis_config={"stroke_width": 0}).move_to([cx, 1.75, 0])
            sl = VMobject(stroke_color=WHITE, stroke_width=3).set_points_as_corners(
                [sa.c2p(a, b) for a, b in zip(xx, sg)])
            fa = Axes(x_range=[0, 17, 1], y_range=[0, 1.1, 1], x_length=4.6, y_length=2.1, tips=False,
                      axis_config={"stroke_width": 2, "color": GREY_B, "include_ticks": False}).move_to([cx, -0.75, 0])
            lo = VGroup(*[Line(fa.c2p(k, 0), fa.c2p(k, hk), color=WHITE, stroke_width=5) for k, hk in zip(K[:8], h[:8])])
            ok = j == 0
            hi = VGroup(*[Line(fa.c2p(k, 0), fa.c2p(k, hk), color=GOOD if ok else NOI, stroke_width=5)
                          for k, hk in zip(K[8:], h[8:])])
            cut = DashedLine(fa.c2p(8.5, 0), fa.c2p(8.5, 1.1), color=GREY_B, stroke_width=2, dash_length=0.06)
            pc = VMobject(stroke_color=OURS, stroke_width=2.5).set_points_as_corners(
                [fa.c2p(k, v) for k, v in zip(K, pat)])
            pc = DashedVMobject(pc, num_dashes=30)
            g = Group(tx(name, 24).next_to(sa, UP, buff=0.15), sl, fa, lo, cut,
                      tx("grid limit", 16, GREY_B).next_to(fa.c2p(8.5, 1.1), UP, buff=0.05),
                      tx("frequency", 16, GREY_B).next_to(fa, DOWN, buff=0.1))
            panels.add(g)
            stems_hi.append((pc, hi, tx("follow the pattern: predictable" if ok else "no pattern: new", 18,
                                        GOOD if ok else NOI).next_to(fa, DOWN, buff=0.45)))
        self.add(tx("cartoon", 16, DIM).to_corner(UR, buff=0.4).shift(DOWN * 0.35))
        self.play(FadeIn(panels), run_time=0.8)
        self.caption("When the grid grows, the new frequencies start as noise. The model must make their signal "
                     "from scratch.")
        self.caption("But part of that signal follows from the frequencies the small grid already has.")
        self.caption("Take two cartoon signals, cut off at the same grid limit.")
        for j, (pc, hi, lab) in enumerate(stems_hi):
            self.play(Create(pc), run_time=0.8)
            self.play(LaggedStart(*[Create(m) for m in hi], lag_ratio=0.15), run_time=1.0)
            self.add(lab)
            if j == 0:
                self.caption("An edge spans many frequencies. Above the limit, its next frequencies follow the "
                             "pattern below.")
            else:
                self.caption("Random texture has no pattern. Its next frequencies are new.")
        self.clear_all()

        # ---- real band images
        self.header("2 · When to grow: only the new part counts")
        R = json.load(open(os.path.join(DATA, "resid.json")))
        heads = VGroup(*[tx(t, 20, c) for t, c in (("true detail", WHITE), ("in the clean guess", SIG),
                                                    ("still new", OURS))])
        xs = (-3.7, -0.55, 2.6)
        for h, x in zip(heads, xs):
            h.move_to([x, 2.75, 0])
        for h in heads:
            h.set_y(2.75)
        self.add(heads, tx("=", 40).move_to([(xs[0] + xs[1]) / 2, 1.3, 0]),
                 tx("+", 40).move_to([(xs[1] + xs[2]) / 2, 1.3, 0]))
        self.add(tx("street, detail in the octave below the grid boundary (pixel view)", 16, DIM)
                 .to_corner(UR, buff=0.4).shift(DOWN * 0.35))
        for r, (key, rec) in enumerate(sorted(R.items(), key=lambda kv: int(kv[0]))):
            st = int(key)
            y = 1.3 - 2.55 * r
            row = Group()
            for name, x in zip(("true", "guess", "resid"), xs):
                im = arr_img(load_rgb(os.path.join(DATA, f"resid_{st:02d}_{name}.png")), 2.4).move_to([x, y, 0])
                row.add(im, framed(im))
            if r == 1:
                self.add(tx("=", 40).move_to([(xs[0] + xs[1]) / 2, y, 0]), tx("+", 40).move_to([(xs[1] + xs[2]) / 2, y, 0]))
            lab = VGroup(tx(f"step {st}", 22), tx(f"grid {rec['n']}", 18, GREY_B)).arrange(DOWN, buff=0.08) \
                .move_to([-5.9, y, 0])
            pct = tx(f"{100 * rec['new_fraction_pixels']:.0f}% new", 26, OURS).move_to([4.9, y, 0])
            self.play(FadeIn(row), FadeIn(lab), run_time=0.7)
            self.add(pct)
            if r == 0:
                self.caption("Now real detail of the street, from the octave just below the grid boundary.")
                self.caption(f"At step {st}, the clean guess has almost none of it. "
                             f"{100 * rec['new_fraction_pixels']:.0f}% is still new.")
            else:
                self.caption(f"At step {st}, the guess already holds the balloon stripes. Only "
                             f"{100 * rec['new_fraction_pixels']:.0f}% is new.")
        self.caption("The rule measures this new fraction, <i>u</i>, just below the grid boundary.")
        self.caption("Only the new part counts: the " + col("residual power", OURS) + " <i>V</i> = <i>u</i>·<i>P</i>.")
        self.caption("Late in sampling, less is new. So the residual is smaller, and the grid can wait longer.")
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
        self.caption("We zoom in on the boundary. There, we have an estimate with an error bar.")
        eb = VGroup(Line(pl.p(b, e - sd), pl.p(b, e + sd), color=OURS, stroke_width=4),
                    Line(pl.p(b, e - sd) + LEFT * 0.1, pl.p(b, e - sd) + RIGHT * 0.1, color=OURS, stroke_width=4),
                    Line(pl.p(b, e + sd) + LEFT * 0.1, pl.p(b, e + sd) + RIGHT * 0.1, color=OURS, stroke_width=4))
        pe = Dot(pl.p(b, e), radius=0.09, color=OURS)
        self.play(Create(eb), FadeIn(pe), run_time=0.6)
        v = e + np.log10(MZ["u"][i])
        lowb = test_value(i, b)
        self.play(pe.animate.move_to(pl.p(b, v)), run_time=0.6)
        self.caption(f"We keep only the new part (<i>u</i> = {MZ['u'][i]:.2f}).")
        self.play(pe.animate.move_to(pl.p(b, lowb)), run_time=0.8)
        tag = tx("tested value", 20, OURS).next_to(pl.p(b, lowb), UL if lowb >= THR[i] else DL, buff=0.15)
        self.add(tag)
        self.caption(f"To be safe, we take the low end of the error bar ({ZM:.1f} standard errors down).")
        thr = pl.hline(THR[i])
        self.play(Create(thr), run_time=0.6)
        self.add(tx("noise level", 20, NOI).next_to(pl.p(pl.xb[1], THR[i]), UP, buff=0.08)
                 .align_to(pl.p(pl.xb[1], 0), RIGHT))
        ok = lowb >= THR[i]
        verdict = tx("above the noise: grow" if ok else "below the noise: wait", 30, GOOD if ok else NOI) \
            .next_to(pl.ax, UP, buff=0.15).align_to(pl.ax, RIGHT)
        self.play(FadeIn(verdict, shift=UP * 0.1), run_time=0.4)
        self.caption("If even the low end is above the noise level, the boundary has emerged. The grid grows.")


def zoom_plot(i, xb=(8, 128), steps=None):
    """spectrum axes zoomed on the boundary test: y range covers the threshold(s) and the estimates."""
    steps = [i] if steps is None else steps
    vals = [THR[j] for j in steps]
    for j in steps:
        n = int(MZ["n"][j])
        t = [b for b in range(n // 2, min(n, NB) + 1) if np.isfinite(MZ["est"][j, b])]
        vals += list((MZ["est"][j, t] + MZ["sd"][j, t]) * LG) + list((MZ["est"][j, t] - MZ["sd"][j, t]) * LG)
    lo, hi = min(vals) - 0.35, max(vals) + 0.45
    return LogPlot(yr=(lo, hi), xb=xb)


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
        truth = pl.curve(BINS, PFIN[2:], SIG, 2, dashed=True, opacity=0.8)
        leg = VGroup(
            VGroup(DashedLine(ORIGIN, RIGHT * 0.5, color=NOI, stroke_width=3, dash_length=0.08),
                   tx("noise level", 18, NOI)).arrange(RIGHT, buff=0.15),
            VGroup(VGroup(Rectangle(width=0.5, height=0.18, stroke_width=0, fill_color=OURS, fill_opacity=0.3),
                          Line(LEFT * 0.25, RIGHT * 0.25, color=OURS, stroke_width=3)),
                   tx("estimate", 18, OURS)).arrange(RIGHT, buff=0.15),
            VGroup(DashedLine(ORIGIN, RIGHT * 0.5, color=SIG, stroke_width=2, dash_length=0.07),
                   tx("true power", 18, SIG)).arrange(RIGHT, buff=0.15),
            VGroup(Dot(radius=0.07, color=OURS), tx("tested value", 18, OURS)).arrange(RIGHT, buff=0.15)) \
            .arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([COL_L, 1.2, 0], aligned_edge=UL)
        cur = state(steps[0])
        self.add(truth, cur, leg)
        self.caption("At each step, the sampler measures the low rings, estimates the boundary power, and tests "
                     "it against the noise.")
        self.caption("The noise level drops each step, because the image gets less noisy.", hold=False)
        for i in steps[1:]:
            self.wait(0.5)
            self.play(Transform(cur, state(i)), run_time=0.7)
        self.caption("The estimate also moves: it gets closer to the true power, because more rings are clear "
                     "each step.")
        self.caption(f"At step {I1}, the tested value passes the noise level. The rule fires.")
        self.caption("All inputs are already in the sampler. The cost is a few Fourier transforms and one small equation solve.")


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
        pl = zoom_plot(i, xb=(n // 2 * 0.8, 128))
        dots, cur, band, _ = step_state(pl, i)
        size = 2.2
        g = tokgrid(n, 3.0 * n / 256, (COL_X, 0.7, 0))
        glab = tx(f"step {i}: grid {n}", 22).next_to(g, DOWN, buff=0.2)
        self.add(pl.parts(), cur, band, pl.hline(THR[i]), g, glab,
                 pl.vline(n // 2, WHITE, 3, dashed=False),
                 tx(f"boundary {n // 2}", 18).next_to(pl.p(n // 2, pl.yr[1]), DR, buff=0.1))
        if intro:
            self.caption(f"At step {i}, the rule fires on grid {n}. A fixed ladder doubles the grid at this point.")
            self.caption("Instead, we test the boundary of every larger grid, in steps of 8, with the same test.",
                         hold=False)
        else:
            self.caption(f"Later, at step {i}, the rule fires on grid {n}.", hold=False)
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
            self.caption("Here, the boundary of every larger grid has emerged. So the grid goes straight to full resolution.")
        g2 = tokgrid(chosen, 3.0 * chosen / 256, (COL_X, 0.7, 0), new_from=n)   # same centre as g
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
                 sp.img, sp.frame, sp.labels(), colorbar().move_to([CBAR_X, 0.2, 0]))
        ghost = sp.stair(ceil_of(P, "speed3"), SPD, 3).set_stroke(opacity=0.5)
        self.add(ghost, side_legend([(legline(SPD, 3, 0.5), "SPEED (fixed)", SPD),
                                     (legline(OURS, 5), "ours", OURS)]))
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
        self.caption("The sampler repeats this at every step: it estimates, tests, and tracks.", hold=False)
        self.play(k.animate.set_value(NSTEP), run_time=7.0, rate_func=linear)
        st.clear_updaters()
        gl.clear_updaters()
        self.caption(f"The staircase follows this image’s frontier: {speedup(P, 'cont2x'):.1f}× less compute than full "
                     f"resolution, against {speedup(P, 'speed3'):.1f}× for SPEED.")

        self.section("results")
        self.clear_all()
        self.header("Results on FLUX.2 [klein], 2048 × 2048")
        rows = [p for p in (2580, 1026) if os.path.exists(os.path.join(DATA, f"final_speed3_{p:06d}.png"))]
        arms = (("fullres", "Full resolution", WHITE), ("speed3", "SPEED, 3 stages", SPD), ("cont2x", "Ours", OURS))
        g = Group()
        xs = (-2.9, 0.0, 2.9)
        for c, (arm, name, colr) in enumerate(arms):
            g.add(tx(name, 22, colr).move_to([xs[c], 2.85, 0]))
        for j, p in enumerate(rows):
            y = 1.45 - 2.45 * j
            g.add(tx(NAMES[p], 18, GREY_B).move_to([COL_X - 0.4, y, 0]))
            for c, (arm, name, colr) in enumerate(arms):
                im = arr_img(final_img(p, arm), 2.1).move_to([xs[c], y, 0])
                x = 1.0 if arm == "fullres" else speedup(p, arm)
                g.add(im, framed(im), tx("1× compute" if arm == "fullres" else f"{x:.1f}× less compute", 16, colr)
                      .next_to(im, DOWN, buff=0.06))
        self.play(FadeIn(g), run_time=0.8)
        nums = VGroup(tx("1.7× faster than a 3-stage schedule · 4.7–5.0× faster than full resolution", 26, OURS),
                      tx("HPSv3 −0.19 vs. the 3-stage schedule · ImageReward and HPSv2.1 unchanged · no training",
                         20, GREY_B)).arrange(DOWN, buff=0.12).move_to([0, -3.2, 0])
        self.add(nums)
        self.wait(4.0)
