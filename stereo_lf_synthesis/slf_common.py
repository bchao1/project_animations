"""Shared style, helpers and cached assets for the stereo -> light field synthesis teasers (5-min and 1-min).

Paper: Chao, Liu, Chen, "Robust Light Field Synthesis From Stereo Images With Left-Right Geometric Consistency",
IEEE ICIP 2021, pp. 1844-1848. National Taiwan University.

Data
  data/hci/full_data/...           HCI 4D light field benchmark (Honauer et al. ACCV 2016, CC BY-NC-SA 4.0), 24 scenes
  data/gif_frames/                 website teaser (bchao1.github.io/images/icip2021-results.gif): the paper's synthesized
                                   9 x 9 light field of HCI origami, 81 raster-ordered 504 x 504 views
  data/fig*.png, data/fig3/        paper figure rasters / panels (scratch/crop_figs.py)
  data/generated/repro/            our short re-run of the authors' code on HCI (scratch/repro/train_hci.py)
  data/generated/*.npy             display caches built here by build_assets() (flag done_v1) and build_repro_assets()
                                   (flag done_v2, needs the re-run); delete a flag to rebuild.
Views are indexed (i, j) = (row, column), HCI camera k = 9 i + j; S_l = V(4, 0), S_r = V(4, 8).
"""
from manim import *
import numpy as np
from PIL import Image
import matplotlib
import fcntl
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GEN = os.path.join(DATA, "generated")
REPRO = os.path.join(GEN, "repro")
ORI = os.path.join(DATA, "hci", "full_data", "test", "origami")

config.background_color = BLACK

# ------------------------------------------------------------------ style (same as the other teasers in this repo)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × − → ≈ ∑ ‖ ∇ ℒ …
SYMFONT = "DejaVu Sans"                 # ✗ ✓ are in neither
DIM = "#8A8A8A"
RED = "#FF5252"
GREEN = "#69F0AE"
BLUE = "#448AFF"
AMBER = "#FFCA28"
CYAN = "#4DD0E1"
PURPLE = "#E040FB"
ORANGE = "#FF9E40"
LEFTC = CYAN        # left stereo view S_l, D_lr, F_l
RIGHTC = ORANGE     # right stereo view S_r, D_rl, F_r

HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈ℝ≪≤↓↔∑∏∫ℱℒ𝔼∗≠∞∂∘⊤●⇒∠∇‖·")
SYMCH = set("✗✓")

N = 9
ROW, JL, JR = 4, 0, 8
AS = np.linspace(-1.6, 1.4, 31)          # refocus slopes a (px per view step) for the focus sweeps
A_WALL, A_CRANE = -1.5, 1.0              # a that focuses the back wall / the origami crane (checked on the GIF)
EPI_Y, EPI_X = 372, (112, 364)           # scanline through the crane wing, zoomed segment (504-px coordinates)
PATCH = (6, 3)                           # PSAS patch (p, q) shown (56 x 56 px patch on the crane)


def mathify(s):
    out = []
    for c in s:
        if c in MATHCH:
            out.append(f'<span font_family="{MATHFONT}">{c}</span>')
        elif c in SYMCH:
            out.append(f'<span font_family="{SYMFONT}">{c}</span>')
        else:
            out.append(c)
    return "".join(out)


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


def I(v):
    return f"<i>{v}</i>"


def B(v):
    return f"<b>{v}</b>"


# notation
SL = I("S") + sub(I("l"))
SR = I("S") + sub(I("r"))
FL = I("F") + sub(I("l"))
FR = I("F") + sub(I("r"))
FC = I("F") + sub(I("c"))
FO = I("F") + sub(I("o"))
FT = I("F") + sub(I("t"))
DLR = I("D") + sub(I("lr"))
DRL = I("D") + sub(I("rl"))
KL = I("K") + sub(I("l"))
KR = I("K") + sub(I("r"))
AL = I("A") + sub(I("l"))
LC = "ℒ" + sub(I("c"))
LT = "ℒ" + sub(I("t"))
LR_ = "ℒ" + sub(I("r"))
V_ = lambda a, b: f"{I('V')}({a}, {b})"

EQ1 = f"{LC} = ‖{FL} − {FT}‖{sub('1')} + ‖{FR} − {FT}‖{sub('1')}"
EQ2 = (f"{LT} = ‖∇{sub(I('x'))}{DLR}‖{sub('1')} + ‖∇{sub(I('y'))}{DLR}‖{sub('1')} + "
       f"‖∇{sub(I('x'))}{DRL}‖{sub('1')} + ‖∇{sub(I('y'))}{DRL}‖{sub('1')}")
EQ3 = f"{KL}({I('i')}, {I('j')}) = ‖({I('i')}, {I('j')}) − (4, 0)‖{sub('2')}"
EQ4 = f"{KR}({I('i')}, {I('j')}) = ‖({I('i')}, {I('j')}) − (4, 8)‖{sub('2')}"
EQA = f"{AL} = {KR} / ({KL} + {KR})"
EQ5 = f"{FC} = {AL} · {FL} + (1 − {AL}) · {FR}"
EQ7 = (f"ℒ = {I('w')}{sub(I('c'))} {LC} + {I('w')}{sub(I('t'))} {LT} + {I('w')}{sub(I('r'))} {LR_},   "
       f"{I('w')}{sub(I('c'))} = 1,  {I('w')}{sub(I('t'))} = 0.001,  {I('w')}{sub(I('r'))} = 1")
EQW = (f"{FL}({I('i')}, {I('j')})({I('x')}, {I('y')}) = {SL}({I('x')} + {DLR}·{I('j')},  "
       f"{I('y')} + {DLR}·({I('i')} − 4))")


# ------------------------------------------------------------------ numpy helpers
def gen(name):
    return os.path.join(GEN, name)


def rgba(a):
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
    return np.ascontiguousarray(a.astype(np.uint8))


def cmap_rgb(a, cmap="inferno"):
    return (matplotlib.colormaps[cmap](np.clip(a, 0, 1))[..., :3] * 255).astype(np.uint8)


def _fshift(a, dx, dy):
    """Periodic sub-pixel shift, out(x) = a(x − d)."""
    F = np.fft.fft2(a, axes=(0, 1))
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    ph = np.exp(-2j * np.pi * (fx * dx + fy * dy))
    if a.ndim == 3:
        ph = ph[..., None]
    return np.real(np.fft.ifft2(F * ph, axes=(0, 1)))


def _resize(a, s):
    return np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((s, s), Image.LANCZOS))


def refocus_stack(views, slopes, size):
    """Shift-and-add refocusing: mean over (i, j) of V(i, j) shifted by a·(i − 4, j − 4)."""
    out = []
    for a in slopes:
        acc = np.zeros(views.shape[2:], np.float64)
        for i in range(N):
            for j in range(N):
                acc += _fshift(views[i, j].astype(np.float64), a * (j - 4), a * (i - 4))
        out.append(_resize(acc / (N * N), size))
    return np.stack(out)


def psas_shuffle(lf):
    """The authors' spatio_angular_shuffle on a (9, 9, H, W, 3) array: new view (p, q) = spatial patch (p, q) of every view,
    tiled by view (u, v). Returns (9, 9, H, W, 3)."""
    n = N
    h, w = lf.shape[2:4]
    x = lf.reshape(n, n, n, h // n, n, w // n, 3)             # (u, v, p, h', q, w', c)
    x = x.transpose(2, 4, 0, 3, 1, 5, 6)                       # (p, q, u, h', v, w', c)
    return x.reshape(n, n, h, w, 3)


def load_gt():
    v = [np.asarray(Image.open(os.path.join(ORI, f"input_Cam{k:03d}.png")).convert("RGB"))[4:508, 4:508] for k in range(81)]
    return np.stack(v).reshape(N, N, 504, 504, 3)


def load_gif():
    v = [np.asarray(Image.open(os.path.join(DATA, "gif_frames", f"{k:03d}.png")).convert("RGB")) for k in range(81)]
    return np.stack(v).reshape(N, N, 504, 504, 3)


def _locked(flag, fn):
    os.makedirs(GEN, exist_ok=True)
    if os.path.exists(gen(flag)):
        return
    with open(gen("lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(gen(flag)):
            return
        fn()
        open(gen(flag), "w").write("ok")


def _build_v1():
    gt = load_gt()
    gif = load_gif()
    np.save(gen("gt_thumbs.npy"), np.stack([[_resize(gt[i, j], 72) for j in range(N)] for i in range(N)]))
    np.save(gen("gt336.npy"), np.stack([[_resize(gt[i, j], 336) for j in range(N)] for i in range(N)]))
    np.save(gen("gif336.npy"), np.stack([[_resize(gif[i, j], 336) for j in range(N)] for i in range(N)]))
    np.save(gen("refocus_gt.npy"), refocus_stack(gt, AS, 336))
    np.save(gen("refocus_gif.npy"), refocus_stack(gif, AS, 336))
    np.save(gen("epi.npy"), np.stack([gt[ROW, j, EPI_Y] for j in range(N)]))                    # (9, 504, 3)
    sh = psas_shuffle(gt)
    np.save(gen("shuffle.npy"), _resize(sh[PATCH[0], PATCH[1]], 504))
    np.save(gen("sl504.npy"), gt[ROW, JL])
    np.save(gen("sr504.npy"), gt[ROW, JR])


def build_assets():
    _locked("done_v1", _build_v1)


def repro_ready():
    return os.path.exists(os.path.join(REPRO, "metrics.json"))


def _build_v2():
    gt = np.load(os.path.join(REPRO, "origami_gt.npy")).reshape(N, N, 504, 504, 3)
    M = {k: np.load(os.path.join(REPRO, f"origami_{k}.npy")).reshape(N, N, 504, 504, 3) for k in ("Fl", "Fr", "Fc", "Fo")}
    for k, v in M.items():
        np.save(gen(f"{k}_thumbs.npy"), np.stack([[_resize(v[i, j], 72) for j in range(N)] for i in range(N)]))
    for k in ("Fl", "Fr"):
        np.save(gen(f"{k}336.npy"), np.stack([[_resize(M[k][i, j], 336) for j in range(N)] for i in range(N)]))
        err = np.abs(M[k].astype(np.float32) - gt.astype(np.float32)).mean(-1) / 255.0
        np.save(gen(f"{k}_err336.npy"), np.stack([[_resize(cmap_rgb(np.sqrt(np.clip(err[i, j] / 0.25, 0, 1))), 336)
                                                    for j in range(N)] for i in range(N)]))
    d_lr = np.load(os.path.join(REPRO, "origami_d_lr.npy"))
    d_rl = np.load(os.path.join(REPRO, "origami_d_rl.npy"))
    lo, hi = np.percentile(np.concatenate([d_lr.ravel(), d_rl.ravel()]), [1, 99])
    np.save(gen("d_range.npy"), np.array([lo, hi]))
    for nm, d in (("d_lr", d_lr), ("d_rl", d_rl)):
        np.save(gen(f"{nm}_rgb.npy"), _resize(cmap_rgb((d - lo) / (hi - lo), "viridis"), 336))
    np.save(gen("d_lr504.npy"), d_lr.astype(np.float32))
    # zoom: 112-px window of the top-right view (far from S_l) where blending removes the most warping error
    # (the re-run's PSAS refiner stayed ~identity, F_o ≈ F_c, so refinement gains are shown from the paper instead)
    i, j = 0, 7
    e_l = np.abs(M["Fl"][i, j].astype(np.float32) - gt[i, j]).mean(-1)
    e_c = np.abs(M["Fc"][i, j].astype(np.float32) - gt[i, j]).mean(-1)
    gain = e_l - e_c
    Z = 112
    best, by, bx = -1e9, 0, 0
    for y in range(0, 504 - Z + 1, 8):
        for x in range(0, 504 - Z + 1, 8):
            g = gain[y:y + Z, x:x + Z].mean()
            if g > best:
                best, by, bx = g, y, x
    np.save(gen("zoom_box.npy"), np.array([i, j, by, bx, Z]))
    for k, src in (("gt", gt[i, j]), ("Fc", M["Fc"][i, j]), ("Fo", M["Fo"][i, j]), ("Fl", M["Fl"][i, j])):
        crop = src[by:by + Z, bx:bx + Z]
        np.save(gen(f"zoom_{k}.npy"), _resize(crop, 336))
        if k != "gt":
            e = np.abs(crop.astype(np.float32) - gt[i, j][by:by + Z, bx:bx + Z]).mean(-1) / 255.0
            np.save(gen(f"zoom_err_{k}.npy"), _resize(cmap_rgb(np.sqrt(np.clip(e / 0.25, 0, 1))), 336))
    np.save(gen("view_Fo_04.npy"), _resize(M["Fo"][i, j], 336))
    np.save(gen("view_gt_04.npy"), _resize(gt[i, j], 336))
    # training snapshots (centre view F_o) and the loss / PSNR log
    snaps = sorted(f for f in os.listdir(REPRO) if f.startswith("snap_"))
    its = [int(f[5:10]) for f in snaps]
    np.save(gen("snap_its.npy"), np.array(its))
    np.save(gen("snaps.npy"), np.stack([_resize(np.load(os.path.join(REPRO, f))[2], 336) for f in snaps]))
    np.save(gen("snaps_d.npy"), np.stack([_resize(cmap_rgb((np.load(os.path.join(REPRO, "snapd_" + f[5:]))
                                                            .astype(np.float32) - lo) / (hi - lo), "viridis"), 336)
                                          for f in snaps]))


def build_repro_assets():
    if repro_ready():
        _locked("done_v2", _build_v2)


def G(name):
    return np.load(gen(name))


def metrics():
    return json.load(open(os.path.join(REPRO, "metrics.json")))


# ------------------------------------------------------------------ mobject helpers
def img(path, h=None, w=None):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    m = ImageMobject(p)
    if h is not None:
        m.set(height=h)
    if w is not None:
        m.set(width=w)
    return m


def arr_img(a, h=None, w=None, nearest=False):
    m = ImageMobject(rgba(a))
    if nearest:
        m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    if h is not None:
        m.set(height=h)
    if w is not None:
        m.set(width=w)
    return m


def framed(m, color=WHITE, sw=1.5):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw)


def live(m, fn):
    """Attach an updater that rewrites the image pixels every frame."""
    m.add_updater(lambda mob: setattr(mob, "pixel_array", rgba(fn())))
    return m


def interp(frames, v):
    v = float(np.clip(v, 0, len(frames) - 1))
    i0 = int(np.floor(v))
    i1 = min(i0 + 1, len(frames) - 1)
    w = v - i0
    return ((1 - w) * frames[i0].astype(np.float32) + w * frames[i1].astype(np.float32)).astype(np.uint8)


def a_index(a):
    return (a - AS[0]) / (AS[1] - AS[0])


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw, max_tip_length_to_length_ratio=0.18, tip_length=0.16)


def fitw(m, w=12.8):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


def label(m, s, size=22, color=WHITE, d=UP, buff=0.15):
    """Label above/below m, placed on a common baseline so labels in a row line up."""
    t = tx(s, size, color)
    plain = re.sub(r"<[^>]+>", "", s)
    desc = any(c in "gjpqy(),;[]{}|µφμβγηρχψξζ" for c in plain) or 'rise="-' in s
    dd, asc = 0.0027 * size, 0.0098 * size
    base = m.get_top()[1] + buff + dd if d is UP else m.get_bottom()[1] - buff - asc
    t.shift(UP * (base - (t.get_bottom()[1] + (dd if desc else 0))))
    t.set_x(m.get_center()[0])
    return t


def note(s, size=17):
    return tx(s, size, DIM)


class ViewGrid:
    """N × N grid of view thumbnails (or empty cells). pos(i, j) gives the centre of cell (i, j); row i grows downwards."""

    def __init__(self, center, cell, thumbs=None, gap=0.03, empty_color="#2A2D33", sw=1.0):
        self.c = np.array([center[0], center[1], 0.0])
        self.cell = cell
        self.step = cell + gap
        self.mobs = Group()
        for i in range(N):
            for j in range(N):
                if thumbs is not None:
                    m = arr_img(thumbs[i, j], h=cell)
                else:
                    m = Square(cell, stroke_color=empty_color, stroke_width=sw, fill_color="#15171B", fill_opacity=1)
                m.move_to(self.pos(i, j))
                self.mobs.add(m)
        self.width = N * self.step - gap

    def pos(self, i, j):
        return self.c + np.array([(j - 4) * self.step, -(i - 4) * self.step, 0])

    def at(self, i, j):
        return self.mobs[i * N + j]

    def box(self, i, j, color=AMBER, sw=3):
        return Square(self.cell + 0.02, color=color, stroke_width=sw).move_to(self.pos(i, j))

    def fill(self, cells, color, opacity=1.0):
        return VGroup(*[Square(self.cell, stroke_width=0, fill_color=color, fill_opacity=opacity).move_to(self.pos(i, j))
                        for i, j in cells])

    def outline(self, color=DIM, sw=1.2):
        return Rectangle(width=self.width + 0.08, height=self.width + 0.08, color=color, stroke_width=sw).move_to(self.c)


def heat_img(H, vmax, cmap="turbo", h=2.4):
    a = cmap_rgb(np.asarray(H) / vmax, cmap)
    return arr_img(a, h=h, nearest=True)


def colorbar(h, cmap="turbo", w=0.18, lo="low", hi="high"):
    a = cmap_rgb(np.linspace(1, 0, 128)[:, None].repeat(4, 1), cmap)
    bar = arr_img(a, h=h)
    bar.stretch_to_fit_width(w)
    g = Group(bar, framed(bar, DIM, 1))
    t1 = tx(hi, 16, DIM).next_to(bar, UP, buff=0.08)
    t2 = tx(lo, 16, DIM).next_to(bar, DOWN, buff=0.08)
    return Group(g, t1, t2)


# ------------------------------------------------------------------ scene base
class Base(Scene):
    def setup(self):
        self.cap = None
        self.hdr = None
        self.clock = ValueTracker(0)
        self.clock.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(self.clock)

    def header(self, s):
        if self.hdr is not None:
            self.remove(self.hdr)
        self.hdr = tx(s, 24, DIM).to_corner(UL, buff=0.35).set_y(HEADER_Y)
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
        k = min(spaces, key=lambda p: abs(p[1] - n / 2))[0]
        return [s[:k], s[k + 1:]]

    def caption(self, s, wait=None, size=28):
        self._hold_caption()
        s = re.sub(r"(\d) × (\d)", "\\1\u00a0×\u00a0\\2", s)
        if self.cap is not None:
            self.remove(self.cap)
        c = tx(s, size)
        if c.width > 13.2:
            parts = self._split(s)
            c = VGroup(*[tx(p, size - 2) for p in parts]).arrange(DOWN, buff=0.12)
            if c.width > 13.2:
                c.scale_to_fit_width(13.2)
        c.move_to([0, CAPTION_Y, 0])
        if c.get_bottom()[1] < -3.85:
            c.shift(UP * (-3.85 - c.get_bottom()[1]))
        self.cap = c
        plain = re.sub(r"<[^>]+>", "", s)
        self.cap_t0 = self.renderer.time
        self.cap_need = len(plain) / 17.0 + 0.8
        self.add(c)
        if wait:
            self.wait(wait)
        return c

    def hold(self, extra=0.0):
        self._hold_caption()
        if extra:
            self.wait(extra)

    def uncaption(self):
        self._hold_caption()
        if self.cap is not None:
            self.remove(self.cap)
            self.cap = None

    def title_card(self, num, title, hold=1.5):
        a = tx(num, 30, DIM)
        b = tx(title, 50, bold=True)
        if b.width > 12.5:
            b.scale_to_fit_width(12.5)
        g = VGroup(a, b).arrange(DOWN, buff=0.3)
        self.add(g)
        self.wait(hold)
        self.remove(g)

    def clear_all(self, keep=(), run_time=0.7):
        self.uncaption()
        keep = set(keep) | {self.hdr, self.clock}
        mobs = [m for m in self.mobjects if m not in keep]
        for m in mobs:
            m.clear_updaters()
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def run(self, tracker, target, t, rate=linear):
        self.play(tracker.animate.set_value(target), run_time=t, rate_func=rate)


def view_path(t, r=3.6, period=2.4):
    """Circular path over the view grid, nearest view (i, j)."""
    th = 2 * np.pi * t / period
    i = int(np.clip(round(4 - r * np.cos(th)), 0, 8))
    j = int(np.clip(round(4 + r * np.sin(th)), 0, 8))
    return i, j


def focus_slider(center, length, a_of_t):
    """Horizontal focus slider: far (wall) on the left, near (crane) on the right."""
    ln = Line(LEFT * length / 2, RIGHT * length / 2, color=DIM, stroke_width=2).move_to(center)
    a2x = lambda a: center[0] + (a - AS[0]) / (AS[-1] - AS[0]) * length - length / 2
    marks = VGroup()
    for a, s, c in ((A_WALL, "back wall", CYAN), (A_CRANE, "crane", AMBER)):
        marks.add(Triangle(color=c, fill_opacity=1, stroke_width=0).scale(0.07).rotate(PI).move_to([a2x(a), center[1] + 0.13, 0]),
                  tx(s, 17, c).move_to([a2x(a), center[1] + 0.36, 0]))
    dot = always_redraw(lambda: Dot([a2x(float(np.clip(a_of_t(), AS[0], AS[-1]))), center[1], 0], radius=0.08, color=WHITE))
    lab = tx("focus", 18, DIM).next_to(ln, LEFT, buff=0.25)
    return VGroup(ln, marks, lab), dot
