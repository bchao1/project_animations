"""Gaussian Wave Splatting — teaser animation, v4.

Source: Choi*, Chao*, Yang, Gopakumar, Wetzstein, "Gaussian Wave Splatting for Computer-Generated
Holography", ACM TOG 44(4) (SIGGRAPH 2025), arXiv 2505.06582.
Paper figures in data/ are crops of the paper (gaussian-wave-splatting/pdfs/). All wave-optics visuals
are computed here with numpy (angular spectrum propagation, closed-form Gaussian spectra, Huygens sums)
and cached in data/generated/ (delete data/generated/done_v3 to rebuild).

Render (see render.sh):
    manim -ql --media_dir media animation_gws_v4.py <Scene>
Scenes, in order: S00_Title ... S09_Outlook.
"""
from manim import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import fcntl
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GEN = os.path.join(DATA, "generated")
FONTB = "/home/brianchc/.local/share/fonts/cmunbx.ttf"

config.background_color = BLACK

# ------------------------------------------------------------------ style
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × − → ≈ ∑ ∏ ℱ 𝒫 𝒢 ℋ ∗ …
DIM = "#8A8A8A"
RED = "#FF5252"
GREEN = "#69F0AE"
BLUE = "#448AFF"
AMBER = "#FFCA28"
CYAN = "#4DD0E1"
PURPLE = "#E040FB"
BOXC = "#1E2A33"

HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈ℝ≪≤↓↔∑∏∬∫ℱ𝒫𝒢𝒩ℋ∗≠∞∂∘⊤●⇒")


def mathify(s):
    return "".join(
        f'<span font_family="{MATHFONT}">{c}</span>' if c in MATHCH else c for c in s)


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


# common symbols
def I(v):
    return f"<i>{v}</i>"


def B(v):
    return f"<b>{v}</b>"


XB = B("x")
KB = B("k")
UHAT = "<i>Û</i>"
SQ = sup("2")
EQ1 = f"{B('c')} = ∑{sub(I('i'))} {B('c')}{sub(I('i'))} α{sub(I('i'))} ∏{sub(I('j') + '&lt;' + I('i'))} (1 − α{sub(I('j'))})"
EQ2 = f"α{sub(I('i'))} = 𝒢{sub(I('i'))}({I('u')}, {I('v')}) · {I('o')}{sub(I('i'))}"
EQ5 = (f"{I('u')}{sub('SLM')}({XB}) = ∑{sub(I('i'))} {B('c')}{sub(I('i'))} {I('o')}{sub(I('i'))} "
       f"𝒫( {I('u')}{sub(I('i'))}({XB}; {I('z')}{sub(I('i'))}) {I('T')}{sub(I('i'))}({XB}) ; −{I('z')}{sub(I('i'))} )")
EQ6 = (f"{I('T')}{sub(I('i'))}({XB}) = ∏{sub(I('j') + '&lt;' + I('i'))} "
       f"(1 − {I('o')}{sub(I('j'))} |{I('u')}{sub(I('j'))}({XB}; {I('z')}{sub(I('j'))})|)")
EQ11 = (f"{UHAT}{sub('SLM')}({KB}) = ∑{sub(I('i'))} {B('c')}{sub(I('i'))} {I('o')}{sub(I('i'))} "
        f"( {UHAT}{sub(I('i'))}({KB}) {col('∗ ℱ(' + I('T') + sub(I('i')) + ')', RED)} ) ℋ({KB}, −{I('z')}{sub(I('i'))})")
EQ12 = (f"{UHAT}{sub('SLM')}({KB}) ≈ ∑{sub(I('i'))} {B('c')}{sub(I('i'))} {I('o')}{sub(I('i'))} "
        f"{UHAT}{sub(I('i'))}({KB}) ℋ({KB}, −{I('z')}{sub(I('i'))})")
EQ13 = f"{B('c')} ≈ ∑{sub(I('i'))} {I('c')}{sub(I('i'))} 𝒢{sub(I('i'))} {I('o')}{sub(I('i'))}"
EQVD = f"α{sub(I('i'))} = 𝒢{sub(I('i'))} · SH({B('o')}{sub(I('i'))}, {B('d')})"

# ------------------------------------------------------------------ optics (numpy)
LAM = 520e-9          # green laser line of the paper's prototype
DX = 8e-6             # pixel pitch of the paper's SLM (Holoeye PLUTO)


def fgrid(N, dx=DX):
    f = np.fft.fftfreq(N, dx)
    return np.meshgrid(f, f)


def asm(u, z, lam=LAM, dx=DX):
    """Angular spectrum method: P(u; z) = F^-1{ F{u} · exp(j 2π f_z z) }, propagating waves only."""
    N = u.shape[0]
    FX, FY = fgrid(N, dx)
    arg = 1 / lam ** 2 - FX ** 2 - FY ** 2
    fz = np.sqrt(np.maximum(arg, 0))
    H = np.where(arg > 0, np.exp(1j * 2 * np.pi * fz * z), 0)
    return np.fft.ifft2(np.fft.fft2(u) * H)


def rot3(rx, ry, rz):
    def Rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rx(rx) @ Ry(ry) @ Rz(rz)


def gws_field(N, mu, su, sv, rx, ry, rz, lam=LAM, dx=DX):
    """Closed-form Gaussian Wave Splatting (paper Eq. 9-10), evaluated directly on the SLM frequency grid:
    U(f) = 2π det(J) det(S) exp(-2π² |S R^-1 (f - f_r)|²) exp(-j2π f·μ), plane-wave illumination f_r."""
    FX, FY = fgrid(N, dx)
    arg = 1 / lam ** 2 - FX ** 2 - FY ** 2
    ok = arg > 0
    FZ = np.sqrt(np.maximum(arg, 0))
    Ri = rot3(rx, ry, rz).T
    Fo = (Ri @ np.stack([FX, FY, FZ], 0).reshape(3, -1)).reshape(3, N, N)
    fr = Ri @ np.array([0, 0, 1 / lam])
    q = np.stack([su * (Fo[0] - fr[0]), sv * (Fo[1] - fr[1])])
    detJ = np.abs(Fo[2]) / np.maximum(FZ, 1e-9)
    spec = 2 * np.pi * detJ * su * sv * np.exp(-2 * np.pi ** 2 * (q ** 2).sum(0))
    spec = spec * np.exp(-1j * 2 * np.pi * (FX * mu[0] + FY * mu[1]))
    spec = spec * np.exp(-1j * 2 * np.pi * FZ * mu[2]) * ok
    return np.fft.fftshift(np.fft.ifft2(spec)) / dx ** 2


def domain_rgb(u, gamma=0.6, amax=None, sat=0.85):
    a = np.abs(u)
    amax = amax or (np.percentile(a, 99.7) + 1e-12)
    v = np.clip(a / amax, 0, 1) ** gamma
    h = (np.angle(u) + np.pi) / (2 * np.pi)
    hsv = np.stack([h, np.full_like(h, sat), v], -1)
    return (matplotlib.colors.hsv_to_rgb(hsv) * 255).astype(np.uint8)


def phase_rgb(phi, cmap="twilight"):
    c = matplotlib.colormaps[cmap]((phi + np.pi) / (2 * np.pi))
    return (c[..., :3] * 255).astype(np.uint8)


def grey(a, gamma=1.0):
    return (np.clip(a, 0, 1) ** gamma * 255).astype(np.uint8)


def wave_lut():
    """Diverging map for real wave fields: blue (−) · black (0) · amber (+)."""
    t = np.linspace(-1, 1, 256)[:, None]
    neg = np.array([0x29, 0x79, 0xFF]) / 255.0
    pos = np.array([0xFF, 0xCA, 0x28]) / 255.0
    rgb = np.where(t < 0, -t * neg, t * pos) ** 0.8
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


WLUT = wave_lut()


def wave_lut_g(g):
    t = np.linspace(-1, 1, 256)[:, None]
    neg = np.array([0x29, 0x79, 0xFF]) / 255.0
    pos = np.array([0xFF, 0xCA, 0x28]) / 255.0
    rgb = np.where(t < 0, (-t) ** g * neg, t ** g * pos)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


WLUT2 = wave_lut_g(1.5)


def wave_rgb(re, scale):
    idx = np.clip((re / scale * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    return WLUT[idx]


def rgba(a):
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
    return np.ascontiguousarray(a.astype(np.uint8))


# ------------------------------------------------------------------ assets
def gen(name):
    return os.path.join(GEN, name)


def _text_img(N, s, size):
    im = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONTB, size)
    bb = d.textbbox((0, 0), s, font=f)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((N - w) // 2 - bb[0], (N - h) // 2 - bb[1]), s, font=f, fill=255)
    return np.asarray(im) / 255.0


def _star_img(N, r0, r1, center, n=5, width=5):
    im = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(im)
    cx, cy = center
    pts = []
    for i in range(2 * n + 1):
        r = r1 if i % 2 == 0 else r0
        a = -np.pi / 2 + i * np.pi / n
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    d.line(pts, fill=255, width=width, joint="curve")
    return np.asarray(im) / 255.0


def gauss2(N, cx, cy, su, sv, th):
    y, x = np.mgrid[0:N, 0:N].astype(float)
    x, y = x - cx, y - cy
    c, s = np.cos(th), np.sin(th)
    u, v = c * x + s * y, -s * x + c * y
    return np.exp(-0.5 * ((u / su) ** 2 + (v / sv) ** 2))


def point_field(W, H, xr, yr, srcs, lam):
    """Top-view complex field of coherent 2D point sources (for Re{U e^{-jωt}} animations)."""
    x = np.linspace(*xr, W)
    y = np.linspace(*yr, H)[::-1]
    X, Y = np.meshgrid(x, y)
    U = 0
    for (sx, sy) in srcs:
        r = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2) + 1e-3
        U = U + np.exp(1j * 2 * np.pi * r / lam) / np.sqrt(r + 0.3)
    return U


def tilted_field():
    """x–z side view: plane wave hits a tilted 2D Gaussian; its wave continues to the SLM (Huygens sum)."""
    W, H = 420, 210                     # z ∈ [0, 40] λ, x ∈ [-10, 10] λ
    z = np.linspace(0, 40, W)
    x = np.linspace(10, -10, H)
    Z, X = np.meshgrid(z, x)
    zc, tilt, sig = 9.0, np.deg2rad(35), 3.0
    s = np.linspace(-9, 9, 240)
    zs = zc + s * np.sin(tilt)
    xs = s * np.cos(tilt)
    amp = np.exp(-0.5 * (s / sig) ** 2)
    U = np.zeros_like(Z, dtype=complex)
    ds = s[1] - s[0]
    for k in range(len(s)):
        dz, dxx = Z - zs[k], X - xs[k]
        r = np.sqrt(dz ** 2 + dxx ** 2) + 1e-3
        obl = np.clip(dz / r, 0, 1)
        U += amp[k] * np.exp(1j * 2 * np.pi * zs[k]) * obl * np.exp(1j * 2 * np.pi * r) / np.sqrt(r) * ds
    line_z = zc + X * np.tan(tilt)             # the disk's plane
    before = Z < line_z
    U = U / np.percentile(np.abs(U[~before]), 99)
    inc = np.exp(1j * 2 * np.pi * Z)
    U = np.where(before, inc, U)
    np.save(gen("tilted_U.npy"), U.astype(np.complex64))
    # disk drawn with brightness = Gaussian amplitude along the disk
    s_of = X / np.cos(tilt)
    disk = np.where(np.abs(Z - line_z) < 0.3, np.exp(-0.5 * (s_of / sig) ** 2), 0.0)
    np.save(gen("tilted_disk.npy"), disk.astype(np.float32))


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    if os.path.exists(gen("done_v3")):
        return
    with open(gen("lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(gen("done_v3")):
            return
        # --- S00 / S01: coherent point sources, top view (λ = 0.9 units)
        U = point_field(480, 270, (-8, 8), (-4.5, 4.5), [(-1.4, -4.9), (1.4, -4.9)], 0.9)
        np.save(gen("two_U.npy"), U.astype(np.complex64))
        U3 = point_field(640, 280, (-8, 8), (-3.5, 3.5), [(-3.2, -3.2), (0.0, 2.6), (3.2, -3.2)], 0.8)
        np.save(gen("three_U.npy"), U3.astype(np.complex64))
        # --- S02 recording: point object + tilted reference (2 µm samples, z = 5 mm)
        N = 360
        xx = (np.arange(N) - N / 2) * 2e-6
        X, Y = np.meshgrid(xx, xx)
        k = 2 * np.pi / LAM
        O = np.exp(1j * k * np.sqrt(X ** 2 + Y ** 2 + 5e-3 ** 2))
        R = np.exp(1j * 2 * np.pi * 60e3 * X)
        Image.fromarray(grey(np.real(O) * 0.5 + 0.5)).save(gen("rec_O.png"))
        Image.fromarray(grey(np.real(R) * 0.5 + 0.5)).save(gen("rec_R.png"))
        Image.fromarray(grey(np.abs(O + R) ** 2 / 4)).save(gen("rec_I.png"))
        # --- S02 3D hologram: star at 12 mm, "3D" at 36 mm (complex field, smooth phase)
        N = 512
        A1 = _star_img(N, 55, 130, (N * 0.33, N * 0.36))
        A2 = np.roll(_text_img(N, "3D", 150), (70, 80), (0, 1))
        Hf = asm(A1, -12e-3) + asm(A2, -36e-3)
        Image.fromarray(phase_rgb(np.angle(Hf))).save(gen("holo_phase.png"))
        zs = np.linspace(0, 45e-3, 91)
        fr = np.stack([grey(np.abs(asm(Hf, z)) ** 2, 0.7) for z in zs])
        np.save(gen("focus_frames.npy"), fr)
        # --- S03 point-based CGH: three points at 3 / 4 / 5.5 mm (2 µm samples)
        N = 360
        xx = (np.arange(N) - N / 2) * 2e-6
        X, Y = np.meshgrid(xx, xx)
        tot = 0
        for (px, py, pz) in [(-0.15e-3, 0.12e-3, 3e-3), (0.12e-3, 0.0, 4e-3), (-0.05e-3, -0.16e-3, 5.5e-3)]:
            r = np.sqrt((X - px) ** 2 + (Y - py) ** 2 + pz ** 2)
            tot = tot + np.exp(1j * k * r)
        Image.fromarray(phase_rgb(np.angle(tot))).save(gen("pts_holo.png"))
        # --- S04 Gaussians (RGBA), for compositing by layering
        N = 400
        for name, (cx, cy, su, sv, th, c, o) in {
            "g_red": (200, 200, 70, 30, 0.5, (255, 82, 82), 0.95),
            "g_amber": (235, 225, 60, 45, -0.3, (255, 202, 40), 0.9),
            "g_blue": (170, 240, 85, 40, 1.2, (68, 138, 255), 0.9),
            "g_one": (200, 200, 95, 40, 0.45, (77, 208, 225), 1.0),
        }.items():
            a = o * gauss2(N, cx, cy, su, sv, th)
            im = np.zeros((N, N, 4), np.uint8)
            im[..., :3] = c
            im[..., 3] = (a * 255).astype(np.uint8)
            Image.fromarray(im, "RGBA").save(gen(name + ".png"))
        # --- S05 tilted Gaussian side view + closed-form SLM field of one Gaussian
        tilted_field()
        u = gws_field(512, (0.0, 0.0, 40e-3), 40e-6, 8e-6, 0.0, np.deg2rad(45), np.deg2rad(30))
        Image.fromarray(domain_rgb(u[128:384, 128:384])).save(gen("gws_slm.png"))
        rec = np.abs(asm(u, 40e-3)) ** 2
        rc = rec[226:286, 226:286]
        Image.fromarray(grey(rc / rc.max(), 0.6)).resize((288, 288), Image.NEAREST).save(gen("gws_rec.png"))
        # --- S06 occlusion toy: red Gaussian (front, 6 mm) partly covers blue Gaussian (10 mm)
        N = 256
        gr = gauss2(N, 110, 128, 34, 34, 0)
        gb = gauss2(N, 150, 128, 42, 42, 0)
        o_r, o_b = 1.0, 1.0
        zr, zb = 6e-3, 10e-3
        T = {"naive": np.ones_like(gr), "silhouette": 1.0 - (o_r * gr > 0.1), "alpha": 1.0 - o_r * gr}
        red_slm = asm(o_r * gr, -zr)
        out = {}
        for key, Tm in T.items():
            blue_slm = asm(o_b * gb * Tm, -zb)
            r_ = np.abs(asm(red_slm, zr))           # focus on each Gaussian's own depth
            b_ = np.abs(asm(blue_slm, zb))
            out[key] = np.stack([r_, 0.25 * b_, b_], -1)
        tgt = np.stack([o_r * gr, 0.25 * o_b * gb * (1 - o_r * gr), o_b * gb * (1 - o_r * gr)], -1)
        out["target"] = tgt
        for key, a in out.items():
            Image.fromarray(grey(a)).save(gen(f"occ_{key}.png"))
        open(gen("done_v3"), "w").write("ok")


build_assets()


def img(path, h=None, w=None):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    m = ImageMobject(p)
    if h is not None:
        m.set(height=h)
    if w is not None:
        m.set(width=w)
    return m


def arr_img(a, h=None, w=None):
    m = ImageMobject(rgba(a))
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


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.16)


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

    # Captions stay up long enough to read: ~17 characters per second + 0.8 s.
    def _hold_caption(self):
        if self.cap is not None:
            left = self.cap_need - (self.renderer.time - self.cap_t0)
            if left > 0.05:
                self.wait(left)

    @staticmethod
    def _split(s):
        """Split at the space nearest the middle that lies outside every markup element."""
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


def label(m, s, size=22, color=WHITE, d=UP, buff=0.15):
    """Label above/below m, placed on a common baseline so labels in a row line up."""
    t = tx(s, size, color)
    plain = re.sub(r"<[^>]+>", "", s)
    desc = any(c in "gjpqy(),;[]{}|µφμβγηρχψξζ" for c in plain) or 'rise="-' in s
    dd, asc = 0.0027 * size, 0.0098 * size        # CMU Serif descender / ascender depth per font-size unit
    base = m.get_top()[1] + buff + dd if d is UP else m.get_bottom()[1] - buff - asc
    t.shift(UP * (base - (t.get_bottom()[1] + (dd if desc else 0))))
    t.set_x(m.get_center()[0])
    return t


# ================================================================== S00 Title
class S00_Title(Base):
    def construct(self):
        title = tx("Gaussian Wave Splatting", 66, bold=True).move_to([0, 2.95, 0])
        subt = tx("for Computer-Generated Holography", 36).move_to([0, 2.2, 0])
        auth = tx("Suyeon Choi*, Brian Chao*, Jacqueline Yang, Manu Gopakumar, Gordon Wetzstein",
                  24, DIM).move_to([0, -2.55, 0])
        aff = tx("Stanford University  ·  ACM Transactions on Graphics (SIGGRAPH 2025)", 22, DIM).move_to([0, -3.0, 0])
        eq = tx("* equal contribution", 18, DIM).move_to([0, -3.4, 0])
        U = np.load(gen("three_U.npy"))
        clk = self.clock
        field = arr_img(wave_rgb(np.real(U), 1.6), h=3.5).move_to([0, -0.1, 0])
        live(field, lambda: wave_rgb(np.real(U * np.exp(-1j * 2 * np.pi * 0.9 * clk.get_value())), 1.6))
        box = framed(field, DIM, 1.2)
        self.add(title, subt, auth, aff, eq)
        self.play(FadeIn(field), Create(box), run_time=1.0)
        self.wait(7.5)


# ================================================================== S01 Waves
class S01_Waves(Base):
    def construct(self):
        self.title_card("Part 1", "Light is a wave")
        self.header("1 · Light is a wave")
        clk = self.clock
        ph = lambda: 2 * PI * 0.5 * clk.get_value()
        ax = Axes(x_range=[0, 4 * PI, PI], y_range=[-1.3, 1.3, 1], x_length=6.0, y_length=2.1,
                  tips=False, axis_config={"color": DIM, "stroke_width": 2}).move_to([-3.5, 1.35, 0])
        wave = always_redraw(lambda: ax.plot(lambda x: np.cos(x - ph()),
                                             x_range=[0, 4 * PI, 0.05], color=AMBER, stroke_width=4))
        ttl = tx(f"{I('u')}({I('x')}, {I('t')}) = {I('A')} cos({I('kx')} − ω{I('t')} + φ)", 28).next_to(ax, UP, buff=0.2)
        self.play(Create(ax), run_time=0.6)
        self.add(wave, ttl)
        self.caption("Light is an oscillating wave with an amplitude and a phase.")

        # phasor
        cen = np.array([-5.3, -1.45, 0])
        circ = Circle(radius=0.85, color=DIM, stroke_width=2).move_to(cen)
        axl = VGroup(Line(cen + LEFT, cen + RIGHT), Line(cen + DOWN, cen + UP)).set_stroke(GREY_D, 1.5)
        vec = always_redraw(lambda: Arrow(cen, cen + 0.85 * np.array([np.cos(-ph()), np.sin(-ph()), 0]), buff=0,
                                          color=AMBER, stroke_width=4, max_tip_length_to_length_ratio=0.25))
        self.play(Create(circ), Create(axl), run_time=0.5)
        self.add(vec)
        c1 = tx(f"{I('u')} = {I('A')} {I('e')}{sup(I('j') + 'φ')}", 34).move_to([-2.6, -1.05, 0])
        c2 = tx(f"{I('A')}: amplitude (how bright)", 22, DIM).move_to([-2.6, -1.65, 0])
        c3 = tx("φ: phase (where the crest is)", 22, DIM).move_to([-2.6, -2.05, 0])
        self.add(c1, c2, c3)
        self.caption(f"As one complex number {I('u')} = {I('A')}{I('e')}{sup(I('j') + 'φ')}, the arrow spins with the wave.")

        # two coherent sources
        U = np.load(gen("two_U.npy"))
        f = arr_img(wave_rgb(np.real(U), 1.4), h=3.6).move_to([3.55, 0.55, 0])
        live(f, lambda: wave_rgb(np.real(U * np.exp(-1j * ph())), 1.4))
        fb = framed(f, DIM, 1.2)
        self.play(FadeIn(f), Create(fb), run_time=0.8)
        lf = label(f, "two coherent point sources (computed)", 22, DIM, DOWN)
        self.add(lf)
        self.caption("Two coherent sources: their waves add, crest on crest or crest on trough.")
        self.wait(1.0)
        Iu = np.abs(U) ** 2
        fi = arr_img((matplotlib.colormaps["inferno"](np.clip(Iu / np.percentile(Iu, 99), 0, 1))[..., :3] * 255
                      ).astype(np.uint8), h=3.6).move_to(f)
        self.play(FadeIn(fi), run_time=1.0)
        f.clear_updaters()
        self.remove(f, lf)
        lf2 = label(fi, f"what a camera records: intensity |{I('u')}|{SQ}", 22, DIM, DOWN)
        self.add(lf2)
        self.caption(f"A camera only records the intensity |{I('u')}|{SQ}: the phase is lost. Holography keeps it.")
        self.clear_all()


# ================================================================== S02 Hologram
class S02_Hologram(Base):
    def construct(self):
        self.title_card("Part 2", "What a hologram does")
        self.header("2 · What a hologram does")
        h = 2.6
        O = img(gen("rec_O.png"), h).move_to([-4.3, 0.75, 0])
        R = img(gen("rec_R.png"), h).move_to([0, 0.75, 0])
        Ih = img(gen("rec_I.png"), h).move_to([4.3, 0.75, 0])
        lo = label(O, f"object wave {I('O')} (a point)", 24)
        lr = label(R, f"tilted reference wave {I('R')}", 24)
        li = label(Ih, f"recorded {I('I')} = |{I('O')} + {I('R')}|{SQ}", 24)
        plus = tx("+", 48).move_to([-2.15, 0.75, 0])
        eqs = tx("=", 48).move_to([2.15, 0.75, 0])
        self.play(FadeIn(O), run_time=0.5)
        self.add(lo)
        self.caption("Recording: the object's light interferes with a clean reference wave.")
        self.play(FadeIn(R), run_time=0.5)
        self.add(lr, plus)
        self.play(FadeIn(Ih), run_time=0.5)
        self.add(li, eqs)
        eq = tx(f"|{I('O')} + {I('R')}|{SQ} = |{I('O')}|{SQ} + |{I('R')}|{SQ} + "
                f"{col(I('O') + ' ' + I('R') + '*', AMBER)} + {col(I('O') + '*' + ' ' + I('R'), AMBER)}", 34
                ).move_to([0, -1.55, 0])
        self.add(eq)
        note = tx("the cross terms carry the phase of " + I("O"), 24, AMBER).next_to(eq, DOWN, buff=0.2)
        self.add(note)
        self.caption("The fringes store the object wave's phase. Light them again and the wave is reborn.")
        self.clear_all()

        # digital holography: SLM phase + propagation
        hp = img(gen("holo_phase.png"), 3.3).move_to([-4.35, 0.6, 0])
        lhp = label(hp, "hologram on the SLM (phase)", 24)
        frames = np.load(gen("focus_frames.npy"))
        z = ValueTracker(0.0)

        def fr():
            v = np.clip(z.get_value(), 0, 45) / 0.5
            i0 = int(np.floor(v))
            i1 = min(i0 + 1, len(frames) - 1)
            w = v - i0
            return ((1 - w) * frames[i0] + w * frames[i1]).astype(np.uint8)
        rec = arr_img(frames[0], 3.3).move_to([4.35, 0.6, 0])
        live(rec, fr)
        rb = framed(rec, DIM, 1.2)
        lrec = label(rec, "the light after a distance " + I("z"), 24)
        ar = arrow([-2.45, 0.6, 0], [2.45, 0.6, 0], color=WHITE)
        prop = tx(f"𝒫({I('u')}; {I('z')}) = ℱ{sup('−1')}{{ ℱ{{{I('u')}}} · exp({I('j')} {I('k')}{sub(I('z'))} {I('z')}) }}", 20
                  ).next_to(ar, UP, buff=0.18)
        prop2 = tx("angular spectrum method", 20, DIM).next_to(ar, DOWN, buff=0.18)
        note = tx("toy simulation: λ = 520 nm, 8 µm pixels", 16, DIM).next_to(hp, DOWN, buff=0.15)
        # z ruler
        nl = NumberLine(x_range=[0, 45, 5], length=7.0, color=DIM, include_numbers=False,
                        stroke_width=2).move_to([0.5, -2.35, 0])
        nlab = tx("z (mm)", 20, DIM).next_to(nl, LEFT, buff=0.25)
        ticks = VGroup(*[tx(str(v), 16, DIM).next_to(nl.n2p(v), DOWN, buff=0.12) for v in range(0, 50, 10)])
        m1 = Triangle(color=AMBER, fill_opacity=1).scale(0.09).rotate(PI).next_to(nl.n2p(12), UP, buff=0.04)
        m2 = Triangle(color=CYAN, fill_opacity=1).scale(0.09).rotate(PI).next_to(nl.n2p(36), UP, buff=0.04)
        s1 = tx("star", 18, AMBER).next_to(m1, UP, buff=0.05)
        s2 = tx("“3D”", 18, CYAN).next_to(m2, UP, buff=0.05)
        dot = always_redraw(lambda: Dot(nl.n2p(np.clip(z.get_value(), 0, 45)), radius=0.08, color=WHITE))
        zval = always_redraw(lambda: tx(f"z = {z.get_value():.1f} mm", 22).next_to(rec, DOWN, buff=0.15))
        self.play(FadeIn(hp), run_time=0.6)
        self.add(lhp, note)
        self.caption("Today a computer calculates the pattern (CGH) and a spatial light modulator (SLM) shows it.")
        self.play(GrowArrow(ar), FadeIn(rec), Create(rb), run_time=0.8)
        self.add(prop, prop2, lrec, nl, nlab, ticks, m1, m2, s1, s2, dot, zval)
        self.caption("Shine a laser on it and let the wave travel a distance z:")
        self.run(z, 45, 5.0, smooth)
        self.run(z, 12, 2.0, smooth)
        self.caption("at 12 mm the star snaps into focus, the text is still a blur ...", 0.5)
        self.run(z, 36, 2.5, smooth)
        self.caption("... at 36 mm, the text. One flat pattern holds a whole 3D scene.", 0.5)
        self.run(z, 24, 1.5, smooth)
        self.run(z, 36, 1.2, smooth)
        rec.clear_updaters()
        self.clear_all()


# ================================================================== S03 Why hard
class S03_WhyHard(Base):
    def construct(self):
        self.title_card("Part 3", "Why photoreal 3D holograms are hard")
        self.header("3 · Why photoreal 3D holograms are hard")
        slm_x = -6.1
        slm = Rectangle(width=0.18, height=4.0, fill_color="#546E7A", fill_opacity=1, stroke_width=0).move_to([slm_x, 0.4, 0])
        lslm = tx("SLM", 22, DIM).next_to(slm, DOWN, buff=0.15)
        pts = [(-3.6, 1.35, RED), (-2.4, 0.2, GREEN), (-4.3, -0.8, BLUE)]
        dots = VGroup(*[Dot([x, y, 0], radius=0.09, color=c) for x, y, c in pts])
        self.play(FadeIn(slm), run_time=0.4)
        self.add(lslm)
        self.play(FadeIn(dots), run_time=0.4)
        rings = []
        for (x, y, c) in pts:
            d = x - (slm_x + 0.12)
            for k in range(1, 4):
                r = d * k / 3
                half = 0.75
                rings.append(Arc(radius=r, start_angle=PI - half, angle=2 * half, arc_center=[x, y, 0],
                                 color=c, stroke_width=2.5, stroke_opacity=1.0 - 0.2 * k))
        self.play(LaggedStart(*[Create(r) for r in rings], lag_ratio=0.06), run_time=1.8)
        ph = img(gen("pts_holo.png"), 3.4).move_to([3.4, 0.45, 0])
        lph = label(ph, "sum of the three spherical waves (phase)", 22, DIM, DOWN)
        ar = arrow([-1.4, 0.45, 0], [1.5, 0.45, 0])
        self.caption("Point-based CGH: every point emits a spherical wave; add them all up at the SLM.")
        self.play(GrowArrow(ar), FadeIn(ph), run_time=0.8)
        self.add(lph)
        self.caption("But detailed scenes need an excessive number of points, and occlusion is hard to model.")
        self.clear_all()

        h = 2.75
        xs = [-4.5, 0, 4.5]
        names = ["point cloud", "mesh, per-face colours", "Gaussian splats"]
        files = ["fig5_lego_target_point.png", "fig5_lego_target_polygon.png", "fig5_lego_target_exact.png"]
        notes = [("holes; poor occlusion", RED), ("overly smooth; many triangles", RED),
                 ("photoreal, from a few photos", GREEN)]
        caps = ["Point clouds leave holes and handle occlusion and view-dependent effects poorly.",
                "Polygon CGH needs per-face colours, so details blur unless triangles multiply.",
                "Meanwhile, neural rendering solved photoreal 3D with Gaussian splatting. Can holography use it?"]
        for x, n, f, (nt, c), cp in zip(xs, names, files, notes, caps):
            m = img(f, h).move_to([x, 0.5, 0])
            self.play(FadeIn(m), run_time=0.4)
            self.add(label(m, n, 26), label(m, nt, 22, c, DOWN))
            self.caption(cp)
        src = tx("target renderings of each input representation (paper Fig. 5)", 18, DIM).move_to([0, -1.9, 0])
        self.add(src)
        self.clear_all()


# ================================================================== S04 Gaussians
class S04_Gaussians(Base):
    def construct(self):
        self.title_card("Part 4", "Scenes made of Gaussians")
        self.header("4 · Scenes made of Gaussians")
        gh = 3.6
        g = img(gen("g_one.png"), gh).move_to([-3.9, 0.75, 0])
        c = g.get_center()
        th = 0.45
        tu = np.array([np.cos(th), -np.sin(th), 0])
        tv = np.array([np.sin(th), np.cos(th), 0])
        L = gh / 400
        au = Arrow(c, c + tu * 95 * L, buff=0, color=AMBER, stroke_width=4, max_tip_length_to_length_ratio=0.15)
        av = Arrow(c, c + tv * 40 * L, buff=0, color=RED, stroke_width=4, max_tip_length_to_length_ratio=0.3)
        self.play(FadeIn(g), run_time=0.7)
        self.play(GrowArrow(au), GrowArrow(av), run_time=0.5)
        lu = tx(f"{B('t')}{sub(I('u'))}, {I('s')}{sub(I('u'))}", 24, AMBER).next_to(au.get_end(), RIGHT, buff=0.08)
        lv = tx(f"{B('t')}{sub(I('v'))}, {I('s')}{sub(I('v'))}", 24, RED).next_to(av.get_end(), UP, buff=0.08)
        lm = tx("<b>μ</b>", 24).next_to(c, DL, buff=0.08)
        self.add(lu, lv, lm, Dot(c, radius=0.05))
        params = tx(f"center <b>μ</b> ∈ ℝ{sup('3')}  ·  tangents {B('t')}{sub(I('u'))}, {B('t')}{sub(I('v'))}  ·  "
                    f"scales {I('s')}{sub(I('u'))}, {I('s')}{sub(I('v'))}", 22).move_to([-3.4, -1.55, 0])
        params2 = tx(f"opacity {I('o')}  ·  colour {B('c')}", 22).move_to([-3.4, -2.0, 0])
        self.add(params, params2)
        self.caption("2D Gaussian splatting (Huang et al. 2024): a scene is a cloud of flat, oriented Gaussian disks.")

        # compositing by layering (z-order = depth; later layers are nearer)
        layers = [img(gen(n), 2.6) for n in ("g_blue", "g_amber", "g_red")]
        spread = [[2.0, 1.9, 0], [3.65, 1.05, 0], [5.3, 0.2, 0]]
        for m, p in zip(layers, spread):
            m.move_to(p)
        tags = [tx("far", 20, DIM).next_to(layers[0], LEFT, buff=-0.3),
                tx("near", 20, DIM).next_to(layers[2], RIGHT, buff=-0.45)]
        self.play(LaggedStart(*[FadeIn(m) for m in layers], lag_ratio=0.3), run_time=1.0)
        self.add(*tags)
        self.caption("Each pixel blends the Gaussians it sees, sorted front to back.")
        self.remove(*tags)
        self.play(*[m.animate.move_to([3.65, 1.0, 0]) for m in layers], run_time=1.3)
        e1 = tx(EQ1, 30).move_to([3.65, -1.2, 0])
        e2 = tx(EQ2, 26).move_to([3.65, -1.9, 0])
        self.add(e1, e2)
        self.caption("Nearer Gaussians hide farther ones through the transmittance ∏(1 − α).")
        self.caption("Photoreal and explicit, but built for rays. What hologram does one Gaussian make?")
        self.clear_all()


# ================================================================== S05 Transform
class S05_Transform(Base):
    def construct(self):
        self.title_card("Part 5", "One Gaussian, one hologram, in closed form")
        self.header("5 · One Gaussian, one hologram, in closed form")
        # --- 1D Fourier pair, computed with numpy FFT
        sig = ValueTracker(1.0)
        n = 1024
        xg = np.linspace(-16, 16, n, endpoint=False)
        dxg = xg[1] - xg[0]
        kg = np.fft.fftshift(np.fft.fftfreq(n, dxg))

        def ft():
            g = np.exp(-xg ** 2 / (2 * sig.get_value() ** 2))
            return np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g)))) * dxg
        ax1 = Axes(x_range=[-4, 4, 1], y_range=[0, 1.2, 0.5], x_length=5.4, y_length=2.2, tips=False,
                   axis_config={"color": DIM, "stroke_width": 2}).move_to([-3.5, 1.05, 0])
        ax2 = Axes(x_range=[-1.2, 1.2, 0.4], y_range=[0, 4.3, 1], x_length=5.4, y_length=2.2, tips=False,
                   axis_config={"color": DIM, "stroke_width": 2}).move_to([3.5, 1.05, 0])
        c1 = always_redraw(lambda: ax1.plot(lambda x: np.exp(-x ** 2 / (2 * sig.get_value() ** 2)),
                                            x_range=[-4, 4, 0.02], color=CYAN, stroke_width=4))

        def c2f():
            G = ft()
            m = (kg >= -1.2) & (kg <= 1.2)
            return ax2.plot_line_graph(kg[m], np.minimum(G[m], 4.3), line_color=AMBER, add_vertex_dots=False,
                                       stroke_width=4)
        c2 = always_redraw(c2f)
        l1 = tx("space: a Gaussian", 24).next_to(ax1, UP, buff=0.15)
        l2 = tx("frequency: its Fourier transform (numpy FFT)", 24).next_to(ax2, UP, buff=0.15)
        self.play(Create(ax1), Create(ax2), run_time=0.6)
        self.add(c1, c2, l1, l2)
        eq7 = tx(f"ℱ{{ exp(−({I('x')}{SQ} + {I('y')}{SQ})/2) }} = 2π exp(−2π{SQ} {KB}{sup('T')}{KB})", 32).move_to([0, -1.05, 0])
        self.add(eq7)
        self.caption("Key fact: the Fourier transform of a Gaussian is again a Gaussian.")
        self.run(sig, 0.45, 1.8, smooth)
        self.caption("Squeeze it in space and it spreads in frequency.")
        self.run(sig, 1.7, 2.2, smooth)
        self.run(sig, 1.0, 1.2, smooth)
        c1.clear_updaters()
        c2.clear_updaters()
        self.clear_all()

        # --- 2D: scale, rotate, shift (live FFT on 256², central crops; spectrum hue = phase)
        N = 256
        th = ValueTracker(0.0)
        asp = ValueTracker(1.0)
        mx = ValueTracker(0.0)
        yy, xx = np.mgrid[0:N, 0:N].astype(float)
        xx, yy = xx - N / 2, N / 2 - yy
        s0 = 9.0

        def space_full():
            c, s = np.cos(th.get_value()), np.sin(th.get_value())
            X, Y = xx - mx.get_value(), yy - 0.5 * mx.get_value()
            u, v = c * X + s * Y, -s * X + c * Y
            return np.exp(-0.5 * ((u / (s0 * asp.get_value())) ** 2 + (v / s0) ** 2))

        def space():
            return grey(space_full()[80:176, 80:176])

        def spec():
            F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(space_full())))[104:152, 104:152]
            return domain_rgb(F, 0.7, amax=np.abs(F).max())
        sp = arr_img(space(), 3.0).move_to([-3.4, 0.95, 0])
        fq = arr_img(spec(), 3.0).move_to([3.4, 0.95, 0])
        live(sp, space)
        live(fq, spec)
        sb, fb = framed(sp, DIM, 1.2), framed(fq, DIM, 1.2)
        ls = label(sp, "2D Gaussian (space)", 24)
        lf = label(fq, "its spectrum (hue = phase)", 24)
        self.play(FadeIn(sp), FadeIn(fq), Create(sb), Create(fb), run_time=0.7)
        self.add(ls, lf)
        ar = arrow([-1.6, 0.95, 0], [1.6, 0.95, 0])
        arl = tx("ℱ", 30).next_to(ar, UP, buff=0.1)
        self.add(ar, arl)
        ea = tx(f"{XB} = {I('R')} {I('S')} {XB}{sub('c')} + <b>μ</b>", 30).move_to([-3.4, -1.2, 0])
        eb = tx(f"Σ = {I('R')} {I('S')} {I('S')}{sup('T')} {I('R')}{sup('T')}", 30).move_to([3.4, -1.2, 0])
        self.add(ea, eb)
        self.caption(f"Stretch it with a scale {I('S')}: the spectrum shrinks the other way (affine theorem).")
        self.run(asp, 2.2, 2.0, smooth)
        self.caption(f"Rotate it with {I('R')}: the spectrum rotates along with it.")
        self.run(th, PI / 3, 2.0, smooth)
        self.caption("Shift it by <b>μ</b>: the shape stays, the spectrum just gains a phase ramp.")
        self.run(mx, 22, 2.2, smooth)
        self.run(mx, -14, 2.0, smooth)
        sp.clear_updaters()
        fq.clear_updaters()
        self.clear_all()

        # --- tilt: side view, computed Huygens field (keeps moving with the scene clock)
        U = np.load(gen("tilted_U.npy"))
        disk = np.load(gen("tilted_disk.npy"))
        clk = self.clock

        def side():
            re = np.real(U * np.exp(-1j * 2 * PI * 0.8 * clk.get_value()))
            a = WLUT2[np.clip((re / 1.05 * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)].astype(np.float32)
            a = a * (1 - disk[..., None]) + 255 * disk[..., None]
            return a.astype(np.uint8)
        sv = arr_img(side(), w=9.6).move_to([-0.4, 0.45, 0])
        live(sv, side)
        slm = Rectangle(width=0.16, height=sv.height, fill_color="#546E7A", fill_opacity=1, stroke_width=0
                        ).next_to(sv, RIGHT, buff=0.05)
        lslm = tx("SLM", 22, DIM).next_to(slm, UP, buff=0.12)
        lpw = tx("plane wave", 22).next_to(sv, UP, buff=0.12).align_to(sv, LEFT)
        lg = tx("tilted 2D Gaussian", 22).next_to(sv, DOWN, buff=0.12).align_to(sv, LEFT).shift(RIGHT * 0.6)
        lr = tx("its wave, computed by summing wavelets", 22, DIM).next_to(sv, DOWN, buff=0.12).align_to(slm, RIGHT)
        self.play(FadeIn(sv), FadeIn(slm), run_time=0.7)
        self.add(lslm, lpw, lg, lr)
        self.caption(f"Splats are tilted. Tilting a plane rotates its wave vectors: "
                     f"{KB}{sub('o')} = {I('R')}{sup('−1')}{KB}.")
        self.caption("The plane-wave illumination just shifts the spectrum. Every step stays analytic.")
        sv.clear_updaters()
        self.clear_all()

        # --- the Gaussian Wave Splatting equation, built term by term
        parts = [
            (f"{UHAT}{sub(I('i'))}({KB}) = 2π ", WHITE, None),
            ("det(" + B("J") + ")", AMBER, "tilt Jacobian"),
            (" det(" + I("S") + sub(I("i")) + ")", GREEN, "scale"),
            (f" exp(−2π{SQ} {KB}{sup('T')} Σ{sub(I('i'))} {KB})", CYAN, "Gaussian shape"),
            (f" exp({I('j')} {KB}·<b>μ</b>{sub(I('i'))})", PURPLE, "position and depth"),
        ]
        EY = 2.2
        full = tx("".join(col(p, c) for p, c, _ in parts), 40).move_to([0, EY, 0])
        x0 = full.get_left()[0]
        shown, prev_w = None, 0.0
        for k in range(1, len(parts) + 1):
            m = tx("".join(col(p, c) for p, c, _ in parts[:k]), 40)
            m.move_to([x0, EY, 0], aligned_edge=LEFT).set_y(EY)
            if shown is not None:
                self.remove(shown)
            self.add(m)
            shown = m
            _, c, name = parts[k - 1]
            if name:
                lab = tx(name, 20, c).move_to([x0 + (prev_w + m.width) / 2, EY - 0.62, 0])
                self.add(lab)
                self.wait(0.7)
            prev_w = m.width
        self.caption("The Gaussian Wave Splatting equation: the exact spectrum of any 2D Gaussian at the SLM, "
                     "no per-primitive FFT or resampling needed.")
        gs = img(gen("gws_slm.png"), 2.7).move_to([-3.0, -0.95, 0])
        gr = img(gen("gws_rec.png"), 2.7).move_to([3.0, -0.95, 0])
        gsf, grf = framed(gs, DIM, 1.2), framed(gr, DIM, 1.2)
        lgs = tx("its field at the SLM (hue = phase)", 20, DIM).next_to(gs, UP, buff=0.12)
        lgr = tx("propagated back: the splat (4× zoom)", 20, DIM).next_to(gr, UP, buff=0.12)
        ar2 = arrow(gs.get_right() + RIGHT * 0.25, gr.get_left() + LEFT * 0.25)
        la2 = tx(f"𝒫(·; {I('z')}{sub(I('i'))})", 22).next_to(ar2, UP, buff=0.1)
        self.play(FadeIn(gs), Create(gsf), run_time=0.6)
        self.add(lgs)
        self.play(GrowArrow(ar2), FadeIn(gr), Create(grf), run_time=0.8)
        self.add(lgr, la2)
        self.caption("Computed from the formula: one tilted splat's wavefront, and the splat it refocuses into.")
        self.clear_all()


# ================================================================== S06 Occlusion
class S06_Occlusion(Base):
    def construct(self):
        self.title_card("Part 6", "Occlusion: alpha wave blending")
        self.header("6 · Occlusion: alpha wave blending")
        e1 = tx("rays:  " + EQ1, 28, DIM).move_to([0, 2.75, 0])
        self.add(e1)
        self.caption("Splatting handles occlusion by alpha blending along each ray.")
        e5 = tx(EQ5, 30).move_to([0, 1.95, 0])
        e6 = tx(EQ6, 30).move_to([0, 1.25, 0])
        self.add(e5, e6)
        self.caption("Waves: splat each Gaussian's wavefront to the SLM, front to back, "
                     "attenuated by the transmittance T of what lies in front.")
        h = 2.3
        xs = [-4.95, -1.65, 1.65, 4.95]
        keys = ["naive", "silhouette", "alpha", "target"]
        names = ["naïve sum", "binary silhouette", "alpha wave blending", "ray-traced target"]
        colors = [RED, RED, GREEN, WHITE]
        for x, k, n, c in zip(xs, keys, names, colors):
            m = img(gen(f"occ_{k}.png"), h).move_to([x, -0.75, 0])
            self.play(FadeIn(m), run_time=0.3)
            self.add(label(m, n, 22, c))
        toy = tx("toy simulation: red Gaussian 4 mm in front of a blue one, each reconstructed in focus", 18, DIM
                 ).move_to([0, -2.1, 0])
        self.add(toy)
        self.caption("No occlusion: the rear Gaussian shines through. A binary mask cuts a hard, wrong edge.")
        self.caption("Alpha wave blending matches the target, smooth falloff included.")
        self.caption("Drop propagation and phase, and it reduces exactly to ray-based alpha blending.")
        self.clear_all()

        xs = [-4.95, -1.65, 1.65, 4.95]
        files = ["fig3_naive.png", "fig3_silhouette.png", "fig3_ours.png", "fig3_target.png"]
        names = ["naïve wave blending", "silhouette (binary mask)", "alpha wave blending", "target rendering"]
        colors = [RED, RED, GREEN, WHITE]
        for x, f, n, c in zip(xs, files, names, colors):
            m = img(f, w=3.15).move_to([x, 0.7, 0])
            self.play(FadeIn(m), run_time=0.3)
            self.add(label(m, n, 20, c))
        src = tx("simulated reconstructions, Mip-NeRF 360 “stump” (paper Fig. 3)", 18, DIM).move_to([0, -0.85, 0])
        self.add(src)
        self.caption("On a real scene, naïve blending oversaturates and silhouettes leave artifacts; "
                     "alpha wave blending matches the target.")
        self.clear_all()


# ================================================================== S07 Fast
class S07_Fast(Base):
    def construct(self):
        self.title_card("Part 7", "Making it fast: splat in the Fourier domain")
        self.header("7 · Making it fast: splat in the Fourier domain")
        e11 = tx(EQ11, 30).move_to([0, 2.6, 0])
        self.add(e11)
        warn = tx("a convolution: every frequency depends on all the others", 22, RED).next_to(e11, DOWN, buff=0.18)
        self.add(warn)
        self.caption("In the Fourier domain the transmittance becomes a convolution, which is slow.")
        e12 = tx(f"{I('T')}{sub(I('i'))} ≈ 1   ⇒   " + EQ12, 30).move_to([0, 1.35, 0])
        self.add(e12)
        self.caption(f"Fast GWS drops it: each frequency {KB} becomes an independent sum over Gaussians.")

        n = 10
        size = 2.5

        def grid(cx):
            cs = size / n
            return VGroup(*[Square(cs * 0.86).set_stroke(DIM, 1).set_fill(BLUE, 0.0).move_to(
                [cx + (j + 0.5) * cs - size / 2, -1.1 + size / 2 - (i + 0.5) * cs, 0]) for i in range(n) for j in range(n)])
        ge = grid(-3.6)
        gf = grid(3.6)
        le = tx("exact: one Gaussian at a time", 22).next_to(ge, UP, buff=0.15)
        lf = tx("fast: one CUDA thread per frequency", 22).next_to(gf, UP, buff=0.15)
        self.play(FadeIn(ge), FadeIn(gf), run_time=0.5)
        self.add(le, lf)
        cnt = ValueTracker(0)
        ctext = always_redraw(lambda: tx(f"Gaussian {int(cnt.get_value())} of many", 20, DIM).next_to(ge, DOWN, buff=0.12))
        self.add(ctext)
        self.play(cnt.animate.set_value(6), *[s.animate.set_fill(BLUE, 0.35) for s in ge], run_time=2.5,
                  rate_func=linear)
        self.play(*[s.animate.set_fill(AMBER, 0.85) for s in gf], run_time=0.6)
        self.caption("Embarrassingly parallel, so it maps straight onto custom CUDA kernels.")
        ctext.clear_updaters()
        self.clear_all()

        e13 = tx(EQ13, 32).move_to([-3.4, 2.5, 0])
        l13 = tx("order-invariant transparency (McGuire &amp; Bavoil 2013)", 20, DIM).next_to(e13, DOWN, buff=0.15)
        evd = tx(EQVD, 32).move_to([3.4, 2.5, 0])
        lvd = tx("view-dependent opacity (spherical harmonics)", 20, DIM).next_to(evd, DOWN, buff=0.15)
        a = img("fig2_naive_oit.png", w=5.4).move_to([-3.4, -0.35, 0])
        b = img("fig2_vd_oit.png", w=5.4).move_to([3.4, -0.35, 0])
        self.add(e13, l13)
        self.play(FadeIn(a), run_time=0.4)
        self.add(label(a, "naïve OIT", 22, RED, DOWN))
        self.caption("Its spatial twin is order-invariant transparency: colours leak, objects vanish.")
        self.add(evd, lvd)
        self.play(FadeIn(b), run_time=0.4)
        self.add(label(b, "OIT with view-dependent opacity (paper Fig. 2)", 22, GREEN, DOWN))
        self.caption("Fix: train the Gaussians (2DGS-OIT) with opacity that depends on the view direction.")
        self.clear_all()

        rows = [("method", "resolution", "primitives", "runtime"),
                ("points (Chen &amp; Wilkinson 2009)", "1024 × 1280", "15k points", "2.47 s"),
                ("polygons (Matsushima et al. 2014)", "2000 × 2000", "15k triangles", "13.5 s"),
                ("GWS exact", "1024 × 1280", "15k Gaussians", "180 s"),
                ("GWS fast", "1024 × 1280", "15k Gaussians", "6.84 s"),
                ("GWS exact", "2000 × 2000", "15k Gaussians", "470 s"),
                ("GWS fast", "2000 × 2000", "15k Gaussians", "18.06 s")]
        xs = [-3.9, 0.3, 2.85, 5.1]
        tbl = VGroup()
        for r, row in enumerate(rows):
            for c, (s, x) in enumerate(zip(row, xs)):
                colr = DIM if r == 0 else (AMBER if "fast" in row[0] else WHITE)
                tbl.add(tx(s, 24, colr).move_to([x, 2.3 - 0.58 * r, 0]))
        hl = Line([-6.4, 2.0, 0], [6.4, 2.0, 0], color=DIM, stroke_width=1.5)
        self.add(tbl, hl)
        src = tx("paper Table 2 (baseline runtimes as reported in their papers)", 18, DIM).move_to([0, -1.9, 0])
        self.add(src)
        self.caption("Table 2: 6.84 s for fast GWS vs 180 s for exact (“nearly 30×”), on par with prior CGH "
                     "at far higher quality.")
        self.clear_all()


# ================================================================== S08 Results
class S08_Results(Base):
    def construct(self):
        self.title_card("Part 8", "Results on a real holographic display")
        self.header("8 · Results on a real holographic display")
        ph = img("setup.png", 4.6).move_to([-3.6, 0.35, 0])
        self.play(FadeIn(ph), run_time=0.6)
        specs = [
            "lasers: 638 / 520 / 488 nm (FISBA READYBeam)",
            "SLM: Holoeye PLUTO, phase-only LCoS",
            "1920 × 1080 pixels, 8 µm pitch",
            "camera: FLIR Grasshopper3",
            "complex field → phase-only via double-phase coding",
            "hologram depth range ≈ 1 cm",
        ]
        sg = VGroup(*[tx(s, 24) for s in specs]).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([2.95, 0.35, 0])
        self.add(sg)
        self.caption("Benchtop prototype: colour is captured one laser at a time and combined.")
        self.clear_all()

        w = 2.95
        xs = [-3.95, -0.85, 2.25, 5.35]
        names = ["point-based", "polygon-based", "fast GWS", "exact GWS"]
        keys = ["point", "polygon", "fast", "exact"]
        colors = [WHITE, WHITE, AMBER, AMBER]
        ims = []
        for x, n, k, c in zip(xs, names, keys, colors):
            m = img(f"fig5_lego_captured_{k}.png", w=w).move_to([x, 1.45, 0])
            ims.append(m)
            self.play(FadeIn(m), run_time=0.3)
            self.add(label(m, n, 24, c))
        src = tx("experimentally captured, front focal slice (paper Fig. 5)", 18, DIM).next_to(ims[0], DOWN, buff=0.12)
        src.set_x((xs[0] + xs[-1]) / 2)
        self.add(src)
        self.caption("Captured holograms: points leave holes, polygons blur details, GWS stays sharp.")
        psnr_b = ["18.34", "20.79", "26.55", "27.30"]
        psnr_m = ["9.45", "15.77", "20.70", "22.37"]
        hdr = tx("simulated PSNR (dB), paper Table 1", 20, DIM).move_to([(xs[0] + xs[-1]) / 2, -0.35, 0])
        vals = VGroup()
        for x, a, b2, c in zip(xs, psnr_b, psnr_m, colors):
            vals.add(tx(a, 28, c).move_to([x, -0.95, 0]), tx(b2, 28, c).move_to([x, -1.6, 0]))
        rb = tx("Blender", 22, DIM).next_to(vals[0], LEFT, buff=0.55)
        rm = tx("Mip-NeRF 360", 22, DIM).next_to(vals[1], LEFT, buff=0.55)
        rb.align_to(rm, RIGHT)
        self.add(hdr, vals, rb, rm)
        self.caption("In simulation GWS wins by 5–11 dB PSNR over point and polygon CGH.")
        self.clear_all()

        # close-up comparison (materials scene)
        a = img("fig5_mat_captured_point.png", h=3.9).move_to([-3.4, 0.35, 0])
        b = img("fig5_mat_captured_exact.png", h=3.9).move_to([3.4, 0.35, 0])
        self.play(FadeIn(a), FadeIn(b), run_time=0.6)
        self.add(label(a, "point-based (captured)", 24), label(b, "exact GWS (captured)", 24, AMBER))
        self.add(tx("Blender “materials”, front focal slice (paper Fig. 5)", 18, DIM).move_to([0, -1.85, 0]))
        self.caption("Up close: speckled, dim points versus clean, glossy spheres.")
        self.clear_all()

        # captured focal sweep + novel views
        big = img("fig6_v0_rear.png", 4.1).move_to([-1.9, 0.3, 0])
        views = [img(f"fig6_v{i}_mid.png", 1.45).move_to([4.6, 1.85 - 1.62 * i, 0]) for i in range(3)]
        self.play(FadeIn(big), *[FadeIn(v) for v in views], run_time=0.7)
        vl = tx("viewpoints", 20, DIM).next_to(views[0], UP, buff=0.12)
        self.add(vl)
        sel = SurroundingRectangle(views[0], buff=0.04, color=AMBER, stroke_width=3)
        self.add(sel)
        flab = tx("rear focus", 26, AMBER).next_to(big, DOWN, buff=0.18)
        self.add(flab)
        src = tx("experimentally captured, exact GWS, Mip-NeRF 360 “garden” (paper Fig. 6)", 18, DIM
                 ).next_to(big, UP, buff=0.15)
        self.add(src)
        self.caption("A captured focal stack of one GWS hologram: the camera refocuses through the scene.")
        order = [(0, ["mid", "front"]), (1, ["front", "rear"]), (2, ["rear", "front"])]
        for vi, seq in order:
            if vi > 0:
                self.play(sel.animate.move_to(views[vi]), run_time=0.35)
                if vi == 1:
                    self.caption("Same Gaussian model, new viewpoints: every hologram stays 3D and refocusable.")
            for s in seq:
                nxt = img(f"fig6_v{vi}_{s}.png", 4.1).move_to(big)
                self.play(FadeIn(nxt), run_time=0.7)
                self.remove(big)
                big = nxt
                self.remove(flab)
                flab = tx(f"{'middle' if s == 'mid' else s} focus", 26, AMBER).next_to(big, DOWN, buff=0.18)
                self.add(flab)
                self.wait(0.5)
        self.clear_all()


# ================================================================== S09 Outlook
class S09_Outlook(Base):
    def construct(self):
        self.title_card("Part 9", "Limitations and outlook")
        self.header("9 · Limitations and outlook")
        items = [
            ("Smooth-phase holograms: sharp in focus, but unnatural defocus blur, small eyebox",
             "→ random-phase GWS"),
            ("Fast GWS is not yet real time", "→ more closed-form tricks, neural networks, custom hardware"),
            ("Small étendue of today's SLMs: view-dependent effects only up to 0th-order SH",
             "→ étendue-expanded, partially coherent displays"),
        ]
        g = VGroup()
        for a, b in items:
            g.add(VGroup(tx(a, 26), tx(b, 24, AMBER)).arrange(DOWN, aligned_edge=LEFT, buff=0.12))
        g.arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        if g.width > 12.8:
            g.scale_to_fit_width(12.8)
        g.move_to([0, 0.6, 0])
        for it in g:
            self.add(it)
            self.wait(1.0)
        self.caption("A first bridge between neural graphics and holographic displays.")
        self.clear_all()

        summ = [
            "1   Gaussian → hologram in closed form (Gaussian Wave Splatting)",
            "2   Occlusion in the wave domain (alpha wave blending)",
            "3   Order-invariant Fourier-domain approximation, custom CUDA kernels",
        ]
        sg = VGroup(*[tx(s, 28) for s in summ]).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([0, 1.2, 0])
        res = tx("5–11 dB PSNR over point / polygon CGH  ·  fast variant nearly 30× faster than exact", 24, AMBER
                 ).move_to([0, -0.55, 0])
        self.add(sg)
        self.wait(1.2)
        self.add(res)
        self.caption("From a few photos to a refocusable, photoreal hologram.")
        self.clear_all()
        self.remove(self.hdr)
        t1 = tx("Gaussian Wave Splatting", 48, bold=True).move_to([0, 2.1, 0])
        t1b = tx("for Computer-Generated Holography", 32).move_to([0, 1.4, 0])
        t2 = tx("arXiv 2505.06582  ·  bchao1.github.io/gaussian-wave-splatting", 28).move_to([0, 0.45, 0])
        t3 = tx("code: github.com/computational-imaging/hsplat", 26, DIM).move_to([0, -0.15, 0])
        t4 = tx("Suyeon Choi*, Brian Chao*, Jacqueline Yang, Manu Gopakumar, Gordon Wetzstein", 24, DIM).move_to([0, -1.2, 0])
        t5 = tx("Stanford University  ·  ACM TOG 44(4), SIGGRAPH 2025", 22, DIM).move_to([0, -1.65, 0])
        self.add(t1, t1b, t2, t3, t4, t5)
        self.wait(4.0)
