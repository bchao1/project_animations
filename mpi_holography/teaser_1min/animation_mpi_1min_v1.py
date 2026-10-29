"""Fast Wave-optics Rendering of Multiplane Images for 3D Holographic Displays — 1-minute teaser, v1.

Source: Chao, Seyb, Matsuda, Cossairt, Zhou, Lanman, Wetzstein, Kuo, Jang, arXiv 2607.19731 (2026).
Condensed from the 5-min teaser (../animation_mpi_v3.py). Helpers are copied from it (not imported); figure crops,
homepage-video frames and toy wave-optics caches are read from ../data (built by the 5-min script; read-only here).
Scenes, in order: T1_Hook, T2_Method, T3_Payoff.
"""
from manim import *
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib
import scipy.fft as sfft
import fcntl
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/brianchc/project_animations/mpi_holography/data"      # parent project's data (read-only)
GEN = os.path.join(DATA, "generated")
CLIPS = os.path.join(DATA, "clips")
ASSET_FLAG = "done_v1"

config.background_color = BLACK

# ------------------------------------------------------------------ style (same as random_phase_wave_splatting)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × − → ≈ ∑ ∏ ℱ 𝒫 𝔼 ∠ △ …
SYMFONT = "DejaVu Sans"                 # ✗ is in neither
DIM = "#8A8A8A"
RED = "#FF5252"
GREEN = "#69F0AE"
BLUE = "#448AFF"
AMBER = "#FFCA28"
CYAN = "#4DD0E1"
PURPLE = "#E040FB"
ORANGE = "#FF9E40"
# method colours used throughout: MPI (ours) green, RGBD red, RPWS orange
C_MPI, C_RGBD, C_RPWS = GREEN, RED, ORANGE

HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈ℝ≪≤≥↓↔∑∏∬∫ℱ𝒫𝒢𝒩ℋ𝒪𝒰𝔼∗≠∞∂∘⊤●⇒∠✓△∝⟵←")
SYMCH = set("✗")


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


def supsub(base, s_):
    """Superscript 'base' followed by a subscript that stays inside the superscript (Pango rise is absolute)."""
    return (f'<span rise="4500"><small><small>{base}</small></small></span>'
            f'<span rise="2600"><small><small><small>{s_}</small></small></small></span>')


def col(x, c):
    return f'<span foreground="{c}">{x}</span>'


def tx(s, size=30, color=WHITE, bold=False, italic=False):
    if bold:
        s = f"<b>{s}</b>"
    if italic:
        s = f"<i>{s}</i>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


def fit(m, w=13.0):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


def src_note(s):
    """Source note, top right (header line)."""
    return tx(s, 16, DIM).to_edge(RIGHT, buff=0.35).set_y(HEADER_Y)


def I(v):
    return f"<i>{v}</i>"


def B(v):
    return f"<b>{v}</b>"


SQ = sup("2")
AI, CI, UI = "α" + sub(I("i")), I("c") + sub(I("i")), I("u") + sub(I("i"))
PHIT = "φ" + sub(I("i")) + sup("(" + I("t") + ")")
EQ_MPI = (f"{I('u')}{sub(I('i') + '−1')} = 𝒫( √(1 − {AI}) {UI} + √{CI} √{AI} "
          f"{I('e')}{supsub(I('i') + 'φ', I('i'))} ; Δ{I('z')} )")
EQ_TM = f"{I('I')} = (1/{I('T')}) ∑{sub(I('t'))} |𝒪( {I('u')}{sub('SLM')}{sup('(' + I('t') + ')')} )|{SQ}"
EQ_ASM = (f"𝒫({I('u')}; {I('z')}) = ℱ{sup('−1')}{{ ℱ{{{I('u')}}} · {I('H')} }},   "
          f"{I('H')} = exp({I('i')}2π{I('f')}{sub(I('z'))}{I('z')})")
EQ_FZ = f"{I('f')}{sub(I('z'))} = √(1/λ{SQ} − {I('f')}{sub(I('x'))}{SQ} − {I('f')}{sub(I('y'))}{SQ})"
EQ_OVER = f"{I('C')} ← α {I('c')} + (1 − α) {I('C')}"

# ------------------------------------------------------------------ optics (numpy)
LAMS = np.array([638e-9, 520e-9, 450e-9])   # toy RGB laser lines
DX = 8e-6                                    # SLM pixel pitch of the paper's prototype (HOLOEYE Pluto-2.1, 8 µm)
N = 512                                      # simulation grid (4.1 mm)
CROP = slice(80, 432)                        # displayed 352 × 352 window
NC = 352
ZL = np.array([4e-3, 9e-3, 14e-3, 20e-3])    # toy MPI: layer 1 (front) … layer 4 (back), distance from the SLM
TFR = 24                                     # time-multiplexed frames (paper: T = 24)
TH_MAX = LAMS[2] / (2 * DX)                  # blue's largest diffraction angle: common angular band of the toy eye
R_EYE = 0.5                                  # eye pupil radius (fraction of TH_MAX) for focal stacks
R_SUB = 0.25                                 # sub-aperture radius for light-field views
S_MAX = 0.7                                  # largest pupil offset (fraction of TH_MAX)
NV = 29                                      # pupil positions
ZS = np.linspace(2e-3, 24e-3, 45)            # focal sweep
ZP = np.linspace(0, 20e-3, 41)               # single-layer propagation sweep
Z_LF = ZL[0]                                 # light-field views focused on the front layer
GAMMA = 1 / 2.0
_f = np.fft.fftfreq(N, DX)
FX, FY = np.meshgrid(_f, _f)


def fft2(a):
    return sfft.fft2(a, workers=16)


def ifft2(a):
    return sfft.ifft2(a, workers=16)


def Hz(z, lam):
    fz = np.sqrt(np.maximum(1 / lam ** 2 - FX ** 2 - FY ** 2, 0))
    return np.exp(1j * 2 * np.pi * fz * z)


def asm(u, z, lam):
    """Angular spectrum method: P(u; z) = F^-1{ F{u} · exp(i 2π f_z z) }."""
    return ifft2(fft2(u) * Hz(z, lam))


def pupil_ang(lam, cx=0.0, cy=0.0, r=R_EYE):
    """Eye pupil in angle space (fractions of TH_MAX): the same physical pupil for every colour."""
    tx_, ty_ = lam * FX / TH_MAX, lam * FY / TH_MAX
    return (((tx_ - cx) ** 2 + (ty_ - cy) ** 2) <= r ** 2).astype(float)


def pupil_ref(r):
    """Mean intensity of a unit random-phase field seen through a pupil of radius r (flat spectrum) per colour."""
    return np.pi * (r * TH_MAX / LAMS) ** 2 * DX ** 2


def rphase(rng):
    return np.exp(1j * rng.uniform(-np.pi, np.pi, (N, N)))


def poly_mask(pts, blur=1.5):
    im = Image.new("L", (N, N), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=255)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(blur))) / 255.0


def star_pts(c, r0, r1, n=5, rot=-np.pi / 2):
    return [(c[0] + (r1 if i % 2 == 0 else r0) * np.cos(rot + i * np.pi / n),
             c[1] + (r1 if i % 2 == 0 else r0) * np.sin(rot + i * np.pi / n)) for i in range(2 * n)]


def toy_mpi():
    """Four RGBA layers (colour = target intensity). 1 star (front), 2 triangle, 3 disc, 4 checkerboard wall (back)."""
    y, x = np.mgrid[0:N, 0:N].astype(float)
    cx = cy = N / 2
    full = lambda r, g, b: np.stack([np.full((N, N), r), np.full((N, N), g), np.full((N, N), b)], -1)
    chk = (((x - cx) // 28 + (y - cy) // 28) % 2)
    win = ((np.abs(x - cx) < 236) & (np.abs(y - cy) < 236)).astype(np.uint8) * 255
    a4 = np.asarray(Image.fromarray(win).filter(ImageFilter.GaussianBlur(2))) / 255.0
    c4 = np.stack([0.05 + 0.16 * chk, 0.10 + 0.26 * chk, 0.28 + 0.40 * chk], -1)
    r = np.hypot(x - (cx + 55), y - (cy - 50))
    a3 = np.clip((95 - r) / 2.0 + 0.5, 0, 1)
    c3 = full(1.0, 0.78, 0.12)
    a2 = poly_mask([(cx - 150, cy + 120), (cx + 60, cy + 120), (cx - 45, cy - 70)])
    c2 = full(0.18, 0.80, 0.36)
    a1 = 0.97 * poly_mask(star_pts((cx + 70, cy + 70), 30, 72), blur=2.0)
    c1 = full(1.0, 0.34, 0.24)
    return [(c1, a1), (c2, a2), (c3, a3), (c4, a4)]


def over(layers, shifts=None):
    """Ray-optics 'over' compositing, back to front (on intensity); optional per-layer horizontal shift (px)."""
    C = np.zeros((N, N, 3))
    for k in range(len(layers) - 1, -1, -1):
        c, a = layers[k]
        if shifts is not None and shifts[k]:
            c, a = np.roll(c, shifts[k], 1), np.roll(a, shifts[k], 1)
        C = a[..., None] * c + (1 - a[..., None]) * C
    return C


def rgbd_from(layers):
    """RGBD: composited image + depth of the front-most layer with α > 0.5; slices = hard depth masks."""
    C = over(layers)
    d = np.full((N, N), len(layers) - 1)
    for k in range(len(layers) - 1, -1, -1):
        d = np.where(layers[k][1] > 0.5, k, d)
    return C, d


def mpi_slm(layers, ch, rng, keep=None):
    """Paper Eq. 2: back to front, u ← P( √(1 − α) u + √c √α e^{iφ} ; Δz ), down to the SLM (z = 0)."""
    lam = LAMS[ch]
    u = np.zeros((N, N), complex)
    for k in range(len(layers) - 1, -1, -1):
        c, a = layers[k]
        u = np.sqrt(1 - a) * u + np.sqrt(c[..., ch] * a) * rphase(rng)
        if keep is not None:
            keep.append(u)
        u = asm(u, -(ZL[k] - (ZL[k - 1] if k > 0 else 0.0)), lam)
    return u


def rgbd_slm(C, d, ch, rng):
    """Conventional layer-based CGH on an RGBD image: depth-sliced random-phase layers, propagated and summed."""
    lam = LAMS[ch]
    u = np.zeros((N, N), complex)
    for k in range(len(ZL)):
        m = (d == k).astype(float)
        u = u + asm(np.sqrt(C[..., ch]) * m * rphase(rng), -ZL[k], lam)
    return u


def domain_rgb(u, gamma=0.7, amax=None, sat=0.85):
    a = np.abs(u)
    amax = amax or (np.percentile(a, 99.7) + 1e-12)
    v = np.clip(a / amax, 0, 1) ** gamma
    h = (np.angle(u) + np.pi) / (2 * np.pi)
    hsv = np.stack([h, np.full_like(h, sat), v], -1)
    return (matplotlib.colors.hsv_to_rgb(hsv) * 255).astype(np.uint8)


def cmap_rgb(a, cmap="inferno"):
    return (matplotlib.colormaps[cmap](np.clip(a, 0, 1))[..., :3] * 255).astype(np.uint8)


def to8(a, gamma=GAMMA):
    return (np.clip(a, 0, 1) ** gamma * 255).astype(np.uint8)


def rgba(a):
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
    return np.ascontiguousarray(a.astype(np.uint8))


# ------------------------------------------------------------------ assets
def gen(name):
    return os.path.join(GEN, name)


SHX, SHY = 0.5, 0.45      # pseudo-3D layer sheet: horizontal squeeze, vertical shear per unit width


def shear_rgba(im):
    """Squeeze horizontally and shear vertically (y rises to the right): a layer seen at an angle."""
    w0, h0 = im.size
    w = int(round(w0 * SHX))
    dy = int(round(SHY * w))
    out_h = h0 + dy
    # output (X, Y) -> input (x, y): x = X / SHX, y = Y - dy + SHY * X  (right edge raised by dy)
    return im.transform((w, out_h), Image.AFFINE, (1 / SHX, 0, 0, SHY, 1, -dy), resample=Image.BICUBIC)


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    if os.path.exists(gen(ASSET_FLAG)):
        return
    with open(gen("lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(gen(ASSET_FLAG)):
            return
        rng = np.random.default_rng(2026)
        layers = toy_mpi()
        C, d = rgbd_from(layers)
        cr = lambda a: a[CROP, CROP]
        # ---- flat images: layers, composite, depth, RGBD slices
        Image.fromarray(to8(cr(C))).save(gen("composite.png"))
        dcol = np.array([[1.0, 0.34, 0.24], [0.18, 0.80, 0.36], [1.0, 0.78, 0.12], [0.30, 0.45, 1.0]])
        dm = cmap_rgb(1 - cr(d) / 3.0, "viridis")
        Image.fromarray(dm).save(gen("depthmap.png"))
        for k, (c, a) in enumerate(layers):
            im = np.concatenate([to8(cr(c)), (cr(a)[..., None] * 255).astype(np.uint8)], -1)
            Image.fromarray(im, "RGBA").save(gen(f"layer{k + 1}.png"))
            shear_rgba(Image.fromarray(im, "RGBA")).save(gen(f"layer{k + 1}_sh.png"))
            m = cr(d == k)
            sl = np.concatenate([to8(cr(C)) * m[..., None], (m[..., None] * 255).astype(np.uint8)], -1)
            Image.fromarray(sl.astype(np.uint8), "RGBA").save(gen(f"slice{k + 1}.png"))
        # ---- ray-optics novel views of the MPI (layer shifts ∝ parallax, back layer fixed)
        par = np.array([30, 19, 9, 0])
        views = [to8(cr(over(layers, [int(round(s * p)) for p in par]))) for s in np.linspace(-1, 1, 41)]
        np.save(gen("mpi_views.npy"), np.stack(views))
        # ---- single-image propagation sweep: smooth vs random phase (full bandwidth)
        prop_s, prop_r = [], []
        ph = rphase(rng)
        for z in ZP:
            s_, r_ = [], []
            for ch in range(3):
                s_.append(np.abs(asm(np.sqrt(C[..., ch]).astype(complex), z, LAMS[ch])) ** 2)
                r_.append(np.abs(asm(np.sqrt(C[..., ch]) * ph, z, LAMS[ch])) ** 2)
            prop_s.append(to8(cr(np.stack(s_, -1))))
            prop_r.append(to8(cr(np.stack(r_, -1))))
        np.save(gen("prop_smooth.npy"), np.stack(prop_s))
        np.save(gen("prop_rand.npy"), np.stack(prop_r))
        # ---- ASM diagram (green channel): field, spectrum, transfer function, result
        u0 = np.sqrt(C[..., 1]) * ph
        Image.fromarray(domain_rgb(cr(u0))).save(gen("asm_field.png"))
        L = np.log10(np.abs(np.fft.fftshift(fft2(u0))) ** 2 + 1e-20)
        L = np.clip((L - (L.max() - 7)) / 7, 0, 1)
        L = L * 0.62 / np.percentile(L, 99)
        Image.fromarray(cmap_rgb(np.clip(L, 0, 1), "magma")).resize((360, 360), Image.BILINEAR).save(gen("asm_spec.png"))
        Hs = np.fft.fftshift(Hz(10e-3, LAMS[1]))
        Hc = Hs[N // 2 - 128:N // 2 + 128, N // 2 - 128:N // 2 + 128]
        hr = (matplotlib.colormaps["twilight"]((np.angle(Hc) + np.pi) / (2 * np.pi))[..., :3] * 255).astype(np.uint8)
        Image.fromarray(hr).resize((360, 360), Image.BILINEAR).save(gen("asm_H.png"))
        Image.fromarray(to8(cr(np.abs(asm(u0, 10e-3, LAMS[1])) ** 2))).save(gen("asm_out.png"))
        # ---- holograms: MPI (Eq. 2) and RGBD, TFR frames each
        Pe = [pupil_ang(l, r=R_EYE) for l in LAMS]
        svals = np.linspace(-S_MAX, S_MAX, NV)
        Psub = [[pupil_ang(l, s, 0, R_SUB) for l in LAMS] for s in svals]
        Hf = [[Hz(z, l) for l in LAMS] for z in ZS]
        Hlf = [Hz(Z_LF, l) for l in LAMS]
        ref_e, ref_s = pupil_ref(R_EYE), pupil_ref(R_SUB)
        fs = {"mpi": 0, "rgbd": 0}
        lf = {"mpi": 0, "rgbd": 0}
        fs_mpi1 = None
        planes = np.zeros((5, NC, NC, 3))       # running-wave intensity at planes 4, 3, 2, 1 and the SLM
        tm_frames = []
        iz_f = int(np.argmin(np.abs(ZS - ZL[0])))
        for t in range(TFR):
            for name in ("mpi", "rgbd"):
                Fs_ = []
                for ch in range(3):
                    if name == "mpi":
                        keep = []
                        u = mpi_slm(layers, ch, rng, keep)
                        for k in range(4):
                            planes[k, ..., ch] += np.abs(keep[k][CROP, CROP]) ** 2
                        planes[4, ..., ch] += np.abs(u[CROP, CROP]) ** 2
                        if t == 0 and ch == 1:
                            Image.fromarray(domain_rgb(cr(u), 0.55, np.percentile(np.abs(cr(u)), 98))).save(gen("slm_field.png"))
                            Image.fromarray(domain_rgb(cr(keep[0]))).save(gen("wave_l4.png"))
                            Lp = np.log10(np.abs(np.fft.fftshift(fft2(u))) ** 2 + 1e-20)
                            Lp = np.clip((Lp - (Lp.max() - 7)) / 7, 0, 1)
                            Lp = Lp * 0.62 / np.percentile(Lp, 99)
                            Image.fromarray(cmap_rgb(np.clip(Lp, 0, 1), "magma")).resize(
                                (360, 360), Image.BILINEAR).save(gen("slm_spec.png"))
                    else:
                        u = rgbd_slm(C, d, ch, rng)
                    Fs_.append(fft2(u))
                st = np.stack([np.stack([np.abs(ifft2(Fs_[ch] * Pe[ch] * Hf[iz][ch]))[CROP, CROP] ** 2 / ref_e[ch]
                                         for ch in range(3)], -1) for iz in range(len(ZS))])
                fs[name] = fs[name] + st
                if name == "mpi":
                    if t == 0:
                        fs_mpi1 = st.copy()
                    tm_frames.append(st[iz_f])
                lf[name] = lf[name] + np.stack([np.stack([np.abs(ifft2(Fs_[ch] * Psub[v][ch] * Hlf[ch]))[CROP, CROP] ** 2
                                                          / ref_s[ch] for ch in range(3)], -1) for v in range(NV)])
        for name in ("mpi", "rgbd"):
            np.save(gen(f"fs_{name}.npy"), to8(fs[name] / TFR))
            np.save(gen(f"lf_{name}.npy"), to8(lf[name] / TFR))
        np.save(gen("fs_mpi1.npy"), to8(fs_mpi1))
        tm = np.stack(tm_frames)
        cum = np.cumsum(tm, 0) / np.arange(1, TFR + 1)[:, None, None, None]
        np.save(gen("tm_frames.npy"), to8(tm))
        np.save(gen("tm_cum.npy"), to8(cum))
        for k in range(5):
            Image.fromarray(to8(planes[k] / TFR)).save(gen(f"plane{k}.png"))
        open(gen(ASSET_FLAG), "w").write("ok")


# assets are built by ../animation_mpi_v3.py (flag data/generated/done_v1); this script only reads them
assert os.path.exists(gen(ASSET_FLAG)), 'run the 5-min script once to build data/generated'


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


def interp(frames, v):
    """Linear interpolation into a stack of uint8 frames at fractional index v."""
    v = float(np.clip(v, 0, len(frames) - 1))
    i0 = int(np.floor(v))
    i1 = min(i0 + 1, len(frames) - 1)
    w = v - i0
    if w < 1e-3:
        return frames[i0]
    return ((1 - w) * frames[i0] + w * frames[i1]).astype(np.uint8)


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.16)


_CLIPS = {}


def clip_frames(key):
    if key not in _CLIPS:
        _CLIPS[key] = np.load(os.path.join(CLIPS, key + ".npy"))
    return _CLIPS[key]


def pingpong(n, t, fps):
    i = int(t * fps) % (2 * n - 2)
    return i if i < n else 2 * n - 2 - i


def video(scene, key, h=None, w=None, fps=25):
    """Homepage teaser clip as a live ImageMobject (ping-pong loop) driven by the scene clock."""
    fr = clip_frames(key)
    m = arr_img(fr[0], h, w)
    t0 = scene.clock.get_value()
    live(m, lambda: fr[pingpong(len(fr), scene.clock.get_value() - t0, fps)])
    return m


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
        s = s.replace("√(1 − α)", "√(1\u00a0−\u00a0α)")
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


def toy_note(s="toy simulation: RGB lasers 638 / 520 / 450 nm, 8 µm pixels"):
    return tx(s, 16, DIM)


def sheet(k, h):
    """Pseudo-3D sheared layer k (1..4) of the toy MPI with its outline, height h (of the unsheared side)."""
    im = Image.open(gen(f"layer{k}_sh.png"))
    m = ImageMobject(gen(f"layer{k}_sh.png"))
    w_px, h_px = im.size
    scale = h / NC
    m.set(height=h_px * scale)
    W, Hh = w_px * scale, h_px * scale
    dy = SHY * NC * SHX * scale
    c = m.get_center()
    x0, x1 = c[0] - W / 2, c[0] + W / 2
    yb, yt = c[1] - Hh / 2, c[1] + Hh / 2
    # left edge spans [yb, yt − dy], right edge spans [yb + dy, yt]
    poly = Polygon([x0, yb, 0], [x1, yb + dy, 0], [x1, yt, 0], [x0, yt - dy, 0], color=DIM, stroke_width=1.5)
    return Group(m, poly)




# ================================================================== T1 Hook
class T1_Hook(Base):
    def construct(self):
        title = fit(tx("Fast Wave-optics Rendering of Multiplane Images", 50, bold=True)).move_to([0, 3.0, 0])
        subt = tx("for 3D Holographic Displays", 36).move_to([0, 2.3, 0])
        a = video(self, "lf", h=3.0).move_to([-2.45, -0.1, 0])
        b = video(self, "fs", h=3.0).move_to([2.45, -0.1, 0])
        self.add(title, subt)
        self.play(FadeIn(a), FadeIn(b), run_time=0.6)
        self.add(label(a, "light field", 22, DIM, d=DOWN), label(b, "focal stack", 22, DIM, d=DOWN),
                 tx("MPI holograms (paper teaser videos)", 16, DIM).move_to([0, 1.65, 0]),
                 fit(tx("Brian Chao, Dario Seyb, Nathan Matsuda, Oliver Cossairt, Yang Zhou, Douglas Lanman, "
                        "Gordon Wetzstein, Grace Kuo, Changwon Jang", 18, DIM)).move_to([0, -2.35, 0]),
                 tx("Stanford University  ·  Meta  ·  arXiv 2607.19731 (2026)", 20, DIM).move_to([0, -2.75, 0]))
        self.caption("Holograms you can refocus and look around, rendered from multiplane images.")
        self.hold(0.5)
        self.clear_all(run_time=0.5)
        w = 5.6
        a = img("teaser_rgbd_front.png", w=w).move_to([-3.0, 0.75, 0])
        b = img("teaser_rpws_front.png", w=w).move_to([3.0, 0.75, 0])
        self.play(FadeIn(a), FadeIn(b), run_time=0.5)
        self.add(label(a, "RGBD layers: fast", 26, C_RGBD), label(b, "RPWS (Gaussians): accurate", 26, C_RPWS),
                 tx("simulated, paper Fig. 1", 16, DIM).move_to([0, -1.35, 0]))
        self.caption("Today's holograms: RGBD layers are fast but leak light at depth edges.")
        self.caption("Gaussian-based RPWS is accurate, but splats millions of primitives: hours per hologram.")
        self.clear_all(run_time=0.5)


# ================================================================== T2 Method
class T2_Method(Base):
    def construct(self):
        sheets = [sheet(k, 2.7) for k in (4, 3, 2, 1)]
        for j, s in enumerate(sheets):
            s.move_to([1.9 - 1.25 * j, 0.7 - 0.26 * j, 0])
        comp = img(gen("composite.png"), 2.7).move_to([-5.0, 0.45, 0])
        self.play(FadeIn(comp), *[FadeIn(s) for s in sheets[::-1]], run_time=0.7)
        self.add(arrow([-3.5, 0.45, 0], [-2.6, 0.45, 0]), label(comp, "image", 22, DIM),
                 tx("multiplane image: a handful of RGBA layers", 24).move_to([-0.5, -1.9, 0]))
        self.caption("The idea: feed the hologram a multiplane image, a handful of RGBA layers.")
        self.clear_all(run_time=0.5)
        e = tx(EQ_MPI, 32).move_to([0, 2.7, 0])
        self.add(e)
        h = 2.05
        xs = [-5.45, -2.72, 0.0, 2.72, 5.45]
        names = ["layer 4 (back)", "layer 3", "layer 2", "layer 1 (front)", "SLM"]
        ims = [img(gen(f"plane{k}.png"), h).move_to([x, 0.55, 0]) for k, x in enumerate(xs)]
        self.play(FadeIn(ims[0]), run_time=0.4)
        self.add(label(ims[0], names[0], 20))
        self.add(toy_note("toy simulation: intensity of the running wave at each plane, averaged over 24 frames"
                          ).move_to([0, -1.0, 0]))
        self.caption("Back to front: propagate, dim by √(1 − α), add each layer's random-phase wave.")
        for k in range(1, 5):
            ar = arrow([xs[k - 1] + h / 2 + 0.02, 0.55, 0], [xs[k] - h / 2 - 0.02, 0.55, 0], color=AMBER, buff=0.04)
            self.play(GrowArrow(ar), FadeIn(ims[k]), run_time=0.5)
            self.add(tx("𝒫", 20, AMBER).next_to(ar, UP, buff=0.05), label(ims[k], names[k], 20, C_MPI if k == 4 else WHITE))
            self.wait(0.25)
        self.caption("One FFT propagation per layer, not one wavefront per primitive.")
        self.clear_all(run_time=0.5)
        # focal sweep, RGBD vs MPI (toy)
        stacks = [np.load(gen(f)) for f in ("fs_rgbd.npy", "fs_mpi.npy")]
        z = ValueTracker(ZL[0] * 1e3)
        zi = lambda: (z.get_value() * 1e-3 - ZS[0]) / (ZS[1] - ZS[0])
        ims = []
        for x, st, nm, c in zip((-2.3, 2.3), stacks, ("RGBD, 24 frames", "MPI, 24 frames"), (C_RGBD, C_MPI)):
            m = arr_img(st[9], 3.4).move_to([x, 0.75, 0])
            live(m, (lambda st_: (lambda: interp(st_, zi())))(st))
            ims.append(m)
            self.add(label(m, nm, 26, c))
        self.play(*[FadeIn(m) for m in ims], run_time=0.4)
        zr = always_redraw(lambda: tx(f"focus {I('z')} = {z.get_value():.1f} mm", 24).move_to([0, -1.35, 0]))
        self.add(zr, toy_note("toy simulation: 4-layer MPI at 4 / 9 / 14 / 20 mm, 638 / 520 / 450 nm, 8 µm pixels"
                              ).move_to([0, -1.85, 0]))
        self.caption("Averaged over 24 frames: RGBD tears at depth edges, the MPI blurs naturally.")
        self.run(z, 20, 3.0, smooth)
        self.run(z, 9, 2.0, smooth)
        for m in ims:
            m.clear_updaters()
        self.clear_all(run_time=0.5)


# ================================================================== T3 Payoff
class T3_Payoff(Base):
    def construct(self):
        w = 5.0
        a = img("teaser_mpi_back.png", w=w).move_to([-2.7, 1.25, 0])
        b = img("teaser_mpi_front.png", w=w).move_to([2.7, 1.25, 0])
        self.play(FadeIn(a), FadeIn(b), run_time=0.5)
        self.add(tx("MPI hologram, 4 layers, simulated (paper Fig. 1)", 16, DIM).next_to(a, DOWN, buff=0.12).set_x(0))
        head = tx("up to 250,000× faster than RPWS", 40, AMBER, bold=True).move_to([0, -1.55, 0])
        sub_ = tx("0.05 s (4 MPI layers) vs ≈ 4 h (RPWS); values read from paper Fig. 3", 20, DIM).move_to([0, -2.2, 0])
        self.add(head, sub_)
        self.caption("Quality close to RPWS, far better than RGBD, and up to 250,000× faster.")
        self.hold(2.0)
        self.clear_all(run_time=0.4)
        t1 = fit(tx("Fast Wave-optics Rendering of Multiplane Images", 46, bold=True)).move_to([0, 1.9, 0])
        t1b = tx("for 3D Holographic Displays", 32).move_to([0, 1.2, 0])
        t2 = tx("arXiv 2607.19731", 30).move_to([0, 0.25, 0])
        t4 = fit(tx("Brian Chao, Dario Seyb, Nathan Matsuda, Oliver Cossairt, Yang Zhou, Douglas Lanman, "
                    "Gordon Wetzstein, Grace Kuo, Changwon Jang", 20, DIM)).move_to([0, -0.8, 0])
        t5 = tx("Stanford University  ·  Meta", 22, DIM).move_to([0, -1.25, 0])
        self.add(t1, t1b, t2, t4, t5)
        self.wait(4.5)
