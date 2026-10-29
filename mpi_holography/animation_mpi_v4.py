"""Fast Wave-optics Rendering of Multiplane Images for 3D Holographic Displays — teaser animation, v4.

Source: Chao, Seyb, Matsuda, Cossairt, Zhou, Lanman, Wetzstein, Kuo, Jang, arXiv 2607.19731 (2026).
Paper figures in data/ are crops of the paper (mpi-holography/pdfs/, via scratch/crop_figs.py); data/clips/*.npy are the
frames of the author-homepage teaser videos (data/videos/). All wave-optics visuals are toy simulations computed here with
numpy (angular spectrum method, 4-layer colour MPI, back-to-front random-phase wave compositing, RGBD layer baseline,
time multiplexing) and cached in data/generated/ (delete data/generated/done_v1 to rebuild).

Render (see render.sh):
    manim -ql --media_dir media animation_mpi_v4.py <Scene>
Scenes, in order: S00_Title ... S09_Outlook.
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
DATA = os.path.join(HERE, "data")
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


# ================================================================== S00 Title
class S00_Title(Base):
    def construct(self):
        title = fit(tx("Fast Wave-optics Rendering of Multiplane Images", 50, bold=True)).move_to([0, 3.0, 0])
        subt = tx("for 3D Holographic Displays", 36).move_to([0, 2.3, 0])
        a = video(self, "lf", h=3.0).move_to([-2.45, -0.2, 0])
        b = video(self, "fs", h=3.0).move_to([2.45, -0.2, 0])
        la = label(a, "light field", 22, DIM, d=DOWN)
        lb = label(b, "focal stack", 22, DIM, d=DOWN)
        auth = fit(tx("Brian Chao, Dario Seyb, Nathan Matsuda, Oliver Cossairt, Yang Zhou, Douglas Lanman, "
                      "Gordon Wetzstein, Grace Kuo, Changwon Jang", 20, DIM)).move_to([0, -2.6, 0])
        aff = tx("Stanford University  ·  Meta  ·  arXiv 2607.19731 (2026)", 22, DIM).move_to([0, -3.05, 0])
        self.add(title, subt, auth, aff)
        self.play(FadeIn(a), FadeIn(b), run_time=0.8)
        self.add(la, lb, tx("MPI holograms (paper teaser videos)", 16, DIM).move_to([0, 1.55, 0]))
        self.wait(7.0)


# ================================================================== S01 Hook
class S01_Hook(Base):
    def construct(self):
        self.title_card("Part 1", "Holograms need fast 3D content")
        self.header("1 · Holograms need fast 3D content")
        a = video(self, "fs", h=3.3).move_to([-3.2, 0.7, 0])
        b = video(self, "lf", h=3.3).move_to([3.2, 0.7, 0])
        self.play(FadeIn(a), FadeIn(b), run_time=0.6)
        self.add(label(a, "refocus", 24), label(b, "look around", 24))
        self.caption("A holographic display rebuilds the light wave of a 3D scene: refocus, look around.")
        self.caption("A computer must first turn 3D content into that wave: computer-generated holography.")
        self.clear_all()

        # --- paper Table 1
        cols = ["3D input", "fast", "focal stack", "light field"]
        rows = [("Point cloud", "△✗✗"), ("RGBD image", "✓△△"), ("Focal stack", "△✓△"), ("Light field", "△✓✓"),
                ("Mesh", "✗✗✗"), ("Gaussians", "✗✓✓"), ("MPIs (this paper)", "✓✓✓")]
        xs = [-3.2, 0.2, 2.3, 4.4]
        y0, dy = 2.45, 0.55
        head = VGroup(*[tx(c, 24, DIM).move_to([x, y0, 0]) for c, x in zip(cols, xs)])
        head[0].align_to([-4.6, 0, 0], LEFT)
        hl = Line([-4.7, y0 - 0.3, 0], [5.3, y0 - 0.3, 0], color=DIM, stroke_width=1.5)
        self.add(head, hl)
        colmap = {"✓": GREEN, "△": AMBER, "✗": RED}
        self.caption("Each 3D format is a trade-off (paper Table 1).")
        rgs = []
        for r, (name, marks) in enumerate(rows):
            y = y0 - dy * (r + 1) - (0.15 if r == 6 else 0)
            g = VGroup(tx(name, 26, GREEN if r == 6 else WHITE).move_to([0, y, 0]).align_to([-4.6, 0, 0], LEFT))
            for mk, x in zip(marks, xs[1:]):
                g.add(tx(mk, 28, colmap[mk]).move_to([x, y, 0]))
            rgs.append(g)
        for g in rgs[:6]:
            self.add(g)
            self.wait(0.3)
        self.caption("RGBD is fast but breaks at occlusions. Gaussians look great but come by the million.")
        self.hold()
        self.add(Line([-4.7, rgs[6].get_y() + 0.34, 0], [5.3, rgs[6].get_y() + 0.34, 0], color=DIM, stroke_width=1.5))
        self.add(rgs[6], SurroundingRectangle(rgs[6], color=AMBER, buff=0.12, stroke_width=2))
        self.caption("This paper's pick: multiplane images, fast and accurate.")
        self.clear_all()


# ================================================================== S02 Waves
class S02_Waves(Base):
    def construct(self):
        self.title_card("Part 2", "Propagating a plane of light")
        self.header("2 · Propagating a plane of light")
        ps = np.load(gen("prop_smooth.npy"))
        pr = np.load(gen("prop_rand.npy"))
        z = ValueTracker(0.0)
        zi = lambda: z.get_value() / 0.5
        h = 3.2
        a = arr_img(ps[0], h).move_to([-2.4, 0.75, 0])
        b = arr_img(pr[0], h).move_to([2.4, 0.75, 0])
        live(a, lambda: interp(ps, zi()))
        live(b, lambda: interp(pr, zi()))
        self.play(FadeIn(a), FadeIn(b), run_time=0.6)
        self.add(label(a, "smooth phase", 24), label(b, "random phase", 24))
        zr = always_redraw(lambda: tx(f"propagation distance {I('z')} = {z.get_value():.1f} mm", 24
                                      ).move_to([0, -1.25, 0]))
        note = toy_note("toy simulation: one RGB image plane, angular spectrum method, 638 / 520 / 450 nm, 8 µm pixels"
                        ).move_to([0, -1.8, 0])
        self.add(zr, note)
        self.caption("Take one image plane of light and let it travel.")
        self.run(z, 20, 5.0, smooth)
        self.caption("Smooth phase rings but stays put. Random phase spreads into every direction.")
        self.run(z, 10, 2.0, smooth)
        for m in (a, b):
            m.clear_updaters()
        self.clear_all()

        # --- ASM diagram
        h = 2.3
        xs = [-5.1, -1.7, 1.7, 5.1]
        files = ["asm_field.png", "asm_spec.png", "asm_H.png", "asm_out.png"]
        names = [f"plane {I('u')} (hue = phase)", f"spectrum ℱ{{{I('u')}}}", "transfer function", f"𝒫({I('u')}; {I('z')})"]
        ims = [img(gen(f), h).move_to([x, 1.0, 0]) for f, x in zip(files, xs)]
        labs = [label(m, n, 22, DIM) for m, n in zip(ims, names)]
        arr_l = ["FFT", "× " + I("H"), "FFT" + sup("−1")]
        e1 = tx(EQ_ASM, 32).move_to([0, -1.05, 0])
        e2 = tx(EQ_FZ, 24, DIM).move_to([0, -1.75, 0])
        self.play(FadeIn(ims[0]), run_time=0.5)
        self.add(labs[0])
        self.caption("The angular spectrum method propagates a whole plane at once.")
        for k in range(3):
            ar = arrow([xs[k] + h / 2 + 0.05, 1.0, 0], [xs[k + 1] - h / 2 - 0.05, 1.0, 0])
            self.play(GrowArrow(ar), FadeIn(ims[k + 1]), run_time=0.5)
            self.add(labs[k + 1], tx(arr_l[k], 20, AMBER).next_to(ar, UP, buff=0.1))
            self.wait(0.5)
        self.add(e1, e2)
        self.caption("Two FFTs and a multiply: the cost depends on pixels, not on content.")
        self.clear_all()

        # --- cost comparison
        rng = np.random.default_rng(3)
        pts = rng.uniform([-6.0, -1.6, 0], [-0.8, 2.4, 0], (700, 3))
        pts[:, 2] = 0
        cols_ = [BLUE, AMBER, RED, GREEN, ORANGE]
        dots = VGroup(*[Dot(p, radius=0.035, color=cols_[i % 5], fill_opacity=0.8) for i, p in enumerate(pts)])
        t1 = tx("primitive-based CGH", 28, C_RPWS).move_to([-3.4, 2.95, 0])
        t1b = tx("millions of primitives, one wavefront each", 22, DIM).move_to([-3.4, -2.1, 0])
        self.add(t1)
        self.play(LaggedStart(*[FadeIn(d_) for d_ in dots], lag_ratio=0.002), run_time=1.5)
        self.add(t1b)
        self.caption("Primitive-based CGH needs one wavefront per point, polygon or Gaussian: millions.")
        sheets = Group(*[sheet(k, 2.6) for k in (4, 3, 2, 1)])
        for j, s in enumerate(sheets):
            s.move_to([5.2 - 0.75 * j, 0.55 - 0.18 * j, 0])
        t2 = tx("MPI-based CGH", 28, C_MPI).move_to([3.4, 2.95, 0])
        t2b = tx("a handful of layers (typically fewer than 100)", 22, DIM).move_to([3.4, -2.1, 0])
        self.hold()
        self.add(t2)
        self.play(LaggedStart(*[FadeIn(s) for s in sheets], lag_ratio=0.3), run_time=1.2)
        self.add(t2b)
        self.caption("A multiplane image needs one propagation per layer, and has only a handful.")
        self.clear_all()


# ================================================================== S03 MPI
class S03_MPI(Base):
    def construct(self):
        self.title_card("Part 3", "Multiplane images")
        self.header("3 · Multiplane images")
        comp = img(gen("composite.png"), 3.0).move_to([-4.7, 0.6, 0])
        self.play(FadeIn(comp), run_time=0.5)
        self.add(label(comp, "an image", 24))
        sheets = [sheet(k, 3.0) for k in (4, 3, 2, 1)]
        for j, s in enumerate(sheets):
            s.move_to([5.0 - 1.35 * j, 0.95 - 0.28 * j, 0])
        ar = arrow([-3.0, 0.6, 0], [-1.7, 0.6, 0])
        self.caption("A multiplane image (MPI) is a stack of RGBA images, each at its own depth.")
        self.play(GrowArrow(ar), *[FadeIn(s) for s in sheets[::-1]], run_time=0.9)
        for s, n in zip(sheets, ["4", "3", "2", "1"]):
            self.add(tx(n, 20, DIM).move_to([s.get_x(), s.get_bottom()[1] - 0.2, 0]))
        self.add(tx(f"layer {I('i')} = 1 (front) … 4 (back):  colour {I('c')}{sub(I('i'))}, opacity α{sub(I('i'))}, "
                    f"depth {I('d')}{sub(I('i'))}", 22).move_to([1.6, -2.3, 0]))
        self.caption("Through transparent regions you see the layers behind, even what the front hides.")
        self.clear_all()

        # --- novel views, ray optics
        views = np.load(gen("mpi_views.npy"))
        s = ValueTracker(0.0)
        v = arr_img(views[20], 3.4).move_to([3.2, 0.75, 0])
        live(v, lambda: interp(views, (s.get_value() + 1) * 20))
        tiles = Group(*[img(gen(f"layer{k}.png"), 1.55) for k in (1, 2, 3, 4)]).arrange_in_grid(2, 2, buff=0.25)
        tiles.move_to([-3.6, 0.75, 0])
        frames_ = VGroup(*[framed(t_, DIM, 1) for t_ in tiles])
        for k, t_ in enumerate(tiles):
            self.add(tx(f"layer {k + 1}", 16, DIM).next_to(t_, UP, buff=0.06))
        self.play(FadeIn(tiles), Create(frames_), FadeIn(v), run_time=0.7)
        self.add(label(v, "rendered view", 24), framed(v, DIM, 1.2))
        eq = tx(EQ_OVER + f",   {I('i')} = {I('L')}, …, 1", 30).move_to([-3.6, -1.75, 0])
        eqn = tx("back to front: the “over” operator", 20, DIM).next_to(eq, DOWN, buff=0.12)
        cam = always_redraw(lambda: tx(f"camera shift: {s.get_value():+.2f}", 22, CYAN).move_to([3.2, -1.4, 0]))
        self.add(eq, eqn, cam)
        self.caption("New view: shift each layer by its parallax, composite back to front.")
        self.run(s, -1, 1.8, smooth)
        self.run(s, 1, 3.0, smooth)
        self.caption("Near layers move more than far ones: parallax and occlusion, for free.")
        self.run(s, -1, 3.0, smooth)
        self.run(s, 0, 1.4, smooth)
        v.clear_updaters()
        self.clear_all()

        # --- real MPIs
        a = img("pipe_a.png", w=6.4).move_to([-3.2, 0.5, 0])
        self.play(FadeIn(a), run_time=0.5)
        self.add(tx("MPI of a real scene (paper Fig. 2a)", 18, DIM).next_to(a, DOWN, buff=0.15))
        items = ["predicted by networks from a stereo pair", "or from a single photo",
                 "or binned from Gaussian splats (this paper)", "typically fewer than 100 layers"]
        g = VGroup(*[tx(("•  " if i < 3 else "") + s_, 22, AMBER if i == 3 else WHITE) for i, s_ in enumerate(items)]
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.35, 0.6, 0])
        self.add(g)
        self.caption("MPIs are cheap to get, and they excel at front-facing views.")
        self.caption("That is exactly what a near-eye display's small eyebox needs.")
        self.clear_all()


# ================================================================== S04 RGBD
class S04_RGBD(Base):
    def construct(self):
        self.title_card("Part 4", "Why not RGBD layers?")
        self.header("4 · Why not RGBD layers?")
        comp = img(gen("composite.png"), 1.9).move_to([-5.4, 1.35, 0])
        dep = img(gen("depthmap.png"), 1.9).move_to([-5.4, -1.1, 0])
        xs = [-2.6, -0.35, 1.9, 4.15]
        sl = [img(gen(f"slice{k}.png"), 1.9).move_to([x, 1.35, 0]) for k, x in zip((1, 2, 3, 4), xs)]
        ly = [img(gen(f"layer{k}.png"), 1.9).move_to([x, -1.1, 0]) for k, x in zip((1, 2, 3, 4), xs)]
        self.play(FadeIn(comp), FadeIn(dep), run_time=0.5)
        self.add(label(comp, "RGB", 20, DIM), label(dep, "depth", 20, DIM))
        self.caption("An RGBD image stores one colour and one depth per pixel.")
        self.hold()
        self.play(*[FadeIn(m) for m in sl], *[Create(framed(m, DIM, 1)) for m in sl], run_time=0.6)
        for k, m in enumerate(sl):
            self.add(label(m, f"slice {k + 1}", 20, C_RGBD))
        self.caption("Slice it by depth: whatever hides behind the front surface is simply missing.")
        self.hold()
        self.play(*[FadeIn(m) for m in ly], *[Create(framed(m, DIM, 1)) for m in ly], run_time=0.6)
        for k, m in enumerate(ly):
            self.add(label(m, f"MPI layer {k + 1}", 20, C_MPI))
        self.caption("MPI layers keep the hidden content, and soft alpha at the edges.")
        self.clear_all()

        # --- light-field views: RGBD vs MPI
        lr = np.load(gen("lf_rgbd.npy"))
        lm = np.load(gen("lf_mpi.npy"))
        s = ValueTracker(0.0)
        idx = lambda: (s.get_value() + S_MAX) / (2 * S_MAX) * (NV - 1)
        h = 3.3
        a = arr_img(lr[NV // 2], h).move_to([-3.4, 0.7, 0])
        b = arr_img(lm[NV // 2], h).move_to([3.4, 0.7, 0])
        live(a, lambda: interp(lr, idx()))
        live(b, lambda: interp(lm, idx()))
        eb = Circle(radius=0.62, color=DIM, stroke_width=2).move_to([0, 0.7, 0])
        pp = always_redraw(lambda: Circle(radius=0.62 * R_SUB, color=CYAN, stroke_width=4, fill_opacity=0.3
                                          ).move_to([s.get_value() * 0.62, 0.7, 0]))
        self.play(FadeIn(a), FadeIn(b), Create(eb), run_time=0.6)
        self.add(label(a, "RGBD hologram (4 slices)", 24, C_RGBD), label(b, "MPI hologram (4 layers)", 24, C_MPI),
                 pp, label(eb, "eyebox", 18, DIM), tx("pupil", 18, CYAN).next_to(eb, DOWN, buff=0.12))
        note = toy_note("toy simulation: 24 frames, focus on the star, pupil moved across the eyebox; 638 / 520 / 450 nm, 8 µm pixels"
                        ).move_to([0, -1.55, 0])
        self.add(note)
        self.caption("As holograms (24 frames each), move the eye's pupil across the eyebox.")
        self.run(s, -S_MAX, 1.8, smooth)
        self.run(s, S_MAX, 3.2, smooth)
        self.caption("RGBD: dark gaps open along occlusion edges. The MPI shows what was behind.")
        self.run(s, -S_MAX, 3.2, smooth)
        self.run(s, 0, 1.4, smooth)
        for m in (a, b):
            m.clear_updaters()
        self.clear_all()

        # --- paper Fig. 4 zoom-ins
        keys = ["gt", "rpws", "rgbd4", "rgbd512", "mpi4"]
        names = ["ground truth", "RPWS", "RGBD, 4 layers", "RGBD, 512 layers", "MPI, 4 layers"]
        colors = [WHITE, C_RPWS, C_RGBD, C_RGBD, C_MPI]
        xs = [-5.3, -2.65, 0, 2.65, 5.3]
        w = 2.5
        for x, k, n, c in zip(xs, keys, names, colors):
            t_ = img(f"dl_{k}_garden_fs_z.png", w=w).move_to([x, 1.1, 0])
            b_ = img(f"dl_{k}_bike_lf_z.png", w=w).move_to([x, -1.0, 0])
            self.add(t_, b_, label(t_, n, 22, c))
        self.add(tx("zoom-ins, simulated: defocused edge (top), light-field view at an occlusion (bottom); paper Fig. 4",
                    18, DIM).move_to([0, -2.1, 0]))
        self.caption("Paper Fig. 4: RGBD leaks and tears at depth edges, even with 512 layers.")
        self.caption("MPI with only 4 layers gives natural blur and clean occlusion, close to RPWS.")
        self.clear_all()


# ================================================================== S05 Compositing
class S05_Compositing(Base):
    def construct(self):
        self.title_card("Part 5", "Wave compositing, back to front")
        self.header("5 · Wave compositing, back to front")
        e = tx(EQ_MPI, 32).move_to([0, 2.7, 0])
        self.add(e)
        h = 2.05
        xs = [-5.45, -2.72, 0.0, 2.72, 5.45]
        names = ["layer 4 (back)", "layer 3", "layer 2", "layer 1 (front)", "SLM"]
        files = ["plane0.png", "plane1.png", "plane2.png", "plane3.png", "plane4.png"]
        ims = [img(gen(f), h).move_to([x, 0.55, 0]) for f, x in zip(files, xs)]
        ops = [f"√({I('c')}{sub('4')}α{sub('4')}) {I('e')}{sup(I('i') + 'φ')}",
               "× √(1 − α) + own wave", "× √(1 − α) + own wave", "× √(1 − α) + own wave", "hologram"]
        self.caption("Start at the farthest layer: √(colour × opacity), times a random phase.")
        self.play(FadeIn(ims[0]), run_time=0.5)
        self.add(label(ims[0], names[0], 20), tx(ops[0], 18, DIM).next_to(ims[0], DOWN, buff=0.15))
        note = toy_note("toy simulation: intensity of the running wave at each plane, averaged over 24 frames"
                        ).move_to([0, -1.25, 0])
        self.add(note)
        self.hold()
        for k in range(1, 5):
            ar = arrow([xs[k - 1] + h / 2 + 0.02, 0.55, 0], [xs[k] - h / 2 - 0.02, 0.55, 0], color=AMBER, buff=0.04)
            self.play(GrowArrow(ar), run_time=0.4)
            self.add(tx("𝒫", 20, AMBER).next_to(ar, UP, buff=0.05))
            self.play(FadeIn(ims[k]), run_time=0.4)
            self.add(label(ims[k], names[k], 20, C_MPI if k == 4 else WHITE),
                     tx(ops[k], 16, DIM).next_to(ims[k], DOWN, buff=0.15))
            if k == 1:
                self.caption("Propagate to the next layer, dim by √(1 − α), add that layer's own wave.")
            elif k == 3:
                self.caption("Each mask sits at its occluder's depth, blocking light from every angle.")
            else:
                self.wait(0.3)
        self.caption(f"Repeat down to the SLM: {I('L')} layers cost {I('L')} propagations per frame.")
        self.clear_all(keep=[e])

        # --- why square roots
        l1 = tx(f"ray optics:   {EQ_OVER}", 30).move_to([0, 1.55, 0])
        l2 = tx(f"wave optics:   {I('u')} ← √(1 − α) {I('u')} + √(α{I('c')}) {I('e')}{sup(I('i') + 'φ')}", 30
                ).move_to([0, 0.75, 0])
        l3 = tx(f"average over φ:   𝔼|{I('u')}|{SQ} = (1 − α) |{I('u')}|{SQ} + α{I('c')}   "
                + col("(cross term → 0)", AMBER), 30).move_to([0, -0.2, 0])
        self.add(l1)
        self.caption("Why square roots? Compare with the over operator on colours.")
        self.hold()
        self.add(l2)
        self.hold(0.5)
        self.add(l3)
        self.add(tx("random-phase alpha blending from Random-phase Wave Splatting (RPWS, Chao et al. 2025)", 20, DIM
                    ).move_to([0, -1.0, 0]))
        self.caption("The cross term averages away: intensities obey the over operator exactly.")
        self.clear_all(keep=[e])

        # --- the resulting hologram
        f1 = img(gen("slm_field.png"), 2.9).move_to([-2.3, 0.35, 0])
        f2 = img(gen("slm_spec.png"), 2.9).move_to([2.3, 0.35, 0])
        self.play(FadeIn(f1), FadeIn(f2), run_time=0.5)
        self.add(label(f1, "one frame at the SLM (hue = phase)", 22), label(f2, "its angular spectrum", 22),
                 toy_note("toy simulation, green channel; spectrum on a log scale").move_to([0, -1.55, 0]))
        self.caption("The result: a random-phase hologram filling the SLM's bandwidth, a wide eyebox.")
        self.clear_all()


# ================================================================== S06 Time multiplexing
class S06_TimeMux(Base):
    def construct(self):
        self.title_card("Part 6", "Time multiplexing and focus")
        self.header("6 · Time multiplexing and focus")
        e = tx(EQ_TM, 30).move_to([0, 2.85, 0])
        self.add(e)
        frames = np.load(gen("tm_frames.npy"))
        cum = np.load(gen("tm_cum.npy"))
        t = ValueTracker(0.0)
        ti = lambda: int(np.clip(t.get_value(), 0, TFR - 1))
        h = 3.0
        a = arr_img(frames[0], h).move_to([-2.2, 0.2, 0])
        b = arr_img(cum[0], h).move_to([2.2, 0.2, 0])
        live(a, lambda: frames[ti()])
        live(b, lambda: cum[ti()])
        la = always_redraw(lambda: label(a, f"frame {ti() + 1}", 24))
        lb = always_redraw(lambda: label(b, f"average of {ti() + 1} frames", 24))
        self.play(FadeIn(a), FadeIn(b), run_time=0.6)
        self.add(la, lb, toy_note("toy simulation: MPI hologram, focus on the star (4 mm), eye pupil half the eyebox"
                                  ).move_to([0, -1.6, 0]))
        self.caption("Each frame gets fresh random phases and is full of speckle.")
        self.run(t, 7.99, 3.5)
        self.caption("Shown in quick succession, the eye averages them. The paper uses 24 frames.")
        self.run(t, 23.99, 4.0, rate=lambda x: x ** 1.3)
        for m in (a, b):
            m.clear_updaters()
        self.clear_all(keep=[e])
        self.remove(e)

        # --- focal sweep
        stacks = [np.load(gen(f)) for f in ("fs_rgbd.npy", "fs_mpi1.npy", "fs_mpi.npy")]
        z = ValueTracker(ZL[0] * 1e3)
        zi = lambda: (z.get_value() * 1e-3 - ZS[0]) / (ZS[1] - ZS[0])
        h = 3.3
        xs = [-4.5, 0, 4.5]
        names = ["RGBD, 24 frames", "MPI, 1 frame", "MPI, 24 frames"]
        cols = [C_RGBD, WHITE, C_MPI]
        ims = []
        for x, st, nm, c in zip(xs, stacks, names, cols):
            m = arr_img(st[0], h).move_to([x, 1.0, 0])
            live(m, (lambda st_: (lambda: interp(st_, zi())))(st))
            ims.append(m)
            self.add(label(m, nm, 24, c))
        self.play(*[FadeIn(m) for m in ims], run_time=0.6)
        nl = NumberLine(x_range=[2, 24, 2], length=8.0, color=DIM, include_numbers=False, stroke_width=2
                        ).move_to([0.6, -1.5, 0])
        nlab = tx(f"focus {I('z')} (mm)", 20, DIM).next_to(nl, LEFT, buff=0.25)
        ticks = VGroup(*[tx(str(v), 16, DIM).next_to(nl.n2p(v), DOWN, buff=0.12) for v in range(4, 25, 4)])
        lc = ["#FF7060", "#40D070", "#FFD040", "#7090FF"]
        lnames = ["star", "triangle", "disc", "wall"]
        marks = VGroup()
        for zz, c, n in zip(ZL * 1e3, lc, lnames):
            tri = Triangle(color=c, fill_opacity=1).scale(0.08).rotate(PI).next_to(nl.n2p(zz), UP, buff=0.04)
            marks.add(tri, tx(n, 16, c).next_to(tri, UP, buff=0.05))
        dot = always_redraw(lambda: Dot(nl.n2p(float(np.clip(z.get_value(), 2, 24))), radius=0.08, color=WHITE))
        zval = always_redraw(lambda: tx(f"z = {z.get_value():.1f} mm", 22).next_to(nl, RIGHT, buff=0.3))
        note = toy_note("toy simulation: 4-layer MPI at 4 / 9 / 14 / 20 mm from the SLM").move_to([0, -2.4, 0])
        self.add(nl, nlab, ticks, marks, dot, zval, note)
        self.caption("Now sweep the focus. Each layer sharpens in turn, with natural blur in between.")
        self.run(z, 20, 4.0, smooth)
        self.caption("24 frames give clean defocus. RGBD shows seams around the star.")
        self.run(z, 4, 4.0, smooth)
        self.run(z, 12, 2.0, smooth)
        for m in ims:
            m.clear_updaters()
        self.clear_all()


# ================================================================== S07 Speed
RT_L = [4, 8, 16, 32, 64, 128, 256, 512]           # values read from paper Fig. 3 (vector data), Mip-NeRF 360 scenes
RT_MPI = [0.050, 0.110, 0.220, 0.441, 0.878, 1.78, 3.61, 7.20]
RT_RGBD = [0.050, 0.110, 0.220, 0.440, 0.880, 1.78, 3.61, 7.22]
RT_RPWS = 1.41e4
PS_MPI = [22.47, 22.59, 22.64, 22.68, 22.72, 22.74, 22.74, 22.72]
PS_RGBD = [21.24, 21.45, 21.50, 21.50, 21.49, 21.48, 21.46, 21.46]
PS_RPWS = 23.50
SS_MPI = [0.508, 0.516, 0.519, 0.520, 0.522, 0.523, 0.524, 0.524]
SS_RGBD = [0.453, 0.467, 0.471, 0.472, 0.472, 0.473, 0.474, 0.476]
SS_RPWS = 0.521


def pow10(k):
    return "10" + sup(("−" if k < 0 else "") + str(abs(k)))


class S07_Speed(Base):
    def construct(self):
        self.title_card("Part 7", "Cost: layers, not primitives")
        self.header("7 · Cost: layers, not primitives")
        ax = Axes(x_range=[2, 9, 1], y_range=[0, 7, 1], x_length=7.6, y_length=4.6, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2, "include_numbers": False}).move_to([-1.6, 0.35, 0])
        xt = VGroup(*[tx(str(2 ** k), 18, DIM).next_to(ax.c2p(k, 0), DOWN, buff=0.12) for k in range(2, 10)])
        yt = VGroup(*[tx(pow10(k), 18, DIM).next_to(ax.c2p(2, k + 2), LEFT, buff=0.12) for k in range(-2, 6)])
        xl = tx("number of layers", 20, DIM).next_to(xt, DOWN, buff=0.15)
        yl = tx("runtime (s), log scale", 20, DIM).rotate(PI / 2).next_to(yt, LEFT, buff=0.15)
        grid = VGroup(*[DashedLine(ax.c2p(2, k + 2), ax.c2p(9, k + 2), color="#333333", stroke_width=1) for k in range(-1, 6)])
        self.play(Create(ax), run_time=0.6)
        self.add(grid, xt, yt, xl, yl)
        self.add(src_note("values read from paper Fig. 3 (Mip-NeRF 360 scenes)"))
        lx = [np.log2(v) for v in RT_L]
        mpi = VMobject(color=C_MPI, stroke_width=4).set_points_as_corners([ax.c2p(x, (np.log10(y) + 2)) for x, y in zip(lx, RT_MPI)])
        rgbd = DashedVMobject(VMobject().set_points_as_corners([ax.c2p(x, (np.log10(y) + 2)) for x, y in zip(lx, RT_RGBD)]),
                              num_dashes=30).set_color(C_RGBD).set_stroke(width=3)
        mdots = VGroup(*[Dot(ax.c2p(x, (np.log10(y) + 2)), radius=0.06, color=C_MPI) for x, y in zip(lx, RT_MPI)])
        self.caption("Runtime of MPI holograms grows linearly with the number of layers.")
        self.play(Create(mpi), FadeIn(mdots), run_time=1.5)
        leg1 = tx("MPI", 22, C_MPI).next_to(ax.c2p(9, (np.log10(7.2) + 2)), RIGHT, buff=0.15).shift(UP * 0.12)
        self.add(leg1)
        self.hold()
        self.play(Create(rgbd), run_time=1.0)
        self.add(tx("RGBD (same)", 22, C_RGBD).next_to(leg1, DOWN, buff=0.08).align_to(leg1, LEFT))
        self.caption("RGBD costs the same: the layer count sets the price.")
        self.hold()
        yr = np.log10(RT_RPWS) + 2
        rp = DashedLine(ax.c2p(2, yr), ax.c2p(9, yr), color=C_RPWS, stroke_width=3, dash_length=0.12)
        self.play(Create(rp), run_time=0.8)
        self.add(tx(f"RPWS ≈ 1.4 × {pow10(4)} s (≈ 4 h)", 22, C_RPWS).next_to(ax.c2p(9, yr), RIGHT, buff=0.15))
        self.caption("RPWS, which splats millions of Gaussians, takes hours per hologram.")
        self.hold()
        a1 = DoubleArrow(ax.c2p(2.35, (np.log10(0.05) + 2)), ax.c2p(2.35, yr), color=AMBER, buff=0.05, stroke_width=3,
                         tip_length=0.18)
        self.play(GrowFromCenter(a1), run_time=0.6)
        self.add(tx("up to", 22, AMBER).move_to(ax.c2p(3.25, 3.95)), tx("250,000×", 34, AMBER, bold=True).move_to(ax.c2p(3.45, 3.35)),
                 tx("4 layers: 0.05 s", 20, C_MPI).next_to(ax.c2p(2.0, (np.log10(0.05) + 2)), RIGHT, buff=0.45).shift(DOWN * 0.22))
        a2 = DoubleArrow(ax.c2p(8.75, (np.log10(7.2) + 2)), ax.c2p(8.75, yr), color=AMBER, buff=0.05, stroke_width=3,
                         tip_length=0.18)
        self.play(GrowFromCenter(a2), run_time=0.5)
        self.add(tx("≈ 2000×", 24, AMBER).next_to(a2, LEFT, buff=0.12))
        self.caption("Up to 250,000× faster with 4 layers, and still about 2000× faster with 512.")
        self.clear_all()

        # --- quality plots
        def qplot(x0, yr_, vals_m, vals_r, v_rp, ttl, ylab, fmt, ticks):
            a = Axes(x_range=[2, 9, 1], y_range=yr_, x_length=5.4, y_length=3.4, tips=False,
                     axis_config={"color": DIM, "stroke_width": 2, "include_numbers": False}).move_to([x0, 0.55, 0])
            g = VGroup(a)
            g.add(*[tx(str(2 ** k), 16, DIM).next_to(a.c2p(k, yr_[0]), DOWN, buff=0.1) for k in (2, 4, 6, 8)])
            g.add(*[tx(fmt.format(v), 16, DIM).next_to(a.c2p(2, v), LEFT, buff=0.1) for v in ticks])
            g.add(tx(ttl, 26).next_to(a, UP, buff=0.2))
            g.add(tx("number of layers", 18, DIM).next_to(a, DOWN, buff=0.45))
            lm = VMobject(color=C_MPI, stroke_width=4).set_points_as_corners([a.c2p(x, y) for x, y in zip(lx, vals_m)])
            lr = DashedVMobject(VMobject().set_points_as_corners([a.c2p(x, y) for x, y in zip(lx, vals_r)]),
                                num_dashes=24).set_color(C_RGBD).set_stroke(width=3)
            dm = VGroup(*[Dot(a.c2p(x, y), radius=0.05, color=C_MPI) for x, y in zip(lx, vals_m)])
            dr = VGroup(*[Dot(a.c2p(x, y), radius=0.045, color=C_RGBD) for x, y in zip(lx, vals_r)])
            rp_ = DashedLine(a.c2p(2, v_rp), a.c2p(9, v_rp), color=C_RPWS, stroke_width=3, dash_length=0.1)
            return a, g, (lm, dm), (lr, dr), rp_
        pa, pg, pm, pr, prp = qplot(-3.4, [21.0, 23.75, 0.5], PS_MPI, PS_RGBD, PS_RPWS, "light-field PSNR (dB)", "",
                                    "{:.1f}", [21.0, 22.0, 23.0])
        sa, sg, sm, sr, srp = qplot(3.4, [0.45, 0.53, 0.02], SS_MPI, SS_RGBD, SS_RPWS, "light-field SSIM", "",
                                    "{:.2f}", [0.45, 0.47, 0.49, 0.51, 0.53])
        self.play(FadeIn(pg), FadeIn(sg), run_time=0.6)
        self.add(src_note("values read from paper Fig. 3 (Mip-NeRF 360 scenes)"))
        leg = VGroup(tx("MPI", 22, C_MPI), tx("RGBD", 22, C_RGBD), tx("RPWS", 22, C_RPWS)).arrange(RIGHT, buff=0.6
                                                                                                  ).move_to([0, -2.55, 0])
        self.add(leg)
        self.play(Create(prp), Create(srp), run_time=0.6)
        self.play(Create(pr[0]), FadeIn(pr[1]), Create(sr[0]), FadeIn(sr[1]), run_time=0.8)
        self.caption("Quality: RGBD stays low however many layers it gets.")
        self.play(Create(pm[0]), FadeIn(pm[1]), Create(sm[0]), FadeIn(sm[1]), run_time=0.8)
        self.caption("MPI beats RGBD at every layer count and matches RPWS in SSIM.")
        v30 = VGroup(*[DashedLine(a_.c2p(np.log2(30), yr0), a_.c2p(np.log2(30), yr1), color=AMBER, stroke_width=2,
                                  dash_length=0.08) for a_, yr0, yr1 in ((pa, 21.0, 23.75), (sa, 0.45, 0.53))])
        self.add(v30, tx("≈ 30 layers", 18, AMBER).next_to(pa.c2p(np.log2(30), 21.0), UR, buff=0.08),
                 tx("≈ 30 layers", 18, AMBER).next_to(sa.c2p(np.log2(30), 0.45), UR, buff=0.08))
        self.caption("The paper finds perceptual quality saturates beyond roughly 30 layers.")
        self.clear_all()


# ================================================================== S08 Results
class S08_Results(Base):
    def construct(self):
        self.title_card("Part 8", "Results")
        self.header("8 · Results")
        # teaser panels (Fig. 1)
        w = 4.3
        pos = {"teaser_rgbd_front": (-2.25, 1.6), "teaser_rpws_front": (2.25, 1.6),
               "teaser_mpi_back": (-2.25, -1.35), "teaser_mpi_front": (2.25, -1.35)}
        ims = {k: img(k + ".png", w=w).move_to([x, y, 0]) for k, (x, y) in pos.items()}
        self.play(FadeIn(ims["teaser_rgbd_front"]), FadeIn(ims["teaser_rpws_front"]), run_time=0.5)
        self.add(src_note("simulated, paper Fig. 1"))
        self.caption("Paper Fig. 1: RGBD leaks light around the vase; RPWS is right but slow.")
        self.hold()
        self.play(FadeIn(ims["teaser_mpi_back"]), FadeIn(ims["teaser_mpi_front"]), run_time=0.5)
        self.caption("The MPI hologram refocuses from grass to vase with natural blur.")
        self.clear_all()

        # simulated focal stacks (Fig. 7), lego
        keys = ["gt", "rpws", "rgbd4", "mpi4"]
        names = ["ground truth", "RPWS", "RGBD, 4 layers", "MPI, 4 layers"]
        colors = [WHITE, C_RPWS, C_RGBD, C_MPI]
        xs = [-4.2, -1.4, 1.4, 4.2]
        w = 2.65
        for x, k, n, c in zip(xs, keys, names, colors):
            a = img(f"sim_{k}_lego_front.png", w=w).move_to([x, 1.2, 0])
            b = img(f"sim_{k}_lego_back.png", w=w).move_to([x, -1.5, 0])
            self.add(a, b, label(a, n, 22, c))
        self.add(tx("front focus", 20, DIM).move_to([-6.2, 1.2, 0]).rotate(PI / 2),
                 tx("back focus", 20, DIM).move_to([-6.2, -1.5, 0]).rotate(PI / 2),
                 tx("simulated focal stack, Blender lego, 24 frames (paper Fig. 7)", 16, DIM).move_to([0, -2.95, 0]))
        self.caption("Simulated focal stacks (paper Fig. 7): with 4 layers, MPI looks like RPWS.")
        self.clear_all()

        # captured (Fig. 8), garden
        keys = ["rpws", "rgbd4", "mpi4"]
        names = ["RPWS", "RGBD, 4 layers", "MPI, 4 layers"]
        colors = [C_RPWS, C_RGBD, C_MPI]
        xs = [-3.9, 0, 3.9]
        w = 3.7
        for x, k, n, c in zip(xs, keys, names, colors):
            a = img(f"exp_{k}_garden_front.png", w=w).move_to([x, 1.35, 0])
            b = img(f"exp_{k}_garden_zA.png", w=w).move_to([x, -1.2, 0])
            self.add(a, b, label(a, n, 22, c))
        self.add(tx("captured, front focus", 18, DIM).move_to([-6.35, 1.35, 0]).rotate(PI / 2),
                 tx("zoom-ins", 18, DIM).move_to([-6.35, -1.2, 0]).rotate(PI / 2),
                 tx("experimentally captured on a benchtop holographic display, 8 µm phase-only SLM (paper Fig. 8)",
                    16, DIM).move_to([0, -2.6, 0]))
        self.caption("Captured on a real holographic display prototype, the trend is the same.")
        self.clear_all()

        # captured light field at two focus depths (Fig. 5)
        t = ValueTracker(0.0)
        vk = ["left", "center", "right"]
        panels = []
        for x, fd in ((-3.35, "front"), (3.35, "back")):
            frs = [np.asarray(Image.open(os.path.join(DATA, f"lf5_{fd}_{v}.png")).convert("RGB").resize((840, 560),
                                                                                                        Image.LANCZOS))
                   for v in vk]
            m = arr_img(frs[1], w=6.3).move_to([x, 0.6, 0])
            live(m, (lambda fr_: (lambda: fr_[[0, 1, 2, 1][int(t.get_value() * 2.2) % 4]]))(frs))
            panels.append(m)
            self.add(label(m, f"{fd} focus", 24))
        self.play(*[FadeIn(m) for m in panels], run_time=0.5)
        vl = always_redraw(lambda: tx("pupil: " + ["left", "centre", "right", "centre"][int(t.get_value() * 2.2) % 4],
                                      22, CYAN).move_to([0, -1.75, 0]))
        self.add(vl, tx("experimentally captured light field, pupil moved at the Fourier plane (paper Fig. 5)", 16, DIM
                        ).move_to([0, -2.25, 0]))
        self.caption("Captured parallax: things off the focal plane shift as the pupil moves.")
        self.run(t, 5.0, 5.0)
        self.caption("Objects at the focused depth stay put, just as ray optics predicts.")
        self.run(t, 9.0, 4.0)
        for m in panels:
            m.clear_updaters()
        self.clear_all()

        # Stereo Magnification MPIs (Fig. 6)
        xs = [-4.3, 0, 4.3]
        w = 4.1
        for k, x in enumerate(xs):
            a = img(f"sm{k}_back.png", w=w).move_to([x, 1.35, 0])
            b = img(f"sm{k}_front.png", w=w).move_to([x, -1.0, 0])
            self.add(a, b)
        self.add(tx("back focus", 20, DIM).move_to([-6.7, 1.35, 0]).rotate(PI / 2),
                 tx("front focus", 20, DIM).move_to([-6.7, -1.0, 0]).rotate(PI / 2),
                 src_note("MPIs predicted by Stereo Magnification (Zhou et al. 2018) → holograms, paper Fig. 6"))
        self.caption("It also works with MPIs from an off-the-shelf network.")
        self.clear_all()

        # teaser videos
        a = video(self, "lf", h=3.6).move_to([-3.35, 0.55, 0])
        b = video(self, "fs", h=3.6).move_to([3.35, 0.55, 0])
        self.play(FadeIn(a), FadeIn(b), run_time=0.5)
        self.add(label(a, "light field", 24), label(b, "focal stack", 24),
                 tx("MPI holograms (paper teaser videos)", 16, DIM).move_to([0, -1.6, 0]))
        self.caption("MPI holograms: accurate focus and parallax, at layer-based speed.")
        self.hold(2.0)
        self.clear_all()


# ================================================================== S09 Outlook
class S09_Outlook(Base):
    def construct(self):
        self.title_card("Part 9", "Limitations and outlook")
        self.header("9 · Limitations and outlook")
        items = [
            ("Up to 250,000× faster, but not yet real time",
             "→ pair with real-time random-phase hologram networks"),
            ("Random phase lowers image contrast",
             "→ learning-based calibration of the display"),
        ]
        g = VGroup()
        for a, b in items:
            g.add(VGroup(tx(a, 26), tx(b, 24, AMBER)).arrange(DOWN, aligned_edge=LEFT, buff=0.12))
        g.arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([0, 1.0, 0])
        for it in g:
            self.add(it)
            self.wait(1.0)
        self.caption("Both are active directions in holographic displays.")
        self.clear_all()

        summ = [
            "1   Multiplane images as the 3D input for holograms",
            "2   Random-phase wave compositing, back to front, layer by layer",
            "3   Time multiplexing: 24 frames average into focus and parallax",
        ]
        sg = VGroup(*[tx(s, 28) for s in summ]).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([0, 1.2, 0])
        res = tx("quality close to RPWS  ·  far better than RGBD  ·  up to 250,000× faster than RPWS", 24, AMBER
                 ).move_to([0, -0.6, 0])
        self.add(sg)
        self.wait(1.2)
        self.add(res)
        self.caption("A handful of layers, a few FFTs each: 3D holograms at layer-based speed.")
        self.clear_all()
        self.remove(self.hdr)
        t1 = fit(tx("Fast Wave-optics Rendering of Multiplane Images", 44, bold=True)).move_to([0, 2.1, 0])
        t1b = tx("for 3D Holographic Displays", 32).move_to([0, 1.4, 0])
        t2 = tx("arXiv 2607.19731", 28).move_to([0, 0.45, 0])
        t4 = fit(tx("Brian Chao, Dario Seyb, Nathan Matsuda, Oliver Cossairt, Yang Zhou, Douglas Lanman, "
                    "Gordon Wetzstein, Grace Kuo, Changwon Jang", 20, DIM)).move_to([0, -0.6, 0])
        t5 = tx("Stanford University  ·  Meta", 22, DIM).move_to([0, -1.05, 0])
        self.add(t1, t1b, t2, t4, t5)
        self.wait(4.0)
