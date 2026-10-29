"""Random-phase Wave Splatting — teaser animation, v3.

Source: Chao*, Yang*, Choi, Gopakumar, Koiso, Wetzstein, "Random-phase Wave Splatting of Translucent Primitives
for Computer-generated Holography", arXiv 2508.17480 (2025). Follow-up of Gaussian Wave Splatting (arXiv 2505.06582).
Paper figures in data/ are crops of the paper (random-phase-wave-splatting/pdfs/, via scratch/crop_figs.py).
All wave-optics visuals are toy simulations computed here with numpy (angular spectrum method, random-phase
wave splatting of a two-layer scene, time multiplexing) and cached in data/generated/ (delete data/generated/done_v1
to rebuild; v2 adds data/generated/done_v2 extras).

Render (see render.sh):
    manim -ql --media_dir media animation_rpws_v3.py <Scene>
Scenes, in order: S00_Title ... S09_Outlook.
"""
from manim import *
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib
import fcntl
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GEN = os.path.join(DATA, "generated")
ASSET_FLAG = "done_v1"

config.background_color = BLACK

# ------------------------------------------------------------------ style (same as gaussian_wave_splatting)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × − → ≈ ∑ ∏ ℱ 𝒫 𝔼 ∠ …
SYMFONT = "DejaVu Sans"                 # ✗ is in neither
DIM = "#8A8A8A"
RED = "#FF5252"
GREEN = "#69F0AE"
BLUE = "#448AFF"
AMBER = "#FFCA28"
CYAN = "#4DD0E1"
PURPLE = "#E040FB"
ORANGE = "#FF9E40"

HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈ℝ≪≤↓↔∑∏∬∫ℱ𝒫𝒢𝒩ℋ𝒪𝒰𝔼∗≠∞∂∘⊤●⇒∠✓")
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


def I(v):
    return f"<i>{v}</i>"


def B(v):
    return f"<b>{v}</b>"


XB = B("x")
KB = B("k")
SQ = sup("2")
W1, W2 = I("w") + sub("1"), I("w") + sub("2")
PH1, PH2 = "φ" + sub("1"), "φ" + sub("2")
A1, A2 = "α" + sub("1"), "α" + sub("2")
C1, C2 = I("c") + sub("1"), I("c") + sub("2")
EQ_GWS = (f"{I('u')}{sub('SLM')} = ∑{sub(I('i'))} 𝒫( {I('c')}{sub(I('i'))} {I('o')}{sub(I('i'))} |{I('u')}{sub(I('i'))}| "
          f"{I('T')}{sub(I('i'))} {I('e')}{supsub(I('ikz'), I('i'))} ; −{I('z')}{sub(I('i'))} )")
EQ_GWS_T = f"{I('T')}{sub(I('i'))} = ∏{sub(I('j') + '&lt;' + I('i'))} (1 − {I('o')}{sub(I('j'))} |{I('u')}{sub(I('j'))}|)"
EQ_CROSS = (f"|{W1} {I('e')}{supsub(I('i') + 'φ', '1')} + {W2} {I('e')}{supsub(I('i') + 'φ', '2')}|{SQ} = {W1}{SQ} + {W2}{SQ} + "
            f"{col('2' + W1 + W2 + ' cos(' + PH1 + ' − ' + PH2 + ')', AMBER)}")
EQ_RPWS4 = (f"{I('g')}{sub(I('i') + '−1')} = 𝒫( {I('M')}{sub(I('i'))} {I('g')}{sub(I('i'))} + "
            f"√{I('c')}{sub(I('i'))} √({I('o')}{sub(I('i'))}|{I('u')}{sub(I('i'))}|) "
            f"exp({I('i')}∠{I('u')}{sub(I('i'))} + {I('i')}φ{sub(I('i'))}{sup('(' + I('t') + ')')}) ; Δ{I('z')} )")
EQ_RPWS5 = f"{I('M')}{sub(I('i'))} = √(1 − {I('o')}{sub(I('i'))} |{I('u')}{sub(I('i'))}|)"
EQ_TM = f"{I('I')} = (1/{I('T')}) ∑{sub(I('t'))} |𝒪( {I('u')}{sub('SLM')}{sup('(' + I('t') + ')')} )|{SQ}"

# ------------------------------------------------------------------ optics (numpy)
LAM = 520e-9          # green laser line
DX = 8e-6             # SLM pixel pitch of the prototype (Holoeye Pluto, 8 µm)
K = 2 * np.pi / LAM
FMAX = 1 / (2 * DX)
N = 512               # simulation grid (4.1 mm)
ZF, ZB = 6e-3, 18e-3  # front star / rear checkerboard, distance from the SLM
TFR = 24              # time-multiplexed frames (the paper's largest setting)
CROP = slice(96, 416) # displayed 320 × 320 window
R_EYE = 0.4           # eye pupil radius for focal stacks (fraction of the SLM bandwidth half-width)
R_SUB = 0.22          # sub-aperture pupil radius for light-field views
S_MAX = 0.72          # largest pupil offset in the eyebox
NV = 41               # pupil positions
ZS = np.linspace(2e-3, 24e-3, 45)   # focal sweep


def _grids(n=N, dx=DX):
    f = np.fft.fftfreq(n, dx)
    FX, FY = np.meshgrid(f, f)
    FZ = np.sqrt(np.maximum(1 / LAM ** 2 - FX ** 2 - FY ** 2, 0))
    return FX, FY, FZ


def asm(u, z, dx=DX):
    """Angular spectrum method: P(u; z) = F^-1{ F{u} · exp(j 2π f_z z) }."""
    FX, FY, FZ = _grids(u.shape[0], dx)
    return np.fft.ifft2(np.fft.fft2(u) * np.exp(1j * 2 * np.pi * FZ * z))


def pupil(cx=0.0, cy=0.0, r=0.5):
    FX, FY, _ = _grids()
    return (((FX / FMAX - cx) ** 2 + (FY / FMAX - cy) ** 2) <= r ** 2).astype(float)


def rphase(shape, rng):
    return np.exp(1j * rng.uniform(-np.pi, np.pi, shape))


def star_mask(n, r0, r1, c, npts=5, blur=1.5):
    im = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(im)
    pts = []
    for i in range(2 * npts):
        r = r1 if i % 2 == 0 else r0
        a = -np.pi / 2 + i * np.pi / npts
        pts.append((c[0] + r * np.cos(a), c[1] + r * np.sin(a)))
    d.polygon(pts, fill=255)
    im = im.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(im) / 255.0


def toy_scene():
    """Front: translucent-edged star (α up to 0.97) at 6 mm. Rear: opaque checkerboard plane at 18 mm."""
    y, x = np.mgrid[0:N, 0:N]
    sq = 32
    chk = (((x - N // 2) // sq + (y - N // 2) // sq) % 2).astype(float)
    half = 140
    win = ((np.abs(x - N / 2 + 0.5) < half) & (np.abs(y - N / 2 + 0.5) < half)).astype(np.uint8) * 255
    win = np.asarray(Image.fromarray(win).filter(ImageFilter.GaussianBlur(2))) / 255.0
    a_b, c_b = win, 0.18 + 0.72 * chk
    a_f = 0.97 * star_mask(N, 34, 80, (N / 2 - 10, N / 2 + 6))
    c_f = np.full((N, N), 0.95)
    return a_f, c_f, a_b, c_b


def smooth_slm(a_f, c_f, a_b, c_b):
    """GWS alpha wave blending: amplitude weights, front-to-back transmittance, phases matched at the SLM."""
    uf = a_f * np.sqrt(c_f) * np.exp(1j * K * ZF)
    ub = (1 - a_f) * a_b * np.sqrt(c_b) * np.exp(1j * K * ZB)
    return asm(uf, -ZF) + asm(ub, -ZB)


def rpws_slm(a_f, c_f, a_b, c_b, rng):
    """RPWS (paper Eqs. 4-5): back to front, sqrt weights, sqrt mask M = √(1 − α) at the occluder's plane."""
    g = np.sqrt(a_b * c_b) * rphase((N, N), rng)
    g = asm(g, -(ZB - ZF))
    g = np.sqrt(1 - a_f) * g + np.sqrt(a_f * c_f) * rphase((N, N), rng)
    return asm(g, -ZF)


def gwssqrt_slm(a_f, c_f, a_b, c_b, rng):
    """Ablation 'GWS with sqrt': sqrt weights, but GWS's transmittance (1 − α_front) applied on the rear plane."""
    uf = np.sqrt(a_f * c_f) * rphase((N, N), rng)
    ub = np.sqrt((1 - a_f) * a_b * c_b) * rphase((N, N), rng)
    return asm(uf, -ZF) + asm(ub, -ZB)


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


def cmap_rgb(a, cmap="inferno"):
    return (matplotlib.colormaps[cmap](np.clip(a, 0, 1))[..., :3] * 255).astype(np.uint8)


def grey(a, gamma=1.0):
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


def gauss2(n, cx, cy, su, sv, th):
    y, x = np.mgrid[0:n, 0:n].astype(float)
    x, y = x - cx, y - cy
    c, s = np.cos(th), np.sin(th)
    u, v = c * x + s * y, -s * x + c * y
    return np.exp(-0.5 * ((u / su) ** 2 + (v / sv) ** 2))


def _xz_maps(rng):
    """x–z side view of a 1D slit object (0.6 mm) with smooth vs random phase, 1D ASM, 8 µm samples."""
    n = 1024
    x = (np.arange(n) - n / 2) * DX
    amp = np.clip((0.3e-3 - np.abs(x)) / 16e-6 + 0.5, 0, 1)
    fx = np.fft.fftfreq(n, DX)
    fz = np.sqrt(np.maximum(1 / LAM ** 2 - fx ** 2, 0))
    zs = np.linspace(0, 20e-3, 420)
    keep = slice(n // 2 - 200, n // 2 + 200)

    def prop_all(u0):
        U = np.fft.fft(u0)
        return np.stack([np.abs(np.fft.ifft(U * np.exp(1j * 2 * np.pi * fz * z)))[keep] ** 2 for z in zs], 1)
    sm = prop_all(amp.astype(complex))
    r1 = prop_all(amp * np.exp(1j * rng.uniform(-np.pi, np.pi, n)))
    ra = 0
    for t in range(TFR):
        ra = ra + prop_all(amp * np.exp(1j * rng.uniform(-np.pi, np.pi, n)))
    ra = ra / TFR

    def out(m, name, scale, gamma=0.6):
        rgb = cmap_rgb((m / scale) ** gamma, "inferno")
        Image.fromarray(rgb).resize((840, 230), Image.BILINEAR).save(gen(name))
    out(sm, "xz_smooth.png", 1.3)
    out(r1, "xz_rand1.png", 1.3)
    out(ra, "xz_rand24.png", 1.3)


def _alpha_toy(rng):
    """Two translucent Gaussians at one depth (orange in front of blue), per-pixel random phase,
    viewed through a band-limited pupil. GT = ray alpha blending on intensity (paper Eq. 7 on c²).
    GWS-NRP: amplitude weights α₁√c₁, (1−α₁)α₂√c₂ with random phase. RPWS: √(α₁c₁), √((1−α₁)α₂c₂)."""
    n = 256
    a1 = 1.0 * gauss2(n, 108, 112, 40, 40, 0)
    a2 = 0.9 * gauss2(n, 150, 150, 52, 52, 0)
    c1 = np.array([1.0, 0.55, 0.18])
    c2 = np.array([0.30, 0.52, 1.0])
    gt = a1[..., None] * c1 + ((1 - a1) * a2)[..., None] * c2
    FX, FY, _ = _grids(n)
    P = ((FX / FMAX) ** 2 + (FY / FMAX) ** 2 <= 0.5 ** 2).astype(float)
    wn1 = a1[..., None] * np.sqrt(c1)
    wn2 = ((1 - a1) * a2)[..., None] * np.sqrt(c2)
    wr1 = np.sqrt(a1[..., None] * c1)
    wr2 = np.sqrt(((1 - a1) * a2)[..., None] * c2)
    accn = np.zeros((n, n, 3))
    accr = np.zeros((n, n, 3))
    avn, avr = [], []
    for t in range(TFR):
        for ch in range(3):
            p1, p2 = rphase((n, n), rng), rphase((n, n), rng)
            for w1, w2, acc in ((wn1, wn2, accn), (wr1, wr2, accr)):
                u = w1[..., ch] * p1 + w2[..., ch] * p2
                acc[..., ch] += np.abs(np.fft.ifft2(np.fft.fft2(u) * P)) ** 2 / (0.5 ** 2 * np.pi / 4)
        avn.append(accn / (t + 1))
        avr.append(accr / (t + 1))
    np.save(gen("toy_gt.npy"), gt.astype(np.float32))
    np.save(gen("toy_nrp.npy"), np.stack(avn).astype(np.float16))
    np.save(gen("toy_rpws.npy"), np.stack(avr).astype(np.float16))


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    if os.path.exists(gen(ASSET_FLAG)):
        return
    with open(gen("lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(gen(ASSET_FLAG)):
            return
        rng = np.random.default_rng(2025)
        a_f, c_f, a_b, c_b = toy_scene()
        FX, FY, FZ = _grids()
        Hz = lambda z: np.exp(1j * 2 * np.pi * FZ * z)
        Pe = pupil(r=R_EYE)
        svals = np.linspace(-S_MAX, S_MAX, NV)
        Psub = [pupil(s, 0, R_SUB) for s in svals]
        Hf = Hz(ZF)
        Hs = [Hz(z) for z in ZS]
        # ---- smooth phase (GWS)
        us = smooth_slm(a_f, c_f, a_b, c_b)
        Fs = np.fft.fft2(us)
        np.save(gen("slm_smooth.npy"), us[CROP, CROP].astype(np.complex64))
        fs_s = np.stack([np.abs(np.fft.ifft2(Fs * Pe * H))[CROP, CROP] ** 2 for H in Hs])
        lf_s = np.stack([np.abs(np.fft.ifft2(Fs * P * Hf))[CROP, CROP] ** 2 for P in Psub])
        # ---- random phase: RPWS and the 'GWS with sqrt' ablation, TFR frames each
        fs_r = np.zeros_like(fs_s)
        fs_r1 = None
        lf_r = np.zeros_like(lf_s)
        lf_g = np.zeros_like(lf_s)
        frames = []
        for t in range(TFR):
            ur = rpws_slm(a_f, c_f, a_b, c_b, rng)
            Fr = np.fft.fft2(ur)
            if t == 0:
                np.save(gen("slm_rand.npy"), ur[CROP, CROP].astype(np.complex64))
                spec_r = np.abs(np.fft.fftshift(Fr)) ** 2
            st = np.stack([np.abs(np.fft.ifft2(Fr * Pe * H))[CROP, CROP] ** 2 for H in Hs])
            if t == 0:
                fs_r1 = st.copy()
            fs_r += st
            frames.append(np.abs(np.fft.ifft2(Fr * Pe * Hf))[CROP, CROP] ** 2)
            lf_r += np.stack([np.abs(np.fft.ifft2(Fr * P * Hf))[CROP, CROP] ** 2 for P in Psub])
            Fg = np.fft.fft2(gwssqrt_slm(a_f, c_f, a_b, c_b, rng))
            lf_g += np.stack([np.abs(np.fft.ifft2(Fg * P * Hf))[CROP, CROP] ** 2 for P in Psub])
        fs_r /= TFR
        lf_r /= TFR
        lf_g /= TFR
        # normalisation: each method by its own in-focus / centre-view level
        iz = int(np.argmin(np.abs(ZS - ZF)))
        ns = np.percentile(fs_s[iz], 99.5)
        nr = np.percentile(fs_r[iz], 99.5)
        np.save(gen("fs_smooth.npy"), grey(fs_s / ns))
        np.save(gen("fs_rand24.npy"), grey(fs_r / nr))
        np.save(gen("fs_rand1.npy"), grey(fs_r1 / nr))
        c0 = NV // 2
        np.save(gen("lf_smooth.npy"), grey(lf_s / np.percentile(lf_s[c0], 99.5)))
        nl = np.percentile(lf_r[c0], 99.5)
        np.save(gen("lf_rpws.npy"), grey(lf_r / nl))
        np.save(gen("lf_gwssqrt.npy"), grey(lf_g / nl))
        frames = np.stack(frames)
        cum = np.cumsum(frames, 0) / np.arange(1, TFR + 1)[:, None, None]
        np.save(gen("tm_frames.npy"), grey(frames / nr))
        np.save(gen("tm_cum.npy"), grey(cum / nr))
        # speckle contrast inside the star (uniform intensity region) vs number of frames
        inner = (a_f > 0.95)[CROP, CROP]
        Cs = np.array([cum[t][inner].std() / cum[t][inner].mean() for t in range(TFR)])
        np.save(gen("speckle_C.npy"), Cs)
        # spectra (eyebox = Fourier plane), log scale over 5 decades
        spec_s = np.abs(np.fft.fftshift(Fs)) ** 2
        for name, sp in (("spec_smooth.png", spec_s), ("spec_rand.png", spec_r)):
            L = np.log10(sp + 1e-20)
            L = np.clip((L - (L.max() - 5)) / 5, 0, 1)
            Image.fromarray(cmap_rgb(L, "magma")).resize((360, 360), Image.BILINEAR).save(gen(name))
        Image.fromarray(phase_rgb(np.angle(us[CROP, CROP]))).save(gen("phase_smooth.png"))
        Image.fromarray(phase_rgb(np.angle(ur[CROP, CROP]))).save(gen("phase_rand.png"))
        # ground-truth layers (for the recap)
        Image.fromarray(grey((a_f * c_f + (1 - a_f) * a_b * c_b)[CROP, CROP])).save(gen("scene_gt.png"))
        # ---- 1D side views, alpha-blending toy
        _xz_maps(rng)
        _alpha_toy(rng)
        # ---- Gaussians (RGBA) for the recap
        n = 400
        for name, (cx, cy, su, sv, th, c, o) in {
            "g_red": (200, 200, 70, 30, 0.5, (255, 82, 82), 0.95),
            "g_amber": (235, 225, 60, 45, -0.3, (255, 202, 40), 0.9),
            "g_blue": (170, 240, 85, 40, 1.2, (68, 138, 255), 0.9),
        }.items():
            a = o * gauss2(n, cx, cy, su, sv, th)
            im = np.zeros((n, n, 4), np.uint8)
            im[..., :3] = c
            im[..., 3] = (a * 255).astype(np.uint8)
            Image.fromarray(im, "RGBA").save(gen(name + ".png"))
        open(gen(ASSET_FLAG), "w").write("ok")


def build_assets_v2():
    """v2 extras: SLM fields in domain colouring and angular spectra over 7 decades (display capped for the flat one)."""
    if os.path.exists(gen("done_v2")):
        return
    with open(gen("lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(gen("done_v2")):
            return
        rng = np.random.default_rng(7)
        a_f, c_f, a_b, c_b = toy_scene()
        us = smooth_slm(a_f, c_f, a_b, c_b)
        ur = rpws_slm(a_f, c_f, a_b, c_b, rng)
        for name, u in (("field_smooth.png", us), ("field_rand.png", ur)):
            Image.fromarray(domain_rgb(u[CROP, CROP], 0.7)).save(gen(name))
        for name, u, cap in (("spec7_smooth.png", us, 1.0), ("spec7_rand.png", ur, 0.62)):
            L = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(u))) ** 2 + 1e-20)
            L = np.clip((L - (L.max() - 7)) / 7, 0, 1)
            if cap < 1:                      # flat spectrum: show its mean level, not its speckle peaks
                L = L * cap / np.percentile(L, 99)
            Image.fromarray(cmap_rgb(np.clip(L, 0, 1), "magma")).resize((360, 360), Image.BILINEAR).save(gen(name))
        open(gen("done_v2"), "w").write("ok")


build_assets()
build_assets_v2()


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
    return ((1 - w) * frames[i0] + w * frames[i1]).astype(np.uint8)


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


def toy_note(s="toy simulation: λ = 520 nm, 8 µm pixels"):
    return tx(s, 16, DIM)


# ================================================================== S00 Title
class S00_Title(Base):
    def construct(self):
        title = tx("Random-phase Wave Splatting", 60, bold=True).move_to([0, 2.95, 0])
        subt = tx("of Translucent Primitives for Computer-generated Holography", 32).move_to([0, 2.22, 0])
        auth = tx("Brian Chao*, Jacqueline Yang*, Suyeon Choi, Manu Gopakumar, Ryota Koiso, Gordon Wetzstein",
                  22, DIM).move_to([0, -2.5, 0])
        aff = tx("Stanford University  ·  KDDI Research  ·  arXiv 2508.17480 (2025)", 22, DIM).move_to([0, -2.95, 0])
        eq = tx("* equal contribution", 18, DIM).move_to([0, -3.35, 0])
        frames = np.load(gen("tm_frames.npy"))
        cum = np.load(gen("tm_cum.npy"))
        t = ValueTracker(0.0)
        h = 3.0
        a = arr_img(frames[0], h).move_to([-2.0, -0.35, 0])
        b = arr_img(cum[0], h).move_to([2.0, -0.35, 0])
        live(a, lambda: frames[int(np.clip(t.get_value(), 0, TFR - 1))])
        live(b, lambda: cum[int(np.clip(t.get_value(), 0, TFR - 1))])
        la = label(a, "one random-phase frame", 22, DIM)
        lb = always_redraw(lambda: label(b, f"average of {int(np.clip(t.get_value(), 0, TFR - 1)) + 1} frames", 22, DIM))
        self.add(title, subt, auth, aff, eq)
        self.play(FadeIn(a), FadeIn(b), Create(framed(a, DIM, 1.2)), Create(framed(b, DIM, 1.2)), run_time=0.8)
        self.add(la, lb)
        self.run(t, TFR - 0.01, 6.0, rate=lambda x: x ** 1.6)
        self.wait(1.5)


# ================================================================== S01 Recap
class S01_Recap(Base):
    def construct(self):
        self.title_card("Part 1", "Recap: Gaussian Wave Splatting")
        self.header("1 · Recap: Gaussian Wave Splatting")
        layers = [img(gen(n), 2.7) for n in ("g_blue", "g_amber", "g_red")]
        for m in layers:
            m.move_to([-4.6, 0.95, 0])
        self.play(LaggedStart(*[FadeIn(m) for m in layers], lag_ratio=0.3), run_time=0.9)
        lg = label(layers[0], "2D Gaussian splats", 24)
        self.add(lg)
        ar = arrow([-2.9, 0.95, 0], [-1.2, 0.95, 0])
        la = VGroup(tx("closed-form", 20, DIM), tx("wavefronts", 20, DIM)).arrange(DOWN, buff=0.06).next_to(ar, UP, buff=0.12)
        ph = img(gen("field_smooth.png"), 2.7).move_to([0.55, 0.95, 0])
        sc = img(gen("scene_gt.png"), 2.7).move_to([4.6, 0.95, 0])
        self.caption("Gaussian Wave Splatting (GWS, SIGGRAPH 2025) turns each 2D Gaussian into its wavefront in closed form.")
        self.play(GrowArrow(ar), FadeIn(ph), run_time=0.7)
        self.add(la, label(ph, "SLM field (hue = phase)", 24))
        self.hold()
        e1 = tx(EQ_GWS, 30).move_to([0, -1.25, 0])
        e2 = tx(EQ_GWS_T, 28).move_to([0, -1.95, 0])
        self.add(e1, e2)
        self.caption("Alpha wave blending adds them up at the SLM, front to back, dimmed by the transmittance T.")
        ar2 = arrow([2.05, 0.95, 0], [3.1, 0.95, 0])
        la2 = tx("𝒫(·; " + I("z") + ")", 20, DIM).next_to(ar2, UP, buff=0.12)
        self.play(GrowArrow(ar2), FadeIn(sc), run_time=0.6)
        self.add(la2, label(sc, "target scene (toy)", 24))
        note = tx("toy scene: star 6 mm and checkerboard 18 mm from the SLM; λ = 520 nm, 8 µm pixels", 16, DIM
                  ).move_to([0, -2.55, 0])
        self.add(note)
        self.caption(f"Each wave carries {I('e')}{supsub(I('ikz'), I('i'))}, so all arrive in step: one smooth phase (a single hue).")
        self.caption("Sharp and photoreal in focus. But what does that smooth phase cost?")
        self.clear_all()


# ================================================================== S02 Phase
class S02_Phase(Base):
    def construct(self):
        self.title_card("Part 2", "Smooth phase vs random phase")
        self.header("2 · Smooth phase vs random phase")
        # --- 1D side views
        w = 8.4
        sm = img(gen("xz_smooth.png"), w=w).move_to([0.9, 1.35, 0])
        r1 = img(gen("xz_rand1.png"), w=w).move_to([0.9, -1.1, 0])
        r24 = img(gen("xz_rand24.png"), w=w).move_to(r1)
        l1 = tx("smooth phase", 26).move_to([-4.9, 1.35, 0])
        l2 = tx("random phase", 26).move_to([-4.9, -1.1, 0])
        l2b = tx("one frame", 20, DIM).next_to(l2, DOWN, buff=0.12)
        zlab = tx(f"propagation distance {I('z')}: 0 → 20 mm", 20, DIM).next_to(sm, UP, buff=0.12)
        note = toy_note("toy simulation: 0.6 mm slit, λ = 520 nm, 8 µm samples; vertical axis (±1.6 mm) stretched"
                        ).next_to(r1, DOWN, buff=0.15)
        self.play(FadeIn(sm), run_time=0.6)
        self.add(l1, zlab, note)
        self.caption("A slit of light with smooth (constant) phase: the beam just keeps going straight.")
        self.hold()
        self.play(FadeIn(r1), run_time=0.6)
        self.add(l2, l2b)
        self.caption("Give every point a random phase: light scatters into every angle the pixels can produce.")
        self.hold()
        self.play(FadeIn(r24), run_time=0.8)
        self.remove(r1, l2b)
        l2b = tx("average of 24 frames", 20, DIM).next_to(l2, DOWN, buff=0.12)
        self.add(l2b)
        self.caption("Averaged over frames: a smooth cone, like light from a real diffuse surface.")
        self.clear_all()

        # --- 2D: SLM phase and angular spectrum
        h = 2.75
        xs = [-5.15, -1.95, 1.95, 5.15]
        files = ["field_smooth.png", "spec7_smooth.png", "field_rand.png", "spec7_rand.png"]
        names = ["SLM field (hue = phase)", "angular spectrum", "SLM field (hue = phase)", "angular spectrum"]
        ims = []
        labs = []
        for x, f, nm in zip(xs, files, names):
            m = img(gen(f), h).move_to([x, 0.45, 0])
            ims.append(m)
            labs.append(label(m, nm, 22, DIM))
        g1 = tx("smooth phase (GWS)", 28).move_to([(xs[0] + xs[1]) / 2, 2.55, 0])
        g2 = tx("random phase", 28).move_to([(xs[2] + xs[3]) / 2, 2.55, 0])
        sep = Line([0, 2.7, 0], [0, -1.5, 0], color=DIM, stroke_width=1.5)
        self.play(FadeIn(ims[0]), FadeIn(ims[1]), run_time=0.6)
        self.add(g1, sep, labs[0], labs[1])
        n1 = tx("energy piled up at the centre", 22, RED).move_to([(xs[0] + xs[1]) / 2, -1.25, 0])
        self.add(n1)
        self.caption("The toy scene's hologram. Smooth phase puts nearly all energy at low spatial frequencies.")
        self.hold()
        self.play(FadeIn(ims[2]), FadeIn(ims[3]), run_time=0.6)
        self.add(g2, labs[2], labs[3])
        n2 = tx("fills the SLM's whole bandwidth", 22, GREEN).move_to([(xs[2] + xs[3]) / 2, -1.25, 0])
        self.add(n2)
        spn = tx("spectra: log scale, 7 decades; the square = every direction the 8 µm pixels can send light", 18, DIM
                 ).move_to([0, -1.9, 0])
        self.add(spn)
        self.caption("Random phase spreads energy evenly over the band: the SLM's space-bandwidth product is used.")
        self.caption("Each frequency is a direction. In a near-eye display, the spectrum becomes the eyebox.")
        self.clear_all()


# ================================================================== S03 Eyebox
class S03_Eyebox(Base):
    def construct(self):
        self.title_card("Part 3", "A full eyebox, with parallax")
        self.header("3 · A full eyebox, with parallax")
        h = 2.6
        lf_s = np.load(gen("lf_smooth.npy"))
        lf_r = np.load(gen("lf_rpws.npy"))
        s = ValueTracker(0.0)
        idx = lambda: (s.get_value() + S_MAX) / (2 * S_MAX) * (NV - 1)
        xs = [-5.25, -2.35, 2.35, 5.25]
        spec_s = img(gen("spec7_smooth.png"), h).move_to([xs[0], 0.5, 0])
        spec_r = img(gen("spec7_rand.png"), h).move_to([xs[2], 0.5, 0])
        vs = arr_img(lf_s[NV // 2], h).move_to([xs[1], 0.5, 0])
        vr = arr_img(lf_r[NV // 2], h).move_to([xs[3], 0.5, 0])
        live(vs, lambda: interp(lf_s, idx()))
        live(vr, lambda: interp(lf_r, idx()))
        fb = [framed(vs, DIM, 1.2), framed(vr, DIM, 1.2)]
        g1 = tx("smooth phase (GWS)", 28).move_to([(xs[0] + xs[1]) / 2, 2.6, 0])
        g2 = tx("random phase (RPWS, 24 frames)", 28).move_to([(xs[2] + xs[3]) / 2, 2.6, 0])
        labs = [label(spec_s, "eyebox (spectrum)", 22, DIM), label(vs, "what the eye sees", 22, DIM),
                label(spec_r, "eyebox (spectrum)", 22, DIM), label(vr, "what the eye sees", 22, DIM)]
        rad = R_SUB * h / 2

        def circ(spec):
            return always_redraw(lambda: Circle(radius=rad, color=CYAN, stroke_width=5).move_to(
                spec.get_center() + RIGHT * s.get_value() * h / 2))
        c1, c2 = circ(spec_s), circ(spec_r)
        sep = Line([0, 2.75, 0], [0, -1.4, 0], color=DIM, stroke_width=1.5)
        self.play(FadeIn(spec_s), FadeIn(spec_r), FadeIn(vs), FadeIn(vr), *[Create(f) for f in fb], run_time=0.8)
        self.add(g1, g2, sep, *labs, c1, c2)
        pl = always_redraw(lambda: tx(f"pupil position: {s.get_value() / S_MAX * 100:+.0f}% of the way to the edge",
                                      22, CYAN).move_to([0, -1.55, 0]))
        note = toy_note("toy simulation: 24 frames, star 6 mm / checkerboard 18 mm from the SLM, focused on the star"
                        ).move_to([0, -2.1, 0])
        self.add(pl, note)
        self.caption("The eye's pupil catches one patch of the eyebox (cyan). Centred, both holograms look fine.")
        self.run(s, -S_MAX, 2.5, smooth)
        self.caption("Move the pupil: the smooth-phase hologram goes dark. Its light never gets there.")
        self.run(s, S_MAX, 4.5, smooth)
        self.caption("The random-phase hologram stays bright everywhere, and the star slides against the checkerboard.")
        self.run(s, -S_MAX, 4.5, smooth)
        self.caption("That is parallax: near and far objects shift differently as your eye moves.")
        self.run(s, 0.0, 2.5, smooth)
        for m in (vs, vr):
            m.clear_updaters()
        self.clear_all()


# ================================================================== S04 Defocus
class S04_Defocus(Base):
    def construct(self):
        self.title_card("Part 4", "Natural defocus blur")
        self.header("4 · Natural defocus blur")
        stacks = [np.load(gen(f)) for f in ("fs_smooth.npy", "fs_rand1.npy", "fs_rand24.npy")]
        z = ValueTracker(ZF * 1e3)
        zi = lambda: (z.get_value() * 1e-3 - ZS[0]) / (ZS[1] - ZS[0])
        h = 3.2
        xs = [-4.5, 0, 4.5]
        names = ["smooth phase (GWS)", "random phase, 1 frame", "random phase, 24 frames"]
        cols = [RED, WHITE, GREEN]
        ims = []
        for x, stck, nm, c in zip(xs, stacks, names, cols):
            m = arr_img(stck[0], h).move_to([x, 0.7, 0])
            live(m, (lambda st: (lambda: interp(st, zi())))(stck))
            ims.append(m)
            self.add(label(m, nm, 24, c))
        self.play(*[FadeIn(m) for m in ims], *[Create(framed(m, DIM, 1.2)) for m in ims], run_time=0.8)
        nl = NumberLine(x_range=[2, 24, 2], length=8.0, color=DIM, include_numbers=False, stroke_width=2
                        ).move_to([0.6, -1.75, 0])
        nlab = tx(f"focus {I('z')} (mm)", 20, DIM).next_to(nl, LEFT, buff=0.25)
        ticks = VGroup(*[tx(str(v), 16, DIM).next_to(nl.n2p(v), DOWN, buff=0.12) for v in range(4, 25, 4)])
        m1 = Triangle(color=AMBER, fill_opacity=1).scale(0.09).rotate(PI).next_to(nl.n2p(6), UP, buff=0.04)
        m2 = Triangle(color=CYAN, fill_opacity=1).scale(0.09).rotate(PI).next_to(nl.n2p(18), UP, buff=0.04)
        s1 = tx("star", 18, AMBER).next_to(m1, UP, buff=0.05)
        s2 = tx("checkerboard", 18, CYAN).next_to(m2, UP, buff=0.05)
        dot = always_redraw(lambda: Dot(nl.n2p(float(np.clip(z.get_value(), 2, 24))), radius=0.08, color=WHITE))
        zval = always_redraw(lambda: tx(f"z = {z.get_value():.1f} mm", 22).next_to(nl, RIGHT, buff=0.3))
        note = toy_note("toy simulation, eye pupil = half the eyebox width").move_to([0, -2.55, 0])
        self.add(nl, nlab, ticks, m1, m2, s1, s2, dot, zval, note)
        self.caption("Focus on the star, 6 mm. The checkerboard sits 12 mm behind it.")
        self.caption("Smooth phase: the far checkerboard barely blurs. It rings instead.")
        self.hold()
        self.run(z, 18, 2.5, smooth)
        self.caption("Focus on the checkerboard: the smooth-phase star stays sharp-edged and ringing.")
        self.caption("Random phase blurs out-of-focus content like a real camera, but one frame is full of speckle.")
        self.run(z, 6, 3.0, smooth)
        self.caption("With 24 frames the speckle averages out: clean, natural defocus blur, a real depth cue.")
        self.run(z, 18, 3.0, smooth)
        self.run(z, 12, 1.5, smooth)
        for m in ims:
            m.clear_updaters()
        self.clear_all()


# ================================================================== S05 Speckle
class S05_Speckle(Base):
    def construct(self):
        self.title_card("Part 5", "Speckle and time multiplexing")
        self.header("5 · Speckle and time multiplexing")
        frames = np.load(gen("tm_frames.npy"))
        cum = np.load(gen("tm_cum.npy"))
        Cs = np.load(gen("speckle_C.npy"))
        t = ValueTracker(0.0)
        ti = lambda: int(np.clip(t.get_value(), 0, TFR - 1))
        h = 3.1
        a = arr_img(frames[0], h).move_to([-4.85, 0.7, 0])
        b = arr_img(cum[0], h).move_to([-1.45, 0.7, 0])
        live(a, lambda: frames[ti()])
        live(b, lambda: cum[ti()])
        la = always_redraw(lambda: label(a, f"frame {ti() + 1}", 24))
        lb = always_redraw(lambda: label(b, f"average of {ti() + 1}", 24))
        ax = Axes(x_range=[0, 24, 4], y_range=[0, 1.0, 0.25], x_length=4.6, y_length=2.9, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2, "include_numbers": False}).move_to([4.2, 0.55, 0])
        xl = tx(f"frames {I('T')}", 20, DIM).next_to(ax, DOWN, buff=0.35)
        yl = tx("speckle contrast", 20, DIM).next_to(ax, UP, buff=0.15)
        xt = VGroup(*[tx(str(v), 16, DIM).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v in (1, 8, 16, 24)])
        yt = VGroup(*[tx(f"{v:.1f}", 16, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0.5, 1.0)])
        c0 = Cs[0]
        theory = ax.plot(lambda x: c0 / np.sqrt(max(x, 1)), x_range=[1, 24, 0.1], color=AMBER, stroke_width=2.5)
        theory = DashedVMobject(theory, num_dashes=40)
        tl = tx(f"∝ 1/√{I('T')}", 22, AMBER).move_to(ax.c2p(17, 0.55))
        dots = always_redraw(lambda: VGroup(*[Dot(ax.c2p(k + 1, Cs[k]), radius=0.05, color=WHITE)
                                              for k in range(ti() + 1)]))
        self.play(FadeIn(a), FadeIn(b), Create(ax), run_time=0.8)
        self.add(la, lb, xl, yl, xt, yt, dots)
        note = toy_note("toy simulation (same scene, focused on the star); contrast = std / mean inside the star"
                        ).move_to([0, -1.75, 0])
        self.add(note)
        self.caption(f"Show {I('T')} holograms in quick succession, each with fresh random phases. The eye averages them.")
        self.run(t, 7.99, 4.0)
        self.add(theory, tl)
        self.caption(f"Speckle contrast falls like 1/√{I('T')}. The paper goes up to 24 frames.")
        self.run(t, 23.99, 4.0)
        self.caption("That is the classic reason to time-multiplex. RPWS finds a deeper one.")
        for m in (a, b):
            m.clear_updaters()
        self.clear_all()


# ================================================================== S06 Alpha blending
class S06_Alpha(Base):
    def construct(self):
        self.title_card("Part 6", "Alpha blending random-phase waves")
        self.header("6 · Alpha blending random-phase waves")
        eq = tx(EQ_CROSS, 34).move_to([0, 2.5, 0])
        self.add(eq)
        lab = tx("cross term", 20, AMBER).next_to(eq, DOWN, buff=0.12).align_to(eq, RIGHT).shift(LEFT * 0.9)
        self.add(lab)
        self.caption(f"Blend two overlapping waves with weights {W1}, {W2}: the intensity picks up a cross term.")
        l1 = VGroup(tx("smooth phase:  cos = 1", 28, RED),
                    tx(f"⇒  |{I('u')}| = {W1} + {W2}", 28),
                    tx("blending is linear in amplitude (GWS)", 22, DIM)).arrange(DOWN, buff=0.18)
        l2 = VGroup(tx(f"random phase:  𝔼[cos] = 0", 28, GREEN),
                    tx(f"⇒  𝔼|{I('u')}|{SQ} = {W1}{SQ} + {W2}{SQ}", 28),
                    tx("blending is linear in intensity, on average", 22, DIM)).arrange(DOWN, buff=0.18)
        l1.move_to([-3.4, 0.6, 0])
        l2.move_to([3.4, 0.6, 0])
        self.hold()
        self.add(l1)
        self.caption("Smooth phase: waves in step, amplitudes add. That is GWS's alpha wave blending.")
        self.hold()
        self.add(l2)
        self.caption("Random phases: the cross term averages to zero over frames, so intensities add.")
        self.hold()
        w_nrp = tx(f"GWS weights:  {W1} = {A1}√{C1},  {W2} = (1 − {A1}) {A2}√{C2}", 26, RED).move_to([0, -1.25, 0])
        w_rp = tx(f"RPWS weights:  {W1} = √({A1}{C1}),  {W2} = √((1 − {A1}) {A2}{C2})", 26, GREEN).move_to([0, -1.9, 0])
        self.add(w_nrp, w_rp)
        self.caption("So RPWS blends in intensity: square-root weights, on primitives fitted to intensity.")
        self.clear_all()

        # --- toy: ground truth / GWS-NRP / RPWS with live frame averaging
        gt = np.load(gen("toy_gt.npy"))
        nrp = np.load(gen("toy_nrp.npy")).astype(np.float32)
        rp = np.load(gen("toy_rpws.npy")).astype(np.float32)
        t = ValueTracker(0.0)
        ti = lambda: int(np.clip(t.get_value(), 0, TFR - 1))
        disp = lambda a: (np.clip(a, 0, 1) ** (1 / 1.8) * 255).astype(np.uint8)

        from scipy.ndimage import gaussian_filter
        gtb = gaussian_filter(gt, (6, 6, 0))
        errs = {}

        def err(which, i):
            if (which, i) not in errs:
                a = (nrp if which == "n" else rp)[i]
                errs[(which, i)] = cmap_rgb(np.abs(gaussian_filter(a, (6, 6, 0)) - gtb).sum(-1) / 0.9, "inferno")
            return errs[(which, i)]
        h = 2.6
        xs = [-4.4, 0, 4.4]
        mg = arr_img(disp(gt), h).move_to([xs[0], 1.05, 0])
        mn = arr_img(disp(nrp[0]), h).move_to([xs[1], 1.05, 0])
        mr = arr_img(disp(rp[0]), h).move_to([xs[2], 1.05, 0])
        live(mn, lambda: disp(nrp[ti()]))
        live(mr, lambda: disp(rp[ti()]))
        en = arr_img(err("n", 0), 1.35).next_to(mn, DOWN, buff=0.45)
        er = arr_img(err("r", 0), 1.35).next_to(mr, DOWN, buff=0.45)
        live(en, lambda: err("n", ti()))
        live(er, lambda: err("r", ti()))
        self.play(FadeIn(mg), FadeIn(mn), FadeIn(mr), run_time=0.7)
        self.add(label(mg, "ray-traced target", 24), label(mn, "GWS-NRP (naïve)", 24, RED),
                 label(mr, "RPWS (√ weights)", 24, GREEN))
        self.add(en, er, framed(en, DIM, 1), framed(er, DIM, 1))
        self.add(tx("error", 20, DIM).next_to(en, LEFT, buff=0.2), tx("error", 20, DIM).next_to(er, LEFT, buff=0.2))
        cnt = always_redraw(lambda: tx(f"frames averaged: {ti() + 1}", 24).move_to([xs[0], -1.0, 0]))
        note = toy_note("toy simulation: orange Gaussian in front of blue, same depth; error = |result − target| after a 6-pixel blur"
                        ).move_to([0, -2.65, 0])
        self.add(cnt, note)
        self.caption("Toy: orange in front of blue. After one frame, both are speckled.")
        self.run(t, 23.99, 5.0, rate=lambda x: x ** 1.5)
        self.caption("GWS with random phase (GWS-NRP) keeps a ring of wrong colour, however many frames.")
        self.caption("RPWS converges to the target: exact alpha blending, in expectation over frames.")
        self.caption("So time multiplexing is not a speckle trick: it is what makes the blending correct.")
        for m in (mn, mr, en, er):
            m.clear_updaters()
        self.clear_all()


# ================================================================== S07 Compositing
class S07_Compositing(Base):
    def construct(self):
        self.title_card("Part 7", "Back-to-front wave compositing")
        self.header("7 · Back-to-front wave compositing")
        e4 = tx(EQ_RPWS4, 30).move_to([0, 2.6, 0])
        e5 = tx(EQ_RPWS5, 28).move_to([0, 1.85, 0])
        self.add(e4, e5)
        # side-view diagram: three primitive planes, SLM on the right
        y0 = -0.35
        slm = Rectangle(width=0.16, height=2.6, fill_color="#546E7A", fill_opacity=1, stroke_width=0).move_to([5.9, y0, 0])
        lslm = tx("SLM", 22, DIM).next_to(slm, DOWN, buff=0.12)
        px = [-4.6, -1.6, 1.4]
        colors = [BLUE, GREEN, ORANGE]
        planes = VGroup(*[DashedLine([x, y0 - 1.25, 0], [x, y0 + 1.25, 0], color=DIM, stroke_width=1.5) for x in px])
        blobs = VGroup(*[Ellipse(width=0.22, height=1.0 - 0.15 * k, color=c, fill_opacity=0.85, stroke_width=0
                                 ).move_to([x, y0 + 0.3 * (1 - k), 0]) for k, (x, c) in enumerate(zip(px, colors))])
        names = VGroup(*[tx(s, 20, DIM).next_to(planes[k], DOWN, buff=0.1) for k, s in
                         enumerate(["far primitive", "primitive " + I("i"), "near primitive"])])
        self.play(FadeIn(slm), Create(planes), FadeIn(blobs), run_time=0.8)
        self.add(lslm, names)
        self.caption("RPWS works like the painter's algorithm, but with waves: start with the farthest primitive.")
        arrs = VGroup()
        labs = VGroup()
        for k in range(3):
            a0 = px[k] + 0.2
            a1 = (px[k + 1] - 0.2) if k < 2 else 5.7
            ar = arrow([a0, y0, 0], [a1, y0, 0], color=AMBER)
            arrs.add(ar)
            labs.add(tx(f"𝒫(·; Δ{I('z')})", 20, AMBER).next_to(ar, UP, buff=0.08))
        masks = VGroup(*[VGroup(tx("× √(1 − α)", 18), tx("+ own wave", 18)).arrange(DOWN, buff=0.05
                                                                                         ).next_to(planes[k], UP, buff=0.08)
                         for k in (1, 2)])
        self.hold()
        self.play(GrowArrow(arrs[0]), run_time=0.5)
        self.add(labs[0], masks[0])
        self.caption("Carry its wave to the next plane, mask by √(1 − α), add that primitive's random-phase wave.")
        self.hold()
        self.play(GrowArrow(arrs[1]), run_time=0.5)
        self.add(labs[1], masks[1])
        self.play(GrowArrow(arrs[2]), run_time=0.5)
        self.add(labs[2])
        self.caption("Repeat until the SLM. Each mask sits at the occluder's depth, blocking light from every angle.")
        self.clear_all()

        # --- toy light-field views: GWS with sqrt vs RPWS
        lf_g = np.load(gen("lf_gwssqrt.npy"))
        lf_r = np.load(gen("lf_rpws.npy"))
        h = 2.15
        vx = [-1.3, 1.25, 3.8]
        idxs = [0, NV // 2, NV - 1]
        heads = ["left of eyebox", "centre", "right of eyebox"]
        rows = [(lf_g, 1.2, "GWS with √ weights", "mask on the far plane", RED),
                (lf_r, -1.15, "RPWS", "mask at the occluder", GREEN)]
        for stck, y, nm, sub_, c in rows:
            for x, i in zip(vx, idxs):
                m = arr_img(stck[i], h).move_to([x, y, 0])
                self.add(m)
            self.add(tx(nm, 26, c).move_to([-4.65, y + 0.2, 0]), tx(sub_, 20, DIM).move_to([-4.65, y - 0.25, 0]))
        for x, hd in zip(vx, heads):
            self.add(tx(hd, 20, DIM).move_to([x, 2.55, 0]))
        note = toy_note("toy simulation, 24 frames, pupil at left / centre / right of the eyebox, focused on the star"
                        ).move_to([0, -2.6, 0])
        self.add(note)
        self.caption("Toy check: masking on the far plane, as GWS does, leaves dark halos in off-axis views.")
        self.caption("Masking at the occluder, back to front, gives clean occlusion and parallax.")
        self.clear_all()

        # --- paper ablation (Fig. 3)
        keys = ["nrp", "gwssqrt", "nosqrt", "full"]
        names = ["GWS-NRP", "GWS with √", "RPWS without √", "RPWS (full)"]
        colors = [RED, RED, RED, GREEN]
        xs = [-4.95, -1.65, 1.65, 4.95]
        w = 3.15
        for x, k, n, c in zip(xs, keys, names, colors):
            a = img(f"abl_{k}_lego.png", w=w).move_to([x, 1.2, 0])
            b = img(f"abl_{k}_garden.png", w=w).move_to([x, -0.95, 0])
            self.add(a, b, label(a, n, 24, c))
        marks = [("✗", "✗", "✓", "✓"), ("✗", "✓", "✗", "✓")]
        rl = ["back-to-front", "√ blending"]
        for r, (row, rname) in enumerate(zip(marks, rl)):
            y = -2.3 - 0.42 * r
            self.add(tx(rname, 20, DIM).move_to([-6.1, y, 0]))
            for x, mk in zip(xs, row):
                self.add(tx(mk, 24, GREEN if mk == "✓" else RED).move_to([x, y, 0]))
        self.add(tx("simulated focal slice (top) and light-field view (bottom), paper Fig. 3", 18, DIM).move_to([0, 2.95, 0]))
        self.caption("Paper ablation: only full RPWS (back to front + √ blending) gets colour and occlusion right.")
        self.clear_all()


# ================================================================== S08 Results
class S08_Results(Base):
    def construct(self):
        self.title_card("Part 8", "Results")
        self.header("8 · Results")
        # teaser comparisons (simulated)
        w = 6.3
        a = img("teaser_gws_front.png", w=w).move_to([-3.3, 0.55, 0])
        b = img("teaser_rpws_back.png", w=w).move_to([3.3, 0.55, 0])
        self.play(FadeIn(a), FadeIn(b), run_time=0.6)
        src = tx("simulated holograms, 2D Gaussian scene, paper Fig. 1", 18, DIM).next_to(a, DOWN, buff=0.15).set_x(0)
        self.add(src)
        self.caption("GWS at front focus: the background barely blurs and rings. RPWS at back focus: natural blur.")
        self.hold(0.5)
        a2 = img("teaser_nrp_front.png", w=w).move_to(a)
        b2 = img("teaser_rpws_front.png", w=w).move_to(b)
        self.play(FadeIn(a2), FadeIn(b2), run_time=0.6)
        self.remove(a, b)
        self.caption("Naïve random phase (GWS-NRP) gets the colours wrong; RPWS gets them right.")
        self.clear_all()

        # simulated light field (Fig. 4b)
        w = 3.5
        pos = {("gws", "center"): (-2.0, 1.2), ("rpws24", "center"): (2.0, 1.2),
               ("gws", "right"): (-2.0, -1.2), ("rpws24", "right"): (2.0, -1.2)}
        ims = {}
        for (k, v), (x, y) in pos.items():
            ims[(k, v)] = img(f"sim_{k}_{v}.png", w=w).move_to([x, y, 0])
        self.play(*[FadeIn(m) for m in ims.values()], run_time=0.6)
        self.add(label(ims[("gws", "center")], "GWS", 26, RED), label(ims[("rpws24", "center")], "RPWS (24 frames)", 26, GREEN))
        self.add(tx("centre view", 22, DIM).next_to(ims[("gws", "center")], LEFT, buff=0.3),
                 tx("right view", 22, DIM).next_to(ims[("gws", "right")], LEFT, buff=0.3))
        self.add(tx("simulated light field,", 18, DIM).move_to([5.45, 0.15, 0]),
                 tx("Mip-NeRF 360 bicycle", 18, DIM).move_to([5.45, -0.15, 0]),
                 tx("(paper Fig. 4)", 18, DIM).move_to([5.45, -0.45, 0]))
        self.caption("Simulated light field: GWS's light sits at the eyebox centre, so its right view is black. RPWS keeps every view.")
        self.clear_all()

        # Table 1
        rows = [("CGH algorithm", "Blender", "Mip-NeRF 360"),
                ("Polygons-RP, 1 frame", "17.02 / 0.14 / 0.75", "12.77 / 0.14 / 0.76"),
                ("Polygons-RP, 8 frames", "20.27 / 0.30 / 0.65", "15.63 / 0.27 / 0.69"),
                ("RPWS-GS, 1 frame", "18.11 / 0.15 / 0.76", "14.17 / 0.17 / 0.76"),
                ("RPWS-GS, 8 frames", "23.42 / 0.36 / 0.64", "19.65 / 0.36 / 0.65"),
                ("RPWS-GS, 24 frames", "24.87 / 0.49 / 0.56", "21.21 / 0.48 / 0.58"),
                ("GWS", "14.35 / 0.18 / 0.63", "10.50 / 0.17 / 0.60")]
        xs = [-3.7, 0.9, 4.6]
        tbl = VGroup()
        for r, row in enumerate(rows):
            cl = DIM if r == 0 else (GREEN if "24" in row[0] else (RED if row[0] == "GWS" else WHITE))
            for c, (s, x) in enumerate(zip(row, xs)):
                tbl.add(tx(s, 24, cl).move_to([x, 2.35 - 0.55 * r, 0]))
        hl = Line([-6.2, 2.07, 0], [6.2, 2.07, 0], color=DIM, stroke_width=1.5)
        self.add(tbl, hl)
        src = tx("paper Table 1: simulated 10 × 10 light fields, PSNR (dB) ↑ / SSIM ↑ / LPIPS ↓", 18, DIM).move_to([0, -1.6, 0])
        self.add(src)
        self.caption("Paper Table 1, dense 10 × 10 light fields: RPWS with 24 frames is best on every metric.")
        self.hold()
        head = tx("+10.5 dB (Blender) and +10.7 dB (Mip-NeRF 360) light-field PSNR over GWS", 26, AMBER).move_to([0, -2.3, 0])
        self.add(head)
        self.caption("About 10.5 dB more light-field PSNR than GWS; more frames, better quality.")
        self.clear_all()

        # experimental captures (Fig. 6)
        w = 3.6
        cells = [("gws", "lego_front", -2.0, 1.2), ("rpws24", "lego_front", 2.0, 1.2),
                 ("gws", "lego_back", -2.0, -1.2), ("rpws24", "lego_back", 2.0, -1.2)]
        ims = {}
        for k, v, x, y in cells:
            ims[(k, v)] = img(f"exp_{k}_{v}.png", w=w).move_to([x, y, 0])
        self.play(*[FadeIn(m) for m in ims.values()], run_time=0.6)
        self.add(label(ims[("gws", "lego_front")], "GWS", 26, RED),
                 label(ims[("rpws24", "lego_front")], "RPWS-GS (24 frames)", 26, GREEN))
        self.add(tx("front focus", 22, DIM).next_to(ims[("gws", "lego_front")], LEFT, buff=0.3),
                 tx("back focus", 22, DIM).next_to(ims[("gws", "lego_back")], LEFT, buff=0.3))
        self.add(tx("experimentally", 18, DIM).move_to([5.5, 0.3, 0]), tx("captured", 18, DIM).move_to([5.5, 0.0, 0]),
                 tx("(paper Fig. 6)", 18, DIM).move_to([5.5, -0.3, 0]))
        self.caption("Captured on a real display: GWS barely changes with focus; RPWS refocuses with natural blur.")
        self.clear_all()

        # time multiplexing captured, zoom
        w = 3.2
        xs = [-3.9, 0, 3.9]
        for x, k, n in zip(xs, ["rpws1", "rpws8", "rpws24"], ["1 frame", "8 frames", "24 frames"]):
            m = img(f"exp_{k}_lego_zA_L.png", w=w).move_to([x, 0.6, 0])
            self.add(m, label(m, "RPWS-GS, " + n, 24))
        self.add(tx("experimentally captured zoom-in (paper Fig. 6)", 18, DIM).move_to([0, -1.4, 0]))
        self.caption("More frames, less speckle: at 24 frames the captures are nearly speckle-free.")
        self.clear_all()

        # triangles (Fig. 5)
        w = 3.0
        xs = [-4.8, -1.6, 1.6, 4.8]
        for i, x in enumerate(xs):
            a = img(f"tri_s{i}_front.png", w=w).move_to([x, 1.15, 0])
            b = img(f"tri_s{i}_rear.png", w=w).move_to([x, -0.95, 0])
            self.add(a, b)
        self.add(tx("front focus", 20, DIM).move_to([-6.6, 1.15, 0]).rotate(PI / 2),
                 tx("rear focus", 20, DIM).move_to([-6.6, -0.95, 0]).rotate(PI / 2))
        self.add(tx("RPWS-Δ: soft-edged triangle splats (Triangle Splatting), simulated, paper Fig. 5", 18, DIM
                    ).move_to([0, -2.35, 0]))
        self.caption("Not just Gaussians: soft-edged triangle splats become holograms too.")
        self.clear_all()


# ================================================================== S09 Outlook
class S09_Outlook(Base):
    def construct(self):
        self.title_card("Part 9", "Limitations and outlook")
        self.header("9 · Limitations and outlook")
        items = [
            ("Exact blending needs many frames, limited by SLM refresh rates",
             "→ fast SLMs: 3600 fps FLCoS, 5760 fps MEMS phase modulators"),
            ("Random phase reduces contrast in experiments",
             "→ learned calibration models that reproduce complex fields"),
        ]
        g = VGroup()
        for a, b in items:
            g.add(VGroup(tx(a, 26), tx(b, 24, AMBER)).arrange(DOWN, aligned_edge=LEFT, buff=0.12))
        g.arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([0, 1.0, 0])
        for it in g:
            self.add(it)
            self.wait(1.0)
        self.caption("Both limits shrink as SLMs get faster and calibration improves.")
        self.clear_all()

        summ = [
            "1   Random-phase wave splatting for any translucent 2D primitive",
            "2   Intensity-domain alpha blending (√ weights), composited back to front",
            "3   Time multiplexing as the exact alpha-blending mechanism",
        ]
        sg = VGroup(*[tx(s, 28) for s in summ]).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([0, 1.2, 0])
        res = tx("full eyebox · parallax · natural defocus · +10.5 dB light-field PSNR over GWS (Blender, 24 frames)", 24, AMBER
                 ).move_to([0, -0.6, 0])
        self.add(sg)
        self.wait(1.2)
        self.add(res)
        self.caption("Random phase, done right: holograms you can look around and focus through.")
        self.clear_all()
        self.remove(self.hdr)
        t1 = tx("Random-phase Wave Splatting", 48, bold=True).move_to([0, 2.1, 0])
        t1b = tx("of Translucent Primitives for Computer-generated Holography", 30).move_to([0, 1.4, 0])
        t2 = tx("arXiv 2508.17480", 28).move_to([0, 0.45, 0])
        t4 = tx("Brian Chao*, Jacqueline Yang*, Suyeon Choi, Manu Gopakumar, Ryota Koiso, Gordon Wetzstein",
                22, DIM).move_to([0, -0.6, 0])
        t5 = tx("Stanford University  ·  KDDI Research", 22, DIM).move_to([0, -1.05, 0])
        t6 = tx("* equal contribution", 18, DIM).move_to([0, -1.45, 0])
        self.add(t1, t1b, t2, t4, t5, t6)
        self.wait(4.0)
