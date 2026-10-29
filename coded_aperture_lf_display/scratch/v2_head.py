"""Time-Division Multiplexing Light Field Display With Learned Coded Aperture — teaser animation, v2.

Source: Chao*, Liu*, Chen, "Time-Division Multiplexing Light Field Display With Learned Coded Aperture",
IEEE Transactions on Image Processing 32, 350-363 (2023); arXiv 2107.06205. National Taiwan University.
Paper figures in data/ are panels exported from the arXiv PDF (scratch/crop_figs.py). The website teaser GIF
(bchao1.github.io/images/tip2021.gif) is in data/tip2021.gif, frames in data/gif_frames/.
All optics visuals are toy simulations computed here and cached in data/generated/ (delete data/generated/done_v1 to rebuild):
  * a layered toy scene rendered into a dense 9 x 9 light field (numpy),
  * refocusing / TDM-LFD / CTDM-LFD image formation in the ray-optics limit of the paper's Eqs. 8-10 (torch FFTs),
  * a toy end-to-end optimisation of k = 4 encoded images and 4 coded apertures with the paper's focus-weighted L1 loss
    (Eqs. 12-15, beta = 2) — images optimised directly for one scene instead of the paper's CNN f,
  * Fourier-optics PSFs |F{P W}|^2 of an open pupil, 4 corner windows and the paper's Fig. 9 binary 9 x 9 apertures (numpy).

Render (see render.sh):
    manim -ql --media_dir media animation_tdmlf_v2.py <Scene>
Scenes, in order: S00_Title ... S09_Outlook.
"""
from manim import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib
import fcntl
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GEN = os.path.join(DATA, "generated")
ASSET_FLAG = "done_v1"

config.background_color = BLACK

# ------------------------------------------------------------------ style (same as random_phase_wave_splatting)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × − → ≈ ∑ ∏ ℱ 𝒫 𝔼 ∇ 𝒲 …
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
MATHCH = set("×⊙→≈∼−∈ℝ≪≤↓↔∑∏∬∫ℱ𝒫𝒢𝒩ℋ𝒪𝒰𝔼∗≠∞∂∘⊤●⇒∠✓∇𝒲ℒ")
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


SQ = sup("2")
PH = I("P\u0302")                     # P-hat (combining circumflex renders in CMU Serif italic)
EQ8 = (f"{I('R')}{sub(I('z'))} = ∑{sub(I('s') + '∈' + I('G'))} |{I('h')}{sub(I('s') + ',' + I('z'))}|{SQ} ∗ "
       f"{I('L')}{sub(I('s'))}")
EQ9 = f"{I('h')}{sub(PH + '<sub>' + I('i') + '</sub>' + ',' + I('z'))} = ℱ{{ {PH}{sub(I('i'))} · {I('W')}({I('z')}) }}"
EQ9s = f"PSF = |ℱ{{ {I('P')} · {I('W')}({I('z')}) }}|{SQ}"
EQ10 = (f"{I('R')}{sub(I('z'))} = ∑{sub(I('t') + ' = 1')}{sup(I('k'))} |{I('h')}{sub(PH + I('t') + ',' + I('z'))}|{SQ} ∗ "
        f"{I('I')}{sub(I('c') + ',' + I('t'))}")
EQ12 = (f"ℒ = 𝒲 ⊙ ‖ {I('F')}{sub(I('c'))} − {I('F')}{sub(I('g'))} ‖{sub('1')}")
EQ13 = (f"{I('U')}{sub(I('j'))} = ‖∇{sub(I('x'))}{I('R')}{sub(I('j'))}‖{sub('1')} + "
        f"‖∇{sub(I('y'))}{I('R')}{sub(I('j'))}‖{sub('1')}")
EQ14 = f"𝒲{sub(I('j'))} = exp(β {I('Ū')}{sub(I('j'))})"

# ------------------------------------------------------------------ toy light field (numpy)
C = 192               # simulation canvas (periodic FFT shifts)
BRD = 16              # border cropped away for display / loss
L = 9                 # aperture grid = angular resolution (paper: l = 9)
US = np.arange(L) - L // 2
CORN = [(0, 0), (0, L - 1), (L - 1, 0), (L - 1, L - 1)]   # n = 4 corner views (paper Fig. 7)
KF = 4                # k = 4 encoded images / coded apertures
D_NEAR, D_MID, D_FAR = 1.0, 0.0, -1.0                      # disparity (px per aperture cell) of star / leaf / letters
AS_T = np.linspace(-1.25, 1.25, 11)                        # focal planes used in the toy loss
AS_S = np.linspace(-1.4, 1.4, 57)                          # focal sweep for display
NIT = 3000
TFONT = "/home/brianchc/.local/share/fonts/cmunbx.ttf"


def _blur(a, s=0.6):
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(s))) / 255.


def _fshift(a, dx, dy):
    """Periodic sub-pixel shift, out(x) = a(x − d)."""
    F = np.fft.fft2(a, axes=(0, 1))
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    ph = np.exp(-2j * np.pi * (fx * dx + fy * dy))
    if a.ndim == 3:
        ph = ph[..., None]
    return np.real(np.fft.ifft2(F * ph, axes=(0, 1)))


def toy_layers():
    """Far: letter wall (d = −1). Middle: striped green leaf on the display plane (d = 0). Near: orange checkered star (d = +1)."""
    im = Image.new("RGB", (C, C), (40, 46, 70))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(TFONT, 22)
    cols = [(255, 202, 40), (77, 208, 225), (255, 255, 255), (255, 82, 82)]
    words = ["LIGHT FIELD", "DISPLAY", "CODED", "APERTURE", "TIME", "DIVISION"]
    for r, y in enumerate(range(-4, C, 24)):
        s = "  ".join(words[(r + j) % len(words)] for j in range(3))
        d.text((-8 - 11 * (r % 3), y), s, fill=cols[r % 4], font=f)
    back = np.asarray(im) / 255.
    yy, xx = np.mgrid[0:C, 0:C]
    mm = ((((xx - 62) / 30.0) ** 2 + ((yy - 120) / 40.0) ** 2) <= 1).astype(float)
    mid = np.zeros((C, C, 3))
    mid[..., 0], mid[..., 1], mid[..., 2] = 0.2, 0.75, 0.35
    mid *= (0.75 + 0.25 * (np.sin((xx + yy) / 3.0) > 0))[..., None]
    sm = Image.new("L", (C, C), 0)
    ds = ImageDraw.Draw(sm)
    cx, cy, r1, r0 = 122, 86, 44, 19
    pts = [(cx + (r1 if i % 2 == 0 else r0) * np.cos(-np.pi / 2 + i * np.pi / 5),
            cy + (r1 if i % 2 == 0 else r0) * np.sin(-np.pi / 2 + i * np.pi / 5)) for i in range(10)]
    ds.polygon(pts, fill=255)
    fm = np.asarray(sm) / 255.
    chk = ((xx // 5 + yy // 5) % 2).astype(float)
    front = np.stack([np.ones((C, C)), 0.62 - 0.25 * chk, np.full((C, C), 0.16)], -1)
    return [(back, np.ones((C, C)), D_FAR), (mid, mm, D_MID), (front, fm, D_NEAR)]


def render_views(layers):
    """Dense L × L light field: back-to-front 'over' compositing of layers shifted by (u, v) · disparity."""
    V = np.zeros((L, L, C, C, 3))
    pre = [(np.stack([_blur(t[..., c]) for c in range(3)], -1), _blur(m), d) for t, m, d in layers]
    for iv, v in enumerate(US):
        for iu, u in enumerate(US):
            img = np.zeros((C, C, 3))
            for t, m, d in pre:
                ms = np.clip(_fshift(m, u * d, v * d), 0, 1)[..., None]
                ts = _fshift(t * m[..., None], u * d, v * d)
                img = img * (1 - ms) + np.clip(ts, 0, 1)
            V[iv, iu] = np.clip(img, 0, 1)
    return V


def cmap_rgb(a, cmap="inferno"):
    return (matplotlib.colormaps[cmap](np.clip(a, 0, 1))[..., :3] * 255).astype(np.uint8)


def rgba(a):
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
    return np.ascontiguousarray(a.astype(np.uint8))


def gen(name):
    return os.path.join(GEN, name)


# ------------------------------------------------------------------ Fourier-optics PSFs (numpy)
NP_PSF = 1024
CELL_PSF = 24
W20S = np.linspace(0, 18, 37)            # defocus in waves (toy)


def _psf_one(args):
    P9, w20 = args
    D = L * CELL_PSF
    p = np.kron(P9, np.ones((CELL_PSF, CELL_PSF)))
    g = np.zeros((NP_PSF, NP_PSF))
    o = (NP_PSF - D) // 2
    g[o:o + D, o:o + D] = p
    y, x = (np.mgrid[0:NP_PSF, 0:NP_PSF] - NP_PSF / 2 + 0.5) / (D / 2)
    out = 0
    for b in (0.85, 0.92, 1.0, 1.08, 1.15):          # small band of defocus values (broadband, incoherent light)
        u = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g * np.exp(1j * np.pi * w20 * b * (x ** 2 + y ** 2)))))
        out = out + np.abs(u) ** 2
    h = out[NP_PSF // 2 - 384:NP_PSF // 2 + 384, NP_PSF // 2 - 384:NP_PSF // 2 + 384].reshape(192, 4, 192, 4).mean((1, 3))
    h = (h / h.sum()) ** 0.5
    return cmap_rgb(h / np.percentile(h, 99.9), "inferno")


# ------------------------------------------------------------------ torch image formation + toy optimisation
def _toy_build():
    import torch
    from multiprocessing.pool import ThreadPool
    torch.set_num_threads(32)
    torch.manual_seed(0)
    rng = np.random.default_rng(2023)
    V = render_views(toy_layers()).astype(np.float32)
    cr = slice(BRD, C - BRD)
    np.save(gen("views.npy"), (V[:, :, cr, cr] * 255).astype(np.uint8))
    Vt = torch.tensor(V).permute(0, 1, 4, 2, 3)                   # [v, u, 3, C, C]
    Vf = torch.fft.rfft2(Vt)
    fy = torch.fft.fftfreq(C)[:, None]
    fx = torch.fft.rfftfreq(C)[None, :]
    uu = torch.tensor(US, dtype=torch.float32)

    def sinc(x):
        return torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.sin(np.pi * x) / (np.pi * x + 1e-12))

    def basis(a):
        """E[v, u, f] = exp(+j2π (fx u + fy v) a) · sinc(fx a) sinc(fy a): aperture cell (v, u) seen at focus a."""
        Bz = (sinc(fx * a) * sinc(fy * a)).to(torch.complex64)
        ex = torch.exp(2j * np.pi * fx[None] * uu[:, None, None] * a)          # [u, 1, F]
        ey = torch.exp(2j * np.pi * fy[None] * uu[:, None, None] * a)          # [v, C, 1]
        return ey[:, None] * ex[None] * Bz                                      # [v, u, C, F]

    ET = torch.stack([basis(a) for a in AS_T])                                # [Z, v, u, C, F]

    def refocus(Es, cells=None):
        if cells is None:
            S = torch.einsum("vucxy,zvuxy->zcxy", Vf, Es) / (L * L)
        else:
            S = sum(Vf[v, u][None] * Es[:, v, u][:, None] for v, u in cells) / len(cells)
        return torch.fft.irfft2(S, s=(C, C))

    def ctdm(Im, P, Es):
        """Eq. 10 (ray-optics limit): R_z = Σ_t I_t ∗ PSF_t,z / 81 — one fully open aperture gives the target brightness."""
        If = torch.fft.rfft2(Im)
        A = torch.einsum("kvu,zvuxy->kzxy", P.to(torch.complex64), Es)
        return torch.fft.irfft2(torch.einsum("kcxy,kzxy->zcxy", If, A), s=(C, C)) / (L * L)

    def crop(x):
        return x[..., cr, cr]

    def psnr(a, b):
        return float(10 * torch.log10(1 / ((crop(a) - crop(b)) ** 2).mean()))

    def weights(Fg, beta=2.0):
        g = Fg.mean(1)
        U = torch.zeros_like(g)
        U[:, :, 1:] += (g[:, :, 1:] - g[:, :, :-1]).abs()
        U[:, 1:, :] += (g[:, 1:, :] - g[:, :-1, :]).abs()
        mn, mx = U.amin((1, 2), keepdim=True), U.amax((1, 2), keepdim=True)
        return torch.exp(beta * (U - mn) / (mx - mn + 1e-8))[:, None]

    def u8(x):
        return (crop(x).clamp(0, 1).permute(*range(x.dim() - 3), -2, -1, -3).numpy() * 255).astype(np.uint8)

    Fg = refocus(ET)
    W = weights(Fg)
    corners = torch.stack([Vt[v, u] for v, u in CORN])
    Fb = refocus(ET, CORN)
    ps = {"base": psnr(Fb, Fg)}
    # focus weight maps (paper Fig. 4 analogue) at the near and far layers
    iN, iF = int(np.argmin(np.abs(AS_T - D_NEAR))), int(np.argmin(np.abs(AS_T - D_FAR)))
    np.save(gen("wmap.npy"), np.stack([cmap_rgb((crop(W[i, 0]).numpy() - 1) / (np.e ** 2 - 1), "inferno") for i in (iN, iF)]))
    np.save(gen("wimg.npy"), np.stack([u8(Fg[i]) for i in (iN, iF)]))
    snaps = sorted(set([0] + [int(round(x)) for x in np.geomspace(1, NIT - 1, 44)]))
    Enear = torch.stack([basis(D_NEAR)])

    def optimise(learn_images):
        Pp = torch.nn.Parameter(torch.randn(KF, L, L) * 0.5)
        if learn_images:
            Ip = torch.nn.Parameter(torch.logit(corners.mean(0, keepdim=True).repeat(KF, 1, 1, 1).clamp(0.01, 0.99)))
            groups = [{"params": [Pp], "lr": 3e-2}, {"params": [Ip], "lr": 2e-2}]
        else:
            Ip = None
            groups = [{"params": [Pp], "lr": 3e-2}]
        opt = torch.optim.Adam(groups)
        hist, sP, sImg = [], [], []
        for it in range(NIT):
            Im = torch.sigmoid(Ip) if learn_images else corners
            P = torch.sigmoid(Pp)
            Fc = ctdm(Im, P, ET)
            if it in snaps:
                with torch.no_grad():
                    sP.append(P.detach().numpy().copy())
                    sImg.append(u8(ctdm(Im.detach(), P.detach(), Enear)[0]))
            if it % 10 == 0:
                hist.append((it, psnr(Fc.detach(), Fg)))
            loss = (crop(W) * (crop(Fc) - crop(Fg)).abs()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            Im = (torch.sigmoid(Ip) if learn_images else corners).detach()
            P = torch.sigmoid(Pp).detach()
            Fc = ctdm(Im, P, ET)
        return Im, P, psnr(Fc, Fg), np.array(hist), np.stack(sP), np.stack(sImg)

    Im_p, P_p, ps["P"], _, _, _ = optimise(False)
    Im_f, P_f, ps["fP"], hist, sP, sImg = optimise(True)
    np.save(gen("train_hist.npy"), hist)
    np.save(gen("train_snapP.npy"), sP)
    np.save(gen("train_snapImg.npy"), sImg)
    np.save(gen("train_snapIt.npy"), np.array(snaps))
    np.save(gen("P_final.npy"), P_f.numpy())
    np.save(gen("P_only.npy"), P_p.numpy())
    np.save(gen("enc_images.npy"), u8(Im_f))
    np.save(gen("corner_views.npy"), u8(corners))
    # focal sweeps for display (57 planes, one at a time)
    st = {"gt": [], "base": [], "P": [], "fP": []}
    for a in AS_S:
        Es = basis(float(a))[None]
        st["gt"].append(u8(refocus(Es)[0]))
        st["base"].append(u8(refocus(Es, CORN)[0]))
        st["P"].append(u8(ctdm(corners, P_p, Es)[0]))
        st["fP"].append(u8(ctdm(Im_f, P_f, Es)[0]))
    for k, v in st.items():
        np.save(gen(f"sweep_{k}.npy"), np.stack(v))
    # per-frame contributions at near focus and their running sum (for the time-multiplexing cycle)
    If = torch.fft.rfft2(Im_f)
    A = torch.einsum("kvu,vuxy->kxy", P_f.to(torch.complex64), basis(D_NEAR))
    parts = torch.fft.irfft2(If * A[:, None], s=(C, C)) / (L * L)
    np.save(gen("cycle_cum.npy"), u8(torch.cumsum(parts, 0)))
    np.save(gen("cycle_part.npy"), u8(parts * KF))
    # baseline TDM cycle: each corner view through its window at near focus, running mean
    pb = torch.stack([torch.fft.irfft2(Vf[v, u] * basis(D_NEAR)[v, u], s=(C, C)) for v, u in CORN])
    np.save(gen("cycle_base_cum.npy"), u8(torch.cumsum(pb, 0) / torch.arange(1, 5)[:, None, None, None]))
    np.save(gen("psnr.npy"), np.array([ps["base"], ps["P"], ps["fP"]]))
    # Fourier-optics PSF sweeps: open pupil, 4 corner windows, paper Fig. 9 aperture 1, sum of all four Fig. 9 apertures
    ap9 = np.load(os.path.join(DATA, "fig9_apertures_9x9.npy")).astype(float)
    corn = np.zeros((L, L))
    for v, u in CORN:
        corn[v, u] = 1
    aps = {"open": np.ones((L, L)), "corners": corn, "coded": ap9[0]}
    with ThreadPool(24) as pool:
        for k, P9 in aps.items():
            np.save(gen(f"psf_{k}.npy"), np.stack(pool.map(_psf_one, [(P9, w) for w in W20S])))


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    if os.path.exists(gen(ASSET_FLAG)):
        return
    with open(gen("lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(gen(ASSET_FLAG)):
            return
        _toy_build()
        open(gen(ASSET_FLAG), "w").write("ok")


build_assets()


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
    return ((1 - w) * frames[i0] + w * frames[i1]).astype(np.uint8)


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.16)


def aperture_rgb(P, color=(77, 208, 225), grid=True, px=12):
    """9 × 9 aperture pattern as an image: open = cyan, closed = near-black, thin grid lines."""
    P = np.clip(np.asarray(P, float), 0, 1)
    c = np.array(color, float)
    base = np.array([18, 20, 26], float)
    cell = base + P[..., None] * (c - base)
    im = np.kron(cell, np.ones((px, px, 1)))
    if grid:
        im[::px, :, :] = 55
        im[:, ::px, :] = 55
        im[-1, :, :] = 55
        im[:, -1, :] = 55
    return im.astype(np.uint8)


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


def toy_note(s):
    return tx(s, 16, DIM)


def star_shape(r=0.14, color=AMBER):
    return Star(5, outer_radius=r, inner_radius=r * 0.45, color=color, fill_opacity=1, stroke_width=0)


def focus_ruler(center, length, f_of_t, labels=True):
    """Depth ruler 'near ↔ far' with star / leaf / letters markers and a moving focus dot.
    f_of_t() returns the focus disparity a in [−1.4, 1.4]; near (a = +1) is on the left."""
    nl = Line(LEFT * length / 2, RIGHT * length / 2, color=DIM, stroke_width=2).move_to(center)
    a2x = lambda a: center[0] - (a / 1.4) * length / 2
    g = VGroup(nl)
    marks = [(D_NEAR, "near: star", AMBER), (D_MID, "middle: leaf", GREEN), (D_FAR, "far: letters", CYAN)]
    for a, s, c in marks:
        tri = Triangle(color=c, fill_opacity=1, stroke_width=0).scale(0.08).rotate(PI).move_to(
            [a2x(a), center[1] + 0.13, 0])
        g.add(tri)
        if labels:
            g.add(tx(s, 17, c).move_to([a2x(a), center[1] + 0.38, 0]))
    dot = always_redraw(lambda: Dot([a2x(float(np.clip(f_of_t(), -1.4, 1.4))), center[1], 0], radius=0.08, color=WHITE))
    lab = tx("focus", 18, DIM).next_to(nl, LEFT, buff=0.25)
    return VGroup(g, lab), dot


def sweep_index(a):
    return (a - AS_S[0]) / (AS_S[1] - AS_S[0])


def fitw(m, w=12.8):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


# ------------------------------------------------------------------ eye diagram (side view, thin-lens optics)
class Eye:
    """Side-view eye facing left. Lens at x_L; rays from an on-axis point at x_o are refracted by a thin lens with focal
    length f and stopped on the eyeball circle (the retina side)."""

    def __init__(self, center=(3.6, 0.35), R=1.35, pupil=0.42, scale=1.0):
        self.c = np.array([center[0], center[1], 0.0])
        self.R = R
        self.xL = center[0] - 0.78 * R
        self.pupil = pupil
        # axial distance lens → back of the eye
        self.dR = center[0] + R - self.xL

    def body(self):
        c, R = self.c, self.R
        ball = Circle(radius=R, color=DIM, stroke_width=2).move_to(c)
        retina = Arc(radius=R, start_angle=-0.95, angle=1.9, color=ORANGE, stroke_width=5).move_to(c, aligned_edge=ORIGIN)
        retina.shift(c - retina.get_arc_center())
        cornea = Arc(radius=0.62 * R, start_angle=PI - 0.9, angle=1.8, color=DIM, stroke_width=2)
        cornea.shift(np.array([self.xL + 0.28 * R, c[1], 0]) - cornea.get_arc_center())
        iris = VGroup(Line([self.xL, c[1] + self.pupil, 0], [self.xL, c[1] + 0.8 * R, 0], color=BLUE, stroke_width=5),
                      Line([self.xL, c[1] - self.pupil, 0], [self.xL, c[1] - 0.8 * R, 0], color=BLUE, stroke_width=5))
        return VGroup(ball, retina, cornea, iris)

    def lens(self, f):
        w = float(np.clip(0.16 + 0.95 * (1 / f - 0.45), 0.12, 0.62))
        return Ellipse(width=w, height=2 * self.pupil + 0.28, color=WHITE, fill_color="#B3E5FC", fill_opacity=0.35,
                       stroke_width=2).move_to([self.xL, self.c[1], 0])

    def hit(self, p0, d):
        """Far intersection of the ray p0 + s d with the eyeball circle."""
        c = self.c[:2]
        p0, d = np.asarray(p0[:2], float), np.asarray(d[:2], float) / np.linalg.norm(d[:2])
        b = 2 * np.dot(d, p0 - c)
        cc = np.dot(p0 - c, p0 - c) - self.R ** 2
        s = (-b + np.sqrt(max(b * b - 4 * cc, 0))) / 2
        return p0 + s * d

    def rays(self, xo, f, n=7, color=AMBER, op=0.85, yo=0.0):
        y0 = self.c[1]
        do = self.xL - xo
        inv = 1 / f - 1 / do
        g = VGroup()
        ends = []
        for h in np.linspace(-self.pupil * 0.92, self.pupil * 0.92, n):
            pL = np.array([self.xL, y0 + h])
            g.add(Line([xo, y0 + yo, 0], [pL[0], pL[1], 0], color=color, stroke_width=2, stroke_opacity=op))
            if abs(inv) < 1e-6:
                d = np.array([1.0, -yo / do])
            else:
                di = 1 / inv
                q = np.array([self.xL + di, y0 - yo * di / do])
                d = (q - pL) * np.sign(di)
            e = self.hit(pL, d)
            ends.append(e[1])
            g.add(Line([pL[0], pL[1], 0], [e[0], e[1], 0], color=color, stroke_width=2, stroke_opacity=op))
        return g, (min(ends), max(ends))

    def f_for(self, xo):
        return 1 / (1 / (self.xL - xo) + 1 / self.dR)


# ================================================================== S00 Title
class S00_Title(Base):
    def construct(self):
        title = fitw(tx("Time-Division Multiplexing Light Field Display", 50, bold=True)).move_to([0, 3.05, 0])
        subt = tx("with Learned Coded Aperture", 36).move_to([0, 2.38, 0])
        frames = [np.asarray(Image.open(os.path.join(DATA, "gif_frames", f"{i:03d}.png")).convert("RGB")) for i in (0, 1)]
        gif = arr_img(frames[0], 3.1).move_to([0, -0.08, 0])
        live(gif, lambda: frames[int(self.clock.get_value()) % 2])
        fr = framed(gif, DIM, 1.2)
        lab = always_redraw(lambda: label(gif, ("baseline TDM-LFD: aliased blur" if int(self.clock.get_value()) % 2 == 0
                                                else "CTDM-LFD (this paper): natural blur"), 24,
                                          RED if int(self.clock.get_value()) % 2 == 0 else GREEN))
        note = tx("near focus, SIGGRAPH16 light field (paper Fig. 12; teaser from bchao1.github.io)", 17, DIM
                  ).next_to(gif, DOWN, buff=0.15)
        auth = tx("Brian Chao*, Chang-Le Liu*, Homer H. Chen", 24, DIM).move_to([0, -2.5, 0])
        aff = tx("National Taiwan University  ·  IEEE Transactions on Image Processing 32 (2023)", 22, DIM).move_to([0, -2.93, 0])
        eq = tx("* equal contribution", 18, DIM).move_to([0, -3.33, 0])
        self.add(title, subt, auth, aff, eq)
        self.play(FadeIn(gif), Create(fr), run_time=0.6)
        self.add(lab, note)
        self.wait(8.0)


# ================================================================== S01 Eye and VAC
