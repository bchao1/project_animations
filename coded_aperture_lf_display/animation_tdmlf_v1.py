"""Time-Division Multiplexing Light Field Display With Learned Coded Aperture — teaser animation, v1.

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
    manim -ql --media_dir media animation_tdmlf_v1.py <Scene>
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
class S01_Eye(Base):
    def construct(self):
        self.title_card("Part 1", "How your eyes sense depth")
        self.header("1 · How your eyes sense depth")
        eye = Eye(center=(3.7, 0.35), R=1.4)
        xo = ValueTracker(-5.2)
        f = ValueTracker(eye.f_for(-5.2))
        body = eye.body()
        lens = always_redraw(lambda: eye.lens(f.get_value()))
        rays = always_redraw(lambda: eye.rays(xo.get_value(), f.get_value())[0])
        src = always_redraw(lambda: star_shape(0.16).move_to([xo.get_value(), eye.c[1], 0]))

        def blurmark():
            lo, hi = eye.rays(xo.get_value(), f.get_value())[1]
            x = eye.c[0] + np.sqrt(max(eye.R ** 2 - ((lo + hi) / 2 - eye.c[1]) ** 2, 0))
            if hi - lo < 0.05:
                return VGroup(Dot([x, (lo + hi) / 2, 0], radius=0.07, color=GREEN))
            seg = Line([x + 0.12, lo, 0], [x + 0.12, hi, 0], color=RED, stroke_width=7)
            return VGroup(seg, tx("blur spot", 20, RED).next_to(seg, RIGHT, buff=0.12))
        bm = always_redraw(blurmark)
        lr = tx("retina", 20, ORANGE).move_to([eye.c[0] + 0.7, eye.c[1] - eye.R - 0.25, 0])
        ll = tx("lens", 20, WHITE).move_to([eye.xL, eye.c[1] + eye.R + 0.05, 0])
        self.play(Create(body), FadeIn(lens), run_time=0.8)
        self.add(lr, ll, rays, src, bm)
        self.caption("Light from a point enters the pupil, and the lens bends it onto one spot of the retina.")
        self.hold(0.3)
        self.run(xo, -0.9, 2.2, smooth)
        self.caption("Bring the point closer: the rays now meet behind the retina, and the point becomes a blur spot.")
        self.hold()
        self.caption("The eye reshapes its lens until the spot is sharp again. That is accommodation.")
        self.run(f, eye.f_for(-0.9), 2.0, smooth)
        self.caption("Focus, and the blur of everything else, tell your brain how far away things are.")
        self.clear_all()

        # --- vergence and the conflict (top view)
        cx = -3.0
        eyes = [np.array([cx - 0.85, -2.25, 0]), np.array([cx + 0.85, -2.25, 0])]

        def top_eye(p):
            return VGroup(Circle(radius=0.3, color=WHITE, stroke_width=2).move_to(p),
                          Dot(p + UP * 0.22, radius=0.08, color=BLUE))
        E = VGroup(*[top_eye(p) for p in eyes])
        obj = star_shape(0.2).move_to([cx, 0.6, 0])
        verg = VGroup(*[DashedLine(p + UP * 0.3, obj.get_center(), color=AMBER, stroke_width=3) for p in eyes])
        acc = DashedLine([cx - 1.6, 0.6, 0], [cx + 1.6, 0.6, 0], color=CYAN, stroke_width=3)
        leg1 = tx("vergence: where the two eyes point", 24, AMBER).move_to([3.1, 1.9, 0])
        leg2 = tx("accommodation: where each eye focuses", 24, CYAN).move_to([3.1, 1.35, 0])
        self.play(FadeIn(E), FadeIn(obj), run_time=0.5)
        self.add(leg1)
        self.play(Create(verg), run_time=0.6)
        self.caption("Your two eyes also rotate to meet at the object: vergence.")
        self.hold()
        self.add(leg2)
        self.play(Create(acc), run_time=0.5)
        ok = tx("real world: both at the same distance ✓", 24, GREEN).move_to([3.1, 0.5, 0])
        self.add(ok)
        self.caption("In the real world, vergence and accommodation always agree.")
        self.hold()
        # stereo display
        self.remove(ok, acc)
        scr_y = 1.9
        scr = Line([cx - 2.0, scr_y, 0], [cx + 2.0, scr_y, 0], color=WHITE, stroke_width=5)
        lscr = tx("display", 20, DIM).next_to(scr, LEFT, buff=0.15)
        virt = star_shape(0.2, color=AMBER).move_to([cx, -0.1, 0]).set_opacity(0.5)
        # screen images seen by each eye (crossed disparity for a point in front of the screen)
        hits = []
        for p in eyes:
            d = virt.get_center() - p
            s = (scr_y - p[1]) / d[1]
            hits.append(p + s * d)
        pix = VGroup(Dot(hits[0], radius=0.09, color=RED), Dot(hits[1], radius=0.09, color=BLUE))
        rays = VGroup(*[DashedLine(p + UP * 0.3, h, color=DIM, stroke_width=2) for p, h in zip(eyes, hits)])
        self.play(FadeOut(obj), FadeOut(verg), FadeIn(scr), run_time=0.5)
        self.add(lscr, pix, rays)
        self.caption("A stereo display shows each eye its own image, drawn on a screen at a fixed distance.")
        self.hold()
        verg2 = VGroup(*[Line(p + UP * 0.3, virt.get_center(), color=AMBER, stroke_width=3) for p in eyes])
        acc2 = DashedLine([cx - 1.9, scr_y - 0.12, 0], [cx + 1.9, scr_y - 0.12, 0], color=CYAN, stroke_width=3)
        self.add(virt, verg2, acc2)
        br = VGroup(Line([cx + 2.25, -0.1, 0], [cx + 2.25, scr_y - 0.12, 0], color=RED, stroke_width=4),
                    Line([cx + 2.1, -0.1, 0], [cx + 2.4, -0.1, 0], color=RED, stroke_width=4),
                    Line([cx + 2.1, scr_y - 0.12, 0], [cx + 2.4, scr_y - 0.12, 0], color=RED, stroke_width=4))
        bad = VGroup(tx("vergence-accommodation", 24, RED), tx("conflict (VAC)", 24, RED)).arrange(DOWN, buff=0.08
                                                                                                    ).move_to([3.1, 0.35, 0])
        self.add(br, bad)
        self.caption("The eyes converge on the virtual object but must focus on the screen: the vergence-accommodation conflict.")
        self.hold()
        harm = tx("eye strain · headache · even loss of depth perception", 22, DIM).move_to([3.1, -0.45, 0])
        self.add(harm)
        self.caption("It can cause eye strain, headache, or even loss of depth perception.")
        self.hold()
        fixes = VGroup(tx("varifocal / multifocal: eye trackers, screen stacks", 20, DIM),
                       tx("holographic: high-resolution SLMs", 20, DIM),
                       tx("light field displays: no extra elements", 20, GREEN)).arrange(DOWN, aligned_edge=LEFT, buff=0.14
                                                                                        ).move_to([3.1, -1.55, 0])
        self.add(fixes)
        self.caption("Several display types fix it. Light field displays do it without trackers, stacks or SLMs.")
        self.clear_all()


# ================================================================== S02 Light fields
class S02_LightField(Base):
    def construct(self):
        self.title_card("Part 2", "Light fields and refocusing")
        self.header("2 · Light fields and refocusing")
        # --- light field display concept: several views inside one pupil
        eye = Eye(center=(4.3, 0.2), R=1.25, pupil=0.5)
        vx = -4.4
        disp_x = -1.2
        vp = star_shape(0.18).move_to([vx, 0.2, 0])
        disp = Line([disp_x, -1.9, 0], [disp_x, 2.3, 0], color=WHITE, stroke_width=5)
        ld = tx("display", 20, DIM).next_to(disp, DOWN, buff=0.12)
        lv = tx("virtual point", 20, AMBER).next_to(vp, DOWN, buff=0.15)
        ys = [-0.34, 0.0, 0.34]
        cols = [RED, GREEN, BLUE]
        fans = VGroup()
        for y, c in zip(ys, cols):
            p_pupil = np.array([eye.xL, eye.c[1] + y, 0])
            d = p_pupil - vp.get_center()
            s = (disp_x - vp.get_center()[0]) / d[0]
            p_disp = vp.get_center() + s * d
            fans.add(DashedLine(vp.get_center(), p_disp, color=c, stroke_width=2, stroke_opacity=0.6),
                     Line(p_disp, p_pupil, color=c, stroke_width=3),
                     Dot(p_disp, radius=0.06, color=c))
        f = eye.f_for(vx)
        inside = VGroup()
        for y, c in zip(ys, cols):
            pL = np.array([eye.xL, eye.c[1] + y])
            di = 1 / (1 / f - 1 / (eye.xL - vx))
            q = np.array([eye.xL + di, eye.c[1]])
            e = eye.hit(pL, q - pL)
            inside.add(Line([pL[0], pL[1], 0], [e[0], e[1], 0], color=c, stroke_width=3))
        self.play(Create(eye.body()), FadeIn(eye.lens(f)), FadeIn(disp), run_time=0.7)
        self.add(ld)
        self.play(Create(fans), run_time=1.0)
        self.add(vp, lv, inside)
        lab = tx("views spaced closer than the pupil", 22, CYAN).move_to([eye.xL - 0.2, 2.2, 0])
        self.add(lab)
        self.caption("A light field display sends several views into each pupil, spaced closer than the pupil itself.")
        self.hold()
        self.caption("The rays of all views meet behind the display, so the eye can focus there: correct accommodation.")
        self.clear_all()

        # --- the light field: 9 × 9 views, refocusing by shift-and-add
        views = np.load(gen("views.npy"))
        g = Group()
        cs = 0.39
        for iv in range(L):
            for iu in range(L):
                m = arr_img(views[iv, iu], cs * 0.94).move_to([-4.3 + (iu - 4) * cs, 0.55 - (iv - 4) * cs, 0])
                g.add(m)
        lg = label(g, "toy light field: 9 × 9 views", 24)
        eqL = tx(f"{I('L')}({I('x')}, {I('y')}, {I('s')}, {I('t')})", 30).move_to([-4.3, -1.55, 0])
        eqd = tx(f"({I('x')}, {I('y')}): position in a view   ({I('s')}, {I('t')}): which view", 18, DIM).move_to([-4.3, -2.0, 0])
        self.play(LaggedStart(*[FadeIn(m) for m in g], lag_ratio=0.01), run_time=1.2)
        self.add(lg, eqL, eqd)
        self.caption(f"A light field {I('L')}({I('x')}, {I('y')}, {I('s')}, {I('t')}) is a grid of views, each from a slightly different position.")
        gt = np.load(gen("sweep_gt.npy"))
        a = ValueTracker(D_NEAR)
        R = arr_img(gt[0], 3.3).move_to([2.6, 0.75, 0])
        live(R, lambda: interp(gt, sweep_index(a.get_value())))
        self.hold()
        self.play(FadeIn(R), run_time=0.5)
        self.add(label(R, "refocused image (all 81 views)", 24), framed(R, DIM, 1.2))
        rl, dot = focus_ruler(np.array([2.6, -1.55, 0]), 4.4, a.get_value)
        self.add(rl, dot)
        note = toy_note("toy simulation: three layers at different depths, 9 × 9 views").move_to([2.6, -2.35, 0])
        self.add(note)
        self.caption("Shift every view by its (s, t) offset and add them up: a photo focused at any depth.")
        self.run(a, D_FAR, 3.0, smooth)
        self.run(a, D_NEAR, 3.0, smooth)
        self.caption("That is what the eye does optically: it sums the views that enter the pupil.")
        self.clear_all()

        # --- sparse views alias
        base = np.load(gen("sweep_base.npy"))
        a = ValueTracker(D_NEAR)
        h = 3.3
        A = arr_img(gt[0], h).move_to([-2.2, 0.75, 0])
        Bm = arr_img(base[0], h).move_to([2.2, 0.75, 0])
        live(A, lambda: interp(gt, sweep_index(a.get_value())))
        live(Bm, lambda: interp(base, sweep_index(a.get_value())))
        ia = arr_img(aperture_rgb(np.ones((L, L)), grid=True, px=10), 1.1, nearest=True).move_to([-5.4, 0.75, 0])
        cm = np.zeros((L, L))
        for v, u in CORN:
            cm[v, u] = 1
        ib = arr_img(aperture_rgb(cm, grid=True, px=10), 1.1, nearest=True).move_to([5.4, 0.75, 0])
        self.play(FadeIn(A), FadeIn(Bm), run_time=0.5)
        self.add(label(A, "all 81 views", 26, GREEN), label(Bm, "only the 4 corner views", 26, RED), ia, ib,
                 label(ia, "views used", 18, DIM), label(ib, "views used", 18, DIM))
        rl, dot = focus_ruler(np.array([0, -1.55, 0]), 5.5, a.get_value)
        self.add(rl, dot)
        self.caption("Now keep only 4 views, the four corners. In-focus parts are still sharp...")
        self.run(a, D_FAR, 2.5, smooth)
        self.caption("...but out-of-focus parts break into ghost copies instead of blur: aliasing.")
        self.run(a, D_NEAR, 2.5, smooth)
        self.run(a, D_MID, 1.2, smooth)
        self.caption("Too few views cannot reproduce natural defocus blur. And blur is exactly the focus cue we need.")
        self.clear_all()


# ================================================================== S03 SDM vs TDM
class S03_Displays(Base):
    def construct(self):
        self.title_card("Part 3", "Space- vs time-division multiplexing")
        self.header("3 · Space- vs time-division multiplexing")
        views = np.load(gen("views.npy"))
        f1b = img("fig1b.png", w=5.6).move_to([-3.4, 0.55, 0])
        self.play(FadeIn(f1b), run_time=0.5)
        self.add(label(f1b, "SDM-LFD: microlens array (paper Fig. 1b)", 22, DIM))
        # 3 × 3 tiled panel
        tiles = Group()
        ts = 0.95
        for i, iv in enumerate((0, 4, 8)):
            for j, iu in enumerate((0, 4, 8)):
                tiles.add(arr_img(views[iv, iu], ts * 0.96).move_to([3.4 + (j - 1) * ts, 0.55 - (i - 1) * ts, 0]))
        self.play(FadeIn(tiles), run_time=0.5)
        self.add(label(tiles, "display panel, 3 × 3 views", 22))
        n9 = tx("each view gets 1/9 of the panel", 24, RED).next_to(tiles, DOWN, buff=0.25)
        self.add(n9)
        self.caption("Space-division: a microlens array splits the panel among the views.")
        self.caption("For a 3 × 3 light field, each view is only one-ninth of the panel: low resolution.")
        self.clear_all()

        f1c = img("fig1c.png", w=5.6).move_to([-3.4, 0.55, 0])
        self.play(FadeIn(f1c), run_time=0.5)
        self.add(label(f1c, "TDM-LFD: switchable shutter (paper Fig. 1c)", 22, DIM))
        t = ValueTracker(0.0)
        ti = lambda: int(t.get_value()) % 4
        cv = [views[v, u] for v, u in CORN]
        panel = arr_img(cv[0], 2.6).move_to([2.2, 0.75, 0])
        live(panel, lambda: cv[ti()])

        def apmask():
            m = np.zeros((L, L))
            v, u = CORN[ti()]
            m[v, u] = 1
            return aperture_rgb(m, px=10)
        ap = arr_img(apmask(), 1.5, nearest=True).move_to([5.3, 0.75, 0])
        live(ap, apmask)
        self.play(FadeIn(panel), FadeIn(ap), run_time=0.5)
        self.add(label(panel, "panel: one full view", 22), label(ap, "shutter", 22, CYAN))
        cnt = always_redraw(lambda: tx(f"time slot {ti() + 1} of 4", 22, AMBER).move_to([3.75, -0.95, 0]))
        self.add(cnt)
        self.caption("Time-division: show the views one after another, each at full panel resolution...")
        self.run(t, 4.0, 2.4)
        self.caption("...while a shutter at the aperture opens the matching window, in sync with the panel.")
        self.run(t, 12.0, 3.2)
        eq = tx(EQ8, 30).move_to([2.2, -1.75, 0])
        self.add(eq)
        self.caption("Persistence of vision adds the frames up: the eye sees a refocused image (paper Eq. 8).")
        self.run(t, 20.0, 2.8)
        self.clear_all()

        f2 = img("fig2_tdm.png", h=3.2).move_to([-2.9, 0.6, 0])
        self.play(FadeIn(f2), run_time=0.5)
        self.add(label(f2, "TDM-LFD refocused (paper Fig. 2a)", 22, RED))
        pros = VGroup(tx("✓ full panel resolution per view", 26, GREEN),
                      tx("✗ tiny window: little light gets through", 26, RED),
                      tx("✗ sparse views: aliasing in defocused regions", 26, RED)).arrange(DOWN, aligned_edge=LEFT, buff=0.3
                                                                                            ).move_to([3.1, 0.6, 0])
        self.add(pros[0])
        self.caption("TDM keeps full resolution, but the window is smaller than the view spacing, so little light passes.")
        self.add(pros[1])
        self.hold()
        self.add(pros[2])
        self.caption("And with sparse views, the defocused background aliases. Look at the shelf behind the green object.")
        self.clear_all()


# ================================================================== S04 Aperture = blur
class S04_Aperture(Base):
    def construct(self):
        self.title_card("Part 4", "Blur takes the shape of the aperture")
        self.header("4 · Blur takes the shape of the aperture")
        psfs = [np.load(gen(f"psf_{k}.npy")) for k in ("open", "corners", "coded")]
        ap9 = np.load(os.path.join(DATA, "fig9_apertures_9x9.npy")).astype(float)
        cm = np.zeros((L, L))
        for v, u in CORN:
            cm[v, u] = 1
        apert = [np.ones((L, L)), cm, ap9[0]]
        names = ["open aperture", "4 corner windows (TDM)", "coded aperture (paper Fig. 9)"]
        colsn = [WHITE, RED, GREEN]
        w = ValueTracker(0.0)
        wi = lambda: w.get_value() / (W20S[1] - W20S[0])
        xs = [-4.4, 0, 4.4]
        h = 2.9
        eq = tx(EQ9s, 30).move_to([0, 2.72, 0])
        self.add(eq)
        ims = []
        for x, st, nm, c, P in zip(xs, psfs, names, colsn, apert):
            m = arr_img(st[0], h).move_to([x, 0.55, 0])
            live(m, (lambda s_: (lambda: interp(s_, wi())))(st))
            ims.append(m)
            a = arr_img(aperture_rgb(P, px=10), 0.95, nearest=True).move_to([x, -1.55, 0])
            self.add(m, framed(m, DIM, 1.0), label(m, nm, 22, c), a)
        self.add(tx("aperture", 18, DIM).move_to([-6.35, -1.55, 0]), tx("blur of", 18, DIM).move_to([-6.35, 0.7, 0]),
                 tx("one point", 18, DIM).move_to([-6.35, 0.4, 0]))
        wl = always_redraw(lambda: tx(f"defocus: {w.get_value():.0f} waves", 22, AMBER).move_to([0, -2.4, 0]))
        note = toy_note("toy Fourier optics: 9 × 9 aperture, defocus phase W, small wavelength band; each panel normalised"
                        ).move_to([0, -2.85, 0])
        self.add(wl, note)
        self.caption("Fourier optics: the blur of one point (its PSF) is set by the aperture P and the defocus W.")
        self.hold()
        self.run(w, 18.0, 5.0, smooth)
        self.caption("Out of focus, a point spreads into a scaled copy of the aperture itself.")
        self.hold()
        self.caption("Open aperture: a smooth blur. Four tiny windows: four separate dots, the ghost copies.")
        self.run(w, 6.0, 2.0, smooth)
        self.run(w, 18.0, 2.0, smooth)
        self.caption("A coded aperture spreads each point into a pattern we get to design.")
        self.caption("So: show each frame through its own coded aperture, and choose the patterns well.")
        self.clear_all()


# ================================================================== S05 CTDM pipeline
class S05_CTDM(Base):
    def construct(self):
        self.title_card("Part 5", "Coded time-division multiplexing")
        self.header("5 · Coded time-division multiplexing")
        cviews = np.load(gen("corner_views.npy"))
        enc = np.load(gen("enc_images.npy"))
        P = np.load(gen("P_final.npy"))
        cum = np.load(gen("cycle_cum.npy"))
        # n views
        vs = Group(*[arr_img(cviews[i], 1.05).move_to([-6.0 + 0.16 * i, 0.55 - 0.16 * i, 0]) for i in range(4)])
        lv = label(vs, f"{I('n')} = 4 views", 22)
        box = Rectangle(width=0.8, height=3.4, color=AMBER, stroke_width=2.5).move_to([-3.95, 0.4, 0])
        bl = tx("encoding network  " + I("f"), 22, AMBER).rotate(PI / 2).move_to(box)
        a1 = arrow([-5.3, 0.4, 0], [-4.4, 0.4, 0])
        ys = [1.75, 0.65, -0.45, -1.55]
        encm = Group(*[arr_img(enc[i], 1.0).move_to([-2.3, y, 0]) for i, y in enumerate(ys)])
        apm = Group(*[arr_img(aperture_rgb(P[i], px=10), 0.95, nearest=True).move_to([-0.95, y, 0]) for i, y in enumerate(ys)])
        tl = VGroup(*[tx(f"{I('t')} = {i + 1}", 18, DIM).next_to(apm[i], RIGHT, buff=0.12) for i in range(4)])
        a2s = VGroup(*[arrow([-3.5, 0.4, 0], [-2.85, y, 0], buff=0.05, sw=2) for y in ys])
        le = tx("encoded images", 20).move_to([-2.3, 2.55, 0])
        la = tx("apertures " + PH + sub(I("t")), 20, CYAN).move_to([-0.95, 2.55, 0])
        self.play(FadeIn(vs), run_time=0.5)
        self.add(lv)
        self.caption(f"CTDM-LFD: an encoding network {I('f')} turns the {I('n')} sparse views into {I('k')} encoded images.")
        self.play(GrowArrow(a1), Create(box), run_time=0.5)
        self.add(bl, a2s)
        self.play(FadeIn(encm), run_time=0.5)
        self.add(le)
        self.hold()
        self.play(FadeIn(apm), run_time=0.5)
        self.add(la, tl)
        self.caption("Each encoded image is shown through its own coded aperture, one time slot after another.")
        # eye / sum
        t = ValueTracker(0.0)
        ti = lambda: int(np.clip(t.get_value(), 0, 3.999))
        R = arr_img(cum[0], 3.0).move_to([3.9, 0.4, 0])
        live(R, lambda: cum[ti()])
        a3 = arrow([0.4, 0.4, 0], [2.3, 0.4, 0])
        la3 = tx("lens + eye", 20, DIM).next_to(a3, UP, buff=0.1)
        hl = always_redraw(lambda: SurroundingRectangle(Group(encm[ti()], apm[ti()]), color=AMBER, buff=0.08, stroke_width=3))
        self.play(GrowArrow(a3), FadeIn(R), run_time=0.5)
        self.add(la3, hl, framed(R, DIM, 1.0))
        lr = always_redraw(lambda: label(R, f"retina, after {ti() + 1} of 4 frames", 22))
        self.add(lr)
        eq = tx(EQ10, 30).move_to([2.6, -2.25, 0])
        self.add(eq)
        self.caption("The eye adds the frames up: persistence of vision (paper Eq. 10).")
        self.run(t, 3.99, 4.0)
        self.caption("The encoded images are not views: they are whatever makes the sum look right at every focus.")
        self.hold()
        note = toy_note("images and apertures from the toy optimisation in Part 6; eye focused on the near star"
                        ).move_to([2.6, -2.8, 0])
        self.add(note)
        self.hold(0.5)
        self.clear_all()
        f3 = img("fig3_pipeline.png", w=11.5).move_to([0, 0.45, 0])
        self.play(FadeIn(f3), run_time=0.6)
        self.add(tx("paper Fig. 3: the CTDM-LFD pipeline (n = 4, k = 4)", 20, DIM).next_to(f3, DOWN, buff=0.2))
        self.caption(f"The paper's pipeline. Both the network {I('f')} and the apertures {PH} are learned, end to end.")
        self.clear_all()


# ================================================================== S06 Learning
class S06_Learning(Base):
    def construct(self):
        self.title_card("Part 6", "Learning the display end to end")
        self.header("6 · Learning the display end to end")
        e12 = tx(EQ12, 32).move_to([-3.2, 2.35, 0])
        d12 = VGroup(tx(f"{I('F')}{sub(I('c'))}: focal stack the display produces", 20, DIM),
                     tx(f"{I('F')}{sub(I('g'))}: focal stack of the full light field", 20, DIM)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(e12, DOWN, buff=0.25)
        e13 = tx(EQ13, 28).move_to([-3.2, -0.2, 0])
        e14 = tx(EQ14, 28).move_to([-3.2, -0.85, 0])
        self.add(e12, d12)
        self.caption("Train on focal stacks: compare what the eye would see at many focus depths with the ground truth.")
        wimg = np.load(gen("wimg.npy"))
        wmap = np.load(gen("wmap.npy"))
        hh = 1.9
        cells = []
        for r, (nm, y) in enumerate((("near focus", 1.3), ("far focus", -1.0))):
            a = arr_img(wimg[r], hh).move_to([2.35, y, 0])
            b = arr_img(wmap[r], hh).move_to([4.45, y, 0])
            cells += [a, b]
            self.add(tx(nm, 20, DIM).rotate(PI / 2).move_to([1.1, y, 0]))
        self.hold()
        self.add(e13, e14)
        self.play(*[FadeIn(m) for m in cells], run_time=0.5)
        self.add(label(cells[0], "ground truth", 20), label(cells[1], "weight 𝒲", 20, AMBER))
        self.add(toy_note("toy scene, β = 2").next_to(cells[3], DOWN, buff=0.12))
        self.caption("A focus measure (image gradients) weights each pixel, so in-focus detail counts more.")
        self.caption("With the paper's β = 2, the sharpest pixels weigh up to e² ≈ 7.4 times more.")
        self.clear_all()

        # --- toy optimisation
        sP = np.load(gen("train_snapP.npy"))
        sI = np.load(gen("train_snapImg.npy"))
        sIt = np.load(gen("train_snapIt.npy"))
        hist = np.load(gen("train_hist.npy"))
        ps = np.load(gen("psnr.npy"))
        it = ValueTracker(0.0)

        def sidx():
            v = it.get_value()
            j = int(np.searchsorted(sIt, v, side="right") - 1)
            j = int(np.clip(j, 0, len(sIt) - 2))
            w = np.clip((v - sIt[j]) / max(sIt[j + 1] - sIt[j], 1), 0, 1)
            return j, w
        aps = []
        for i in range(4):
            m = arr_img(aperture_rgb(sP[0, i], px=10), 1.3, nearest=True).move_to([-5.55 + 1.5 * (i % 2), 1.0 - 1.5 * (i // 2), 0])
            live(m, (lambda k: (lambda: aperture_rgb((lambda j, w: (1 - w) * sP[j, k] + w * sP[j + 1, k])(*sidx()), px=10)))(i))
            aps.append(m)
        R = arr_img(sI[0], 3.0).move_to([-0.9, 0.35, 0])
        live(R, lambda: (lambda j, w: ((1 - w) * sI[j] + w * sI[j + 1]).astype(np.uint8))(*sidx()))
        ax = Axes(x_range=[0, NIT, 1000], y_range=[14, 34, 5], x_length=4.3, y_length=3.0, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2, "include_numbers": False}).move_to([4.2, 0.35, 0])
        xt = VGroup(*[tx(str(v), 16, DIM).next_to(ax.c2p(v, 14), DOWN, buff=0.1) for v in (0, 1000, 2000, 3000)])
        yt = VGroup(*[tx(str(v), 16, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (15, 20, 25, 30)])
        xl = tx("iteration", 18, DIM).next_to(ax, DOWN, buff=0.4)
        yl = tx("toy focal-stack PSNR (dB)", 18, DIM).next_to(ax, UP, buff=0.12)
        bl = DashedLine(ax.c2p(0, ps[0]), ax.c2p(NIT, ps[0]), color=RED, stroke_width=2)
        bll = tx("baseline TDM", 16, RED).next_to(ax.c2p(NIT, ps[0]), UP, buff=0.06).shift(LEFT * 0.6)
        curve = always_redraw(lambda: ax.plot_line_graph(
            hist[: max(2, int(np.searchsorted(hist[:, 0], it.get_value())) + 1), 0],
            np.clip(hist[: max(2, int(np.searchsorted(hist[:, 0], it.get_value())) + 1), 1], 14, 34),
            add_vertex_dots=False, line_color=GREEN, stroke_width=3))
        self.play(*[FadeIn(m) for m in aps], FadeIn(R), Create(ax), run_time=0.7)
        self.add(tx("coded apertures", 20, CYAN).move_to([-4.8, 2.25, 0]), label(R, "eye focused on the near star", 20),
                 framed(R, DIM, 1.0), xt, yt, xl, yl, bl, bll, curve)
        cnt = always_redraw(lambda: tx(f"iteration {int(it.get_value())}", 22, AMBER).move_to([-4.8, -1.75, 0]))
        note = toy_note("toy optimisation: 4 images + 4 apertures fitted to one scene through the display model "
                        "(the paper trains a CNN f over a dataset)").move_to([0, -2.6, 0])
        self.add(cnt, note)
        self.caption("Toy: start from random apertures and flat images, and descend the gradient through the display model.")
        self.run(it, 60, 3.0, rate=lambda x: x)
        self.run(it, NIT - 1, 7.0, rate=lambda x: x ** 2.2)
        for m in aps + [R]:
            m.clear_updaters()
        self.caption("The apertures grow into large blobs in different parts of the aperture: more light, and blur instead of ghosts.")
        self.clear_all()

        # --- comparison with focus sweep
        st = [np.load(gen(f"sweep_{k}.npy")) for k in ("base", "fP", "gt")]
        names = ["baseline TDM-LFD (4 windows)", f"learned images + apertures {PH}", "ground truth (81 views)"]
        colsn = [RED, GREEN, DIM]
        vals = [f"{ps[0]:.1f} dB", f"{ps[2]:.1f} dB", ""]
        a = ValueTracker(D_NEAR)
        xs = [-4.4, 0, 4.4]
        h = 3.4
        ims = []
        for x, s_, nm, c, v in zip(xs, st, names, colsn, vals):
            m = arr_img(s_[0], h).move_to([x, 0.75, 0])
            live(m, (lambda q: (lambda: interp(q, sweep_index(a.get_value()))))(s_))
            ims.append(m)
            self.add(m, label(m, nm, 20, c))
            if v:
                self.add(tx("toy PSNR " + v, 18, c).next_to(m, DOWN, buff=0.12))
        rl, dot = focus_ruler(np.array([0, -1.85, 0]), 6.0, a.get_value)
        self.add(rl, dot)
        self.caption("Sweep the focus through the toy scene. The baseline breaks the blur into ghost copies.")
        self.run(a, D_FAR, 3.0, smooth)
        self.caption("The learned display keeps focus sharp and blurs smoothly, far closer to the ground truth.")
        self.run(a, D_NEAR, 3.0, smooth)
        self.run(a, D_FAR, 2.5, smooth)
        self.clear_all()


# ================================================================== S07 Results
class S07_Results(Base):
    def construct(self):
        self.title_card("Part 7", "Results")
        self.header("7 · Results")
        rows = [("Method  (PSNR / SSIM / LPIPS)", "INRIA-syn", "INRIA-real", "SIGGRAPH16"),
                (f"TDM-LFD baseline, {I('n')} = 4", "28.56 / 0.8485 / 0.2626", "26.39 / 0.8932 / 0.1434", "32.92 / 0.9285 / 0.1199"),
                (f"CTDM-LFD, {I('n')} = 4, {I('k')} = 4", "32.29 / 0.9374 / 0.1284", "30.21 / 0.9602 / 0.0736", "37.57 / 0.9705 / 0.03119"),
                (f"TDM-LFD baseline, {I('n')} = 9", "34.73 / 0.9688 / 0.1385", "36.32 / 0.9872 / 0.0363", "40.61 / 0.9858 / 0.0397"),
                (f"CTDM-LFD, {I('n')} = 9, {I('k')} = 8", "36.41 / 0.9858 / 0.0632", "39.97 / 0.9891 / 0.0186", "41.21 / 0.9739 / 0.0107")]
        xs = [-4.55, -0.9, 2.35, 5.45]
        tbl = VGroup()
        for r, row in enumerate(rows):
            cl = DIM if r == 0 else (GREEN if "CTDM" in row[0] else RED)
            for c, (s, x) in enumerate(zip(row, xs)):
                tbl.add(tx(s, 20, cl).move_to([x, 2.3 - 0.62 * r, 0]))
        hl = Line([-6.6, 2.0, 0], [6.6, 2.0, 0], color=DIM, stroke_width=1.5)
        self.add(tbl, hl)
        src = tx("paper Table V: focal stacks of the displayed light fields, PSNR (dB) ↑ / SSIM ↑ / LPIPS ↓", 18, DIM
                 ).move_to([0, -0.85, 0])
        self.add(src)
        self.caption("Paper Table V: on three light field datasets, CTDM-LFD beats the TDM baseline.")
        self.hold()
        head = fitw(tx(f"{I('n')} = 4 views:  +3.7 dB (INRIA-syn)  ·  +3.8 dB (INRIA-real)  ·  +4.7 dB (SIGGRAPH16)", 26, AMBER)
                    ).move_to([0, -1.75, 0])
        self.add(head)
        self.caption("With 4 views, about 4 dB more PSNR. Better on every metric, except SSIM for n = 9 on SIGGRAPH16.")
        self.clear_all()

        # Fig. 10 chairs
        w = 3.5
        xs = [-4.2, 0, 4.2]
        for x, m_, n_, c in zip(xs, ["base", "ours", "gt"], ["baseline TDM-LFD", "CTDM-LFD (ours)", "ground truth"], [RED, GREEN, DIM]):
            a = img(f"fig10_r0_{m_}.png", w=w).move_to([x, 0.95, 0])
            za = img(f"fig10_r0_{m_}_zA.png", w=1.7).move_to([x - 0.88, -1.75, 0])
            zb = img(f"fig10_r0_{m_}_zB.png", w=1.7).move_to([x + 0.88, -1.75, 0])
            self.add(a, za, zb, label(a, n_, 24, c))
        self.add(tx("INRIA-syn, n = 4 (paper Fig. 10)", 18, DIM).move_to([0, -2.85, 0]))
        self.caption("Synthetic scene: the baseline's defocused background shows stripes of aliasing; CTDM blurs it naturally.")
        self.clear_all()

        # Fig. 11 letters near / far
        w = 2.55
        xs = [-3.0, 0, 3.0]
        for r, (nm, y) in enumerate((("mom_near", 1.25), ("mom_far", -1.35))):
            for x, m_ in zip(xs, ["base", "ours", "gt"]):
                a = img(f"fig11_{nm}_{m_}.png", w=w).move_to([x, y, 0])
                self.add(a)
                if r == 0:
                    self.add(label(a, {"base": "baseline TDM-LFD", "ours": "CTDM-LFD (ours)", "gt": "ground truth"}[m_], 22,
                                   {"base": RED, "ours": GREEN, "gt": DIM}[m_]))
        self.add(tx("near focus", 20, DIM).move_to([-5.3, 1.25, 0]), tx("far focus", 20, DIM).move_to([-5.3, -1.35, 0]),
                 tx("INRIA Lytro,", 18, DIM).move_to([5.45, 0.15, 0]), tx("n = 4, k = 4", 18, DIM).move_to([5.45, -0.15, 0]),
                 tx("(paper Fig. 11)", 18, DIM).move_to([5.45, -0.45, 0]))
        self.caption("Real Lytro light field. Near focus: the baseline's far letters double up; CTDM's blur smoothly.")
        self.caption("Far focus: the letters are sharp for both, and CTDM's near blocks blur like the ground truth.")
        self.clear_all()

        # Fig. 17 focal sweeps (frame swaps)
        rows17 = ["chairs", "mom", "poppy"]
        stacks = [[np.asarray(Image.open(os.path.join(DATA, f"fig17_{r}_{j}.png")).convert("RGB")) for j in range(9)] for r in rows17]
        j = ValueTracker(0.0)
        xs = [-4.3, 0, 4.3]
        ims = []
        for x, stck in zip(xs, stacks):
            m = arr_img(stck[0], 3.4).move_to([x, 0.6, 0])
            live(m, (lambda s_: (lambda: s_[int(np.clip(round(j.get_value()), 0, 8))]))(stck))
            ims.append(m)
            self.add(m)
        lab = always_redraw(lambda: tx(f"focal plane {int(np.clip(round(j.get_value()), 0, 8)) + 1} of 9", 24, AMBER
                                       ).move_to([0, -1.55, 0]))
        self.add(lab, tx("CTDM-LFD continuous focal stacks, n = 4, k = 4 (paper Fig. 17)", 18, DIM).move_to([0, -2.05, 0]))
        self.caption("Paper Fig. 17: sweeping focus through CTDM-LFD light fields, front to back.")
        self.run(j, 8, 3.2)
        self.run(j, 0, 3.2)
        self.run(j, 8, 3.2)
        self.clear_all()

        # Zemax optics simulation
        lens = img("fig8_lens.png", w=4.6).move_to([-4.2, 1.05, 0])
        aps = Group(*[img(f"fig9_ap{i}.png", h=0.95) for i in range(4)]).arrange(RIGHT, buff=0.15).move_to([-4.2, -1.35, 0])
        self.add(lens, label(lens, "unit-magnification relay lens (Fig. 8)", 20, DIM), aps,
                 label(aps, "optimised 9 × 9 binary apertures (Fig. 9)", 20, DIM))
        n1 = img("fig13_near_ours.png", h=2.7).move_to([0.95, 0.5, 0])
        n2 = img("fig13_far_ours.png", h=2.7).move_to([4.35, 0.5, 0])
        self.add(n1, n2, label(n1, "near focus", 22, GREEN), label(n2, "far focus", 22, GREEN),
                 tx("Zemax optics simulation of CTDM-LFD (paper Fig. 13)", 18, DIM).move_to([2.65, -1.2, 0]))
        self.caption("A real optical design: a relay lens with only spherical surfaces and common glasses, aperture at the centre.")
        self.caption("Simulated in Zemax with binary apertures, the display refocuses with natural bokeh.")
        self.clear_all()


# ================================================================== S08 Ablations
class S08_Ablations(Base):
    def construct(self):
        self.title_card("Part 8", "What matters")
        self.header("8 · What matters")
        ax = Axes(x_range=[0, 3, 1], y_range=[26, 33, 1], x_length=5.4, y_length=3.6, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2, "include_numbers": False}).move_to([-3.4, 0.4, 0])
        yt = VGroup(*[tx(str(v), 16, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (26, 28, 30, 32)])
        vals = [(28.56, "baseline", RED), (29.85, f"learned {PH}", WHITE), (32.29, f"learned {I('f')} and {PH}", GREEN)]
        bars = VGroup()
        labs = VGroup()
        for i, (v, n, c) in enumerate(vals):
            p0, p1 = ax.c2p(i + 0.5, 26), ax.c2p(i + 0.5, v)
            b = Rectangle(width=1.1, height=p1[1] - p0[1], fill_color=c, fill_opacity=0.85, stroke_width=0)
            b.move_to((p0 + p1) / 2)
            bars.add(b)
            labs.add(tx(f"{v:.2f}", 20, c).next_to(b, UP, buff=0.08), tx(n, 18, c).next_to(p0, DOWN, buff=0.15))
        self.play(Create(ax), run_time=0.5)
        self.add(yt, tx("PSNR (dB)", 18, DIM).next_to(ax, UP, buff=0.12).shift(LEFT * 1.6))
        self.add(tx("paper Table VI, INRIA-syn, n = 4, k = 4, l = 9", 18, DIM).next_to(ax, DOWN, buff=0.65))
        for i in range(3):
            self.play(GrowFromEdge(bars[i], DOWN), run_time=0.4)
            self.add(labs[2 * i], labs[2 * i + 1])
            if i == 1:
                self.caption(f"Learning only the apertures gains 1.29 dB over the baseline.")
        self.caption(f"Learning the encoding network {I('f')} as well adds another 2.44 dB.")
        self.hold()
        rt = [("binary vs continuous apertures", "31.24 vs 32.29 dB"),
              ("mirrored (symmetric) vs free apertures", "31.06 vs 32.29 dB"),
              ("loss weight β = 0 / 2 / 5", "31.25 / 32.29 / 32.07 dB"),
              (f"{I('k')} = 4 / 8 / 16 frames ({I('n')} = 4)", "32.29 / 33.12 / 32.43 dB")]
        g = VGroup()
        for a_, b_ in rt:
            g.add(VGroup(tx(a_, 22), tx(b_, 22, AMBER)).arrange(DOWN, aligned_edge=LEFT, buff=0.08))
        g.arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([3.4, 0.4, 0])
        self.add(g[0])
        self.caption("Binary apertures (for a binary liquid-crystal shutter) cost about 1 dB; symmetric ones cost more.")
        self.add(g[1])
        self.hold()
        self.add(g[2])
        self.caption("β = 2 gives the best PSNR; β = 5 keeps a little more in-focus detail.")
        self.add(g[3])
        self.caption("More frames is not always better, and the required frame rate grows linearly with k.")
        self.clear_all()


# ================================================================== S09 Outlook
class S09_Outlook(Base):
    def construct(self):
        self.title_card("Part 9", "Summary")
        self.header("9 · Summary")
        summ = [
            "1   Coded time-division multiplexing: each frame through its own coded aperture",
            "2   Encoding network and apertures learned end to end with Fourier optics",
            "3   Focus-weighted focal-stack loss: sharp in focus, natural blur out of focus",
        ]
        sg = VGroup(*[fitw(tx(s, 26)) for s in summ]).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, 1.6, 0])
        for s in sg:
            self.add(s)
            self.wait(0.8)
        lim = VGroup(tx("Trade-off: more frames k needs a faster display and shutter", 22, DIM),
                     tx("Validated with Fourier-optics and Zemax simulations", 22, DIM)).arrange(DOWN, buff=0.12
                                                                                              ).move_to([0, -0.2, 0])
        self.add(lim)
        res = fitw(tx("+3.7 dB PSNR over TDM-LFD with 4 views (INRIA-syn), and correct defocus blur for accommodation", 24, AMBER)
                   ).move_to([0, -1.2, 0])
        self.add(res)
        self.caption("Coded apertures plus learning: light field displays whose blur finally looks right.")
        self.clear_all()
        self.remove(self.hdr)
        frames = [np.asarray(Image.open(os.path.join(DATA, "gif_frames", f"{i:03d}.png")).convert("RGB")) for i in (0, 1)]
        gif = arr_img(frames[1], 2.4).move_to([0, 1.55, 0])
        t1 = tx("Time-Division Multiplexing Light Field Display", 40, bold=True).move_to([0, -0.2, 0])
        t1b = tx("with Learned Coded Aperture", 30).move_to([0, -0.8, 0])
        t2 = tx("IEEE Transactions on Image Processing 32, 350–363 (2023)  ·  arXiv 2107.06205", 22).move_to([0, -1.45, 0])
        t4 = tx("Brian Chao*, Chang-Le Liu*, Homer H. Chen", 22, DIM).move_to([0, -2.05, 0])
        t5 = tx("National Taiwan University", 22, DIM).move_to([0, -2.47, 0])
        t6 = tx("* equal contribution", 18, DIM).move_to([0, -2.85, 0])
        self.add(gif, t1, t1b, t2, t4, t5, t6)
        self.wait(4.0)
