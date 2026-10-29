"""Prototype of the wave-optics assets (copied into the animation script once it looks right)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
from matplotlib import cm
import os

OUT = "/home/brianchc/project_animations/gaussian_wave_splatting/scratch/proto"
os.makedirs(OUT, exist_ok=True)
FONTB = "/home/brianchc/.local/share/fonts/cmunbx.ttf"

LAM = 520e-9
DX = 8e-6


def fgrid(N, dx=DX):
    f = np.fft.fftfreq(N, dx)
    return np.meshgrid(f, f)


def asm(u, z, lam=LAM, dx=DX):
    """Angular spectrum propagation by distance z (band-limited to propagating waves)."""
    N = u.shape[0]
    FX, FY = fgrid(N, dx)
    arg = 1 / lam ** 2 - FX ** 2 - FY ** 2
    prop = arg > 0
    fz = np.sqrt(np.maximum(arg, 0))
    H = np.where(prop, np.exp(1j * 2 * np.pi * fz * z), 0)
    return np.fft.ifft2(np.fft.fft2(u) * H)


def domain_rgb(u, gamma=0.6, amax=None):
    a = np.abs(u)
    amax = amax or np.percentile(a, 99.5)
    v = np.clip(a / amax, 0, 1) ** gamma
    h = (np.angle(u) + np.pi) / (2 * np.pi)
    hsv = np.stack([h, np.full_like(h, 0.85), v], -1)
    return (matplotlib.colors.hsv_to_rgb(hsv) * 255).astype(np.uint8)


def phase_rgb(phi, cmap="twilight"):
    c = matplotlib.colormaps[cmap]((phi + np.pi) / (2 * np.pi))
    return (c[..., :3] * 255).astype(np.uint8)


def save(a, name):
    Image.fromarray(a).save(os.path.join(OUT, name))


# ---------- two point sources (top view), Re{U e^{-jwt}}
def two_sources():
    W, H = 480, 270
    x = np.linspace(-8, 8, W)
    y = np.linspace(-4.5, 4.5, H)
    X, Y = np.meshgrid(x, y)
    k = 2 * np.pi / 0.9
    U = 0
    for sx in (-1.4, 1.4):
        r = np.sqrt((X - sx) ** 2 + (Y + 4.8) ** 2) + 1e-3
        U = U + np.exp(1j * k * r) / np.sqrt(r)
    re = np.real(U)
    lut = matplotlib.colormaps["RdBu_r"]
    save((lut(np.clip(re / 1.2 * 0.5 + 0.5, 0, 1))[..., :3] * 255).astype(np.uint8), "two_re.png")
    I = np.abs(U) ** 2
    save((matplotlib.colormaps["inferno"](np.clip(I / np.percentile(I, 99), 0, 1))[..., :3] * 255).astype(np.uint8), "two_I.png")


def text_img(N, s, size, pos=None):
    im = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONTB, size)
    bb = d.textbbox((0, 0), s, font=f)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    px = pos or ((N - w) // 2 - bb[0], (N - h) // 2 - bb[1])
    d.text(px, s, font=f, fill=255)
    return np.asarray(im) / 255.0


def star_img(N, r0, r1, n=5, width=4, center=None):
    im = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(im)
    cx, cy = center or (N / 2, N / 2)
    pts = []
    for i in range(2 * n + 1):
        r = r1 if i % 2 == 0 else r0
        a = -np.pi / 2 + i * np.pi / n
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    d.line(pts, fill=255, width=width, joint="curve")
    return np.asarray(im) / 255.0


def focus_sweep():
    N = 512
    A1 = star_img(N, 55, 130, width=5, center=(N * 0.33, N * 0.36))     # front
    A2 = text_img(N, "3D", 150)                                         # back
    A2 = np.roll(A2, (70, 80), (0, 1))
    z1, z2 = 12e-3, 36e-3
    Hf = asm(A1, -z1) + asm(A2, -z2)
    save(domain_rgb(Hf), "holo_complex.png")
    save(phase_rgb(np.angle(Hf)), "holo_phase.png")
    for z in (0, 6e-3, 12e-3, 24e-3, 36e-3, 45e-3):
        r = np.abs(asm(Hf, z)) ** 2
        save((np.clip(r / 1.0, 0, 1) ** 0.7 * 255).astype(np.uint8), f"rec_{z*1e3:.0f}mm.png")


def recording():
    N = 400
    x = (np.arange(N) - N / 2) * 2e-6
    X, Y = np.meshgrid(x, x)
    z = 5e-3
    k = 2 * np.pi / LAM
    r = np.sqrt((X - 0.0) ** 2 + Y ** 2 + z ** 2)
    O = np.exp(1j * k * r)
    R = np.exp(1j * 2 * np.pi * 60e3 * X)
    I = np.abs(O + R) ** 2
    g = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
    save(g(np.real(O) * 0.5 + 0.5), "rec_O.png")
    save(g(np.real(R) * 0.5 + 0.5), "rec_R.png")
    save(g(I / 4), "rec_I.png")


def gws_field(N, mu, su, sv, rx, ry, rz, lam=LAM, dx=DX):
    """Closed-form angular spectrum of a 2D Gaussian (Eq. 10), evaluated on the FFT grid,
    returned as the field at the SLM (z = 0) after propagating back by mu_z."""
    FX, FY = fgrid(N, dx)
    arg = 1 / lam ** 2 - FX ** 2 - FY ** 2
    ok = arg > 0
    FZ = np.sqrt(np.maximum(arg, 0))
    def Rx(a):
        c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def Ry(a):
        c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def Rz(a):
        c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = Rx(rx) @ Ry(ry) @ Rz(rz)
    Ri = R.T
    F = np.stack([FX, FY, FZ], 0).reshape(3, -1)
    Fo = (Ri @ F).reshape(3, N, N)
    # on-axis plane-wave illumination f_r = (0, 0, 1/lam): shifts the local spectrum
    fr = Ri @ np.array([0, 0, 1 / lam])
    S = np.diag([su, sv])
    q = np.einsum("ij,jmn->imn", S, Fo[:2] - fr[:2, None, None])
    fo_z = np.abs(Fo[2])
    detJ = np.where(FZ > 0, fo_z / np.maximum(FZ, 1e-9), 0)
    spec = 2 * np.pi * detJ * su * sv * np.exp(-2 * np.pi ** 2 * (q ** 2).sum(0))
    spec = spec * np.exp(-1j * 2 * np.pi * (FX * mu[0] + FY * mu[1]))      # lateral shift
    spec = spec * np.exp(-1j * 2 * np.pi * FZ * mu[2]) * ok                # back to the SLM
    return np.fft.ifft2(spec) / dx ** 2


def gws_demo():
    N = 512
    u = gws_field(N, (0.3e-3, -0.2e-3, 10e-3), 90e-6, 25e-6, 0.0, np.deg2rad(50), np.deg2rad(30))
    save(domain_rgb(u), "gws_slm.png")
    for z in (5e-3, 10e-3, 15e-3):
        r = np.abs(asm(u, z)) ** 2
        save((np.clip(r / r.max(), 0, 1) ** 0.6 * 255).astype(np.uint8), f"gws_rec_{z*1e3:.0f}.png")


if __name__ == "__main__":
    two_sources()
    focus_sweep()
    recording()
    gws_demo()
    print(sorted(os.listdir(OUT)))
