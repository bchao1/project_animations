"""Prototype of the RPWS toy wave-optics simulations (later pasted into build_assets()).
λ = 520 nm, 8 µm pixels, N = 512 grid, angular spectrum method.
Scene: soft-edged star (front, z_f = 6 mm) over a checkerboard plane (rear, z_b = 18 mm)."""
import numpy as np
import time
import matplotlib
from PIL import Image, ImageDraw, ImageFilter
import os

OUT = "/home/brianchc/project_animations/random_phase_wave_splatting/scratch/proto"
os.makedirs(OUT, exist_ok=True)
LAM, DX, N = 520e-9, 8e-6, 512
K = 2 * np.pi / LAM
FMAX = 1 / (2 * DX)
f = np.fft.fftfreq(N, DX)
FX, FY = np.meshgrid(f, f)
FZ = np.sqrt(np.maximum(1 / LAM ** 2 - FX ** 2 - FY ** 2, 0))


def H(z):
    return np.exp(1j * 2 * np.pi * FZ * z)


def asm(u, z):
    return np.fft.ifft2(np.fft.fft2(u) * H(z))


def pupil(cx=0.0, cy=0.0, r=0.5):
    return (((FX / FMAX - cx) ** 2 + (FY / FMAX - cy) ** 2) <= r ** 2).astype(float)


def star_mask(N, r0, r1, c, n=5, blur=1.5):
    im = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(im)
    pts = []
    for i in range(2 * n):
        r = r1 if i % 2 == 0 else r0
        a = -np.pi / 2 + i * np.pi / n
        pts.append((c[0] + r * np.cos(a), c[1] + r * np.sin(a)))
    d.polygon(pts, fill=255)
    im = im.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(im) / 255.0


def scene():
    y, x = np.mgrid[0:N, 0:N]
    sq = 20
    chk = (((x - N // 2) // sq + (y - N // 2) // sq) % 2).astype(float)
    cb = 0.18 + 0.72 * chk
    half = 140
    win = ((np.abs(x - N / 2 + 0.5) < half) & (np.abs(y - N / 2 + 0.5) < half)).astype(float)
    win = np.asarray(Image.fromarray((win * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))) / 255.0
    a_b = win                       # rear plane: opaque inside its window
    c_b = cb                        # rear intensity colour
    a_f = 0.97 * star_mask(N, 34, 80, (N / 2 - 10, N / 2 + 6))
    c_f = np.full((N, N), 0.95)
    return a_f, c_f, a_b, c_b


ZF, ZB = 6e-3, 18e-3


def smooth_slm(a_f, c_f, a_b, c_b):
    """GWS alpha wave blending (amplitude domain, front to back, phases matched at the SLM)."""
    uf = a_f * np.sqrt(c_f) * np.exp(1j * K * ZF)
    ub = (1 - a_f) * a_b * np.sqrt(c_b) * np.exp(1j * K * ZB)
    return asm(uf, -ZF) + asm(ub, -ZB)


def rpws_slm(a_f, c_f, a_b, c_b, rng):
    """RPWS: back to front, sqrt weights, sqrt transmittance mask at the front plane (paper Eqs. 4-5)."""
    pb = np.exp(1j * rng.uniform(-np.pi, np.pi, (N, N)))
    pf = np.exp(1j * rng.uniform(-np.pi, np.pi, (N, N)))
    g = np.sqrt(a_b * c_b) * pb                      # rear primitive
    g = asm(g, -(ZB - ZF))                           # to the front primitive's plane
    g = np.sqrt(1 - a_f) * g + np.sqrt(a_f * c_f) * pf
    return asm(g, -ZF)                               # to the SLM


def gwssqrt_slm(a_f, c_f, a_b, c_b, rng):
    """Ablation 'GWS with sqrt': sqrt weights but GWS's front-to-back transmittance applied on the rear plane."""
    pb = np.exp(1j * rng.uniform(-np.pi, np.pi, (N, N)))
    pf = np.exp(1j * rng.uniform(-np.pi, np.pi, (N, N)))
    uf = np.sqrt(a_f * c_f) * pf
    ub = np.sqrt((1 - a_f) * a_b * c_b) * pb
    return asm(uf, -ZF) + asm(ub, -ZB)


def view(u_slm_F, z, P):
    """Image at distance z from the SLM seen through pupil P (applied in the SLM's Fourier plane)."""
    return np.abs(np.fft.ifft2(u_slm_F * P * H(z))) ** 2


if __name__ == "__main__":
    t0 = time.time()
    a_f, c_f, a_b, c_b = scene()
    rng = np.random.default_rng(0)
    us = smooth_slm(a_f, c_f, a_b, c_b)
    Pe = pupil(r=0.5)
    C = slice(96, 416)
    def sv(a, name, g=1.0, m=None):
        m = m or np.percentile(a, 99.5)
        Image.fromarray((np.clip(a / m, 0, 1) ** g * 255).astype(np.uint8)).save(f"{OUT}/{name}.png")
    Fs = np.fft.fft2(us)
    for z, nm in [(ZF, "f"), (ZB, "b"), (12e-3, "m")]:
        sv(view(Fs, z, Pe)[C, C], f"smooth_{nm}")
    acc = {k: 0 for k in ["f", "b", "m"]}
    accs = {k: 0 for k in ["f", "b", "m"]}
    for t in range(24):
        ur = rpws_slm(a_f, c_f, a_b, c_b, rng)
        Fr = np.fft.fft2(ur)
        ug = gwssqrt_slm(a_f, c_f, a_b, c_b, rng)
        Fg = np.fft.fft2(ug)
        for z, nm in [(ZF, "f"), (ZB, "b"), (12e-3, "m")]:
            v = view(Fr, z, Pe)
            acc[nm] = acc[nm] + v
            if t == 0:
                sv(v[C, C], f"rand1_{nm}")
    for nm in acc:
        sv(acc[nm][C, C] / 24, f"rand24_{nm}")
    # spectra
    sv(np.log10(np.abs(np.fft.fftshift(Fs)) ** 2 + 1e-9), "spec_smooth", m=None)
    sv(np.log10(np.abs(np.fft.fftshift(Fr)) ** 2 + 1e-9), "spec_rand")
    # light-field views, focus on the front plane, small shifted pupil
    for s in [-0.7, 0.0, 0.7]:
        Pl = pupil(s, 0, 0.22)
        sv(view(Fs, ZF, Pl)[C, C], f"lf_smooth_{s:+.1f}", m=np.percentile(view(Fs, ZF, pupil(0, 0, 0.22)), 99.5))
        vr = 0
        vg = 0
        rng2 = np.random.default_rng(1)
        for t in range(12):
            vr = vr + view(np.fft.fft2(rpws_slm(a_f, c_f, a_b, c_b, rng2)), ZF, Pl)
            vg = vg + view(np.fft.fft2(gwssqrt_slm(a_f, c_f, a_b, c_b, rng2)), ZF, Pl)
        sv(vr[C, C] / 12, f"lf_rpws_{s:+.1f}")
        sv(vg[C, C] / 12, f"lf_gwssqrt_{s:+.1f}")
    print("done", time.time() - t0)
