"""Prototype of the toy MPI wave-optics simulation (colour, 3 wavelengths). Writes scratch/proto/*.png."""
import numpy as np
import scipy.fft as sfft
from PIL import Image, ImageDraw, ImageFilter
import os, time

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto")
os.makedirs(OUT, exist_ok=True)
LAMS = np.array([638e-9, 520e-9, 450e-9])
DX = 8e-6
N = 512
CROP = slice(80, 432)
ZL = np.array([4e-3, 9e-3, 14e-3, 20e-3])       # layer 1 (front) ... layer 4 (back)
TH_MAX = LAMS[2] / (2 * DX)                       # blue's max diffraction angle (rad) -> common angular band


def fft2(a):
    return sfft.fft2(a, workers=16)


def ifft2(a):
    return sfft.ifft2(a, workers=16)


f = np.fft.fftfreq(N, DX)
FX, FY = np.meshgrid(f, f)


def H(z, lam):
    fz = np.sqrt(np.maximum(1 / lam ** 2 - FX ** 2 - FY ** 2, 0))
    return np.exp(1j * 2 * np.pi * fz * z)


def pupil_ang(lam, cx=0.0, cy=0.0, r=0.4):
    """Pupil in angle space (fractions of blue's max angle), same physical pupil for every colour."""
    tx, ty = lam * FX / TH_MAX, lam * FY / TH_MAX
    band = (np.abs(FX) <= 1 / (2 * DX)) & (np.abs(FY) <= 1 / (2 * DX))
    return ((((tx - cx) ** 2 + (ty - cy) ** 2) <= r ** 2) & band).astype(float)


def poly_mask(pts, blur=1.5):
    im = Image.new("L", (N, N), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=255)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(blur))) / 255.0


def star(c, r0, r1, n=5, rot=-np.pi / 2):
    return [(c[0] + (r1 if i % 2 == 0 else r0) * np.cos(rot + i * np.pi / n),
             c[1] + (r1 if i % 2 == 0 else r0) * np.sin(rot + i * np.pi / n)) for i in range(2 * n)]


def toy_mpi():
    y, x = np.mgrid[0:N, 0:N].astype(float)
    cx = cy = N / 2
    # back: checkerboard wall
    sq = 28
    chk = (((x - cx) // sq + (y - cy) // sq) % 2)
    win = ((np.abs(x - cx) < 200) & (np.abs(y - cy) < 200)).astype(np.uint8) * 255
    win = np.asarray(Image.fromarray(win).filter(ImageFilter.GaussianBlur(2))) / 255.0
    c4 = np.stack([0.10 + 0.25 * chk, 0.22 + 0.35 * chk, 0.45 + 0.40 * chk], -1)
    a4 = win
    # layer 3: yellow disc
    r = np.hypot(x - (cx + 55), y - (cy - 50))
    a3 = np.clip((95 - r) / 2.0 + 0.5, 0, 1)
    c3 = np.stack([np.full((N, N), 1.0), np.full((N, N), 0.80), np.full((N, N), 0.20)], -1)
    # layer 2: green triangle
    a2 = poly_mask([(cx - 150, cy + 120), (cx + 60, cy + 120), (cx - 45, cy - 70)])
    c2 = np.stack([np.full((N, N), 0.30), np.full((N, N), 0.85), np.full((N, N), 0.45)], -1)
    # layer 1: red-orange star (soft-edged)
    a1 = 0.97 * poly_mask(star((cx + 70, cy + 70), 30, 72), blur=2.0)
    c1 = np.stack([np.full((N, N), 1.0), np.full((N, N), 0.42), np.full((N, N), 0.30)], -1)
    return [(c1, a1), (c2, a2), (c3, a3), (c4, a4)]


def over(layers):
    """Ray-optics 'over' compositing, back to front (on intensity)."""
    C = np.zeros((N, N, 3))
    for c, a in layers[::-1]:
        C = a[..., None] * c + (1 - a[..., None]) * C
    return C


def rgbd_from(layers):
    """Depth map = front-most layer with alpha > 0.5; RGBD slices = hard masks of the composited image."""
    C = over(layers)
    d = np.full((N, N), len(layers) - 1)
    for i in range(len(layers) - 1, -1, -1):
        d = np.where(layers[i][1] > 0.5, i, d)
    return C, d


def mpi_slm(layers, ch, rng):
    lam = LAMS[ch]
    u = np.zeros((N, N), complex)
    L = len(layers)
    for i in range(L - 1, -1, -1):
        c, a = layers[i]
        u = np.sqrt(1 - a) * u + np.sqrt(c[..., ch] * a) * np.exp(1j * rng.uniform(-np.pi, np.pi, (N, N)))
        dz = ZL[i] - (ZL[i - 1] if i > 0 else 0.0)
        u = ifft2(fft2(u) * H(-dz, lam))
    return u


def rgbd_slm(C, d, ch, rng):
    lam = LAMS[ch]
    u = np.zeros((N, N), complex)
    for i in range(len(ZL)):
        m = (d == i).astype(float)
        ui = np.sqrt(C[..., ch]) * m * np.exp(1j * rng.uniform(-np.pi, np.pi, (N, N)))
        u = u + ifft2(fft2(ui) * H(-ZL[i], lam))
    return u


def view(Us, z, cx=0.0, r=0.4):
    out = []
    for ch in range(3):
        F = fft2(Us[ch])
        out.append(np.abs(ifft2(F * pupil_ang(LAMS[ch], cx, 0, r) * H(z, LAMS[ch])))[CROP, CROP] ** 2)
    return np.stack(out, -1)


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(1)
    layers = toy_mpi()
    C, d = rgbd_from(layers)
    Image.fromarray((np.clip(C[CROP, CROP], 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)).save(f"{OUT}/gt.png")
    T = 12
    res = {}
    for name in ("mpi", "rgbd"):
        acc = {}
        for t in range(T):
            Us = [mpi_slm(layers, ch, rng) if name == "mpi" else rgbd_slm(C, d, ch, rng) for ch in range(3)]
            for key, z, cx, r in (("f1", ZL[0], 0, 0.4), ("f4", ZL[3], 0, 0.4), ("f2", ZL[1], 0, 0.4),
                                  ("lfL", ZL[0], -0.7, 0.25), ("lfR", ZL[0], 0.7, 0.25), ("lfC", ZL[0], 0, 0.25)):
                acc[key] = acc.get(key, 0) + view(Us, z, cx, r)
        res[name] = acc
    for name, acc in res.items():
        for key, v in acc.items():
            r = 0.25 if key.startswith("lf") else 0.4
            refk = T * np.pi * (r * TH_MAX / LAMS) ** 2 * DX ** 2      # flat random-phase spectrum through the pupil
            im = np.clip(v / refk, 0, 1) ** (1 / 2.2)
            Image.fromarray((im * 255).astype(np.uint8)).save(f"{OUT}/{name}_{key}.png")
    print("done", time.time() - t0)
