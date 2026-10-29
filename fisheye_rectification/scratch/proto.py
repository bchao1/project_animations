"""Prototype: procedural toy scene, equidistant fisheye warp, flow field, rectification. Writes scratch/proto_*.png."""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
import matplotlib

L = 3.3          # canvas half-extent (normalised units; the fisheye / rectified frame is [-1, 1])
PPU = 220        # canvas pixels per unit
KF = 1.4         # toy lens strength: r_d = atan(k r_u) / atan(k)


def draw_scene(seed=0, ss=2, sc=0.42, voff=0.7):
    """Procedural street facade full of straight lines. Scene units are scaled by sc into normalised coords."""
    n = int(2 * L * PPU) * ss
    s = PPU * ss
    X = lambda u: (u * sc + L) * s
    Y = lambda v: (L - (v - voff) * sc) * s
    im = Image.new("RGB", (n, n))
    d = ImageDraw.Draw(im)
    for j in range(n):                      # sky
        v = (L - j / s) / sc + voff
        t = np.clip((v + 0.2) / 3.5, 0, 1)
        c = (1 - t) * np.array([205, 225, 242]) + t * np.array([80, 140, 205])
        d.line([(0, j), (n, j)], fill=tuple(int(x) for x in c))
    ground = -0.95
    W = lambda k: max(1, int(k * ss * sc / 0.6))
    # side buildings (grey-blue, taller)
    for (a, b, top, colr) in ((-5.0, -2.35, 2.2, (120, 132, 150)), (2.35, 5.0, 2.6, (132, 124, 118)),
                               (-9, -5.0, 1.4, (150, 150, 140)), (5.0, 9, 1.8, (110, 120, 135))):
        d.rectangle([X(a), Y(top), X(b), Y(ground)], fill=colr)
        for u0 in np.arange(a + 0.2, b - 0.3, 0.5):
            for v0 in np.arange(ground + 0.35, top - 0.35, 0.55):
                d.rectangle([X(u0), Y(v0 + 0.32), X(u0 + 0.28), Y(v0)], fill=(55, 65, 80), outline=(215, 215, 210), width=W(4))
    # ground: pavement tiles
    d.rectangle([0, Y(ground), n, n], fill=(150, 146, 140))
    for v in np.arange(ground - 0.25, -9, -0.25):
        d.line([(0, Y(v)), (n, Y(v))], fill=(108, 104, 98), width=W(3))
    for u in np.arange(-9, 9.01, 0.25):
        d.line([(X(u), Y(ground)), (X(u), n)], fill=(108, 104, 98), width=W(3))
    top = 1.35
    d.rectangle([X(-2.2), Y(top), X(2.2), Y(ground)], fill=(178, 92, 70))
    d.rectangle([X(-0.42), Y(2.15), X(0.42), Y(top)], fill=(160, 80, 60))
    d.polygon([(X(-0.5), Y(2.15)), (X(0.5), Y(2.15)), (X(0), Y(2.5))], fill=(70, 80, 95))
    d.ellipse([X(-0.2), Y(1.95), X(0.2), Y(1.55)], fill=(235, 230, 215), outline=(60, 50, 45), width=W(4))
    d.line([(X(0), Y(1.75)), (X(0), Y(1.9))], fill=(40, 40, 40), width=W(4))
    d.line([(X(0), Y(1.75)), (X(0.1), Y(1.75))], fill=(40, 40, 40), width=W(4))
    for v in (top, 0.62, -0.12):
        d.rectangle([X(-2.25), Y(v + 0.04), X(2.25), Y(v - 0.04)], fill=(228, 214, 190))
    for u in np.arange(-2.2, 2.21, 0.55):
        d.rectangle([X(u - 0.03), Y(top), X(u + 0.03), Y(ground)], fill=(205, 180, 150))
    for u0 in np.arange(-2.2, 2.2 - 0.01, 0.55):
        for (v0, v1) in ((0.72, 1.22), (-0.02, 0.5)):
            a, b = u0 + 0.13, u0 + 0.42
            d.rectangle([X(a), Y(v1), X(b), Y(v0)], fill=(52, 70, 92), outline=(240, 232, 215), width=W(5))
            d.line([(X((a + b) / 2), Y(v1)), (X((a + b) / 2), Y(v0))], fill=(240, 232, 215), width=W(4))
            d.line([(X(a), Y((v0 + v1) / 2)), (X(b), Y((v0 + v1) / 2))], fill=(240, 232, 215), width=W(4))
    d.rectangle([X(-0.22), Y(-0.28), X(0.22), Y(ground)], fill=(90, 55, 40), outline=(240, 232, 215), width=W(5))
    d.line([(X(0), Y(-0.28)), (X(0), Y(ground))], fill=(240, 232, 215), width=W(4))
    for u0 in (-2.0, -1.2, 0.6, 1.4):
        d.rectangle([X(u0), Y(-0.3), X(u0 + 0.6), Y(-0.85)], fill=(70, 96, 110), outline=(240, 232, 215), width=W(5))
    for u in (-1.35, 1.35):
        d.rectangle([X(u - 0.025), Y(0.1), X(u + 0.025), Y(ground - 0.3)], fill=(35, 38, 42))
        d.ellipse([X(u - 0.08), Y(0.22), X(u + 0.08), Y(0.06)], fill=(255, 230, 150))
    im = im.resize((n // ss, n // ss), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def sample(img_or_canvas, u, v, extent, fill=0.0):
    """Bilinear sample an image covering [-extent, extent]^2 at normalised coords (u right, v up)."""
    n = img_or_canvas.shape[0]
    col = (u + extent) / (2 * extent) * n - 0.5
    row = (extent - v) / (2 * extent) * n - 0.5
    out = np.stack([map_coordinates(img_or_canvas[..., c], [row, col], order=1, mode="nearest")
                    for c in range(img_or_canvas.shape[2])], -1)
    return np.clip(out, 0, 1)


def grid(n):
    t = (np.arange(n) + 0.5) / n * 2 - 1
    u, v = np.meshgrid(t, -t)
    return u, v


def r_d(ru, k):
    return ru if k < 1e-6 else np.arctan(k * ru) / np.arctan(k)


def r_u(rd, k):
    if k < 1e-6:
        return rd
    th = np.clip(rd * np.arctan(k), 0, np.pi / 2 - 1e-3)
    return np.tan(th) / k


def fisheye(canvas, k, n=512):
    u, v = grid(n)
    rd = np.hypot(u, v) + 1e-9
    s = r_u(rd, k) / rd
    return sample(canvas, u * s, v * s, L)


def flow_true(n, k):
    """Rectifying flow in rectified coords: x~(p) = x(p + f_p), p + f_p = p * r_d(|p|)/|p|."""
    u, v = grid(n)
    r = np.hypot(u, v) + 1e-9
    s = r_d(r, k) / r
    return np.stack([u * (s - 1), v * (s - 1)], -1)


def warp(img, f):
    n = f.shape[0]
    u, v = grid(n)
    return sample(img, np.clip(u + f[..., 0], -0.999, 0.999), np.clip(v + f[..., 1], -0.999, 0.999), 1.0)


if __name__ == "__main__":
    cv = draw_scene()
    Image.fromarray((cv * 255).astype(np.uint8)).resize((640, 640)).save("scratch/proto_canvas.png")
    fe = fisheye(cv, KF)
    f = flow_true(512, KF)
    rec = warp(fe, f)
    gt = sample(cv, *grid(512), L)
    print("rect vs gt RMSE (0-255):", np.sqrt(((rec - gt) ** 2).mean()) * 255)
    row = np.concatenate([gt, fe, rec], 1)
    Image.fromarray((row * 255).astype(np.uint8)).save("scratch/proto_row.png")
